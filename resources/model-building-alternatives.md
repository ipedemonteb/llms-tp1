# Alternativas de Diseño Arquitectónico y Modelado

**73.69 Large Language Models — Trabajo Práctico 1**  
**Ejercicio 2: Desarrollo del Sistema de Predicción de BTR**  
**Dataset:** `resources/clean_dataset.csv` (21 variables)

---

## 1. Contexto y Diagnóstico de Datos

El objetivo del sistema es predecir el **Buy Through Rate (BTR)**, modelado como la probabilidad de que un producto impreso sea comprado (`bought = 1`). A partir del análisis exploratorio (EDA) y la extracción de features (`clean_dataset.py`), las 20 variables de entrada se dividen naturalmente en tres modalidades con propiedades estadísticas y semánticas distintas:

```
                              ┌──> Texto Libre / Secuencias (title_clean, description, ingredients)
                              │
Variables de Entrada (X) ─────┼──> Variables Numéricas Continuas (price, price_span, price_per_oz, volume, etc.)
                              │
                              └──> Variables Categóricas (category, brand, storage_type, title_tag, etc.)
```

### Justificación del Sesgo Inductivo (*Inductive Bias*)

| Modalidad | Variables | Modelo Adecuado | Justificación Teórica y Práctica |
| :--- | :--- | :--- | :--- |
| **Texto y Secuencias** | `title_clean`, `description`, `ingredients` | **Transformer (Self-Attention)** | Captura dependencias sintácticas y semánticas de largo alcance, orden de palabras, claims de marketing e interacciones contextuales que los modelos tabulares planos no pueden inferir. |
| **Numéricas Continuas** | `price`, `price_span`, `price_per_oz`, `net_weight_oz`, `volume`, `nutrition_score`, `num_ingredients` | **MLP / Capas Lineales** o **GBDT (LightGBM/XGBoost)** | Gran eficiencia para aprender relaciones de escala, particiones por umbrales rígidos (ej. precios límite) y no linealidades continuas sin requerir tokenización artificial. |
| **Categóricas Estructuradas** | `category`, `brand`, `storage_type`, `unit_of_measure`, `title_tag`, `country_of_origin`, `day_of_week` | **Entity Embeddings (`nn.Embedding`)** o **Codificación Nativa GBDT** | Mapea categorías de baja y media cardinalidad a un espacio latente continuo y denso, evitando la dispersión del One-Hot Encoding y aprendiendo similitudes semánticas entre categorías. |

---

## 2. Alternativas Arquitectónicas Propuestas

A continuación se detallan 4 estrategias concretas de integración del Transformer con otros modelos para resolver la tarea de predicción.

---

### Alternativa 1: Red Neuronal Multimodal End-to-End con Fusión Tardía (*Late Fusion*)

Entrenamiento conjunto de un único grafo computacional en PyTorch donde las representaciones densas de texto y variables tabulares se extraen en ramas independientes y se fusionan mediante concatenación antes de la cabeza clasificadora.

```
[ title_clean + description + ingredients ]
                     │
            Tokenizador + Positional Encoding
                     │
          Transformer Encoder (L capas, H cabezales)
                     │
          Pooling (Mean / CLS Token) ───────────────> e_text (dim = d_text)
                                                             │
                                                             ├──> Concatenación [e_text || e_tab] ──> MLP Head ──> Sigmoid (BTR)
                                                             │
[ Categóricas ] ──> nn.Embedding ──────┐                     │
                                       ├──> MLP Tabular ───> e_tab (dim = d_tab)
[ Numéricas ] ───> BatchNorm + Linear ─┘
```

* **Rama de Texto:**
  * Secuencia de entrada: Concatenación formateada de `title_clean`, `description` e `ingredients`.
  * Encoder Transformer liviano ($d_{\text{model}} \in [64, 128]$, $L \in [2, 4]$ capas, $H \in [2, 4]$ cabezales).
  * Salida: Vector contextual $e_{\text{text}} \in \mathbb{R}^{d_{\text{text}}}$.
* **Rama Tabular:**
  * Categóricas proyectadas con `nn.Embedding` (dimensión según regla heurística $\min(50, \lfloor \text{cardinalidad}/2 \rfloor)$).
  * Numéricas estandarizadas procesadas con `nn.BatchNorm1d` + capas lineales con activación GELU y Dropout.
  * Salida: Vector tabular $e_{\text{tab}} \in \mathbb{R}^{d_{\text{tab}}}$.
* **Módulo de Fusión y Salida:**
  $$e_{\text{fused}} = [e_{\text{text}} \,\|\, e_{\text{tab}}] \in \mathbb{R}^{d_{\text{text}} + d_{\text{tab}}}$$
  $$\hat{y} = \sigma(\text{MLP}_{\text{head}}(e_{\text{fused}}))$$
* **Ventajas:** Optimización *end-to-end* con retropropagación directa, gradientes compartidos y simplicidad de implementación.

---

### Alternativa 2: Fusión Cruzada con Atención (*Cross-Attention Multimodal*)

En lugar de una simple concatenación estática, se utiliza un mecanismo de **Cross-Attention** donde las variables tabulares condicionan dinámicamente qué partes del texto son más relevantes para la decisión de compra.

```
[ Secuencia de Tokens de Texto ] ──> Transformer Encoder ──> K, V (Secuencia de estados ocultos)
                                                                    │
                                                                    ▼
                                                            Cross-Attention
                                                                    ▲
                                                                    │
[ Vector Tabular e_tab ] ──────────────────────────────────────────> Q (Query proyectado)
                                                                    │
                                                                    ▼
                                                            e_cross_attended ──> MLP Head ──> Sigmoid (BTR)
```

* **Mecanismo:**
  * El vector tabular $e_{\text{tab}}$ se proyecta como **Query ($Q$)**.
  * La secuencia completa de representaciones de tokens del texto actúa como **Keys ($K$)** y **Values ($V$)**:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
* **Intuición Semántica:**
  * Si el producto tiene un precio elevado (`price` alto), la atención puede enfocarse en términos de valor agregado como *"organic"*, *"imported"*, *"premium"*.
  * Si el producto pertenece a la categoría `Produce` y requiere almacenamiento `Refrigerated`, la atención ponderará frases sobre frescura o conservación.
* **Ventajas:** Mayor capacidad expresiva e interpretabilidad mediante la inspección de los mapas de atención cruzada.

---

### Alternativa 3: Enfoque Secuencial en Dos Etapas (*Two-Stage: Transformer Extractor + GBDT*)

Desacopla la extracción semántica profunda del modelado tabular, combinando un Transformer en PyTorch con algoritmos de árboles de decisión potenciados por gradiente (**LightGBM / XGBoost / CatBoost**).

```
PASO 1: MODELO DE TEXTO (PyTorch)
[ Textos del Producto ] ──> Transformer Encoder ──> Clasificador Previo
                                    │
                                    └── Extraer Embeddings (e_text) para cada producto

PASO 2: MODELO TABULAR Y FINAL (LightGBM / XGBoost)
[ Features Numéricas ] ────────┐
[ Features Categóricas ] ──────┼──> Dataset Extendido [X_tab || e_text] ──> GBDT ──> Predicción BTR
[ Embeddings de Texto e_text ] ─┘
```

* **Fase 1 (Extractor de Embeddings):**
  * Se entrena un Transformer en la tarea de predecir BTR utilizando únicamente texto (o texto + título) o mediante una tarea auto-supervisada.
  * Se extrae el vector latente $e_{\text{text}} \in \mathbb{R}^{d}$ de la capa de pooling para cada fila del dataset.
* **Fase 2 (Entrenamiento de GBDT):**
  * Se construye una matriz de características extendida: $X_{\text{final}} = [X_{\text{num}} \,\|\, X_{\text{cat}} \,\|\, e_{\text{text}}]$.
  * Se entrena un modelo LightGBM/XGBoost optimizando log-loss / binary cross-entropy.
* **Ventajas:** Aprovecha el rendimiento estado del arte de los modelos de árboles para datos tabulares y umbrales numéricos complejos, con un entrenamiento extremadamente rápido en la Fase 2.

---

### Alternativa 4: Ensamble por Stacking / Blending (*Model Stacking*)

Entrena dos o más modelos independientes especializados en sus respectivos dominios y combina sus probabilidades predichas mediante un meta-modelo o promedio ponderado.

```
                     ┌──> Modelo A: Transformer (Texto + Tags) ───> Prob_Transformer (p_A) ──┐
                     │                                                                       ├──> Meta-Clasificador ──> Predicción Final
[ clean_dataset ] ───┤                                                                       │    (Regresión Logística
                     └──> Modelo B: LightGBM (Tabular + Estadísticos) ──> Prob_GBDT (p_B) ───┘     o Promedio Ponderado)
```

* **Modelo A (Especialista en Lenguaje):** Red Transformer optimizada para extraer patrones en títulos, descripciones y claims.
* **Modelo B (Especialista Tabular):** LightGBM/XGBoost entrenado sobre variables numéricas, ratios (`price_per_oz`, `price_span`), categorías e información estadística de texto (longitud de descripción, cantidad de ingredientes).
* **Meta-Modelo:**
  $$\hat{y}_{\text{final}} = \sigma\left(w_0 + w_1 \cdot \text{logit}(p_A) + w_2 \cdot \text{logit}(p_B)\right)$$
* **Ventajas:** Muy robusto contra sobreajuste, alta interpretabilidad de la contribución relativa de cada modalidad.

---

## 3. Matriz Comparativa de Alternativas

| Dimensión | Alt. 1: Late Fusion End-to-End | Alt. 2: Cross-Attention | Alt. 3: Two-Stage (Transformer + GBDT) | Alt. 4: Stacking / Ensamble |
| :--- | :--- | :--- | :--- | :--- |
| **Rol del Transformer** | Extractor de texto end-to-end | Encoder de texto + Cross-Attention | Extractor de features congelado | Predictor independiente de texto |
| **Modelo Tabular** | Embeddings + MLP | Embeddings + MLP | GBDT (LightGBM/XGBoost) | GBDT / Random Forest |
| **Complejidad de Implementación** | Media | Alta | Media | Baja-Media |
| **Costo Computacional** | Bajo-Medio ($d_{\text{model}} < 100$) | Medio | Bajo (tras extraer embeddings) | Bajo |
| **Facilidad para Ablación** | **Excelente** | **Excelente** | Buena | Buena |
| **Adecuación a la Consigna** | **Directa y completa** | **Directa y avanzada** | Muy buena | Buena |

---

## 4. Plan de Experimentación y Estudio de Ablación

Para dar cumplimiento exhaustivo a los requisitos del Ejercicio 2, se estructurará la experimentación en las siguientes fases:

### 4.1. Baselines de Referencia (Punto de Partida)
1. **Baseline Tabular Puro (Sin Transformer):** Modelo MLP o LightGBM entrenado únicamente con variables numéricas y categóricas.
2. **Baseline Texto Puro (Solo Transformer):** Transformer procesando únicamente secuencias de texto libre, sin metadatos tabulares.

### 4.2. Modelo Principal Híbrido
* Implementación de la **Alternativa 1 (Late Fusion)** como arquitectura central de referencia.

### 4.3. Experimentos de Ablación (Comparación de Módulos)
* **Ablación de Fusión:** Late Fusion (concatenación) vs. Cross-Attention (Alternativa 2).
* **Ablación de Componentes del Transformer:**
  * Variación de $d_{\text{model}}$: 32 vs. 64 vs. 128.
  * Cantidad de capas ($L$): 1 vs. 2 vs. 4.
  * Cantidad de cabezales de atención ($H$): 2 vs. 4 vs. 8.
  * Mecanismo de Pooling: `Mean Pooling` vs. `Max Pooling` vs. `Token [CLS]`.
  * Inclusión vs. exclusión de *Positional Encoding*.
* **Ablación de Features:**
  * Impacto de agregar `ingredients` y `title_tag` a la entrada de texto.
  * Impacto de las variables derivadas (`price_per_oz`, `price_span`, `volume`).

### 4.4. Métricas de Evaluación
De acuerdo con la consigna, la evaluación se realizará sobre el conjunto de **Test** (con partición cronológica) monitoreando:
* **PR-AUC (Precision-Recall AUC):** Métrica primaria por el desbalance natural de compras ($BTR \ll 50\%$).
* **ROC-AUC:** Capacidad discriminativa global del ranking de probabilidades.
* **Binary Cross-Entropy Loss (BCE):** Monitoreo de curvas de entrenamiento y validación para control de *overfitting* / *underfitting*.
