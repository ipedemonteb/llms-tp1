# Comparativa Técnica: Late Fusion vs. Cross-Attention Multimodal

**73.69 Large Language Models — Trabajo Práctico 1**  
**Ejercicio 2: Desarrollo del Sistema de Predicción de BTR**  
**Documento de Diseño:** Integración de Texto y Variables Tabulares

---

## 1. Introducción y Contexto

En el problema de predicción de **Buy Through Rate (BTR)** (`bought \in \{0, 1\}`), el dataset combina dos tipos de señales complementarias:
1. **Señales Textuales / Semánticas:** `title_clean`, `description` e `ingredients` (secuencias de longitud variable con dependencias lingüísticas y claims de marketing).
2. **Señales Tabulares / Estructuradas:** Variables numéricas continuas (`price`, `price_span`, `price_per_oz`, `volume`, `nutrition_score`) y categóricas (`category`, `brand`, `storage_type`, `title_tag`, etc.).

El desafío central de diseño arquitectónico radica en **cómo integrar de manera óptima la representación del texto extraída por el Transformer con las características tabulares**. Las dos alternativas principales en discusión son:

* **Alternativa 1: Fusión Tardía (*Late Fusion / Concatenación*)** — Fusión estática post-pooling.
* **Alternativa 2: Fusión Cruzada (*Cross-Attention Multimodal*)** — Fusión dinámica donde las variables tabulares condicionan la atención sobre los tokens del texto.

---

## 2. Flujo de Información y Esquemas Conceptuales

### 2.1. Alternativa 1: Late Fusion (Fusión Tardía)

En este enfoque, cada modalidad se procesa de forma completamente aislada e independiente en su propia rama hasta la última etapa, donde sus vectores representativos se colapsan y concatenan:

```
[ Secuencia de Tokens de Texto ]
               │
      Transformer Encoder
               │
    Pooling (Mean / CLS Token) ─────────> e_text  (vector fijo en R^d_text)
                                                 │
                                                 ├──> Concatenación [e_text || e_tab] ──> MLP Head ──> Sigmoid (BTR)
                                                 │
[ Categóricas ] ──> nn.Embedding ──────┐         │
                                       ├──> MLP ─> e_tab   (vector fijo en R^d_tab)
[ Numéricas ] ───> BatchNorm + Linear ─┘
```

> **Principio de Funcionamiento:** El texto se comprime en un único vector $e_{\text{text}}$ **sin tener conocimiento previo** del precio, categoría, marca o perfil del producto. La interacción multimodal ocurre únicamente en las capas densas finales.

---

### 2.2. Alternativa 2: Cross-Attention Multimodal

En este enfoque, se preserva la secuencia completa de representaciones de tokens producida por el Transformer. El vector tabular actúa como una "consulta" (*Query*) que busca activamente qué partes del texto son relevantes en función de las características del producto:

```
[ Secuencia de Tokens de Texto ]
               │
      Transformer Encoder
               │
  H_text = [h_1, h_2, ..., h_N] ───────> Keys (K) & Values (V) (en R^(N x d_k))
                                                │
                                                ▼
                                         Cross-Attention
                                                ▲
                                                │
[ Vector Tabular e_tab ] ──────────────> Query (Q) (en R^(1 x d_k))
                                                │
                                                ▼
                                          e_cross_attended (en R^d_model)
                                                │
                                                ├──> Concatenación [e_cross || e_tab] ──> MLP Head ──> Sigmoid (BTR)
                                                │
                                             e_tab
```

> **Principio de Funcionamiento:** La atención modula dinámicamente la importancia de cada palabra o frase de la descripción/título en función de las variables numéricas y categóricas del producto antes de tomar la decisión final.

---

## 3. Formulación Matemática de Ambos Métodos

### 3.1. Late Fusion
1. **Representación de Texto:**
   $$\mathbf{H} = \text{TransformerEncoder}(\mathbf{X}_{\text{tokens}}) \in \mathbb{R}^{N \times d_{\text{model}}}$$
   $$\mathbf{e}_{\text{text}} = \text{MeanPooling}(\mathbf{H}) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{h}_i \in \mathbb{R}^{d_{\text{model}}}$$
2. **Representación Tabular:**
   $$\mathbf{e}_{\text{tab}} = \text{MLP}_{\text{tab}}([\mathbf{e}_{\text{cat}} \,\|\, \mathbf{x}_{\text{num}}]) \in \mathbb{R}^{d_{\text{tab}}}$$
3. **Fusión y Predicción:**
   $$\mathbf{e}_{\text{fused}} = [\mathbf{e}_{\text{text}} \,\|\, \mathbf{e}_{\text{tab}}] \in \mathbb{R}^{d_{\text{model}} + d_{\text{tab}}}$$
   $$\hat{y} = \sigma\left(\mathbf{W}_{\text{out}} \cdot \text{GELU}(\mathbf{W}_1 \mathbf{e}_{\text{fused}} + \mathbf{b}_1) + b_{\text{out}}\right)$$

---

### 3.2. Cross-Attention
1. **Representación de Texto (Secuencia no colapsada):**
   $$\mathbf{H} = [h_1, h_2, \dots, h_N] \in \mathbb{R}^{N \times d_{\text{model}}}$$
2. **Representación Tabular (Query):**
   $$\mathbf{e}_{\text{tab}} \in \mathbb{R}^{d_{\text{tab}}}$$
3. **Proyecciones de Atención:**
   $$\mathbf{Q} = \mathbf{e}_{\text{tab}} \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k}$$
   $$\mathbf{K} = \mathbf{H} \mathbf{W}_K \in \mathbb{R}^{N \times d_k}, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V \in \mathbb{R}^{N \times d_v}$$
4. **Cálculo de Pesos y Agregación:**
   $$\alpha_i = \frac{\exp\left(\frac{\mathbf{Q} \mathbf{K}_i^T}{\sqrt{d_k}}\right)}{\sum_{j=1}^{N} \exp\left(\frac{\mathbf{Q} \mathbf{K}_j^T}{\sqrt{d_k}}\right)}$$
   $$\mathbf{e}_{\text{cross}} = \sum_{i=1}^{N} \alpha_i \mathbf{V}_i \in \mathbb{R}^{1 \times d_v}$$
5. **Predicción con Conexión Residual:**
   $$\mathbf{e}_{\text{fused}} = [\mathbf{e}_{\text{cross}} \,\|\, \mathbf{e}_{\text{tab}}]$$
   $$\hat{y} = \sigma(\text{MLP}_{\text{head}}(\mathbf{e}_{\text{fused}}))$$

---

## 4. Análisis Comparativo en Profundidad

### A. Capacidad Expresiva e Interacciones Cruzadas
* **Late Fusion:** Al promediar los estados ocultos de la secuencia antes de interactuar con las variables tabulares, se produce un "cuello de botella de información" (*information bottleneck*). Toda la descripción y el título quedan resumidos en un punto fijo del espacio latente. Si un término (ej. *"Sin TACC"*, *"Orgánico"*, *"Pack Familiar"*) adquiere relevancia solo cuando `allergens != None` o cuando `price_per_oz` es bajo, el MLP posterior debe inferir esa relación no lineal a partir de representaciones ya comprimidas.
* **Cross-Attention:** Permite un alineamiento de grano fino (*fine-grained multimodal alignment*). La representación del texto resultante es **específica y personalizada para el perfil del producto**. Para un producto caro, la atención enfatizará claims de procedencia o calidad; para un producto de limpieza o despensa, enfatizará volumen o rendimiento.

---

### B. Interpretabilidad y Explicabilidad (XAI)
* **Late Fusion:** Ofrece interpretabilidad limitada a la auto-atención interna del texto (cómo las palabras se relacionan sintácticamente entre sí), pero es una caja negra respecto a cómo influyen las variables numéricas y de negocio en dicha interpretación.
* **Cross-Attention:** **Altamente interpretable.** Los coeficientes de atención cruzada $\alpha_i$ permiten visualizar mapas de calor exactos sobre el texto del producto, respondiendo a la pregunta: *¿Qué palabras o frases del catálogo guiaron la decisión de compra dado el precio y categoría de este artículo?*

---

### C. Complejidad Computacional y Entrenamiento
* **Late Fusion:**
  * Menor cantidad de parámetros entrenables.
  * Costo computacional mínimo tras la salida del Transformer.
  * Muy estable numéricamente y de convergencia rápida con AdamW.
* **Cross-Attention:**
  * Introduce matrices de proyección adicionales ($W_Q, W_K, W_V$), capas de LayerNorm y posiblemente múltiples cabezales de atención cruzada (*Multi-Head Cross-Attention*).
  * Tiempo de cómputo por batch ligeramente superior debido a las operaciones matriciales sobre secuencias de longitud $N$.
  * Requiere ajuste cuidadoso de regularización (*Dropout* en la matriz de atención) para evitar que memorice combinaciones espurias en datasets medianos.

---

## 5. Matriz Resumen de Trade-Offs

| Dimensión de Análisis | Alternativa 1: Late Fusion | Alternativa 2: Cross-Attention |
| :--- | :--- | :--- |
| **Punto de Interacción Multimodal** | Al final (post-pooling en MLP Head) | Intermedio (a nivel de tokens con atención cruzada) |
| **Pérdida de Información Textual** | Media-Alta (debido al Mean/CLS Pooling) | Muy Baja (mantiene la secuencia completa $N \times d$) |
| **Capacidad de Acondicionamiento** | Estática / Global | Dinámica (guiada por el Query tabular) |
| **Interpretabilidad (XAI)** | 🔴 Baja (caja negra multimodal) | 🟢 Alta (pesos $\alpha_i$ visualizables por producto) |
| **Cantidad de Parámetros Adicionales** | 🟢 Mínima | 🟡 Moderada ($W_Q, W_K, W_V + \text{LayerNorm}$) |
| **Riesgo de Sobreajuste (*Overfitting*)** | 🟢 Bajo | 🟡 Medio (requiere Dropout de atención) |
| **Tiempo de Entrenamiento / Época** | 🟢 Más Rápido | 🟡 Ligeramente más lento (~15-25% extra) |
| **Facilidad de Depuración** | 🟢 Muy Sencilla | 🟡 Requiere control de dimensiones de tensores |

---

## 6. Estrategia Metodológica para el Trabajo Práctico

La relación entre ambas alternativas es **sinérgica y responde perfectamente a la consigna del trabajo**:

1. **Fase 1 — Establecer el Modelo Base (*Core Baseline*):**
   * Se implementa la **Alternativa 1 (Late Fusion)**.
   * Esto permite validar de punta a punta el pipeline de datos, tokenización, embeddings categóricos, normalización numérica, optimizador y el cómputo de métricas (**PR-AUC**, **ROC-AUC**, BCE Loss).
   * Al ser computacionalmente liviana y estable ($d_{\text{model}} < 100$), garantiza iteraciones rápidas y resultados confiables.

2. **Fase 2 — Estudio de Ablación y Variantes Arquitectónicas:**
   * Se evalúa la **Alternativa 2 (Cross-Attention)** como una variante directa del módulo de fusión, manteniendo idénticos el tokenizador, los hiperparámetros del Transformer y la rama tabular.
   * Se comparan formalmente las curvas de aprendizaje y las métricas en el conjunto de prueba (**Test Set temporal**):
     $$\Delta \text{PR-AUC} = \text{PR-AUC}_{\text{Cross-Attention}} - \text{PR-AUC}_{\text{Late-Fusion}}$$
     $$\Delta \text{ROC-AUC} = \text{ROC-AUC}_{\text{Cross-Attention}} - \text{ROC-AUC}_{\text{Late-Fusion}}$$

3. **Fase 3 — Valor para la Presentación Oral (25 min):**
   * Presentar la evolución conceptual de **Late Fusion $\to$ Cross-Attention**.
   * Exponer la justificación técnica del sesgo inductivo.
   * Mostrar los gráficos de mapas de atención cruzada como evidencia empírica de explicabilidad e interpretabilidad del modelo.
