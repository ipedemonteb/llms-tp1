# Análisis Forense de la Causa Raíz del Sobreajuste en el Modelo Híbrido

## 1. Resumen Ejecutivo del Problema

Durante el entrenamiento del modelo híbrido simple (**Late Fusion / Concatenación directa** de Texto Transformer + Embeddings Tabulares), se observaron tres anomalías fundamentales:
1. **Sobreajuste Severo:** En todas las variantes de hiperparámetros y dimensiones ($d_{\text{model}} \in [16, 32, 64, 96]$), la métrica de entrenamiento alcanza $\text{Train PR-AUC} \approx 0.85 - 0.90$, mientras que en Test se estanca en $\text{Test PR-AUC} \approx 0.70 - 0.72$ ($\Delta \text{Gap} \approx 0.13 - 0.18$).
2. **Degradación respecto a la Rama Tabular Pura:** El modelo híbrido rinde **significativamente peor** en Test que el modelo puramente tabular ($0.7005$ vs $0.7452$), consumiendo 60 veces más parámetros (75k-288k vs 4.6k).
3. **Invarianza a la Reducción de Capacidad:** Achicar el Transformer a $d_{\text{model}} = 16$ (36k parámetros) redujo el tamaño del modelo pero no eliminó la memorización en Train.

---

## 2. La Señal Real en el Dataset (Análisis de Datos)

El análisis matemático de correlaciones con la variable objetivo `bought` revela la estructura latente del problema:

### A. Correlaciones con Variables Numéricas y de Marca
* **Variables Numéricas (`price`, `nutrition_score`, `volume`, `price_per_oz`, `net_weight_oz`):** Correlación lineal prácticamente nula con la compra ($|r| < 0.02$).
* **Variables de Marca (`brand`):** Tasas de compra homogéneas entre marcas ($\mu \approx 11\% - 17\%$).

### B. La Variable Determinante: `title_tag` (Nivel de Reputación)
La compra no está distribuida uniformemente: está concentrada casi exclusivamente en 4 categorías de reputación:

| `title_tag` / Reputación | Muestras en Train | Tasa Real de Compra (BTR) | Comportamiento |
| :--- | :---: | :---: | :--- |
| **Customer Favorite** | 343 | **68.5%** | Alta probabilidad de compra |
| **Best Seller** | 334 | **65.8%** | Alta probabilidad de compra |
| **#1 Pick** | 352 | **63.1%** | Alta probabilidad de compra |
| **Top Rated** | 338 | **62.4%** | Alta probabilidad de compra |
| **Well Reviewed, Shopper Favorite, etc.** | ~1.400 | **2.0% - 3.0%** | Baja probabilidad |
| **Otros 12 tags (Clearance Listing, Rarely Reordered...)** | ~4.200 | **0.0%** | Cero compras |

---

## 3. Asimetría de Representación: Tabular vs. Texto

### A. Procesamiento en la Rama Tabular (Por qué funciona con 0 Overfitting)
* `title_tag` entra como una codificación **One-Hot** directa (40 dimensiones binarias).
* Cuatro columnas binarias indican de forma determinística la pertenencia al grupo de alta compra.
* La cabeza clasificadora lineal solo necesita aprender 4 pesos positivos.
* **Resultado:** $\text{Test PR-AUC} = 0.7452 \pm 0.017$ con $\text{Train PR-AUC} = 0.7600$ ($\Delta = 0.015$, casi nulo sobreajuste).

### B. Procesamiento en la Rama de Texto (Por qué falla y sobreajusta)
1. **Limpieza del Título:** La función de preprocesamiento elimina el tag explícito del título (`clean_title` remueve `(Best Seller)`).
2. **Señal Enterrada en la Descripción:** La señal de reputación queda colocada como la última oración de un texto plantilla sintético de 50 palabras:
   > *"Frozen mac and cheese in a 10 oz package for online grocery orders. Listed under frozen and intended for frozen storage. **One of the most repurchased items in its aisle.** | Ingredients..."*
3. **Falla del Mean Pooling:** Al aplicar promedio simple sobre los 50 tokens, las 8 palabras clave de reputación se diluyen con 42 palabras de ruido contextual ("package", "online grocery", "ambient storage", "mac and cheese").
4. **Memorización de Nombres de Alimentos:** Para minimizar la pérdida en Train, el Transformer con auto-atención libre busca atajos y memoriza los nombres de productos específicos que aparecieron en ejemplos positivos durante el entrenamiento (ej. "mac and cheese", "brioche buns"). Al evaluar en Test con productos nuevos, estos tokens no coinciden y el vector latente $e_{\text{text}}$ inyecta ruido.

---

## 4. Dinámica de Fusión: Inanición de Gradientes (*Gradient Starvation*)

En la concatenación simple $[e_{\text{text}} \parallel e_{\text{tab}}] \to \text{ClassifierHead}$:
1. **Fase Temprana de Train:** La rama de texto, con alta capacidad paramétrica, memoriza rápidamente los tokens de Train y reduce el residuo de pérdida $(\hat{p} - y) \to 0$.
2. **Inanición de Gradientes:** Al anularse el residuo, el gradiente $\frac{\partial \mathcal{L}}{\partial \theta_{\text{tab}}} = (\hat{p} - y) \cdot W_{\text{head}} \cdot \frac{\partial e_{\text{tab}}}{\partial \theta_{\text{tab}}}$ se apaga prematuramente. La rama tabular deja de optimizar sus pesos.
3. **Fase de Inferencia en Test:** El texto produce un vector latente $e_{\text{text}}$ ruidoso por la presencia de vocabulario no visto. La cabeza clasificadora, condicionada a confiar en el texto durante Train, traslada ese ruido a la predicción final, degradando la señal pura del One-Hot tabular.

---

## 5. Matriz de Síntesis del Diagnóstico

```
┌──────────────────────────────┬────────────────────────────────────┬────────────────────────────────────┐
│ DIMENSIÓN DE ANÁLISIS        │ RAMA TABULAR                       │ RAMA DE TEXTO (TRANSFORMER)        │
├──────────────────────────────┼────────────────────────────────────┼────────────────────────────────────┤
│ Representación del Tag       │ One-Hot directo (40 binarios)      │ 3ra oración en texto de 50 tokens  │
│ Relación Señal / Ruido (SNR) │ Máxima (Señal pura sin ruido)      │ Muy baja (Diluida por Mean Pooling)│
│ Parámetros Libres            │ 311 params en embeddings           │ 68.000 a 277.000 params            │
│ Modo de Falla                │ Ninguno (Generaliza perfecto)      │ Memorización de nombres de comida  │
│ Efecto en Late Fusion        │ Desplazada por gradientes de texto │ Contamina la decisión en Test      │
└──────────────────────────────┴────────────────────────────────────┴────────────────────────────────────┘
```
