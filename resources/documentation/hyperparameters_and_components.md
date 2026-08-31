# Catálogo de Hiperparámetros, Componentes y Algoritmos para Estudio de Ablación

**73.69 Large Language Models — Trabajo Práctico 1**  
**Sistema de Predicción de Buy Through Rate (BTR)**  
**Módulos del Sistema:** `src/hybrid_transformer/`, `src/training/`, `src/tokenizer/`

---

## 1. División Estricta de Variables por Modalidad

Todas las 21 columnas del dataset (`clean_dataset.csv`) están asignadas a una rama y tratamiento específico sin solapamientos:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          DATASET LIMPIO (21 COLUMNAS)                                            │
├───────────────────────┬─────────────────────────────┬────────────────────────┬───────────────────────────────────┤
│ MODALIDAD / RAMA      │ TRATAMIENTO ESPECÍFICO      │ CANTIDAD               │ VARIABLES EXACTAS                 │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 1. TEXTO TRANSFORMER  │ Tokenización Byte-Level BPE │ 3 campos               │ title_clean                       │
│                       │ + Self-Attention L capas    │                        │ description                       │
│                       │ + Positional Encoding       │                        │ ingredients                       │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 2. TABULAR CATEGÓRICA │ Entity Embeddings           │ 4 variables            │ brand                             │
│                       │ (nn.Embedding por variable) │                        │ category                          │
│                       │                             │                        │ country_of_origin                 │
│                       │                             │                        │ allergens                         │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 3. TABULAR CATEGÓRICA │ One-Hot Encoding            │ 4 variables            │ storage_type                      │
│                       │ (F.one_hot sin clase 0 OOV) │                        │ unit_of_measure                   │
│                       │                             │                        │ title_tag                         │
│                       │                             │                        │ day_of_week                       │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 4. TABULAR NUMÉRICA   │ log1p (en asimétricas)      │ 7 variables            │ price                             │
│                       │ + Estandarización Z-score   │                        │ price_span                        │
│                       │ + BatchNorm1d opcional      │                        │ price_per_oz (con log1p)          │
│                       │                             │                        │ net_weight_oz (con log1p)         │
│                       │                             │                        │ volume (con log1p)                │
│                       │                             │                        │ num_ingredients                   │
│                       │                             │                        │ nutrition_score                   │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 5. TABULAR DIRECTA    │ Passthrough float 0.0 / 1.0 │ 1 variable             │ has_allergens                     │
├───────────────────────┼─────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ 6. TARGET Y METADATA  │ Variable objetivo / Split   │ 2 columnas             │ bought (Target BTR y ∈ {0, 1})    │
│                       │                             │                        │ timestamp (Orden temporal)        │
└───────────────────────┴─────────────────────────────┴────────────────────────┴───────────────────────────────────┘
```

---

## 2. Mapa Global de la Arquitectura Multimodal (Conexión Directa)

Por defecto, **no hay un MLP intermedio en la rama tabular**: las representaciones tabulares (embeddings, one-hot, numéricas y directas) se concatenan directamente con la salida del Transformer $e_{\text{text}}$ y entran al **único MLP final (cabeza clasificadora)** para predecir el BTR.

```
                                  PRODUCTO DEL CATÁLOGO (Fila i)
                                                │
               ┌────────────────────────────────┴────────────────────────────────┐
               ▼                                                                 ▼
     SEÑALES TEXTUALES (3 campos)                                      VARIABLES TABULARES (16 vars)
 [title_clean | description | ingredients]                   [price, volume, brand, category, storage_type, ...]
               │                                                                 │
               ▼                                                                 ▼
    Byte-Level BPE Tokenizer                                            TabularPreprocessor
    (input_ids, attention_mask)                                       (log1p → z-score, entity/oh IDs)
               │                                                                 │
               ▼                                                                 ▼
     TextTransformerEncoder                                               TabularEncoder
  (Embeddings + PosEnc + L x Blocks)                                 (Embeddings + One-Hot + Numéricas)
               │                                                                 │
               ▼                                                                 ▼
      e_text (B, d_model)                                           e_tab crudo (B, input_dim ~ 59)
               │                                                                 │
               └───────────────────────────────┬─────────────────────────────────┘
                                               ▼
                                  Concatenación Directa (~123 dims)
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │   ÚNICO MLP FINAL    │
                                    │ (Linear->GELU->1 dim)│
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                     logit de compra (B,)
                                               │
                                               ▼
                                      BCEWithLogitsLoss
```

---

## 3. Catálogo de la Rama de Texto / Transformer

### 3.1. Componentes y Mecanismos Internos

| Componente | Implementación en Código | Alternativas / Algoritmos Soportados | Rol e Impacto en la Representación |
| :--- | :--- | :--- | :--- |
| **Tokenizador** | `src/tokenizer/bpe.py` (`ByteLevelBPETokenizer`) | • **Byte-Level BPE** con byte-fallback (cero tokens `[UNK]`).<br>• Variación de `vocab_size` (ej. 1024, 1720, 2048) reentrenando con `min_frequency`. | Convierte la secuencia `title_clean \| description \| ingredients` en tokens discretos preservando números, símbolos y mayúsculas. |
| **Lookup de Embeddings** | `TextTransformerEncoder.embedding` (`nn.Embedding`) | • `vocab_size` $\times$ `d_model`<br>• `padding_idx = 0` (gradiente nulo en `[PAD]`)<br>• Escalamiento opcional por $\sqrt{d_{\text{model}}}$ (`scale_embedding=True/False`) | Mapea cada token a un vector denso continuo. Multiplicar por $\sqrt{d_{\text{model}}}$ equilibra las activaciones con la codificación posicional. |
| **Positional Encoding (PE)** | `src/hybrid_transformer/text_encoder.py` (`PositionalEncoding`) | 1. **`"sinusoidal"`** (Vaswani et al., 2017): Funciones trigonométricas fijas ($0$ parámetros).<br>2. **`"learned"`**: Matriz `nn.Embedding(max_seq_len, d_model)` aprendida.<br>3. **`"none"`**: Identidad / Ablación sin orden temporal. | Inyecta noción de orden de palabras. Permite contrastar si el orden sintáctico es crucial o si una bolsa latente de palabras clave basta. |
| **Multi-Head Self-Attention (MHSA)** | `src/hybrid_transformer/text_encoder.py` (`MultiHeadSelfAttention`) | • Scaled Dot-Product: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$<br>• Máscara aditiva de padding ($M_{ij} = -10^9$ para `[PAD]`).<br>• $H$ cabezales en paralelo ($d_k = d_{\text{model}} / H$).<br>• Retorno opcional de mapas de atención para explicabilidad (XAI). | Modela relaciones semánticas no locales entre palabras distantes del título, descripción e ingredientes. |
| **Position-wise Feed-Forward (FFN)** | `src/hybrid_transformer/text_encoder.py` (`PositionwiseFeedForward`) | • Dos capas lineales: $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$<br>• Activación: **`"gelu"`** (continua/suave) vs. **`"relu"`** (clásica).<br>• Dropout en capas intermedias. | Aplica transformaciones no lineales posición por posición para proyectar a subespacios de mayor capacidad. |
| **Topología de Capas** | `src/hybrid_transformer/text_encoder.py` (`TransformerEncoderBlock`) | 1. **Pre-LayerNorm** (`norm_first=True`, default): $\text{LN}(x) \to \text{SubLayer} \to \text{Residual}$. Mayor estabilidad de gradientes.<br>2. **Post-LayerNorm** (`norm_first=False`, Vaswani 2017): $\text{SubLayer} \to \text{Residual} \to \text{LN}$. | Controla el flujo del gradiente a través de las conexiones residuales. |
| **Mecanismo de Pooling** | `TextTransformerEncoder.pool_sequence` | 1. **`"mean"`** (Default): Promedio enmascarado sobre tokens reales.<br>2. **`"cls"`**: Vector en posición 0 ($h_0$, token `[CLS]`).<br>3. **`"max"`**: Máximo sobre tokens válidos.<br>4. **`"none"`**: Secuencia completa $(B, T, d_{\text{model}})$ para Cross-Attention. | Colapsa la matriz temporal $\mathbf{H} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ en un único vector de producto $e_{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$. |

---

### 3.2. Tabla Maestra de Hiperparámetros de Texto

| Hiperparámetro | Flag CLI | Clave JSON | Tipo | Default | Valores / Rango para Ablación | Impacto y Comportamiento |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Dimensión Latente ($d_{\text{model}}$)** | `--d_model` | `"d_model"` | `int` | `64` | `[32, 48, 64, 96, 128]` | Dimensión del espacio vectorial. Debe ser divisible por `n_heads`. Consigna: $d_{\text{model}} < 100$. Mayor dimensión aumenta capacidad pero incrementa riesgo de overfitting. |
| **Cantidad de Cabezales ($H$)** | `--n_heads` | `"n_heads"` | `int` | `4` | `[1, 2, 4, 8]` | Número de subespacios de atención paralelos ($d_k = d_{\text{model}} / H$). Con $d=64$, $H=4 \implies d_k=16$. |
| **Dimensión Feed-Forward ($d_{\text{ff}}$)** | `--d_ff` | `"d_ff"` | `int` | `256` | `[128, 256, 512]` | Dimensión de la capa interna del FFN (habitualmente $4 \times d_{\text{model}}$ o $2 \times d_{\text{model}}$). |
| **Cantidad de Capas ($L$)** | `--num_layers` | `"num_layers"` | `int` | `2` | `[1, 2, 3, 4]` | Profundidad del encoder apilado. $L=1$ es un modelo ultraliviano; $L=4$ captura interacciones sintácticas de mayor nivel. |
| **Tipo de Positional Encoding** | `--pos_encoding` | `"pos_encoding"` | `str` | `"sinusoidal"` | `["sinusoidal", "learned", "none"]` | Mecanismo de posición. `"none"` permite medir el impacto de ignorar el orden secuencial. |
| **Estrategia de Pooling** | `--pooling` | `"pooling"` | `str` | `"mean"` | `["mean", "cls", "max"]` | Forma de colapsar la secuencia temporal en el vector $e_{\text{text}}$. Se fuerza a `"none"` automáticamente si `fusion="cross"`. |
| **Longitud Máxima Secuencia** | `--max_length` | `"max_length"` | `int` | `128` | `[64, 96, 128, 192, 256]` | Truncamiento y padding de la secuencia tokenizada. Cubre el 99% del catálogo con 128 tokens. |
| **Dropout de Texto** | `--dropout` | `"dropout"` | `float` | `0.1` | `[0.0, 0.05, 0.1, 0.2, 0.3]` | Probabilidad de dropout aplicada a embeddings, atención y capas FFN. Clave para regularizar con 7.000 ejemplos de train. |
| **Función de Activación** | *(Config Python)* | `"activation"` | `str` | `"gelu"` | `["gelu", "relu"]` | Función no lineal en la red Feed-Forward. |
| **Pre-LN vs Post-LN** | *(Config Python)* | `"norm_first"` | `bool` | `true` | `[true, false]` | Ubicación de la normalización por capa. `true` = Pre-LN, `false` = Post-LN. |
| **Escalamiento Embedding** | *(Config Python)* | `"scale_embedding"`| `bool` | `true` | `[true, false]` | Si multiplica por $\sqrt{d_{\text{model}}}$ antes de sumar posiciones. |

---

## 4. Catálogo de la Rama Tabular

### 4.1. Distribución de Variables y Preprocesamiento (`TabularPreprocessor`)

El preprocesamiento se ajusta **estrictamente sobre el split de entrenamiento** para evitar fuga de información (*data leakage*):

```
                                  VARIABLES TABULARES (16)
                                             │
      ┌───────────────────────┬──────────────┴──────────────┬───────────────────────┐
      ▼                       ▼                             ▼                       ▼
NUMÉRICAS (7)            DIRECTAS (1)               ENTITY EMBEDDINGS (4)     ONE-HOT ENCODING (4)
[price, price_span,     [has_allergens]             [brand, category,        [storage_type,
 price_per_oz,                                       country_of_origin,       unit_of_measure,
 net_weight_oz, volume,                              allergens]               title_tag, day_of_week]
 num_ingredients,                                           │                       │
 nutrition_score]                                           │                       │
      │                       │                             │                       │
      ▼                       ▼                             ▼                       ▼
log1p (en asimétricas)    Passthrough                nn.Embedding(card+1, d)    F.one_hot(card+1)[:, 1:]
      │                   (float 0/1)                       │                       │
      ▼                       │                             │                       │
Estandarización Z-score       │                             │                       │
      │                       │                             │                       │
      └───────────────────────┴──────────────┬──────────────┴───────────────────────┘
                                             ▼
                        Concatenación Directa: e_tab (~59 dims)
                     (Pasa directo al Clasificador Final sin MLP intermedio)
```

---

### 4.2. Tabla Maestra de Hiperparámetros Tabulares

| Hiperparámetro | Flag CLI | Clave JSON | Tipo | Default | Valores / Rango para Ablación | Impacto y Comportamiento |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **MLP Tabular Intermedio** | `--tabular_mlp`<br>`--no_tabular_mlp` | `"tabular_mlp"` | `bool` | `false` | `[false (directo), true (con MLP)]` | **`false`**: las variables tabulares se concatenan directamente a la cabeza.<br>**`true`**: pasan por un MLP previo que las comprime a `d_tab`. |
| **Dimensión de Salida ($d_{\text{tab}}$)** | `--d_tab` | `"d_tab"` | `int` | `32` | `[16, 32, 64]` | Dimensión de salida si `--tabular_mlp` está activo. Si está desactivado, la dimensión es la suma exacta de features (~59). |
| **Dimensiones de Entity Embeddings** | *(Automático)* | `"embedding_dims"`| `list[int]` | $\min(50, \lceil c/2 \rceil)$ | Auto o manual `[8, 8, 8, 8]` | Dimensión del vector de embedding asignado a cada variable categórica. |
| **Uso de BatchNorm en Numéricas** | *(Config Python)* | `"use_batchnorm"` | `bool` | `true` | `[true, false]` | Aplica `nn.BatchNorm1d` sobre las numéricas antes de concatenar. |
| **Dropout Tabular** | `--dropout` | `"dropout"` | `float` | `0.1` | `[0.0, 0.1, 0.2, 0.3]` | Regularización por dropout en la rama tabular. |

---

## 5. Catálogo del Módulo de Fusión y Cabeza Clasificadora

### 5.1. Mecanismos de Fusión Disponibles

```
                        ALTERNATIVA 1: LATE FUSION (Concatenación Directa)
               e_text: (B, d_model)  ──┐
                                       ├──► [e_text ‖ e_tab]: (B, d_model + input_dim ~ 123) ──► Único MLP Head ──► Logit
               e_tab : (B, input_dim)──┘

                    ALTERNATIVA 2: CROSS-ATTENTION (Atención Cruzada)
                                           e_tab: (B, input_dim)
                                                  │
                                                  ▼
                                         Query: Q = e_tab * W_Q (B, 1, H, d_k)
                                                  │
        H_text: (B, T, d_model) ──► Keys/Values: K, V (B, T, H, d_k)
                                                  │
                                                  ▼
                                       Scaled Dot-Product Attention
                                                  │
                                                  ▼
                                       e_cross: (B, d_model)
                                                  │
                                                  ├──► [e_cross ‖ e_tab] ──► Único MLP Head ──► Logit
                                       e_tab   ───┘
```

---

### 5.2. Cabeza Clasificadora (`ClassifierHead`) y Baselines Aislados

| Componente / Modo | Flag CLI | Descripción |
| :--- | :--- | :--- |
| **Modelo Multimodal Híbrido** | *(Default)* | Utiliza ambas ramas: `use_text=True`, `use_tabular=True`. |
| **Baseline Solo Texto** | `--no_tabular`<br>`"use_tabular": false` | Desactiva la rama tabular. Aísla el poder predictivo del Transformer puro sobre el texto. |
| **Baseline Solo Tabular** | `--no_text`<br>`"use_text": false` | Desactiva el Transformer. Aísla el poder predictivo del clasificador sobre variables tabulares. |
| **Capas Ocultas de la Cabeza** | *(Config Python)* `"hidden_dims"` | Capas del MLP final antes del logit escalar (default: `[64]`). Proyecta $d_{\text{fused}} \to 64 \to 1$. |

---

## 6. Catálogo de Entrenamiento, Optimización y Regularización

### 6.1. Componentes de Optimización y Loss

| Componente | Implementación | Alternativas / Configuración | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Criterio de Pérdida** | `nn.BCEWithLogitsLoss` | • `pos_weight=None` (sin ponderar)<br>• `pos_weight=w` ($w \approx \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 3.76$ para BTR de $\approx 21\%$) | Aplica el truco numérico **log-sum-exp** combinando sigmoide y binary cross-entropy, evitando saturación de gradientes. |
| **Optimizador** | `torch.optim.AdamW` | • `lr` ($10^{-4}$ a $3\times 10^{-3}$)<br>• `weight_decay` ($0.0$ a $0.1$) | Desacopla la regularización L2 del cálculo adaptativo de momentos de gradiente (estándar para Transformers). |
| **Clipping de Gradientes** | `clip_grad_norm_` | • `grad_clip = 1.0` (Default)<br>• `grad_clip = None` (Desactivado) | Previene explosión de gradientes en los primeros pasos de optimización. |
| **Early Stopping** | `src/training/trainer.py` | • Métrica de monitoreo: **PR-AUC de validación**<br>• `patience = 5` épocas<br>• Restaura automáticamente el mejor checkpoint. | Evita el sobreajuste y garantiza que el modelo evaluado en test sea el óptimo según la métrica objetivo. |

---

### 6.2. Tabla Maestra de Hiperparámetros de Entrenamiento

| Hiperparámetro | Flag CLI | Clave JSON | Tipo | Default | Valores / Rango para Ablación | Impacto en la Optimización |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Learning Rate ($\eta$)** | `--lr` | `"lr"` | `float` | `0.001` | `[1e-4, 3e-4, 5e-4, 1e-3, 3e-3]` | Velocidad de actualización. |
| **Weight Decay ($\lambda$)** | `--weight_decay` | `"weight_decay"` | `float` | `0.01` | `[0.0, 1e-4, 0.01, 0.05, 0.1]` | Penalización L2 desacoplada. Subir a $0.05$ combate activamente el sobreajuste. |
| **Dropout Global** | `--dropout` | `"dropout"` | `float` | `0.1` | `[0.0, 0.1, 0.2, 0.3]` | Probabilidad de desactivación neuronal. |
| **Tamaño de Lote ($B$)** | `--batch_size` | `"batch_size"` | `int` | `64` | `[16, 32, 64, 128]` | Muestras por paso de optimización. |
| **Épocas Máximas** | `--epochs` | `"epochs"` | `int` | `20` | `[10, 20, 30, 50]` | Límite superior de iteraciones completas. |
| **Paciencia Early Stopping** | `--patience` | `"patience"` | `int` | `5` | `[3, 5, 8, 10]` | Épocas consecutivas sin superar la mejor PR-AUC en validación. |
| **Peso Positivo Loss ($w_{\text{pos}}$)** | `--pos_weight` | `"pos_weight"` | `float` | `null` | `[null, 2.0, 3.76, 5.0]` | Ponderación de la clase positiva para desbalance. |
| **Semilla Aleatoria ($seed$)** | `--seed` | `"seed"` | `int` | `42` | `[7, 42, 123, 456, 999]` | Fija generadores de números aleatorios para significancia estadística. |
| **Dispositivo de Cómputo** | `--device` | `"device"` | `str` | `null` | `["cuda", "mps", "cpu"]` | Hardware de ejecución (`null` autodetecta GPU CUDA o Apple Silicon MPS). |

---

## 7. Guía de Ejecución y Plan de Experimentos de Ablación

```bash
# 1. Modelo Híbrido Directo (Sin MLP tabular intermedio - DEFAULT)
uv run python -m src.training.train --config late_fusion

# 2. Ablación: Comparar con MLP tabular intermedio activado
uv run python -m src.training.train --config late_fusion --tabular_mlp --run_name con_tabular_mlp

# 3. Baselines de referencia
uv run python -m src.training.train --config baseline_tabular
uv run python -m src.training.train --config baseline_texto

# 4. Fusión por Cross-Attention
uv run python -m src.training.train --config cross_attention

# 5. Ablación de Pooling y Positional Encoding
uv run python -m src.training.train --config late_fusion --pooling cls --run_name pool_cls
uv run python -m src.training.train --config late_fusion --pos_encoding none --run_name pos_none
```
