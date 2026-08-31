# Configuración de Hiperparámetros — Estudio de Ablación del Transformer (Texto Puro)

**Ubicación:** `resources/documentation/transformer_ablation_setup.md`  
**Script Ejecutor:** `src/training/transformer_ablation.py`  
**Directorio de Figuras:** `results/figures/transformer_ablation/`  
**Directorio de CSVs:** `results/aggregate/transformer_ablation/`  

---

## 1. Parámetros Globales Comunes

Todos los experimentos se ejecutaron sobre la rama de texto aislada (**sin variables tabulares**) compartiendo los siguientes parámetros fijos:

* **Modalidad:** `use_text = True`, `use_tabular = False`
* **Tokenizador:** Byte-Level BPE (`vocab_size = 1720`, `max_length = 128`, `pad_token_id = 0`)
* **Optimizador:** `AdamW` (`lr = 0.001`, `weight_decay = 0.01`)
* **Función de Pérdida:** `BCEWithLogitsLoss`
* **Regularización:** `dropout = 0.1` en embeddings, atención y FFN
* **Activación FFN:** `GELU`
* **Topología:** Pre-LayerNorm (`norm_first = True`)
* **Cabeza Clasificadora:** `ClassifierHead(input_dim = d_model, hidden_dims = [64], dropout = 0.1, activation = "gelu")`
* **Batch Size:** `64`
* **Épocas Máximas:** `15`
* **Early Stopping:** `patience = 5` épocas (monitoreando PR-AUC de validación)
* **Semillas Aleatorias:** `seeds = [7, 42, 123]` (3 corridas independientes por combinación)

---

## 2. Detalle de Hiperparámetros por Estudio

### 🔹 Estudio 1: Dimensión Latente ($d_{\text{model}}$)
* **Figura generada:** `01_transformer_d_model_pr_auc.png`
* **CSV generado:** `transformer_ablation_d_model.csv`
* **Valores evaluados:**
  * $d_{\text{model}} \in \{32, 48, 64, 96\}$
* **Parámetros fijados en este estudio:**
  * Dimensión FFN: $d_{\text{ff}} = 4 \times d_{\text{model}} \in \{128, 192, 256, 384\}$
  * Cantidad de capas: $L = 2$
  * Cabezales de atención: $H = 4$
  * Positional Encoding: `"sinusoidal"`
  * Pooling: `"mean"`
* **Total de corridas:** 4 configuraciones $\times$ 3 semillas = **12 corridas**
* **Valor fijado para los siguientes estudios:** $d_{\text{model}} = 96$

---

### 🔹 Estudio 2: Cantidad de Capas ($L$)
* **Figura generada:** `02_transformer_layers_pr_auc.png`
* **CSV generado:** `transformer_ablation_layers.csv`
* **Valores evaluados:**
  * $L \in \{1, 2, 3, 4\}$
* **Parámetros fijados en este estudio:**
  * Dimensión latente: $d_{\text{model}} = 96$
  * Dimensión FFN: $d_{\text{ff}} = 384$
  * Cabezales de atención: $H = 4$
  * Positional Encoding: `"sinusoidal"`
  * Pooling: `"mean"`
* **Total de corridas:** 4 configuraciones $\times$ 3 semillas = **12 corridas**
* **Valor fijado para los siguientes estudios:** $L = 1$

---

### 🔹 Estudio 3: Cabezales de Atención ($H$)
* **Figura generada:** `03_transformer_heads_pr_auc.png`
* **CSV generado:** `transformer_ablation_heads.csv`
* **Valores evaluados:**
  * $H \in \{1, 2, 4, 8\}$
  * Dimensión por cabezal ($d_k = d_{\text{model}} / H$): $d_k \in \{96, 48, 24, 12\}$
* **Parámetros fijados en este estudio:**
  * Dimensión latente: $d_{\text{model}} = 96$
  * Dimensión FFN: $d_{\text{ff}} = 384$
  * Cantidad de capas: $L = 1$
  * Positional Encoding: `"sinusoidal"`
  * Pooling: `"mean"`
* **Total de corridas:** 4 configuraciones $\times$ 3 semillas = **12 corridas**
* **Valor fijado para los siguientes estudios:** $H = 1$

---

### 🔹 Estudio 4: Positional Encoding
* **Figura generada:** `04_transformer_pos_encoding_pr_auc.png`
* **CSV generado:** `transformer_ablation_pos_encoding.csv`
* **Valores evaluados:**
  * $\text{pos\_encoding} \in \{\text{"sinusoidal"}, \text{"learned"}, \text{"none"}\}$
* **Parámetros fijados en este estudio:**
  * Dimensión latente: $d_{\text{model}} = 96$
  * Dimensión FFN: $d_{\text{ff}} = 384$
  * Cantidad de capas: $L = 1$
  * Cabezales de atención: $H = 1$
  * Pooling: `"mean"`
* **Total de corridas:** 3 configuraciones $\times$ 3 semillas = **9 corridas**
* **Valor fijado para el siguiente estudio:** $\text{pos\_encoding} = \text{"sinusoidal"}$

---

### 🔹 Estudio 5: Estrategia de Pooling
* **Figura generada:** `05_transformer_pooling_pr_auc.png`
* **CSV generado:** `transformer_ablation_pooling.csv`
* **Valores evaluados:**
  * $\text{pooling} \in \{\text{"mean"}, \text{"cls"}, \text{"max"}\}$
* **Parámetros fijados en este estudio:**
  * Dimensión latente: $d_{\text{model}} = 96$
  * Dimensión FFN: $d_{\text{ff}} = 384$
  * Cantidad de capas: $L = 1$
  * Cabezales de atención: $H = 1$
  * Positional Encoding: `"sinusoidal"`
* **Total de corridas:** 3 configuraciones $\times$ 3 semillas = **9 corridas**

---

## 3. Resumen de Ejecución

$$\text{Total de corridas entrenadas} = 12 + 12 + 12 + 9 + 9 = \mathbf{54\text{ corridas}}$$
