# Plan de Experimentación y Estudio de Ablación

**73.69 Large Language Models — Trabajo Práctico 1**  
**Sistema de Predicción de Buy Through Rate (BTR)**  
**Ubicación:** `resources/documentation/ablation_study_plan.md`

---

## 1. Principio Metodológico: Control de Variables

Para que los resultados sean interpretables, científicamente válidos y presentables de forma clara en la exposición oral, se aplica la metodología **One-Factor-at-a-Time (OFAT)**:

1. Se define un **Modelo Ancla de Referencia** (*Anchor Baseline*).
2. Se altera **un único eje o hiperparámetro por experimento**, manteniendo constantes todos los demás.
3. Se selecciona la mejor variante según la métrica primaria (**PR-AUC en validación**) y se utiliza como nueva base para la siguiente etapa.
4. Se verifica la significancia estadística final mediante **múltiples semillas aleatorias** (*multi-seed*).

```
┌───────────────────────────┐
│ Etapa 1: Baselines & Ancla│ ──► Aislamiento del aporte de cada modalidad
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Etapa 2: Capacidad (d, L) │ ──► Dimensión latente y profundidad óptimas (<100)
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Etapa 3: Mecanismos MHSA  │ ──► Cabezales (H), Positional Encoding y Pooling
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Etapa 4: Fusión Multimodal│ ──► Late Fusion (estática) vs Cross-Attention (dinámica)
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Etapa 5: Robustez y Seeds │ ──► Regularización y significancia estadística (5 seeds)
└───────────────────────────┘
```

---

## 2. Roadmap Detallado en 5 Etapas

### 1️⃣ Etapa 1: Establecer los Baselines y el Modelo Ancla

**Objetivo:** Medir el piso de rendimiento de cada modalidad por separado antes de realizar cualquier ajuste fino.

| Experimento | Configuración | Justificación / Pregunta que Responde |
| :--- | :--- | :--- |
| **B1: Solo Tabular** | `use_text: false`, `use_tabular: true` | ¿Cuánto predicen las variables estructuradas (precio, marca, categoría, storage) sin procesar texto? |
| **B2: Solo Texto** | `use_text: true`, `use_tabular: false` | ¿Cuánto predice el Transformer leyendo únicamente título, descripción e ingredientes? |
| **B3: Híbrido Base (Ancla)** | `use_text: true`, `use_tabular: true`, `d_model: 64`, `L: 2`, `H: 4`, `fusion: "late"` | ¿La combinación multimodal supera significativamente a ambas modalidades aisladas? |

---

### 2️⃣ Etapa 2: Capacidad y Escala del Transformer

**Objetivo:** Encontrar el tamaño óptimo del espacio latente y la profundidad del encoder antes de entrar en sobreajuste (*overfitting*), respetando la consigna ($d_{\text{model}} < 100$).

#### A. Dimensión Latente ($d_{\text{model}}$)
* *Fijar:* `num_layers = 2`, `n_heads = 4`, `pooling = "mean"`.
* *Variar:* `d_model ∈ [32, 48, 64, 96]`.
* *Análisis:* Evaluar si $d=32$ subrepresenta el vocabulario o si $d=96$ incrementa el sobreajuste sin ganar PR-AUC.

#### B. Cantidad de Capas ($L$ / `num_layers`)
* *Fijar:* $d_{\text{model}}^*$ (ganador del paso A), `n_heads = 4`.
* *Variar:* `num_layers ∈ [1, 2, 3, 4]`.
* *Análisis:* ¿El texto de e-commerce requiere abstracción jerárquica profunda ($L \ge 3$) o un encoder liviano ($L=1$ o $L=2$) generaliza mejor?

---

### 3️⃣ Etapa 3: Mecanismos Internos de la Atención

**Objetivo:** Evaluar cómo influyen las decisiones arquitectónicas del Transformer sobre la secuencia de texto.

#### A. Cantidad de Cabezales de Atención ($H$ / `n_heads`)
* *Fijar:* $d_{\text{model}}^*$ y $L^*$ óptimos.
* *Variar:* `n_heads ∈ [1, 2, 4, 8]` (asegurando $d_k = d_{\text{model}} / H$).
* *Análisis:* Comparar Single-Head ($H=1$) contra Multi-Head ($H>1$). Demuestra si proyectar a múltiples subespacios semánticos en paralelo es crucial para el problema.

#### B. Positional Encoding (`pos_encoding`)
* *Variar:* `["sinusoidal", "learned", "none"]`.
* *Análisis:* Con `pos_encoding = "none"`, el Transformer se convierte en un *Set / Bag of Words Transformer*. ¿El orden secuencial de las palabras aporta señal predictiva o las palabras clave aisladas dominan la decisión de compra?

#### C. Estrategia de Pooling (`pooling`)
* *Variar:* `["mean", "cls", "max"]`.
* *Análisis:* Determinar la forma más efectiva de colapsar la matriz temporal $\mathbf{H} \in \mathbb{R}^{B \times T \times d}$ en el vector $e_{\text{text}}$.

---

### 4️⃣ Etapa 4: Mecanismo de Fusión Multimodal

**Objetivo:** Comparar la interacción estática vs. dinámica entre modalidades.

```
                    ALTERNATIVA 1: LATE FUSION (Concatenación)
  e_text (B, d_model) ──┐
                        ├──► [e_text ‖ e_tab] ──► MLP Head ──► Logit BTR
  e_tab  (B, input_dim) ──┘

                 ALTERNATIVA 2: CROSS-ATTENTION MULTIMODAL
  e_tab (Query) ──────┐
                      ├──► Scaled Dot-Product ──► e_cross ──► [e_cross ‖ e_tab] ──► Logit BTR
  H_text (Keys/Values)┘
```

* **Late Fusion:** Fusión simple post-pooling. El texto se comprime a ciegas sin conocer las variables del producto.
* **Cross-Attention:** Las variables tabulares actúan como consulta (*Query*) para atender de forma selectiva tokens del texto (ej. claims relevantes según la categoría o el precio).
* *Métricas a comparar:* $\Delta\text{PR-AUC} = \text{PR-AUC}_{\text{cross}} - \text{PR-AUC}_{\text{late}}$ y mapas de atención para explicabilidad (XAI).

---

### 5️⃣ Etapa 5: Regularización, Optimización y Significancia Estadística

**Objetivo:** Garantizar la generalización y demostrar solidez estadística para la presentación.

1. **Regularización y Dropout:**
   * Probar: `dropout ∈ [0.0, 0.1, 0.2]`, `weight_decay ∈ [0.0, 0.01, 0.05]`.
   * Monitorear la brecha $\text{Loss}_{\text{train}} - \text{Loss}_{\text{val}}$.
2. **Evaluación Multi-Seed (5 semillas):**
   * Semillas: `[7, 42, 123, 456, 999]`.
   * Correr el **Modelo Ganador** vs **Baseline Tabular** vs **Baseline Texto**.
   * Reportar media $\pm$ desvío estándar ($\mu \pm \sigma$) en Test Set y realizar test de significancia estadística (test $t$ pareado o Wilcoxon).

---

## 3. Matriz Maestra de Experimentos

| ID | Nombre de Corrida | Variable Alterada | Valor | $d_{\text{model}}$ | $L$ | $H$ | Fusión | PosEnc | Pooling |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1.1** | `base_tabular` | Modalidad | Solo Tabular | - | - | - | - | - | - |
| **E1.2** | `base_texto` | Modalidad | Solo Texto | 64 | 2 | 4 | - | `sinusoidal` | `mean` |
| **E1.3** | `anchor_hibrido` | Modelo Ancla | Ambos | 64 | 2 | 4 | `late` | `sinusoidal` | `mean` |
| **E2.1** | `dmodel_32` | $d_{\text{model}}$ | 32 | **32** | 2 | 4 | `late` | `sinusoidal` | `mean` |
| **E2.2** | `dmodel_48` | $d_{\text{model}}$ | 48 | **48** | 2 | 4 | `late` | `sinusoidal` | `mean` |
| **E2.3** | `dmodel_96` | $d_{\text{model}}$ | 96 | **96** | 2 | 4 | `late` | `sinusoidal` | `mean` |
| **E2.4** | `layers_1` | Capas $L$ | 1 | $d^*$ | **1** | 4 | `late` | `sinusoidal` | `mean` |
| **E2.5** | `layers_3` | Capas $L$ | 3 | $d^*$ | **3** | 4 | `late` | `sinusoidal` | `mean` |
| **E3.1** | `heads_1` | Cabezales $H$ | 1 (Single) | $d^*$ | $L^*$ | **1** | `late` | `sinusoidal` | `mean` |
| **E3.2** | `heads_8` | Cabezales $H$ | 8 | $d^*$ | $L^*$ | **8** | `late` | `sinusoidal` | `mean` |
| **E3.3** | `pos_none` | Pos. Encoding | `"none"` | $d^*$ | $L^*$ | $H^*$ | `late` | **`none`** | `mean` |
| **E3.4** | `pos_learned` | Pos. Encoding | `"learned"` | $d^*$ | $L^*$ | $H^*$ | `late` | **`learned`** | `mean` |
| **E3.5** | `pool_cls` | Pooling | `"cls"` | $d^*$ | $L^*$ | $H^*$ | `late` | $P^*$ | **`cls`** |
| **E3.6** | `pool_max` | Pooling | `"max"` | $d^*$ | $L^*$ | $H^*$ | `late` | $P^*$ | **`max`** |
| **E4.1** | `fusion_cross` | Fusión | `"cross"` | $d^*$ | $L^*$ | $H^*$ | **`cross`**| $P^*$ | `none` |
| **E5.1-5**| `final_seeds` | Semillas | 5 seeds | $d^*$ | $L^*$ | $H^*$ | $F^*$ | $P^*$ | $\text{Pool}^*$ |

---

## 4. Comandos de Ejecución con la CLI

```bash
# -------------------------------------------------------------
# ETAPA 1: BASELINES Y MODELO ANCLA
# -------------------------------------------------------------
uv run python -m src.training.train --config baseline_tabular --run_name base_tabular
uv run python -m src.training.train --config baseline_texto   --run_name base_texto
uv run python -m src.training.train --config late_fusion       --run_name anchor_hibrido

# -------------------------------------------------------------
# ETAPA 2: CAPACIDAD (d_model y capas)
# -------------------------------------------------------------
uv run python -m src.training.train --config late_fusion --d_model 32 --d_ff 128 --run_name dmodel_32
uv run python -m src.training.train --config late_fusion --d_model 48 --d_ff 192 --run_name dmodel_48
uv run python -m src.training.train --config late_fusion --d_model 96 --d_ff 384 --run_name dmodel_96
uv run python -m src.training.train --config late_fusion --num_layers 1 --run_name layers_1
uv run python -m src.training.train --config late_fusion --num_layers 3 --run_name layers_3

# -------------------------------------------------------------
# ETAPA 3: MECANISMOS DE ATENCIÓN
# -------------------------------------------------------------
uv run python -m src.training.train --config late_fusion --n_heads 1 --run_name heads_1
uv run python -m src.training.train --config late_fusion --n_heads 8 --run_name heads_8
uv run python -m src.training.train --config late_fusion --pos_encoding none    --run_name pos_none
uv run python -m src.training.train --config late_fusion --pos_encoding learned --run_name pos_learned
uv run python -m src.training.train --config late_fusion --pooling cls          --run_name pool_cls
uv run python -m src.training.train --config late_fusion --pooling max          --run_name pool_max

# -------------------------------------------------------------
# ETAPA 4: FUSIÓN MULTIMODAL
# -------------------------------------------------------------
uv run python -m src.training.train --config cross_attention --run_name fusion_cross

# -------------------------------------------------------------
# ETAPA 5: MULTI-SEED (Ejemplo con semillas 7, 42, 123, 456, 999)
# -------------------------------------------------------------
for s in 7 42 123 456 999; do
  uv run python -m src.training.train --config late_fusion --seed $s --run_name final_seed_$s
done
```

---

## 5. Recomendaciones para el Reporte y la Presentación Oral

1. **Métrica Clave:** Dado el desbalance intrínseco del target $BTR$ ($\approx 21\%$ positivos), la métrica central de comparación debe ser **PR-AUC**, acompañada por **ROC-AUC** y las curvas de pérdida.
2. **Detección de Overfitting:** Mostrar en las diapositivas gráficos comparativos de curvas de aprendizaje ($\text{Loss}_{\text{train}}$ vs $\text{Loss}_{\text{val}}$) para justificar por qué una arquitectura más grande no siempre es mejor.
3. **Historia Narrativa de la Presentación:**
   * *Problema & Baseline Tabular:* ¿Cuánto sabemos sin texto?
   * *Incorporación del Transformer:* ¿Cuánto aporta el modelado semántico?
   * *Ablaciones del Transformer:* ¿Por qué elegimos estos cabezales y encoding posicional?
   * *Cross-Attention:* ¿Cómo ayuda la atención cruzada a alinear el perfil tabular con los claims del catálogo?
   * *Resultados Finales:* Tabla de significancia estadística con intervalos de confianza.
