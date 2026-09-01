# Transformer "Pelado" sobre Serialización Total — Diseño, Pipeline y Resultados

**Ubicación:** `resources/documentation/raw_transformer.md`  
**Módulo:** `src/raw_transformer/` (plan de trabajo detallado en `src/raw_transformer/PLAN.md`)  
**Tests:** `tests/test_raw_transformer.py` (18 tests)  
**Directorio de Corridas:** `results/runs_raw/`  
**Directorio de Tablas:** `results/aggregate/raw_vs_hybrid_*.{md,csv}`  

---

## 1. Qué se Evalúa

El `hybrid_transformer` separa el problema en dos ramas — texto → Transformer, variables tabulares → MLP — y las fusiona al final. Este experimento prueba la **alternativa de arquitectura opuesta**: convertir **todas** las variables de la fila (numéricas y categóricas incluidas) a texto plano y entregárselas a un único Transformer, sin ramas, sin embeddings de categorías, sin normalización de numéricas y sin ninguna feature derivada.

```
title: Cedar House Steamable Pepperoni Pizza - 10 oz (Popular Choice) | description: ... |
price: 8.3 | category: Frozen | brand: Cedar House | net_weight_oz: 10.14 | nutrition_score: 36 | ...
                                    │
                        TextTransformerEncoder (reutilizado)
                                    │
                          ClassificationHead → logit → σ → BTR
```

**Hipótesis (H1, la esperada):** la rama tabular explícita aporta un sesgo inductivo que el Transformer pelado no puede recuperar — el precio `8.3` deja de ser un número con orden y pasa a ser los tokens `['Ġ8', '.', '3']`, cuya magnitud hay que aprender desde 7.000 ejemplos.

Es material directo para la consigna de *"comparación de alternativas de los distintos módulos"*: no es una variante de un módulo, es una alternativa a la arquitectura entera.

---

## 2. Decisiones de Diseño

| # | Decisión | Elección | Motivo |
|---|---|---|---|
| D1 | Formato de los valores | **Crudo**, sin redondear ni bucketizar (`price: 8.3`) | Es el test más honesto: cero ayuda al modelo. Variantes (decimales fijos, buckets) quedan como ablation. |
| D2 | Campos serializados | Los **20 campos del CSV crudo**, menos `cart` (leakage de embudo) y `bought` (target) | Sin features derivadas ni `title_clean`: el badge queda dentro del título como texto. |
| D2' | Presets de campos | `all` (20 campos) y `product_only` (14, sin `query_id`, `filter_*`, `timestamp`) | El hybrid descarta el contexto de búsqueda; `product_only` es la comparación controlada. |
| D3 | Tokenizador | **BPE propio** entrenado sobre el corpus serializado de train (`bpe_tokenizer_raw.json`) | El del hybrid nunca vio dígitos ni nombres de campo. Mismo presupuesto de vocabulario (2048). |
| D4 | `max_seq_len` | **256** | Las filas serializadas miden 198 tokens en media y 212 como máximo; con 128 no entraría ninguna completa. |
| D5 | Encoder | **`TextTransformerEncoder` importado sin modificar** | Lo único que cambia es cómo entra la información, no el Transformer. |
| D6 | Entrenamiento | **`src.training.trainer.Trainer` reutilizado** | Mismo loop, mismas métricas, mismos defaults que el hybrid; el dataset emite `input_ids` / `attention_mask` / `labels`. |

**Nulos:** se escriben explícitamente como `None` (solo afecta a `allergens`, 44,5 % de las filas) para que todas las secuencias tengan la misma estructura.

**Split temporal:** replica el de `build_transformer_dataset.py` (orden cronológico estable, 70/15/15). Se verificó que las filas de cada split son **idénticas y en el mismo orden** en los tres datasets (`transformer_*`, `raw_*`, `raw_po_*`).

---

## 3. Pipeline y Reproducción

```bash
# 1. Serializar la fila completa (preset all → raw_*.csv; product_only → raw_po_*.csv)
uv run python -m src.raw_transformer.serialize
uv run python -m src.raw_transformer.serialize --field_preset product_only --prefix raw_po

# 2. Entrenar el BPE propio sobre train (+ análisis de longitudes y fragmentación numérica)
uv run python -m src.raw_transformer.train_tokenizer

# 3. Smoke tests del modelo y del dataset
uv run python -m src.raw_transformer.model
uv run python -m src.raw_transformer.dataset

# 4. Entrenar (defaults idénticos al hybrid; el run_name se deriva de los hiperparámetros)
uv run python -m src.raw_transformer.train --seed 42                       # → results/runs_raw/raw_d64_L2_H4_s42/
uv run python -m src.raw_transformer.train --data_prefix raw_po --seed 42  # → results/runs_raw/raw_po_d64_L2_H4_s42/
uv run python -m src.raw_transformer.train --auto_pos_weight               # ablation del desbalance (pos_weight ≈ 6,6)

# 5. Corridas del hybrid sobre los mismos splits (requiere clean_dataset + build_transformer_dataset)
uv run python -m src.training.train --config late_fusion --seed 42         # ídem baseline_texto, baseline_tabular, cross_attention

# 6. Comparación multi-semilla pareada (raw vs hybrid)
uv run python -m src.raw_transformer.compare                 # métrica test_pr_auc
uv run python -m src.raw_transformer.compare --metric val_pr_auc

# Tests
uv run pytest tests/test_raw_transformer.py
```

### Archivos del módulo

| Archivo | Rol |
|---|---|
| `serialize.py` | Fila del CSV crudo → `campo: valor \| campo: valor \| ...`; presets `all` / `product_only`; split temporal. |
| `train_tokenizer.py` | BPE sobre el corpus serializado + reporte de longitudes y de cómo se fragmentan los números. |
| `model.py` | `RawTransformerClassifier` = encoder reutilizado + `ClassificationHead`; devuelve logits crudos. |
| `dataset.py` | `RawSerializedDataset` (tokenización anticipada, padding fijo 256) y DataLoaders, parametrizados por prefijo. |
| `train.py` | CLI fina sobre el `Trainer` común; resultados en `results/runs_raw/<run_name>/` con el mismo `summary.json` que el hybrid. |
| `compare.py` | Carga raw + hybrid, valida la grilla y produce tablas descriptivas y pareadas por semilla (reutiliza `aggregate.py`). |
| `PLAN.md` | Plan de trabajo con las decisiones, la evidencia de cada fase y las predicciones anotadas antes de correr. |

---

## 4. Hallazgos Intermedios (Fases 1–4)

* **Longitud:** 104 palabras / ~198 tokens por fila (vs. ~42 palabras del hybrid) → **2,5×** más largo.
* **Andamiaje:** los nombres de campo y separadores, sin ningún valor, consumen **95 de los ~198 tokens (48 %)**. Casi la mitad del cómputo se gasta en repetir `title:`, `nutrition_score:`, `|` en cada fila — el costo estructural de serializar.
* **Fragmentación numérica** (evidencia central del experimento):

  ```
  price: 8.25          -> ['price', ':', 'Ġ8', '.', '25']
  nutrition_score: 61  -> [..., ':', 'Ġ61']
  nutrition_score: 69  -> [..., ':', 'Ġ69']       ← IDs arbitrarios, sin relación de orden entre sí
  category: Frozen     -> ['category', ':', 'ĠFrozen']   ← un único token; las categóricas sobreviven intactas
  ```

* **Vocabulario efectivo:** 2048 (el corpus serializado agota el presupuesto; el del hybrid quedó en 1720).
* **Modelo:** 235.521 parámetros con `d_model=64`, 2 capas, 4 cabezales — **56 % son la tabla de embeddings** (2048 × 64).
* **Cobertura:** cero secuencias truncadas en los tres splits con `max_seq_len=256`.

---

## 5. Configuración Experimental

Todas las corridas — raw y hybrid — comparten:

* **Splits:** idénticos fila por fila (70/15/15 temporal; BTR 13,16 % / 12,20 % / 13,13 %)
* **Semillas:** `{42, 7, 123}` (3 corridas por configuración, 18 en total)
* **Arquitectura del Transformer:** `d_model=64`, `n_heads=4`, `d_ff=256`, `num_layers=2`, pooling `mean`, positional `sinusoidal`, Pre-LN, GELU
* **Optimización:** `AdamW` (`lr=1e-3`, `weight_decay=0.01`), `BCEWithLogitsLoss` **sin ponderar**, `batch_size=64`, `dropout=0.1`, gradient clipping 1.0
* **Early stopping:** `patience=5` sobre PR-AUC de validación; test evaluado una sola vez con el mejor checkpoint
* **Diferencias inevitables:** `max_seq_len` 256 (raw) vs. 128 (hybrid); tokenizador propio en cada caso

Configuraciones comparadas:

| Configuración | Entrada | Params |
|---|---|---:|
| `raw_all` | 20 campos crudos como texto (incluye contexto de búsqueda) | 235.521 |
| `raw_product_only` | 14 campos crudos como texto — **la comparable con el hybrid** | 235.521 |
| `baseline_texto` | Transformer sobre `title_clean \| description \| ingredients` | 214.401 |
| `baseline_tabular` | MLP sobre las features curadas, sin Transformer | 4.664 |
| `late_fusion` | texto + tabular, concatenación | 218.936 |
| `cross_attention` | texto + tabular, cross-attention con Query tabular | 235.832 |

---

## 6. Resultados

### 6.1 Media por configuración (3 semillas)

**`test_pr_auc`** (predictor constante = 0,1313; ROC-AUC = 0,5):

| Configuración | Params | media ± σ | min–max | ROC-AUC | Mejor época (mediana) |
|:---|---:|---:|---:|---:|---:|
| **`baseline_tabular`** | **4.664** | **0,7476 ± 0,0292** | 0,7303–0,7813 | **0,9710** | 5 |
| `raw_product_only` | 235.521 | 0,7070 ± 0,0236 | 0,6809–0,7268 | 0,9654 | 5 |
| `raw_all` | 235.521 | 0,7006 ± 0,0491 | 0,6602–0,7552 | 0,9625 | 4 |
| `cross_attention` | 235.832 | 0,6988 ± 0,0160 | 0,6803–0,7081 | 0,9618 | 2 |
| `late_fusion` | 218.936 | 0,6928 ± 0,0103 | 0,6856–0,7046 | 0,9644 | 3 |
| `baseline_texto` | 214.401 | 0,6843 ± 0,0282 | 0,6648–0,7167 | 0,9591 | 3 |

**`val_pr_auc`** (mejor época de cada corrida):

| Configuración | media ± σ | min–max |
|:---|---:|---:|
| **`baseline_tabular`** | **0,7223 ± 0,0318** | 0,7028–0,7590 |
| `late_fusion` | 0,6922 ± 0,0264 | 0,6719–0,7221 |
| `cross_attention` | 0,6814 ± 0,0081 | 0,6740–0,6901 |
| `raw_all` | 0,6800 ± 0,0297 | 0,6465–0,7032 |
| `raw_product_only` | 0,6747 ± 0,0114 | 0,6628–0,6855 |
| `baseline_texto` | 0,6667 ± 0,0358 | 0,6357–0,7059 |

### 6.2 Diferencias pareadas por semilla (`test_pr_auc`)

Δ = referencia − otra configuración, calculado semilla a semilla (positivo = la referencia es mejor). IC95% e *p* del t-test de una muestra sobre las tres diferencias.

**Referencia `raw_product_only`:**

| vs. | Δ media ± σ | IC95% | Semillas a favor | p |
|:---|---:|---:|---:|---:|
| `baseline_texto` | +0,0227 ± 0,0507 | [−0,103, +0,149] | 2/3 | 0,52 |
| `late_fusion` | +0,0142 ± 0,0247 | [−0,047, +0,076] | 2/3 | 0,42 |
| `cross_attention` | +0,0082 ± 0,0315 | [−0,070, +0,086] | 2/3 | 0,70 |
| `raw_all` | +0,0064 ± 0,0702 | [−0,168, +0,181] | 2/3 | 0,89 |
| `baseline_tabular` | **−0,0406 ± 0,0333** | [−0,123, +0,042] | **0/3** | 0,17 |

**Referencia `baseline_tabular`:**

| vs. | Δ media ± σ | IC95% | Semillas a favor | p |
|:---|---:|---:|---:|---:|
| `baseline_texto` | +0,0633 ± 0,0512 | [−0,064, +0,190] | 3/3 | 0,17 |
| **`late_fusion`** | **+0,0548 ± 0,0190** | **[+0,008, +0,102]** | **3/3** | **0,038** |
| `cross_attention` | +0,0488 ± 0,0452 | [−0,063, +0,161] | 3/3 | 0,20 |
| `raw_all` | +0,0470 ± 0,0727 | [−0,134, +0,228] | 2/3 | 0,38 |
| `raw_product_only` | +0,0406 ± 0,0333 | [−0,042, +0,123] | 3/3 | 0,17 |

**Referencia `late_fusion`** (el híbrido de referencia): ninguna diferencia contra los otros Transformers supera p = 0,4; la única significativa es contra `baseline_tabular` (Δ = −0,055, p = 0,038). En validación, `late_fusion` le gana a `baseline_texto` en 3/3 semillas (Δ = +0,026, p = 0,048): la rama tabular sí suma sobre el texto solo.

---

## 7. Conclusiones

1. **El Transformer pelado iguala al híbrido.** `raw_product_only` vs `late_fusion`: Δ = +0,014 ± 0,025 en test (p = 0,42) y −0,018 ± 0,036 en validación (p = 0,49). Con la fila cruda como texto — precio como caracteres, categorías como palabras, badge dentro del título, cero feature engineering — el modelo **no pierde nada** frente a la arquitectura de dos ramas con fusión. La hipótesis H1 ("la rama tabular aporta un sesgo inductivo irrecuperable") **no se confirma** en este dataset.

2. **El mejor modelo es el MLP tabular solo**, con 4.664 parámetros (50× menos que cualquier Transformer). Le gana a `late_fusion` en 3/3 semillas con **p = 0,038** — la única diferencia estadísticamente significativa de toda la grilla — y a `raw_product_only` en 3/3 semillas (Δ = 0,041, p = 0,17). La señal de este dataset está en los atributos estructurados curados (badge como categórica, precio, categoría, etc.), y agregarles un Transformer — sobre el texto o sobre la fila serializada — **no suma**, solo agrega parámetros y varianza.

3. **El contexto de búsqueda no aporta.** `raw_all` vs `raw_product_only`: Δ = −0,006 (p = 0,89). `raw_all` es además la configuración más inestable (σ = 0,049, rango 0,66–0,76): el 0,755 de la semilla 42 era un outlier. `query_id` y los `filter_*` agregan varianza, no información.

4. **Todos los Transformers sobreajustan temprano.** Tocan su máximo de validación entre las épocas 2 y 6 y después divergen (train PR-AUC → 0,80–0,85 mientras val cae hasta ~0,57). El raw es el que más cae. El MLP tabular casi no sobreajusta.

5. **Lectura general:** las cinco variantes con Transformer caen en una banda de 0,68–0,71 de test PR-AUC, indistinguibles entre sí con n = 3. Lo que ordena los modelos no es la arquitectura del Transformer ni cómo entra la información, sino si se usan o no las features tabulares curadas — y el modelo que usa *solo* eso es el mejor.

**Salvedad:** con tres semillas, solo diferencias mayores a ~0,05 alcanzan significancia. Las conclusiones 1 y 3 son "no se detecta diferencia", no "no hay diferencia"; la 2 sí tiene respaldo estadístico.

---

## 8. Registro de lo Realizado

| Commit | Contenido |
|---|---|
| `ebbd110` | Plan del experimento y serializador de fila completa (presets `all` / `product_only`). |
| `a8ef170` | BPE propio sobre el corpus serializado + análisis de longitudes y fragmentación numérica. |
| `823ed70` | `RawTransformerClassifier` reutilizando el encoder del hybrid; smoke tests. |
| `bf9bc19` | `RawSerializedDataset` y DataLoaders compatibles con el `Trainer` común. |
| `b94d48f` | CLI de entrenamiento sobre el `Trainer` común, con los mismos defaults que el hybrid. |
| `704aadf` | 16 tests del módulo y soporte de `--data_prefix` para correr ambos presets. |
| `2c1cd2a` | Primeras corridas (seed 42) de `all` y `product_only` registradas en el plan. |
| `fb8efb0` | Comparación contra las 4 configs del hybrid con seed 42 sobre splits idénticos. |
| `593ca74` | `compare.py` (comparación multi-semilla pareada) + 2 tests; resultados con 3 semillas. |

Ningún archivo de `src/hybrid_transformer/`, `src/training/`, `src/tokenizer/`, `src/data_extraction/` ni `config/` fue modificado: el módulo solo **importa** de ellos.

---

## 9. Pendientes

* **Ablation por campo del raw** (para la predicción de que la brecha viene de las numéricas): `product_only` sin `price` / `net_weight_oz` / `nutrition_score` / `dimensions_in` vs. sin las categóricas.
* **Variantes de serialización** (D1): decimales fijos y bucketizado de numéricas; nombres de campo abreviados para recuperar parte del 48 % de andamiaje.
* **Curvas de aprendizaje comparativas** raw vs. hybrid → `results/figures/`.
* Más semillas si se quiere resolver diferencias menores a 0,05.
