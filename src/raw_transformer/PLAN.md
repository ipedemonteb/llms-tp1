# PLAN — `raw_transformer`: Transformer "pelado" sobre serialización total

**73.69 Large Language Models — TP1**
**Ubicación:** `src/raw_transformer/` (hermano de `src/hybrid_transformer/`)

---

## 1. La idea en una frase

En lugar de separar el problema en dos ramas (texto → Transformer, tabular → MLP) y
fusionarlas al final, **convertimos TODAS las variables a texto plano** y se las damos
a un Transformer solo, sin ninguna ayuda estructural.

```
title: Cedar House Steamable Pepperoni Pizza | brand: Cedar House | category: Frozen |
price: 8.3 | storage: Frozen | weight: 10.14 oz | nutrition: 36 | origin: United States |
allergens: Wheat | ingredients: Prepared ingredients, Spices, Salt | description: ...
                                    │
                              TRANSFORMER
                                    │
                              MLP head → σ → BTR
```

El precio `8.30` deja de ser un número y pasa a ser **la cadena de caracteres "8.3"**.
La categoría `Frozen` deja de ser un embedding aprendido y pasa a ser **la palabra "Frozen"**.

---

## 2. Qué estamos testeando (la hipótesis)

> **H0:** Un Transformer con suficiente capacidad puede inferir la estructura tabular
> por sí solo desde el texto serializado, sin necesidad de ramas separadas ni
> ingeniería de features.
>
> **H1 (lo que esperamos):** No. La rama tabular explícita del `hybrid_transformer`
> aporta un *sesgo inductivo* que el modelo pelado no puede recuperar, especialmente
> en las variables numéricas.

Sea cual sea el resultado, **es material directo para la presentación**. El enunciado
pide explícitamente *"comparación de alternativas de los distintos módulos"*: esto no es
una variante de un módulo, es una alternativa a la arquitectura entera.

---

## 3. Contraste con `hybrid_transformer`

| | `hybrid_transformer` | `raw_transformer` |
|---|---|---|
| **Entrada** | 2 tensores: `input_ids` + vector tabular | 1 tensor: `input_ids` |
| **Precio $8.30** | float normalizado (`BatchNorm`) | los tokens `"8"`, `"."`, `"3"` |
| **Categoría `Frozen`** | `nn.Embedding` de categoría | la palabra `"Frozen"` tokenizada |
| **Orden entre números** | implícito ($5 < 8 < 12$ por construcción) | debe **aprenderse desde cero** |
| **Ramas** | Transformer + MLP tabular + fusión | solo Transformer |
| **Longitud de secuencia** | ~42 palabras | **~79 palabras** (medido) |
| **Feature engineering** | `price_per_oz`, `volume`, `has_allergens`... | ninguno |

### El punto pedagógico central: **cómo un Transformer "ve" un número**

Un `nn.Linear` que recibe `price=8.30` sabe, por construcción del espacio vectorial,
que 8.30 está entre 5 y 12. Un Transformer que recibe los tokens `["8", ".", "3"]`
**no sabe nada de eso**. Tiene que aprender, a partir de 7.000 ejemplos de entrenamiento,
que la secuencia de caracteres "8.3" representa una cantidad mayor que "5.5".

Esto es exactamente la limitación conocida de los LLMs con aritmética, y este experimento
la va a exhibir en un caso concreto y medible. **Ese es el mejor slide de la presentación.**

---

## 4. Decisiones de diseño

### D1 — Formato de serialización ✅ DECIDIDO: **crudo**
Se escribe el valor tal cual viene del CSV, sin redondear, sin fijar decimales y sin
bucketizar: `price: 8.3 | weight: 10.14 oz | nutrition: 36`.

Es la versión más "pelada" y por lo tanto el test más honesto de la hipótesis: cero ayuda
al modelo, que tiene que aprender la magnitud numérica desde los caracteres.

*Variantes descartadas por ahora, disponibles como ablation posterior:* decimales fijos
(`8.30`, tokenización más regular) y bucketizado (`price: cheap`, que le devolvería el
orden al modelo pero ya sería feature engineering).

### D2 — Qué campos incluir ✅ DECIDIDO: **los 20 crudos del CSV**
Todo lo que hay en `supermarket_products.csv` **menos `cart` y `bought`**. Sin features
derivadas: nada de `price_per_oz`, `volume`, `num_ingredients` ni `title_clean`.

El contraste queda nítido: el `hybrid` recibe features curadas por un humano, el `raw`
recibe el CSV crudo y se arregla solo.

> ⚠️ **Confounder detectado:** el `hybrid` descarta `query_id`, `filter_category` y
> `filter_storage_type`, pero el `raw` los incluiría. Eso le daría al `raw` **más
> información**, y entonces la diferencia de performance ya no sería atribuible solo a
> la arquitectura.
>
> **Solución:** el serializador soporta dos presets de campos. `all` (default, todo el CSV)
> y `product_only` (sin contexto de búsqueda ni timestamp). Corremos los dos y comparamos;
> el segundo es la comparación controlada, el primero es el "pelado total".

### D3 — Tokenizador ✅ RESUELTO
El tokenizador del hybrid (`resources/tokenizer/bpe_tokenizer.json`) se entrenó sobre la
columna `text` = solo campos textuales. **Nunca vio dígitos ni nombres de campo.**
→ Hay que **reentrenar un BPE propio** sobre el corpus serializado, y guardarlo aparte
(`resources/tokenizer/bpe_tokenizer_raw.json`) para no pisar el del hybrid.
Se reutiliza la clase `ByteLevelBPETokenizer` tal cual, solo cambia el corpus.

> 📌 Se pide `vocab_size=2048` en ambos tokenizadores, pero el vocabulario **efectivo**
> depende de cuántos merges dé el corpus (el del hybrid quedó en 1720). El modelo toma
> `tokenizer.vocab_size` real al construirse, así que la comparación sigue siendo justa:
> mismo presupuesto de vocabulario, corpus distinto.

### D4 — `max_seq_len` ✅ DECIDIDO: **256**
Medición sobre el corpus serializado ya tokenizado (Fase 2):

| | tokens |
|---|---|
| media / p50 | 198 |
| p95 | 204 |
| p99 | 207 |
| **max** | **212** |

| `max_seq_len` | secuencias completas | costo $O(N^2)$ relativo |
|---|---|---|
| 128 (el del hybrid) | **0,0%** | 1,0× |
| 192 | 13,7% | 2,2× |
| **256** | **100,0%** | **4,0×** |

Con 128 tokens **ninguna** secuencia entra completa: el hybrid trunca, pero el raw perdería
casi la mitad de cada fila. Con 256 entran todas con margen, al costo de cuadruplicar el
bloque de atención. Se asume ese costo: truncar invalidaría el experimento.

> 📌 **Hallazgo:** el andamiaje (nombres de campo + separadores, sin ningún valor) consume
> **95 de los ~198 tokens**, es decir **~48% de la secuencia**. Casi la mitad del cómputo se
> gasta en repetir `title:`, `nutrition_score:`, `|`… en cada fila. Es el precio estructural
> de serializar, y es un argumento cuantitativo a favor de la rama tabular del hybrid.
> *Ablation posible:* acortar los nombres de campo y medir cuánto se recupera.

### D5 — Reutilizar el encoder ✅ DECIDIDO
`TextTransformerEncoder` de `hybrid_transformer/text_encoder.py` ya hace todo lo necesario
(embeddings, positional encoding, N bloques, pooling configurable). **Lo importamos, no lo
duplicamos.** `raw_transformer` solo agrega: serializador + dataset + cabeza de clasificación.

### D6 — Reutilizar el Trainer ✅ DECIDIDO
`src/training/trainer.py` ya implementa el loop completo (AdamW, BCEWithLogitsLoss con
`pos_weight`, early stopping por PR-AUC de validación, restauración del mejor checkpoint)
y `src/training/metrics.py` las métricas con baseline. El dataset del raw emite los batches
con las mismas claves (`input_ids`, `attention_mask`, `labels`) para que `Trainer` los
consuma sin ninguna adaptación. `raw_transformer/train.py` queda como una CLI fina.

Los resultados van a `results/runs_raw/` (separados de `results/runs/`, cuyo agregador
está acoplado al parser del hybrid), con el mismo formato interno de `summary.json`.

---

## 5. Controles para que la comparación sea justa

Si `raw` pierde contra `hybrid`, tiene que ser por la arquitectura y no por otra cosa.
Se fija idéntico entre ambos:

- [x] Mismo split temporal (mismas filas en train/val/test)
- [ ] Mismo `d_model`, `n_heads`, `num_layers`, `dropout`
- [x] Mismo presupuesto de vocabulario del BPE (2048 pedidos)
- [ ] Misma seed, mismo optimizador (AdamW), mismo LR, mismas épocas
- [ ] Mismas métricas: **PR-AUC** (principal, por el desbalance 13%), ROC-AUC, BCE loss

Lo único que cambia: **cómo entra la información al modelo.**

Se reporta también el conteo de parámetros de cada uno — si `raw` tiene muchos menos
(no tiene rama tabular ni fusión), es un dato a mencionar, no un problema.

---

## 6. Fases

### Fase 1 — Serializador ✅
`serialize.py`: convierte una fila del CSV crudo → string `campo: valor | campo: valor | ...`
- ✅ D1 (crudo) y D2 (20 campos) aplicadas; presets `all` y `product_only`
- ✅ Nulos escritos explícitamente como `None` (solo afecta a `allergens`, 44,5% del dataset),
  para que la estructura de la secuencia sea constante en todas las filas
- ✅ Split temporal replicado con orden estable → mismas filas que `build_transformer_dataset.py`
- ✅ **Checkpoint superado.** Resultados medidos:
  - Longitud: **104 palabras** de media (709 chars), contra 42 del `hybrid` → **2,5×**
  - Balance estable entre splits: BTR 13,16% / 12,20% / 13,13% (sin drift temporal del target)
  - Salidas: `resources/datasets/raw_{train,val,test}.csv`

### Fase 2 — Tokenizador propio ✅
`train_tokenizer.py`: BPE entrenado **solo sobre train** (7.000 secuencias), presupuesto de
vocabulario 2048 (igual que el hybrid), guardado en `resources/tokenizer/bpe_tokenizer_raw.json`.
- ✅ Vocabulario efectivo: **2048** (el corpus serializado sí agota el presupuesto;
  el del hybrid quedó en 1720)
- ✅ D4 confirmada: longitudes media 198 / p99 207 / max 212 → `max_seq_len = 256`
- ✅ **Checkpoint superado.** Evidencia sobre la fragmentación numérica:

```
price: 8.25           -> ['price', ':', 'Ġ8', '.', '25']
price: 13.33          -> ['price', ':', 'Ġ13', '.', '33']
nutrition_score: 61   -> ['n','ut','rition','_','score', ':', 'Ġ61']
nutrition_score: 69   -> ['n','ut','rition','_','score', ':', 'Ġ69']
category: Frozen      -> ['category', ':', 'ĠFrozen']          ← 1 solo token
```

**El resultado clave:** `61` y `69` son dos IDs de vocabulario **arbitrarios y sin ninguna
relación entre sí**. Para el modelo, la distancia entre 61 y 69 es exactamente igual que la
distancia entre 61 y `Frozen`: cero estructura de orden. Toda la noción de magnitud tiene
que aprenderse desde los datos.

En cambio las categóricas sobreviven intactas (`ĠFrozen` es un único token), lo que **respalda
la predicción #2**: la brecha contra el hybrid debería venir de las numéricas, no de las
categóricas.

### Fase 3 — Modelo ✅
`model.py`: `RawTransformerClassifier` = `TextTransformerEncoder` (importado sin modificar)
+ `ClassificationHead` (LayerNorm → Dropout → Linear → GELU → Dropout → Linear).

Devuelve **logits crudos**, no probabilidades, para poder usar `BCEWithLogitsLoss`
(estable numéricamente y con soporte de `pos_weight` para el desbalance del 13%).

- ✅ **Checkpoint superado.** 5 smoke tests: shapes, rango [0,1] de la sigmoide,
  máscara de padding efectiva (delta = 0), flujo de gradientes hasta los embeddings,
  y exposición de los mapas de atención para XAI.

**Conteo de parámetros** (`d_model=64`, 2 capas, `max_seq_len=256`):

| Componente | Parámetros | |
|---|---|---|
| Tabla de embeddings | 131.072 | 56% |
| Encoder (2 bloques) | 100.096 | 42% |
| Cabeza de clasificación | 4.353 | 2% |
| **Total** | **235.521** | |

> 📌 Más de la mitad de los parámetros son la tabla de embeddings (2048 × 64). El modelo
> gasta la mayor parte de su capacidad en representar tokens — incluidos los fragmentos
> numéricos `Ġ61`, `Ġ69`, `.`, `25` — antes de razonar sobre ellos.

### Fase 4 — Dataset + DataLoader ✅
`dataset.py`: `RawSerializedDataset` con tokenización anticipada (todo el split de una vez
en el constructor, no por fila en `__getitem__`) y batches con claves
`input_ids` / `attention_mask` / `labels`, compatibles con el `Trainer` común (D6).
Padding a longitud fija 256; shuffle solo en train, val/test en orden cronológico.

- ✅ **Checkpoint superado:**

| split | filas | BTR | truncadas | batches (bs=32) |
|---|---|---|---|---|
| train | 7.000 | 13,16% | **0** | 219 |
| val | 1.500 | 12,20% | **0** | 47 |
| test | 1.500 | 13,13% | **0** | 47 |

- Cero secuencias truncadas → `max_seq_len=256` confirmado en la práctica
- Tokens reales por fila: min 189, media 197, max 204
- `pos_weight = 6,600` (calculado **solo sobre train**) para `BCEWithLogitsLoss`
- Verificación end-to-end: el modelo consume un batch real y devuelve logits `(32,)`

### Fase 5 — Entrenamiento ✅
`train.py`: CLI fina sobre `src.training.trainer.Trainer` (D6), resultados en `results/runs_raw/`
con el mismo formato de `summary.json` / `history.csv` / `checkpoint.pt` que las corridas del hybrid.

- Defaults de entrenamiento idénticos a `src/training/train.py` (AdamW lr=1e-3, wd=0.01,
  batch_size=64, 20 épocas, patience=5, BCE sin ponderar) para que la comparación sea justa
- `--auto_pos_weight` calcula n_neg/n_pos sobre train (≈ 6,6) como ablation del desbalance
- ✅ **Checkpoint superado** (smoke run de 2 épocas en CPU, ~105 s/época): el `Trainer`
  común consume los batches sin adaptación, la loss baja (0.287 → 0.160) y el pipeline
  completo guarda checkpoint, resumen e historial.

> ⚠️ Ya con 2 épocas el preset `all` da PR-AUC de test ≈ 0.67 (lift 5x). Recordar el
> confounder de D2: `all` incluye el contexto de búsqueda (`query_id`, `filter_*`) que el
> hybrid no ve. La comparación honesta contra el hybrid es con el preset `product_only`.

### Fase 6 — Evaluación y comparación ✅ (3 semillas, pareado)

Todas las corridas con **los mismos splits (verificado fila por fila)**, las mismas tres
semillas (42, 7, 123) y los mismos hiperparámetros: `d_model=64`, 2 capas, 4 cabezales,
AdamW lr=1e-3, wd=0.01, batch 64, dropout 0.1, BCE sin ponderar, early stopping con
patience 5. Tablas generadas por `compare.py`, que reutiliza la estadística pareada de
`src/training/aggregate.py` (Δ semilla a semilla, IC95% y t-test de una muestra).

**Media por configuración — `test_pr_auc`** (predictor constante = 0,1313):

| Configuración | Entrada | Params | media ± σ | min–max | ROC-AUC | Mejor ép. |
|:---|:---|---:|---:|---:|---:|---:|
| **`baseline_tabular`** | MLP sobre features curadas, sin Transformer | **4.664** | **0,7476 ± 0,0292** | 0,7303–0,7813 | **0,9710** | 5 |
| **`raw_product_only`** | 14 campos crudos como texto (comparable al hybrid) | 235.521 | 0,7070 ± 0,0236 | 0,6809–0,7268 | 0,9654 | 5 |
| `raw_all` | 20 campos crudos como texto (incluye búsqueda) | 235.521 | 0,7006 ± 0,0491 | 0,6602–0,7552 | 0,9625 | 4 |
| `cross_attention` | texto + tabular, cross-attention | 235.832 | 0,6988 ± 0,0160 | 0,6803–0,7081 | 0,9618 | 2 |
| `late_fusion` | texto + tabular, concatenación | 218.936 | 0,6928 ± 0,0103 | 0,6856–0,7046 | 0,9644 | 3 |
| `baseline_texto` | Transformer sobre `title_clean \| description \| ingredients` | 214.401 | 0,6843 ± 0,0282 | 0,6648–0,7167 | 0,9591 | 3 |

En validación (`val_pr_auc`) el orden es: tabular 0,722 > late_fusion 0,692 > cross 0,681
> raw_all 0,680 > raw_po 0,675 > texto 0,667 — mismo ganador, y todos los Transformers
apretados en 0,67–0,69.

**Diferencias pareadas por semilla — `test_pr_auc`** (Δ = referencia − otra; positivo =
la referencia es mejor):

| `raw_product_only` vs. | Δ media ± σ | IC95% | Semillas a favor | p |
|:---|---:|---:|---:|---:|
| `baseline_texto` | +0,0227 ± 0,0507 | [−0,103, +0,149] | 2/3 | 0,52 |
| `late_fusion` | +0,0142 ± 0,0247 | [−0,047, +0,076] | 2/3 | 0,42 |
| `cross_attention` | +0,0082 ± 0,0315 | [−0,070, +0,086] | 2/3 | 0,70 |
| `raw_all` | +0,0064 ± 0,0702 | [−0,168, +0,181] | 2/3 | 0,89 |
| `baseline_tabular` | **−0,0406 ± 0,0333** | [−0,123, +0,042] | **0/3** | 0,17 |

| `baseline_tabular` vs. | Δ media ± σ | IC95% | Semillas a favor | p |
|:---|---:|---:|---:|---:|
| `baseline_texto` | +0,0633 ± 0,0512 | [−0,064, +0,190] | 3/3 | 0,17 |
| **`late_fusion`** | **+0,0548 ± 0,0190** | **[+0,008, +0,102]** | **3/3** | **0,038** |
| `cross_attention` | +0,0488 ± 0,0452 | [−0,063, +0,161] | 3/3 | 0,20 |
| `raw_all` | +0,0470 ± 0,0727 | [−0,134, +0,228] | 2/3 | 0,38 |
| `raw_product_only` | +0,0406 ± 0,0333 | [−0,042, +0,123] | 3/3 | 0,17 |

**Contraste con las predicciones de la sección 7:**

1. ❌ **#1 no se confirma.** `raw_product_only` vs `late_fusion`: Δ = +0,014 ± 0,025 en
   test (2/3 semillas a favor del raw, p = 0,42) y −0,018 ± 0,036 en validación
   (p = 0,49). **Indistinguibles.** El Transformer pelado, con la fila como texto y sin
   ninguna ingeniería de features, **iguala a la arquitectura híbrida completa** en las
   tres semillas. Es el resultado central para la presentación.
2. ⏳ **#2 sigue sin poder testearse** (hace falta el ablation por campo, ver pendientes).
3. ✅ **#3 confirmada.** Todos los Transformers tocan su máximo entre las épocas 2 y 6 y
   después divergen (train PR-AUC → 0,80–0,85 mientras val cae). El raw es el que más
   cae. El MLP tabular casi no sobreajusta.
4. ✅ **#4 es el hallazgo real, y ahora con respaldo estadístico.** El **MLP tabular solo,
   con 4.664 parámetros (50× menos), es el mejor modelo** en test y en validación:
   le gana a `late_fusion` en 3/3 semillas con **p = 0,038** — la única diferencia
   significativa de toda la grilla — y a `raw_product_only` en 3/3 semillas
   (Δ = 0,041, p = 0,17). La señal de este dataset está en los atributos estructurados
   curados (badge como categórica, precio, etc.); agregarles un Transformer, sea sobre
   el texto o sobre la fila serializada, **no suma**.

**Sobre el contexto de búsqueda (`raw_all` vs `raw_product_only`):** con tres semillas
la ventaja desaparece — Δ = −0,006 (p = 0,89) — y `raw_all` es la configuración **más
inestable** de todas (σ = 0,049, rango 0,66–0,76). El 0,755 de la semilla 42 era un
outlier, no una señal. `query_id` y los `filter_*` no aportan información útil promedio,
solo varianza.

**Lectura general:** todas las variantes con Transformer quedan en una banda de
0,68–0,71 de test PR-AUC, indistinguibles entre sí con n = 3 (ningún p < 0,4 entre
ellas). Lo que ordena los modelos no es la arquitectura del Transformer ni cómo entra
la información (texto crudo vs. ramas separadas), sino si se usan o no las features
tabulares curadas — y el modelo que solo usa eso es el mejor.

Artefactos (gitignorados): `results/runs_raw/` (6 corridas raw), `results/runs/`
(12 corridas hybrid), `results/aggregate/raw_vs_hybrid_*.{md,csv}`.

Pendiente:
- [ ] Ablation por campo del raw para la predicción #2: `product_only` sin los numéricos
  (`price`, `net_weight_oz`, `nutrition_score`, `dimensions_in`) vs sin los categóricos
- [ ] Gráficos comparativos (curvas de aprendizaje raw vs hybrid) → `results/`

---

## 7. Qué esperamos encontrar (predicciones a verificar)

Anotadas **antes** de correr nada, para poder contrastarlas después:

1. **`raw` va a rendir peor en PR-AUC.** El sesgo inductivo de la rama tabular es real
   y 7.000 ejemplos de entrenamiento son pocos para que el modelo aprenda la magnitud
   numérica desde caracteres.
2. **La brecha va a venir sobre todo de las variables numéricas** (`price`, `nutrition_score`,
   `net_weight_oz`), no de las categóricas — `Frozen` como palabra es casi tan buena señal
   como `Frozen` como embedding.
3. **`raw` va a sobreajustar antes**, porque desperdicia capacidad aprendiendo a parsear
   texto en vez de a predecir.
4. Si `raw` **empata o gana**, la conclusión también es valiosa: significaría que en este
   dataset la señal está mayormente en el texto (los badges de social proof) y que la
   rama tabular aporta poco. Habría que verificarlo con un ablation de features.

---

## 8. Archivos previstos

```
src/raw_transformer/
├── PLAN.md          ← este documento
├── __init__.py
├── serialize.py        Fase 1 — fila del CSV → string
├── train_tokenizer.py  Fase 2 — BPE propio + análisis de fragmentación
├── model.py            Fase 3 — encoder importado + cabeza de clasificación
├── dataset.py          Fase 4 — Dataset / DataLoader de PyTorch
├── train.py            Fase 5 — CLI de entrenamiento sobre el Trainer común
└── compare.py          Fase 6 — comparación multi-semilla pareada contra el hybrid
```

**Se reutiliza sin duplicar:**
- `src/tokenizer/bpe.py` → `ByteLevelBPETokenizer`
- `src/hybrid_transformer/text_encoder.py` → `TextTransformerEncoder`, `TextTransformerConfig`
- `src/training/trainer.py` → `Trainer`, `TrainerConfig`, `set_seed`
- `src/training/metrics.py` → `compute_metrics`, `lift_over_baseline`
