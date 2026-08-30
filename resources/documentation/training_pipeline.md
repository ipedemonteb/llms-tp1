# Pipeline de Entrenamiento y Evaluación

**73.69 Large Language Models — Trabajo Práctico 1**
**Ejercicio 2: Desarrollo del Sistema de Predicción de BTR**
**Módulos:** `src/hybrid_transformer/fusion.py`, `src/training/`

---

## 1. Arquitectura del Sistema Completo

```
                          UN PRODUCTO (una fila del dataset)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
   6 campos de texto concatenados            7 numéricas + 8 categóricas
                 │                                           │
      ByteLevelBPETokenizer                        TabularPreprocessor
      (input_ids, attention_mask)                  (log1p → z-score, índices)
                 │                                           │
      TextTransformerEncoder                          TabularEncoder
         (Transformer, 210k)                            (MLP, 6,7k)
                 │                                           │
           e_text (B, 64)                              e_tab (B, 32)
                 │                                           │
                 └─────────────────┬─────────────────────────┘
                                   │
                        Módulo de Fusión
                    ┌──────────────┴──────────────┐
                    │                             │
              mode='late'                   mode='cross'
        concat [e_text ‖ e_tab]      e_tab como Query sobre H_text
                    └──────────────┬──────────────┘
                                   │
                         ClassifierHead (MLP)
                                   │
                            logit (B,)  ──► BCEWithLogitsLoss
```

El modelo completo (`BTRModel`) admite desactivar cualquiera de las dos ramas, lo que produce
los baselines de texto puro y tabular puro sin código adicional.

---

## 2. Componentes

### 2.1. `src/hybrid_transformer/fusion.py`

| Clase | Rol |
| :--- | :--- |
| `FusionConfig` | Hiperparámetros de la fusión y de la cabeza. Valida que `d_text` sea divisible por `n_heads` en modo cross. |
| `CrossAttentionFusion` | Cross-attention multi-cabezal: `e_tab` proyectado como Query, la secuencia `H_text` como Keys y Values. Devuelve opcionalmente los pesos para los mapas de atención. |
| `ClassifierHead` | MLP que proyecta el vector fusionado a **un logit sin activar**. |
| `BTRModel` | Ensambla ambas ramas, la fusión y la cabeza. Expone `param_breakdown()` para el informe. |

**Por qué logits y no probabilidades:** la sigmoide vive dentro de `BCEWithLogitsLoss`, que aplica
el truco log-sum-exp. Calcular `sigmoid` y después `log` por separado pierde precisión cuando las
probabilidades saturan cerca de 0 o 1.

**Modo `cross` y pooling:** el cross-attention necesita la secuencia completa `(B, T, d_model)`, no
el vector colapsado. `BTRModel` fuerza automáticamente `pooling_mode='none'` en el encoder de texto
cuando se selecciona ese modo.

### 2.2. `src/training/dataset.py`

`SupermarketDataset` tokeniza una única vez en el constructor (con 10.000 filas el costo de memoria
es despreciable y evita repetir el trabajo en cada época) y devuelve diccionarios con las claves que
consume `BTRModel.forward`.

`build_dataloaders()` es el punto de entrada: lee los tres CSV, **ajusta el `TabularPreprocessor`
solo con train**, construye los tres loaders y devuelve además los metadatos que necesita el modelo
(`vocab_size`, `cardinalities`, `num_numeric`, tasas de positivos).

> La composición de la secuencia de texto se resuelve **acá**, en tiempo de entrenamiento, mediante
> el parámetro `text_fields`. Eso permite variar los campos entre experimentos manteniendo idéntica
> la partición train/val/test, condición necesaria para que la ablación sea comparable.

### 2.3. `src/training/metrics.py`

Calcula PR-AUC, ROC-AUC y BCE, más la **línea base de PR-AUC** (la prevalencia de la clase positiva)
y el `lift` sobre ella.

**Por qué PR-AUC es la métrica primaria:** con un BTR global de ~13%, un clasificador que siempre
predice la clase negativa alcanza 87% de accuracy sin ser útil. La ROC-AUC también resulta optimista
con clases desbalanceadas porque premia el manejo de los negativos, que son abundantes. La PR-AUC se
concentra en la clase positiva y su línea base es directamente la prevalencia: un valor de 0,13
equivale a no haber aprendido nada.

Si un split tiene una sola clase presente, las AUC quedan en `NaN` en lugar de lanzar excepción, para
que el entrenamiento continúe y el problema quede visible en el log.

### 2.4. `src/training/trainer.py`

| Decisión | Motivo |
| :--- | :--- |
| `BCEWithLogitsLoss` | Estabilidad numérica frente a `BCELoss` + sigmoide separada. |
| `AdamW` en vez de `Adam` | Desacopla el weight decay del paso adaptativo, que es la forma correcta de regularizar con optimizadores adaptativos. |
| Early stopping sobre **PR-AUC de validación** | La métrica que reporta el trabajo debe ser la que elige el checkpoint, no la loss. |
| Restauración del mejor checkpoint | Test se evalúa una sola vez, al final, con los pesos del mejor epoch de validación. |
| `clip_grad_norm_` | Evita pasos destructivos en las primeras épocas. |

---

## 3. Uso

### 3.1. Pipeline completo desde cero

```bash
uv sync                                                        # dependencias
uv run python -m src.data_extraction.clean_dataset             # crudo → clean_dataset.csv
uv run python -m src.data_extraction.build_transformer_dataset # → splits temporales
uv run python -m src.tokenizer.bpe --train_file resources/datasets/transformer_train.csv
uv run python -m src.training.train                            # entrena y evalúa
```

### 3.2. Configuraciones de experimento (`--config`)

Los hiperparámetros pueden definirse en un archivo JSON dentro de `config/`, en lugar de escribir
comandos con veinte flags. Cada experimento queda así versionado y reproducible.

```bash
uv run python -m src.training.train --config late_fusion
```

`--config late_fusion` se resuelve como `config/late_fusion.json`. También se acepta una ruta
explícita (`--config experimentos/prueba.json`).

**Precedencia de valores**, de menor a mayor prioridad:

```
defaults del parser  <  archivo de --config  <  flags explícitos de la CLI
```

Eso permite reutilizar una configuración variando un único hiperparámetro, sin duplicar archivos:

```bash
# Cinco semillas sobre la misma configuración
for s in 1 2 3 4 5; do
  uv run python -m src.training.train --config late_fusion --seed $s --run_name lf_seed$s
done
```

**Formato del archivo.** Las claves son los mismos nombres que los flags, sin los guiones. Las que
empiezan con `_` se ignoran, lo que permite dejar notas dentro del archivo:

```json
{
  "_descripcion": "Modelo híbrido de referencia.",
  "use_text": true,
  "use_tabular": false,
  "d_model": 64,
  "num_layers": 2,
  "text_fields": ["title_clean", "badge", "description"],
  "lr": 0.001
}
```

Tres detalles del formato:
- Los flags `--no_text` / `--no_tabular` se expresan como `"use_text": false` / `"use_tabular": false`.
- `text_fields` acepta una **lista** en JSON (por CLI sigue siendo una string separada por comas).
- Si la config no define `run_name`, se usa el nombre del archivo. Así `--config modelo_chico`
  guarda en `results/runs/modelo_chico/` sin configurar nada más.

Una clave mal escrita **aborta la corrida con un error**, en lugar de ignorarse en silencio. Una
config inexistente lista las disponibles en el mensaje de error.

#### Configuraciones incluidas

| Archivo | Experimento |
| :--- | :--- |
| `late_fusion.json` | Modelo híbrido de referencia. |
| `baseline_texto.json` | Solo Transformer, sin rama tabular. |
| `baseline_tabular.json` | Solo MLP tabular, sin Transformer. |
| `cross_attention.json` | Ablación del módulo de fusión. |
| `regularizacion_alta.json` | Dropout 0,3 y weight decay 0,05 contra el overfitting. |
| `modelo_chico.json` | Ablación de capacidad: `d_model=32`, 1 capa, 2 cabezales. |
| `texto_con_brand.json` | Ablación de features: `brand` como séptimo campo de texto. |

### 3.3. Experimentos por flags

Los flags siguen soportados en su totalidad y pueden combinarse con `--config`:

```bash
# Baselines
uv run python -m src.training.train --no_tabular --run_name baseline_texto
uv run python -m src.training.train --no_text    --run_name baseline_tabular

# Ablación de la arquitectura del Transformer
uv run python -m src.training.train --d_model 32 --num_layers 1 --n_heads 2 --run_name small
uv run python -m src.training.train --pooling cls --run_name pooling_cls
uv run python -m src.training.train --pos_encoding none --run_name sin_posicional

# Ablación de features de texto
uv run python -m src.training.train \
  --text_fields title_clean,badge,description,ingredients,country_of_origin,allergens,brand \
  --run_name texto_con_brand

# Ablación de la codificación categórica
uv run python -m src.training.train --cat_encoding embedding --run_name entity_embeddings
```

Ver la lista completa de flags con `uv run python -m src.training.train --help`.
Restricción a respetar: **`d_model` debe ser divisible por `n_heads`**.

Cada corrida guarda en `results/runs/<run_name>/`:
- `checkpoint.pt` — pesos del mejor epoch más el historial.
- `summary.json` — argumentos, desglose de parámetros, métricas de test e historial completo.
- `history.csv` — curvas por época, listas para graficar.

---

### 3.4. Figuras para la presentación

```bash
uv run python -m src.training.plots                              # todas las corridas
uv run python -m src.training.plots --runs late_fusion baseline_texto
uv run python -m src.training.plots --skip_predictions           # solo la figura 1 (rápido)
```

Las figuras se guardan en `results/figures/training/`.

| Figura | Qué muestra | Para qué sección de la presentación |
| :--- | :--- | :--- |
| `01_curvas_aprendizaje_<run>.png` | Loss y PR-AUC de train vs val por época, con la mejor época marcada y la zona posterior sombreada. | *"¿Cómo evalúo la performance teniendo en cuenta overfitting y underfitting?"* |
| `02_curvas_pr_roc_<run>.png` | Curvas Precision-Recall y ROC sobre test, cada una con su línea base dibujada. | De dónde salen las dos métricas reportadas, y por qué la ROC se ve más optimista. |
| `03_comparacion_modelos.png` | PR-AUC de todas las corridas, con la línea base y el conteo de parámetros. | Comparación de alternativas arquitectónicas. |
| `04_curva_top_n_<run>.png` | Precisión y recall según cuántos productos se promocionan. | Traducción del modelo a la decisión de negocio que originó el problema. |

Las figuras 2 y 4 necesitan las predicciones sobre test: se recalculan cargando el checkpoint de
la corrida, **sin reentrenar**. Con `--skip_predictions` se omiten.

**Criterios de diseño aplicados.** La paleta se validó con el script de la metodología de
visualización (modo claro, superficie `#fcfcfb`): separación CVD ΔE 9,2 y de visión normal ΔE 27,6
en el peor par adyacente. Se usan dos series (azul `#2a78d6` para train/precisión, naranja
`#eb6834` para validación/recall), marcas finas de 2 px, grilla hairline sólida, ejes recesivos y
etiquetado directo selectivo. En la comparativa de modelos **todas las barras comparten color**:
el largo ya codifica la magnitud, de modo que un degradado por valor sería doble codificación.

---

## 4. Garantías Anti-Leakage

| Artefacto | Se ajusta con | Se aplica a |
| :--- | :--- | :--- |
| Partición train/val/test | Orden cronológico estricto, sin shuffle | — |
| Vocabulario BPE | `transformer_train.csv` | val, test |
| Medias y desvíos (z-score) | Split de train | val, test |
| Vocabularios categóricos | Split de train | val, test |
| Selección de checkpoint | Split de val | — |
| Evaluación final | — | test, **una sola vez** |

El test `test_build_dataloaders_no_filtra_informacion_entre_splits` verifica activamente esta
propiedad: si el preprocesador se hubiera ajustado con todos los datos, val quedaría centrado en 0 y
el test falla.

---

## 5. Tests

```bash
uv run pytest              # suite completa
uv run pytest -v           # con detalle por test
uv run pytest tests/test_metrics.py
```

**60 tests** distribuidos en cuatro archivos:

| Archivo | Cubre |
| :--- | :--- |
| `test_metrics.py` | Valores conocidos analíticamente: clasificador perfecto → AUC 1.0, invertido → ROC-AUC 0.0, sin señal → 0.5, PR-AUC base = prevalencia. La BCE se contrasta contra `BCEWithLogitsLoss` de PyTorch. |
| `test_preprocessor.py` | Estandarización correcta, `log1p` solo en los campos marcados, ausencia de leakage entre splits, categorías no vistas → índice 0, round-trip del artefacto JSON, columnas constantes sin división por cero. |
| `test_model.py` | Dimensiones de salida, flujo de gradientes a **ambas** ramas, los pesos de atención suman 1 y respetan la máscara, las filas sin padding no se ven afectadas por enmascarar, independencia entre muestras del lote. |
| `test_training.py` | Alineación features/etiquetas, integridad de los splits, early stopping, restauración del mejor checkpoint, reproducibilidad con semilla fija, y el **test de sobreajuste**. |
| `test_config.py` | Resolución de nombres y rutas, precedencia CLI sobre archivo, claves desconocidas y JSON inválido que fallan ruidosamente, y validación de que todas las configs versionadas en `config/` cargan sin error. |

> **El test más importante es `test_el_modelo_puede_sobreajustar_un_lote_chico`.** Con 32 ejemplos y
> sin regularización, un modelo correctamente cableado debe memorizarlos (loss < 50% de la inicial,
> PR-AUC > 0,9). Si falla, hay un error estructural —gradientes que no fluyen, etiquetas
> desalineadas, optimizador mal conectado— que ninguna verificación de dimensiones detecta.

Los tests que dependen de artefactos generados (`transformer_train.csv`, `bpe_tokenizer.json`) se
saltean automáticamente si no existen, de modo que la suite corre en un clon limpio.

---

## 6. Resultados Preliminares

Configuración base: `d_model=64`, `L=2`, `H=4`, `d_ff=256`, `d_tab=32`, `lr=1e-3`, `batch=64`,
early stopping con `patience=5` sobre PR-AUC de validación.

| Modelo | Parámetros | PR-AUC (test) | ROC-AUC (test) | Lift sobre base |
| :--- | ---: | ---: | ---: | ---: |
| Línea base (prevalencia) | — | 0,1313 | 0,5000 | 1,00× |
| **Tabular puro** (sin Transformer) | 8.879 | 0,1798 | 0,5769 | 1,37× |
| **Texto puro** (solo Transformer) | 214.401 | 0,6802 | 0,9566 | 5,18× |
| **Late fusion** (texto + tabular) | 223.151 | **0,7605** | **0,9726** | **5,79×** |

### Lectura de los resultados

**La señal principal está en el texto.** El baseline tabular apenas supera la prevalencia (lift
1,37×), consistente con lo que anticipaba el EDA: las siete numéricas tienen correlaciones lineales
con el target por debajo de |0,03|, y la señal categórica fuerte (`category`, con 13,4 pp de spread)
ya está enunciada en la descripción del producto, que el Transformer lee.

**Pero las modalidades son complementarias.** El resultado más interesante surge de comparar las
tres filas: la rama tabular **sola** rinde 0,1798, casi nada. Sumada al texto, sin embargo, lleva la
PR-AUC de 0,6802 a 0,7605, **+0,08 absoluto (+11,8% relativo)**. Es decir que aporta información que
el Transformer no extrae por su cuenta, aun cuando por sí misma esa información sea insuficiente
para predecir. Ese es el argumento empírico que justifica la arquitectura híbrida por sobre un
Transformer solo, y responde directamente a la consigna de comparar alternativas.

**El modelo sobreajusta a partir de la época 3-5.** En late fusion, la PR-AUC de train sube de 0,74
a 0,91 mientras la de validación cae de 0,71 a 0,65; en texto puro el patrón se repite antes (mejor
epoch: el 3). El early stopping restauró el mejor checkpoint en ambos casos. Es el comportamiento
esperado con 7.000 filas y ~220k parámetros, y justifica explorar mayor dropout y weight decay.

> **Salvedad metodológica:** los valores de arriba corresponden a **una única corrida con semilla
> fija (42)**. La PR-AUC de validación fluctúa notablemente entre épocas contiguas (0,69–0,71 en late
> fusion), de modo que la diferencia de +0,08 debería confirmarse promediando varias semillas antes
> de afirmarla en el informe. Es el siguiente experimento a correr.

---

## 7. Plan de Ablación Pendiente

| Eje | Variantes | Flag |
| :--- | :--- | :--- |
| Rama activa | texto / tabular / ambas | `--no_text`, `--no_tabular` |
| Módulo de fusión | late vs cross-attention | `--fusion` |
| Profundidad | `num_layers` ∈ {1, 2, 4} | `--num_layers` |
| Ancho | `d_model` ∈ {32, 64, 96} | `--d_model` |
| Cabezales | `n_heads` ∈ {2, 4, 8} | `--n_heads` |
| Pooling | mean / cls / max | `--pooling` |
| Codificación posicional | sinusoidal / learned / none | `--pos_encoding` |
| Codificación categórica | one-hot vs entity embeddings | `--cat_encoding` |
| Campos de texto | con y sin `brand` | `--text_fields` |
| Regularización | `dropout`, `weight_decay` | `--dropout`, `--weight_decay` |

Cada corrida deja su `summary.json`, de modo que la tabla comparativa final se arma agregando esos
archivos.
