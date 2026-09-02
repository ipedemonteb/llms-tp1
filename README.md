# TP1 - Large Language Models

El presente trabajo implementa y evalúa una arquitectura de Deep Learning basada en Transformers para predecir la tasa de conversión (*Buy-Through Rate* o BTR) de productos de supermercado en e-commerce. A partir de un catálogo multimodal compuesto por textos descriptivos (título, descripción, lista de ingredientes) y metadatos tabulares (precios, categorías, procedencia, alérgenos, puntuaciones nutricionales), se diseña desde cero un pipeline integral que incluye tokenización Byte-Level BPE, codificadores tabulares con *Entity Embeddings*, bloques Transformer con atención multi-cabezal, y mecanismos de integración multimodal (*Late Fusion* y *Cross-Attention*), complementado con un estudio comparativo riguroso frente a un modelo *Raw Transformer* serializado.

Las funcionalidades incluidas son las siguientes:

- <b>Tokenizador Byte-Level BPE desde cero</b>: Implementación pura en Python de Byte-Level Byte-Pair Encoding (`ByteLevelBPETokenizer`) con control exacto de vocabulario ($V=1000$), tokens especiales (`<pad>`, `<unk>`, `<cls>`, `<sep>`), serialización JSON y análisis de entropía y fertilidad.
- <b>Pipeline de datos y preprocesamiento multimodal</b>: Limpieza automatizada del catálogo, extracción de features de texto enriquecidas, escalado y transformaciones logarítmicas de variables continuas, y codificación de variables categóricas ajustada exclusivamente sobre el conjunto de entrenamiento.
- <b>Arquitectura Transformer Encoder & Híbrida modular</b>: Bloques Transformer custom (`TextTransformerEncoder`) con multi-head self-attention, codificaciones posicionales sinusoidales o aprendidas, capas de feed-forward con activación GELU, normalización de capa y múltiples modos de agregación (*mean*, *cls*, *max* pooling).
- <b>Mecanismos de fusión multimodal</b>: Soporte para integración tardía por concatenación latente (*Late Fusion*) y atención cruzada (*Cross-Attention*) donde las representaciones tabulares consultan la secuencia completa de tokens de texto sin compresión previa.
- <b>Motor de entrenamiento reproducible y métricas de negocio</b>: `Trainer` en PyTorch con optimizador AdamW, programación de tasa de aprendizaje, *early stopping* sobre PR-AUC de validación, checkpointing del mejor estado y evaluación de métricas de ranking y calibración (PR-AUC, ROC-AUC, BCE Loss, Brier Score y Lift sobre baseline).
- <b>Suite de ablaciones y barridos multi-semilla</b>: Scripts automatizados para aislar y cuantificar el aporte de cada componente mediante corridas sobre 5 semillas aleatorias fijas: dimensión de *Entity Embeddings* ($d_{\text{emb}}$), hiperparámetros del Transformer ($d_{\text{model}}$, $L$, $H$, pooling, posición), estrategias de regularización anti-overfitting y desredundancia tabular.
- <b>Baseline Raw Transformer y comparación pareada</b>: Pipeline completo de serialización de catálogo en lenguaje natural (*Raw text*), tokenizador especializado y comparativa estadística pareada ($t$-test y test de Wilcoxon) frente al modelo híbrido.
- <b>Sistema de visualización y resultados estructurados</b>: Generación automatizada de curvas de aprendizaje, gráficos de barras de error (media ± desvío), matrices de correlación y paneles comparativos con estilo visual sobrio y accesible en `results/figures/` y tablas consolidadas en `results/aggregate/`.

<details>
  <summary>Contenidos</summary>
  <ol>
    <li><a href="#instalación">Instalación</a></li>
    <li><a href="#instrucciones">Instrucciones</a></li>
    <li><a href="#manual-de-usuario">Manual de Usuario</a></li>
    <li><a href="#integrantes">Integrantes</a></li>
  </ol>
</details>

## Instalación

Clonar el repositorio:

- HTTPS:
  ```sh
  git clone https://github.com/ipedemonteb/llms-tp1.git
  ```
- SSH:
  ```sh
  git clone git@github.com:ipedemonteb/llms-tp1.git
  ```

Configurar el entorno con `uv` (recomendado):

```sh
cd llms-tp1
uv sync
```

O utilizando entorno virtual estándar con `venv` y `pip` (Python >= 3.12):

```sh
cd llms-tp1
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -e .
```

> **Requisitos**: Python 3.12+ (compatible con PyTorch 2.x, NumPy, Pandas, Scikit-Learn y Matplotlib). El gestor de paquetes `uv` es recomendado para reproducibilidad exacta del entorno.

<p align="right">(<a href="#tp1---large-language-models">Volver</a>)</p>

## Instrucciones

Todos los comandos deben ejecutarse desde la raíz del repositorio con el entorno virtual activado (`uv run` ejecuta automáticamente en el entorno).

- Preparación y particionado del dataset:
  ```sh
  uv run python -m src.data_extraction.clean_dataset
  uv run python -m src.data_extraction.build_transformer_dataset
  ```
- Entrenamiento del tokenizador Byte-Level BPE:
  ```sh
  uv run python -m src.tokenizer.bpe
  ```
- Análisis exploratorio de datos (EDA):
  ```sh
  uv run python -m src.data_analysis.eda_runner
  ```
- Entrenamiento de una corrida individual (ej. modelo híbrido Late Fusion):
  ```sh
  uv run python -m src.training.train --config late_fusion
  ```
- Generación de curvas de aprendizaje de corridas individuales:
  ```sh
  uv run python -m src.training.plots
  ```
- Estudio de ablación de la rama tabular (5 semillas):
  ```sh
  uv run python -m src.training.experiments.tabular_ablation
  ```
- Estudio de ablación del Transformer de texto (5 semillas):
  ```sh
  uv run python -m src.training.experiments.transformer_ablation --study all
  ```
- Evaluación del modelo híbrido consolidado (Late Fusion vs. Cross-Attention):
  ```sh
  uv run python -m src.training.experiments.run_cross_attention_evaluation
  ```
- Barrido de escalabilidad y sobreajuste por $d_{\text{model}}$:
  ```sh
  uv run python -m src.training.experiments.run_dmodel_sweep
  ```
- Barrido de regularización anti-overfitting:
  ```sh
  uv run python -m src.training.experiments.run_regularization_sweep
  ```
- Experimento con rama tabular desredundada:
  ```sh
  uv run python -m src.training.experiments.run_non_redundant_tabular_experiment
  ```
- Pipeline y comparativa pareada del Raw Transformer:
  ```sh
  uv run python -m src.raw_transformer.train_tokenizer
  uv run python -m src.raw_transformer.train --seed 42
  uv run python -m src.raw_transformer.compare
  ```
- Ejecución de la suite completa de tests:
  ```sh
  uv run pytest
  ```

<p align="right">(<a href="#tp1---large-language-models">Volver</a>)</p>

## Manual de Usuario

### 1. Extracción de Datos y Tokenizador BPE

```sh
# 1. Limpieza y generación de features de texto
uv run python -m src.data_extraction.clean_dataset

# 2. División estratificada en train (70%), val (15%) y test (15%)
uv run python -m src.data_extraction.build_transformer_dataset

# 3. Entrenamiento del tokenizador BPE sobre el texto de entrenamiento
uv run python -m src.tokenizer.bpe
```

Archivos generados:

- `resources/datasets/transformer_{train,val,test}.csv`: particiones del catálogo con columnas de texto unificado y variables tabulares.
- `resources/tokenizer/bpe_tokenizer.json`: vocabulario BPE ($V=1000$), pares de fusión aprendidos y tokens especiales.
- `results/figures/tokenizer/`: histogramas de distribución de longitud en tokens y curvas de fertilidad.

### 2. Análisis Exploratorio de Datos (EDA)

```sh
uv run python -m src.data_analysis.eda_runner
```

- Analiza el balance del target BTR, embudo de conversión (Cart vs Buy), distribuciones numéricas, cardinalidades categóricas y consistencia entre campos de texto.
- Genera 9 figuras científicas de diagnóstico en `results/figures/dataset_info/`.
- Scripts ad-hoc adicionales disponibles en `src/data_analysis/scripts/` (ej. `brand_title_consistency.py`, `description_structure.py`).

### 3. Entrenamiento Individual de Modelos

```sh
uv run python -m src.training.train [OPCIONES]
```

Parámetros principales:

- `--config`: configuración base JSON predefinida en `config/` (`late_fusion`, `cross_attention`, `baseline_texto`, `baseline_tabular`).
- `--d_model`: dimensión latente de los embeddings de texto (default: `64`).
- `--num_layers`: cantidad de capas / bloques del Transformer Encoder (default: `2`).
- `--n_heads`: cantidad de cabezales de atención multi-head (default: `4`).
- `--pooling`: método de agregación de la secuencia (`mean`, `cls`, `max`).
- `--pos_encoding`: codificación posicional (`sinusoidal`, `learned`, `none`).
- `--fusion`: tipo de integración multimodal (`late` para concatenación, `cross` para atención cruzada).
- `--embedding_dim`: dimensión de los Entity Embeddings tabulares (default: `None`, cálculo dinámico $\min(16, \lceil c/2 \rceil)$).
- `--lr`: tasa de aprendizaje inicial para AdamW (default: `1e-3`).
- `--weight_decay`: factor de decaimiento de pesos $L_2$ (default: `0.01`).
- `--dropout`: probabilidad de dropout (default: `0.1`).
- `--epochs`: máximo de épocas de entrenamiento (default: `20`).
- `--patience`: épocas de tolerancia para early stopping sobre `val_pr_auc` (default: `5`).
- `--seed`: semilla aleatoria para inicialización de pesos y split loaders (default: `42`).
- `--run_name`: identificador de la corrida (si se omite, se genera automáticamente a partir de la arquitectura).

Archivos generados en `results/runs/<run_name>/`:

- `history.json`: historial de métricas por época (pérdidas, PR-AUC, ROC-AUC en train y val).
- `summary.json`: métricas finales consolidadas en train, val y test, cantidad de parámetros y tiempos.
- `best_model.pt`: checkpoint con los pesos del modelo en la época de óptimo PR-AUC de validación.
- `predictions_test.csv`: probabilidades predichas $\hat{y}$ vs etiquetas reales $y$ en el conjunto de test.

### 4. Estudio de Ablación Tabular (Entity Embeddings)

```sh
uv run python -m src.training.experiments.tabular_ablation \
  --seeds 7 42 123 456 999 \
  --epochs 20
```

- Evalúa sistemáticamente la rama tabular aislada conectada directamente al clasificador final variando la dimensión de Entity Embeddings: One-Hot crudo ($d_{\text{emb}}=0$), $d_{\text{emb}} \in [2, 4, 8, 16, 32]$ y asignación adaptativa (`auto`).
- Reporta PR-AUC, ROC-AUC y BCE Loss en Train, Val y Test con media ± desvío estándar multi-semilla.
- Salidas en `results/aggregate/tabular_ablation/` (`tabular_ablation_summary.csv`) y figuras en `results/figures/tabular_ablation/`.

### 5. Estudio de Ablación del Transformer de Texto

```sh
# Ejecutar todos los estudios de texto
uv run python -m src.training.experiments.transformer_ablation --study all

# O ejecutar un estudio específico: d_model, layers, heads, pos_encoding, pooling
uv run python -m src.training.experiments.transformer_ablation --study d_model
```

- Evalúa la rama de texto puro en 5 dimensiones arquitecturales independientes:
  1. Dimensión latente ($d_{\text{model}} \in [32, 48, 64, 96]$)
  2. Profundidad ($L \in [1, 2, 3, 4]$ capas)
  3. Cabezales de atención ($H \in [1, 2, 4, 8]$)
  4. Positional Encoding (*sinusoidal* vs. *learned* vs. *none*)
  5. Pooling de texto (*mean* vs. *cls* vs. *max*)
- Salidas en `results/aggregate/transformer_ablation/` y figuras en `results/figures/transformer_ablation/`.

### 6. Evaluación y Barridos del Modelo Híbrido

```sh
# Evaluación general multi-semilla del modelo híbrido
uv run python -m src.training.experiments.hybrid_evaluation \
  --d_model 96 --num_layers 1 --n_heads 1 --epochs 15

# Comparación directa Late Fusion vs. Cross-Attention
uv run python -m src.training.experiments.run_cross_attention_evaluation

# Barrido de escalabilidad y sobreajuste por d_model
uv run python -m src.training.experiments.run_dmodel_sweep

# Barrido de regularización (dropout, learning rate, weight decay)
uv run python -m src.training.experiments.run_regularization_sweep

# Experimento con variables tabulares estrictamente desredundadas
uv run python -m src.training.experiments.run_non_redundant_tabular_experiment
```

- Cuantifica el balance entre capacidad de memorización y generalización, analizando la brecha $\Delta(\text{Train} - \text{Test})$ de PR-AUC y la dinámica de la función de pérdida.
- Salidas en `results/aggregate/hybrid_baseline/` y figuras comparativas en `results/figures/hybrid_baseline/`.

### 7. Baseline Raw Transformer y Comparación Pareada

```sh
# 1. Entrenamiento del tokenizador para texto serializado
uv run python -m src.raw_transformer.train_tokenizer

# 2. Entrenamiento multi-semilla del Raw Transformer
uv run python -m src.raw_transformer.train --seed 7
uv run python -m src.raw_transformer.train --seed 42
uv run python -m src.raw_transformer.train --seed 123

# 3. Comparación estadística pareada (Raw vs. Hybrid)
uv run python -m src.raw_transformer.compare
```

- Serializa el producto completo en formato textual plano (*pseudo-JSON* o plantilla clave-valor).
- Realiza contrastes de hipótesis pareados ($t$-test dependiente y Wilcoxon) semilla a semilla calculando intervalos de confianza del 95% para la diferencia $\Delta \text{PR-AUC} = \text{PR-AUC}_{\text{hybrid}} - \text{PR-AUC}_{\text{raw}}$.
- Salidas en `results/aggregate/` (`pareado.csv`, `resumen.csv`) y curvas en `results/figures/training/`.

### 8. Visualización y Curvas de Aprendizaje

```sh
# Generar curvas de todas las corridas individuales en results/runs/
uv run python -m src.training.plots

# Filtrar por corridas específicas
uv run python -m src.training.plots --runs late_fusion_s42 baseline_texto_s42 baseline_tabular_s42

# Modo rápido (solo curvas de pérdida y PR-AUC por época)
uv run python -m src.training.plots --skip_predictions
```

- Produce figuras individuales en `results/figures/training/<run_name>/`:
  - `01_learning_curves.png`: dinámica de pérdida BCE y PR-AUC por época (Train vs. Val).
  - `02_pr_roc_curves.png`: curvas Precision-Recall (con línea de prevalencia base) y ROC en el split de Test.
  - `03_top_n_curve.png`: tasa de captura de conversiones según el percentil de productos recomendados.
- Produce además el gráfico comparativo global `results/figures/training/comparacion_modelos.png`.

### 9. Suite de Tests Automatizados

```sh
uv run pytest
```

- Ejecuta los 147 tests unitarios y de integración distribuidos en:
  - `tests/test_tokenizer.py`: entrenamiento BPE, idempotencia de tokenización, encode/decode.
  - `tests/test_preprocessor.py`: ajuste y transformación de features continuas y categóricas.
  - `tests/test_model.py`: formas de tensores, atención multi-cabezal, máscaras de padding y forward pass multimodal.
  - `tests/test_training.py`: ciclo del `Trainer`, *early stopping*, reproducibilidad con semilla fija y test de sobreajuste en batch único.
  - `tests/test_metrics.py`: cálculo de PR-AUC, ROC-AUC, Brier score y lift.
  - `tests/test_config.py`: resolución y validación de esquemas JSON y nombrado determinista de corridas.
  - `tests/test_aggregate.py`: agregación estadística, cálculo de media/desvío e intervalos de confianza.
  - `tests/test_raw_transformer.py`: serialización de catálogo y pipeline de texto crudo.

<p align="right">(<a href="#tp1---large-language-models">Volver</a>)</p>

## Integrantes

Martín Alejandro Barnatán (64463) - mbarnatan@itba.edu.ar  
Juan Ignacio Cantarella (64509) - jcantarella@itba.edu.ar  
Ignacio Pedemonte Berthoud (64908) - ipedemonteberthoud@itba.edu.ar

<p align="right">(<a href="#tp1---large-language-models">Volver</a>)</p>