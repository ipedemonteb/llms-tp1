# Diseño e Implementación del Tokenizador Byte-Level BPE

**73.69 Large Language Models — Trabajo Práctico 1**  
**Módulo:** Tokenización para Arquitecturas Transformer  
**Archivo de Implementación:** `src/tokenizer/bpe.py`  
**Artefacto Serializado:** `resources/tokenizer/bpe_tokenizer.json`

---

## 1. Introducción y Justificación Teórica

Para procesar las señales de texto del catálogo de supermercado (`title_clean`, `badge`, `description`, `ingredients`, `country_of_origin`, `allergens`), se implementó un algoritmo de tokenización a nivel de subpalabras basado en **Byte-Level BPE (Byte-Pair Encoding)**.

### ¿Por qué Byte-Level BPE frente a otras alternativas?

```
                               ┌──> 1. Word-Level ──────> OOV Severo en ingredientes raros / Vocabulario gigante
                               │
Alternativas de Tokenización ──┼──> 2. Char-Level ──────> Secuencias muy largas (O(N^2) en atención) / Pobreza semántica
                               │
                               └──> 3. Byte-Level BPE ──> Vocabulario compacto + Secuencias balanceadas + CERO OOV
```

1. **Garantía Matemática de Cero OOV (*Out-Of-Vocabulary*):**
   * El BPE tradicional arranca a nivel de caracteres Unicode. Si en *Test* aparece un símbolo o caracter no visto en *Train* (ej. `™`, `°`, `½`, tildes raras), genera un token `[UNK]`, perdiendo información.
   * **Byte-Level BPE** utiliza como alfabeto base los **256 valores posibles de un byte UTF-8** (`0x00` a `0xFF`). Cualquier texto, símbolo o caracter arbitrario en UTF-8 se puede descomponer en bytes, por lo que **el modelo jamás emite un token `[UNK]`**.
2. **Dimensionamiento Óptimo de la Matriz de Embedding ($|V| \times d_{\text{model}}$):**
   * En lugar de usar un tokenizador sobredimensionado de un modelo preentrenado (ej. BERT con $|V|=30.522$, que requeriría casi 2 millones de parámetros en `nn.Embedding` para $d_{\text{model}}=64$), Byte-Level BPE nos permite entrenar un vocabulario a medida. El artefacto entrenado sobre nuestro corpus tiene $|V|=1.720$, lo que reduce la matriz de embedding a $1.720 \times 64 = 110.080$ parámetros.
3. **Manejo de Morfología de Catálogo e Ingredientes:**
   * Descompone palabras compuestas o técnicas de ingredientes (*"polyphosphate"* $\to$ `["poly", "phosphate"]`, *"steamable"* $\to$ `["steam", "able"]`) conservando su raíz semántica.

---

## 2. Arquitectura del Pipeline de Tokenización

La clase `ByteLevelBPETokenizer` en `src/tokenizer/bpe.py` encapsula un pipeline de 4 etapas optimizado:

```
[ Texto Crudo ]
       │
       ▼
1. Pre-Tokenization (ByteLevel) ───> Segmenta espacios/palabras usando byte-fallback (prefijo 'Ġ')
       │
       ▼
2. BPE Model & Merges ─────────────> Aplica reglas de fusión aprendidas sobre 256 bytes iniciales
       │
       ▼
3. Post-Processing (Template) ─────> Inserta tokens de control: [CLS] al inicio, [SEP] al final
       │
       ▼
4. Padding & Truncation ───────────> Genera Tensores PyTorch (input_ids, attention_mask)
```

### Tokens Especiales Configurados

| Token | ID Numérico | Rol en la Arquitectura Transformer |
| :--- | :---: | :--- |
| **`[PAD]`** | `0` | Relleno para secuencias de longitud fija dentro de un mini-batch. |
| **`[UNK]`** | `1` | Token de fallback (reservado, no utilizado en BBPE por cobertura total de bytes). |
| **`[CLS]`** | `2` | Token de inicio de secuencia (*Classification Token* / agregación global). |
| **`[SEP]`** | `3` | Token de fin de secuencia o separador de campos. |
| **`[MASK]`** | `4` | Token reservado para tareas auto-supervisadas de enmascaramiento (*Masked LM*). |

---

## 3. Hiperparámetros Configurables

La implementación permite parametrizar los siguientes hiperparámetros tanto desde CLI como desde código:

| Hiperparámetro | Tipo | Default | Descripción e Impacto |
| :--- | :---: | :---: | :--- |
| **`vocab_size`** | `int` | `2048` | Tamaño **máximo** del vocabulario final; es un techo, no un objetivo alcanzado. Con `min_frequency=2` el corpus **satura en 1.720 tokens**: no quedan pares de bytes que aparezcan al menos dos veces y el entrenamiento se detiene antes del techo. Subir este valor a 4.096 no cambia nada salvo que se baje `min_frequency` en paralelo — tenerlo en cuenta al diseñar la ablación. |
| **`min_frequency`** | `int` | `2` | Frecuencia mínima que debe tener un par de bytes en el corpus de Train para calificar a una fusión (*merge*). Evita memorizar combinaciones raras o erróneas. |
| **`max_length`** | `int` | `128` | Longitud máxima de secuencia (en tokens). Controla el límite de truncamiento y padding. Sobre la composición actual de 6 campos, las secuencias miden **52,5 tokens en promedio y 65 como máximo**, de modo que 128 no trunca ninguna fila de ningún split. Hay margen de sobra para incorporar más campos. |
| **`special_tokens`** | `list[str]` | `["[PAD]", ...]` | Lista ordenada de tokens especiales asignados a los primeros IDs (`0, 1, 2, ...`). |
| **`text_column`** | `str` | `"text"` | Nombre de la columna en el archivo CSV de datos que contiene la secuencia de texto unificada. |

---

## 4. Instrucciones de Uso y Ejecución

### 4.1. Entrenamiento desde la Línea de Comandos (CLI)

Para entrenar el tokenizador sobre el conjunto de entrenamiento (`resources/datasets/transformer_train.csv`) y guardarlo:

```bash
# Ejecución estándar con hiperparámetros por defecto
uv run python -m src.tokenizer.bpe

# Ejecución personalizada (ejemplo para estudio de ablación con vocab_size=4096 y max_length=128)
uv run python -m src.tokenizer.bpe \
    --train_file resources/datasets/transformer_train.csv \
    --text_column text \
    --save_path resources/tokenizer/bpe_tokenizer_4096.json \
    --vocab_size 4096 \
    --min_frequency 2 \
    --max_length 128
```

---

### 4.2. Llamada desde Código Python (en cualquier módulo Transformer)

Cualquier modelo o script dentro del proyecto (ej. `src/hybrid_transformer/`) puede importar y utilizar el tokenizador directamente:

```python
import torch
from src.tokenizer.bpe import ByteLevelBPETokenizer

# 1. Cargar el tokenizador pre-entrenado
tokenizer = ByteLevelBPETokenizer.from_file(
    path="resources/tokenizer/bpe_tokenizer.json",
    max_length=128
)

# 2. Tokenizar un único texto (inspección de tokens)
texto = (
    "Cedar House Steamable Pepperoni Pizza | Well Reviewed | "
    "Steamable pepperoni pizza in a 10 oz package. Listed under frozen. | "
    "Prepared ingredients, Spices, Salt | United States | Wheat"
)
encoding = tokenizer.encode(texto)
print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)

# 3. Tokenización por Lotes (Batch) lista para PyTorch
batch_textos = [
    texto,
    "Sunny Basket Ready To Heat Waffles | Customer Favorite | "
    "Ready to heat waffles in a 6 ct package. Listed under frozen. | "
    "Flour, Sugar, Eggs | Canada | Milk",
]

batch = tokenizer.encode_batch(
    texts=batch_textos,
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors="pt"  # Devuelve tensores de PyTorch
)

input_ids = batch["input_ids"]           # torch.LongTensor de dimensión (batch_size, 128)
attention_mask = batch["attention_mask"] # torch.LongTensor de dimensión (batch_size, 128)

print("Shape input_ids:", input_ids.shape)
print("Shape attention_mask:", attention_mask.shape)

# 4. Decodificar IDs a texto original
texto_recuperado = tokenizer.decode(input_ids[0])
print("Texto decodificado:", texto_recuperado)
```

---

### 4.3. Integración en un `PyTorch Dataset` / `DataLoader`

Ejemplo de cómo integrar el tokenizador en la clase de carga de datos para alimentar el Transformer:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.tokenizer.bpe import ByteLevelBPETokenizer

class SupermarketTextDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: ByteLevelBPETokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.df["text"].fillna("").tolist()
        self.labels = self.df["bought"].astype(int).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def get_collate_fn(tokenizer: ByteLevelBPETokenizer, max_length: int = 128):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer.encode_batch(
            texts=list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        }
    return collate_fn

# Uso con DataLoader
tokenizer = ByteLevelBPETokenizer.from_file("resources/tokenizer/bpe_tokenizer.json")
dataset = SupermarketTextDataset("resources/datasets/transformer_train.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=get_collate_fn(tokenizer))
```

---

## 5. Prevención de Data Leakage

Para garantizar la validez metodológica del trabajo y respetar la consigna:
* El vocabulario y las reglas de fusión de BPE se ajustan **estrictamente sobre el split de entrenamiento (`transformer_train.csv`)**.
* Los splits de **Validación (`transformer_val.csv`)** y **Test (`transformer_test.csv`)** únicamente se codifican utilizando las reglas aprendidas en Train, simulando con precisión el comportamiento en inferencia real.

---

## 6. Reentrenamiento ante Cambios en la Composición del Texto

El vocabulario BPE es **específico del corpus sobre el que se entrenó**. Si cambia la lista de
campos que componen la columna `text` (`DEFAULT_TEXT_FIELDS` o el flag `--text_fields` de
`build_transformer_dataset.py`), el tokenizador **debe reentrenarse**:

```bash
uv run python -m src.tokenizer.bpe --train_file resources/datasets/transformer_train.csv
```

No hacerlo no produce un error —el byte-level garantiza cobertura total y nunca emite `[UNK]`—
pero degrada silenciosamente la representación: los valores nunca vistos se fragmentan en bytes
sueltos en lugar de recibir tokens dedicados. Medición real al incorporar `country_of_origin` y
`allergens` a la secuencia:

| Valor | Vocabulario desactualizado | Reentrenado |
| :--- | :---: | :---: |
| `No Allergens` | 8 tokens | **3** |
| `United States` | 6 tokens | **3** |
| `Tree nuts` | 6 tokens | **3** |
| `Shellfish` | 5 tokens | **3** |
| **Longitud media de secuencia** | 58,81 | **52,46** (−10,8%) |

El impacto de fondo no es el ahorro de cómputo sino la calidad de la representación: con el
vocabulario viejo el modelo debía componer 8 fragmentos de bytes sin significado para recuperar
el concepto "sin alérgenos", presente en el 44% de las filas de entrenamiento. Con el vocabulario
reentrenado ese concepto es **un único token** al que la atención puede apuntar directamente.

El entrenamiento es determinista: dos corridas sobre el mismo `transformer_train.csv` producen
artefactos byte-idénticos, de modo que `bpe_tokenizer.json` es reproducible desde el pipeline
versionado.
