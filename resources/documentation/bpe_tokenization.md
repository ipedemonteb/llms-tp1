# Diseño e Implementación del Tokenizador Byte-Level BPE

**73.69 Large Language Models — Trabajo Práctico 1**  
**Módulo:** Tokenización para Arquitecturas Transformer  
**Archivo de Implementación:** `src/tokenizer/bpe.py`  
**Artefacto Serializado:** `resources/tokenizer/bpe_tokenizer.json`

---

## 1. Introducción y Justificación Teórica

Para procesar las señales de texto del catálogo de supermercado (`title_clean`, `badge`, `description`, `ingredients`), se implementó un algoritmo de tokenización a nivel de subpalabras basado en **Byte-Level BPE (Byte-Pair Encoding)**.

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
   * En lugar de usar un tokenizador sobredimensionado de un modelo preentrenado (ej. BERT con $|V|=30.522$, que requeriría casi 2 millones de parámetros en `nn.Embedding` para $d_{\text{model}}=64$), Byte-Level BPE nos permite entrenar un vocabulario a medida ($|V| \in [1.024, 4.096]$), reduciendo los parámetros a solo $\approx 130.000$.
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
| **`vocab_size`** | `int` | `2048` | Tamaño máximo del vocabulario final. Determina la cantidad de fusiones (*merges*) aprendidas. A mayor tamaño, secuencias más cortas pero mayor memoria en la capa de Embedding. Rango sugerido para ablación: `[1024, 2048, 4096]`. |
| **`min_frequency`** | `int` | `2` | Frecuencia mínima que debe tener un par de bytes en el corpus de Train para calificar a una fusión (*merge*). Evita memorizar combinaciones raras o erróneas. |
| **`max_length`** | `int` | `128` | Longitud máxima de secuencia (en tokens). Controla el límite de truncamiento y padding. En nuestro corpus de supermercado, 128 cubre holgadamente el 99% de las secuencias combinadas (`title + badge + description + ingredients`). |
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
texto = "Cedar House Steamable Pepperoni Pizza | Best Seller | Crispy crust | Flour, Yeast"
encoding = tokenizer.encode(texto)
print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)

# 3. Tokenización por Lotes (Batch) lista para PyTorch
batch_textos = [
    "Cedar House Steamable Pepperoni Pizza | Best Seller | Crispy crust | Flour, Yeast",
    "Organic Whole Milk | Customer Favorite | Fresh dairy | Grade A Milk"
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
