# Arquitectura y Diseño del Transformer Encoder de Texto

**73.69 Large Language Models — Trabajo Práctico 1**  
**Módulo:** Rama de Procesamiento de Lenguaje Natural (NLP)  
**Archivo de Implementación:** `src/hybrid_transformer/text_encoder.py`  
**Paquete:** `src/hybrid_transformer`  

---

## 1. Introducción y Rol en la Arquitectura Global

En el marco del sistema de predicción de **Buy Through Rate (BTR)** (`bought \in {0, 1}`), el catálogo de productos de supermercado contiene un conjunto rico de señales textuales y semánticas no estructuradas:
* `title_clean`: Título comercial del producto limpio. Incluye la marca como prefijo literal en el 100% de las filas, de modo que `brand` entra al encoder por esta vía.
* `badge` / `title_tag`: Etiquetas de prueba social (*"Best Seller"*, *"Customer Favorite"*, *"Top Rated"*, etc.).
* `description`: Descripción comercial detallada (atributos de sabor, textura, preparación). Enuncia además `category`, `storage_type` y `unit_of_measure` en prosa.
* `ingredients`: Listado de materias primas y aditivos alimentarios.
* `country_of_origin`: País de origen. No aparece en ningún otro campo textual, por lo que se incorpora explícitamente.
* `allergens`: Alérgenos declarados. Solo el 35% de las filas que declaran alérgeno lo mencionan en la prosa, por lo que también se incorpora.

El módulo **`TextTransformerEncoder`** implementa un modelo **Transformer Encoder-Only** (siguiendo los principios de Vaswani et al., 2017 y BERT) encargado de procesar la salida del tokenizador Byte-Level BPE y mapear las secuencias de texto a un espacio latente continuo y denso, produciendo el vector contextual $e_{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$.

Este vector $e_{\text{text}}$ representa la semántica unificada del producto y está diseñado para acoplarse con los embeddings de las variables categóricas y las proyecciones numéricas en la etapa de **Fusión Multimodal** (Late Fusion o Cross-Attention) antes de la cabeza clasificadora MLP.

```
[ "Cedar House Pizza | Well Reviewed | ... frozen storage. | Flour, Yeast | United States | Wheat" ]
                                      │
                                      ▼
                        Byte-Level BPE Tokenizer
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
            input_ids: (B, T)              attention_mask: (B, T)
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
                        TextTransformerEncoder (L capas)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
       Pooling = "mean" / "cls" / "max"        Pooling = "none"
                      │                               │
                      ▼                               ▼
       Vector e_text: (B, d_model)        Secuencia H_text: (B, T, d_model)
                      │                               │
                      ▼                               ▼
              [ LATE FUSION ]                [ CROSS-ATTENTION ]
        (Concatenación con e_tab)       (Atención guiada por Query e_tab)
```

---

## 2. Diagrama Arquitectónico y Flujo de Tensores

El modelo está implementado modularmente en PyTorch desde sus bloques fundamentales para garantizar interpretabilidad total, control de gradientes y soporte para ablación:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 input_ids: (B, T)                      │
                  └───────────────────────────┬────────────────────────────┘
                                              ▼
                             nn.Embedding(vocab_size, d_model)
                                              ▼
                             Escalamiento: x * sqrt(d_model)
                                              ▼
                        + PositionalEncoding(sinusoidal / learned)
                                              ▼
                                       Dropout(p=0.1)
                                              │
                      ┌───────────────────────┴────────────────────────────┐
                      │  Stack de L x TransformerEncoderBlock              │
                      │                                                    │
                      │   ┌──────────────────────────────────────────┐     │
                      │   │ Pre-LayerNorm 1                          │     │
                      │   │                   │                      │     │
                      │   │ Multi-Head Self-Attention (Q, K, V)      │     │
                      │   │                   │ (con attention_mask) │     │
                      │   │ Conexión Residual + Dropout              │     │
                      │   │                   │                      │     │
                      │   │ Pre-LayerNorm 2                          │     │
                      │   │                   │                      │     │
                      │   │ Position-wise Feed-Forward (GELU/ReLU)   │     │
                      │   │                   │                      │     │
                      │   │ Conexión Residual + Dropout              │     │
                      │   └──────────────────────────────────────────┘     │
                      └───────────────────────┬────────────────────────────┘
                                              ▼
                                  Final LayerNorm(d_model)
                                              ▼
                      Mecanismo de Pooling ('mean' | 'cls' | 'max' | 'none')
                                              ▼
                                 Salida e_text / H_text
```

---

## 3. Detalle de los Módulos Internos

### 3.1. Capa de Embedding (`nn.Embedding`)
* Utiliza la clase estándar nativa de PyTorch: `nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_token_id)`.
* **Manejo de Padding:** Los tokens `[PAD]` (ID 0) se fijan con gradiente nulo (`padding_idx=0`), impidiendo que el relleno afecte la optimización de los pesos.
* **Escalamiento por $\sqrt{d_{\text{model}}}$:** De acuerdo con Vaswani et al. (2017), los vectores de embedding se multiplican por $\sqrt{d_{\text{model}}}$ para balancear la magnitud de las activaciones frente a la codificación posicional.

---

### 3.2. Codificación Posicional (`PositionalEncoding`)
Dado que la auto-atención es una operación invariante ante permutaciones, se debe inyectar la noción de orden secuencial:

1. **Sinusoidal Fija (Vaswani et al., 2017) — `pos_encoding_type="sinusoidal"` (Default):**
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
   * **Ventaja:** No consume parámetros entrenables ($0$ params extra), generaliza a longitudes no vistas y preserva distancias relativas por propiedades trigonométricas ($\cos(\alpha+\beta)$).
2. **Positional Embeddings Aprendibles (BERT-style) — `pos_encoding_type="learned"`:**
   * Matriz `nn.Embedding(max_seq_len, d_model)` ajustada por retropropagación.
3. **Ablación Sin Posición — `pos_encoding_type="none"`:**
   * Desactiva la suma posicional para cuantificar empíricamente el valor del orden sintáctico en el catálogo.

---

### 3.3. Auto-Atención Multi-Cabezal (`MultiHeadSelfAttention`)
Permite al modelo capturar dependencias entre tokens lejanos (ej. relacionar la marca del inicio con un alérgeno al final de la descripción):

* **Dimensiones internas:**
  $$d_k = d_v = \frac{d_{\text{model}}}{H}$$
  Para nuestra arquitectura base con $d_{\text{model}}=64$ y $H=4$, cada cabezal opera en $d_k=16$.
* **Mecanismo Scaled Dot-Product con Máscara de Padding:**
  $$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
  $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)$$
  $$\mathbf{M}_{ij} = \begin{cases} 0 & \text{si el token } j \text{ es válido} \\ -10^9 & \text{si el token } j \text{ es } \text{[PAD]} \end{cases}$$
  $$\text{Output} = (\mathbf{A} \mathbf{V}) \mathbf{W}_O$$
* **Explicabilidad (XAI):** Permite retornar opcionalmente la matriz completa de pesos de atención $\mathbf{A} \in \mathbb{R}^{B \times H \times T \times T}$ mediante el flag `return_attention=True` para graficar mapas de atención e interpretar qué palabras concentran el interés del modelo.

---

### 3.4. Red Feed-Forward Posicional (`PositionwiseFeedForward`)
Aplica dos transformaciones lineales continuas separadas por una función de activación no lineal y Dropout:
$$\text{FFN}(x) = \text{Dropout}\left(\sigma(x \mathbf{W}_1 + \mathbf{b}_1)\right) \mathbf{W}_2 + \mathbf{b}_2$$
* $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (donde típicamente $d_{\text{ff}} = 4 \times d_{\text{model}} = 256$).
* $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.
* **Activación ($\sigma$):** Soporta `GELU` (Gaussian Error Linear Unit, estándar moderno más suave) y `ReLU`.

---

### 3.5. Topología del Bloque: Pre-LN vs. Post-LN
* **Pre-LayerNorm (`norm_first=True`, Default):**
  $$x^{(1)} = x + \text{Dropout}(\text{SelfAttention}(\text{LayerNorm}_1(x)))$$
  $$x^{(2)} = x^{(1)} + \text{Dropout}(\text{FFN}(\text{LayerNorm}_2(x^{(1)})))$$
  * La normalización se aplica antes de cada subcapa. Esto mantiene un camino de gradiente directo a través de la autopista residual, previniendo la explosión/desvanecimiento de gradientes y eliminando la necesidad de un *warmup* agresivo del optimizador.
* **Post-LayerNorm (`norm_first=False`):**
  $$x^{(1)} = \text{LayerNorm}_1(x + \text{Dropout}(\text{SelfAttention}(x)))$$
  $$x^{(2)} = \text{LayerNorm}_2(x^{(1)} + \text{Dropout}(\text{FFN}(x^{(1)})))$$
  * Topología clásica de Vaswani et al. (2017), disponible para comparar en el estudio de ablación.

---

### 3.6. Mecanismos de Pooling para Extracción de $e_{\text{text}}$
Para condensar la matriz de salida $\mathbf{H} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ en un único vector por producto $e_{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$:

| Modo (`pooling_mode`) | Operación Matemática | Justificación y Uso |
| :--- | :--- | :--- |
| **`"mean"` (Default)** | $e_{\text{text}} = \frac{\sum_{i=1}^T \text{mask}_i \cdot h_i}{\sum_{i=1}^T \text{mask}_i}$ | Promedia uniformemente todas las representaciones contextuales de los tokens reales, ignorando los `[PAD]`. Muy robusto para descripciones largas. |
| **`"cls"`** | $e_{\text{text}} = h_0$ (Vector en la posición del token `[CLS]`) | Paradigma estándar de BERT. Permite que el token especial agregue la información global mediante auto-atención. |
| **`"max"`** | $e_{\text{text}} = \max_{i \in \text{valid}} h_{i}$ | Captura la señal o característica más saliente a lo largo del texto (ej. claims muy destacados). |
| **`"none"`** | Retorna $\mathbf{H} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ sin colapsar | Requerido para la **Alternativa 2 (Cross-Attention)**, donde el vector tabular actúa como Query sobre la secuencia completa. |

---

## 4. Inventario Completo de Hiperparámetros Configurables

La clase `TextTransformerConfig` centraliza todos los hiperparámetros con valores por defecto acordes a la consigna ($d_{\text{model}} < 100$):

| Hiperparámetro | Tipo | Default Base | Opciones / Rango de Ablación | Impacto Arquitectónico y Computacional |
| :--- | :---: | :---: | :---: | :--- |
| **`vocab_size`** | `int` | `2048` | `1720` (valor real del tokenizador) | Cantidad de filas en `nn.Embedding`. **Debe pasarse siempre `tokenizer.vocab_size`**, no el default: el corpus satura en 1.720 tokens con `min_frequency=2`, así que fijar 2.048 desperdiciaría 328 filas de embedding nunca indexadas. Para variarlo en la ablación hay que reentrenar el tokenizador bajando `min_frequency`. |
| **`max_seq_len`** | `int` | `128` | `[64, 128, 256]` | Longitud máxima de secuencia en tokens. Define el tamaño del buffer posicional y la dimensión temporal de los tensores. |
| **`d_model`** | `int` | `64` | `[32, 64, 96, 128]` | Dimensión del espacio latente y de los estados ocultos. Debe ser divisible por `n_heads`. Respeta $d_{\text{model}} < 100$ de la consigna. |
| **`n_heads`** | `int` | `4` | `[2, 4, 8]` | Cantidad de cabezales de auto-atención paralelos. Define $d_k = d_{\text{model}} / n_{\text{heads}}$. |
| **`d_ff`** | `int` | `256` | `[128, 256, 512]` | Dimensión oculta de la capa intermedia del Feed-Forward (habitualmente $4 \times d_{\text{model}}$). |
| **`num_layers`** | `int` | `2` | `[1, 2, 3, 4]` | Cantidad de bloques `TransformerEncoderBlock` apilados en profundidad. |
| **`dropout`** | `float` | `0.1` | `[0.0, 0.1, 0.2, 0.3]` | Probabilidad de regularización por dropout en embeddings, matrices de atención y capas lineales del FFN. |
| **`activation`** | `str` | `"gelu"` | `["gelu", "relu"]` | Función no lineal en la red Feed-Forward. GELU ofrece transiciones continuas más suaves que ReLU. |
| **`norm_first`** | `bool` | `True` | `[True (Pre-LN), False (Post-LN)]` | Ubicación de la capa LayerNorm. Pre-LN favorece estabilidad numérica en arquitecturas profundas. |
| **`pos_encoding_type`** | `str` | `"sinusoidal"` | `["sinusoidal", "learned", "none"]` | Mecanismo de inyección de información de orden de palabras. |
| **`pooling_mode`** | `str` | `"mean"` | `["mean", "cls", "max", "none"]` | Estrategia de reducción temporal para obtener el vector $e_{\text{text}}$. |
| **`pad_token_id`** | `int` | `0` | `int >= 0` | ID numérico del token `[PAD]` (ignorado en embeddings mediante `padding_idx=0`). |
| **`scale_embedding`** | `bool` | `True` | `[True, False]` | Multiplicación de embeddings por $\sqrt{d_{\text{model}}}$ previo a la codificación posicional. |

---

## 5. Conteo y Desglose de Parámetros

Para la configuración base recomendada:
* `vocab_size = 1720` (valor real del tokenizador entrenado), `d_model = 64`, `n_heads = 4`, `d_ff = 256`, `num_layers = 2`.

```
1. Matriz de Embedding (nn.Embedding):
   vocab_size * d_model = 1,720 * 64 = 110,080 parámetros

2. Por cada Bloque Encoder (x2 capas):
   - Multi-Head Attention:
     * W_Q, W_K, W_V: 3 * (64 * 64 + 64) = 12,480
     * W_O: 64 * 64 + 64 = 4,160
   - LayerNorm 1: 2 * 64 = 128
   - Position-wise Feed-Forward:
     * Linear 1 (64 -> 256): 64 * 256 + 256 = 16,640
     * Linear 2 (256 -> 64): 256 * 64 + 64 = 16,448
   - LayerNorm 2: 2 * 64 = 128
   Total por bloque = 49,984 parámetros
   Total 2 bloques = 99,968 parámetros

3. LayerNorm Final:
   2 * 64 = 128 parámetros

========================================================================
TOTAL GENERAL:            210,176 parámetros (~0.21 M)
TOTAL SIN EMBEDDINGS:     100,096 parámetros (~0.10 M)
========================================================================
```

> Valores verificados con `model.get_num_params()` instanciando el encoder con
> `vocab_size=tokenizer.vocab_size`. Si se fijara el default `vocab_size=2048` en lugar del valor
> real del tokenizador, el total subiría a 231.168 parámetros, de los cuales 20.992 corresponderían
> a filas de embedding que ningún token indexa jamás.

> [!NOTE]
> Con solo $\approx 100\text{k}$ parámetros en el núcleo del Transformer, el modelo es extremadamente liviano, permitiendo iteraciones de entrenamiento de pocos minutos en GPU/MPS o CPU, facilitando un estudio de ablación exhaustivo tal como exige `assignment.md`.

---

## 6. Guía de Uso e Integración

### 6.1. Inicialización y Paso Forward Básico

```python
import torch
from src.hybrid_transformer import TextTransformerConfig, TextTransformerEncoder
from src.tokenizer.bpe import ByteLevelBPETokenizer

# 1. Cargar el tokenizador pre-entrenado
tokenizer = ByteLevelBPETokenizer.from_file("resources/tokenizer/bpe_tokenizer.json")

# 2. Configurar e instanciar el Transformer Encoder
config = TextTransformerConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=128,
    d_model=64,
    n_heads=4,
    d_ff=256,
    num_layers=2,
    dropout=0.1,
    pos_encoding_type="sinusoidal",
    pooling_mode="mean",
    pad_token_id=tokenizer.pad_token_id or 0,
)
model = TextTransformerEncoder(config)

# 3. Tokenizar un lote de productos de ejemplo
texts = [
    "Cedar House Steamable Pepperoni Pizza | Well Reviewed | "
    "Steamable pepperoni pizza in a 10 oz package. Listed under frozen. | "
    "Prepared ingredients, Spices, Salt | United States | Wheat",
    "Sunny Basket Ready To Heat Waffles | Customer Favorite | "
    "Ready to heat waffles in a 6 ct package. Listed under frozen. | "
    "Flour, Sugar, Eggs | Canada | Milk",
]
encoded = tokenizer.encode_batch(texts, max_length=128, return_tensors="pt")

# 4. Inferencia: Obtener el vector contextual e_text
model.eval()
with torch.no_grad():
    e_text = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"]
    )

print("Shape de e_text:", e_text.shape)  # torch.Size([2, 64])
```

---

### 6.2. Extracción de Mapas de Atención para Explicabilidad (XAI)

Para inspeccionar qué palabras del título, descripción o ingredientes capturan mayor peso atencional:

```python
# Forward con retorno de pesos de atención
model.eval()
with torch.no_grad():
    e_text, attention_maps = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        return_attention=True
    )

# attention_maps es una lista de longitud `num_layers`
# Cada elemento tiene dimensión (batch_size, n_heads, seq_len, seq_len)
print(f"Capas de atención retornadas: {len(attention_maps)}")
print(f"Shape capa 1: {attention_maps[0].shape}")  # torch.Size([2, 4, 128, 128])

# Promediar cabezales para el primer producto
layer1_product0_attn = attention_maps[0][0].mean(dim=0)  # Shape (128, 128)
```

---

### 6.3. Ejecución de Smoke Tests desde CLI

El módulo cuenta con una rutina de verificación integrada para validar shapes, backward pass y conectividad con el tokenizador real:

```bash
uv run python -m src.hybrid_transformer.text_encoder
```

---

## 7. Alineación con las Consignas del Trabajo Práctico (`assignment.md`)

1. **Comprensión de la Arquitectura Transformer:**
   * La implementación modular explícita de `MultiHeadSelfAttention`, `PositionwiseFeedForward`, `PositionalEncoding` y `TransformerEncoderBlock` demuestra el dominio de cada subcomponente matemático sin depender de abstracciones opacas.
2. **Restricción de Recursos y Arquitectura Base Pequeña:**
   * Se inicia con $d_{\text{model}} = 64 < 100$ y $L = 2$ capas, garantizando tiempos de cómputo ágiles para validar el pipeline completo.
3. **Soporte Nativo para el Estudio de Ablación:**
   * Permite alternar fácilmente entre variantes arquitectónicas:
     * Sinusoidal vs. Learned vs. No Positional Encoding.
     * Mean Pooling vs. CLS Pooling vs. Max Pooling vs. Secuencia Completa.
     * Pre-LayerNorm vs. Post-LayerNorm.
     * Variación de $d_{\text{model}} \in \{32, 64, 96\}$, $L \in \{1, 2, 3, 4\}$, $H \in \{2, 4, 8\}$.
4. **Preparación para Fusión Multimodal:**
   * La salida $e_{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$ está estandarizada para conectarse de manera inmediata con la rama tabular (numéricas y embeddings categóricos) en la siguiente etapa del desarrollo.
