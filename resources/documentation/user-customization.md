# Personalización del Usuario en la Predicción de BTR

**73.69 Large Language Models — Trabajo Práctico 1**  
**Ejercicio 3: Personalización**  
**Documento de Diseño:** Incorporación del Factor de Usuario

---

## 1. Contexto y Motivación

El sistema actual de predicción de BTR opera exclusivamente sobre **atributos del producto**: texto
(título, descripción, ingredientes) y variables tabulares (precio, categoría, marca, etc.). Esto
significa que para un mismo producto, el modelo predice siempre la misma probabilidad de compra,
independientemente de quién lo esté mirando.

En la realidad, la probabilidad de compra depende fuertemente del **usuario**. Un usuario vegano
tiene alta probabilidad de comprar leche de almendras; uno que prioriza precio buscará marcas
económicas. Incorporar este factor de personalización requiere:

1. Contar con **datos del usuario** (actualmente no disponibles en el dataset).
2. Definir **qué features del usuario** se utilizarán.
3. Decidir **dónde y cómo** integrar la representación del usuario en la arquitectura existente.

---

## 2. Features de Personalización del Usuario

Se propone representar al usuario mediante un conjunto de **features agregadas** que resumen su
comportamiento histórico en la plataforma. Estas variables se calculan como estadísticos sobre el
historial de interacciones del usuario (productos vistos, añadidos al carrito y comprados).

### 2.1. Features Numéricas Continuas

| Feature | Descripción | Justificación |
|:---|:---|:---|
| `avg_purchase_price` | Precio promedio de los productos comprados | Sensibilidad al precio del usuario |
| `purchase_count` | Cantidad total de compras históricas | Nivel de actividad / engagement |
| `avg_nutrition_score` | Promedio de puntuación nutricional de productos comprados | Preferencia por productos saludables |
| `purchase_frequency` | Compras por semana (promedio) | Frecuencia de uso de la plataforma |
| `avg_basket_size` | Cantidad promedio de productos por sesión de compra | Patrón de compra (bulk vs. puntual) |
| `category_diversity` | Cantidad de categorías distintas compradas / total de categorías | Amplitud de intereses del usuario |
| `organic_ratio` | Proporción de productos orgánicos comprados | Preferencia por productos orgánicos |
| `days_since_last_purchase` | Días transcurridos desde la última compra | Recencia de la actividad |

Estas variables recibirían el mismo tratamiento que las numéricas del producto en el
`TabularEncoder` actual: compresión `log1p` para las que presentan asimetría
(`purchase_count`, `days_since_last_purchase`) y estandarización z-score ajustada sobre
el split de entrenamiento.

### 2.2. Features Categóricas

| Feature | Descripción | Codificación |
|:---|:---|:---|
| `preferred_category` | Categoría con mayor frecuencia de compra | Entity Embedding |
| `preferred_brand` | Marca con mayor frecuencia de compra | Entity Embedding |
| `preferred_storage_type` | Tipo de almacenamiento más frecuente (`Ambient`, `Refrigerated`, `Frozen`) | One-Hot |

Los vocabularios de estas variables categóricas son los mismos que ya existen en el
`TabularPreprocessor` del producto, por lo que no se introducen categorías nuevas.

### 2.3. ¿Por qué un MLP y no un Transformer?

Estas features son un **vector tabular de dimensión fija**: un conjunto de estadísticos
precomputados que no tienen estructura secuencial ni dependencias posicionales entre sí. La
relación entre `avg_purchase_price` y `organic_ratio` no depende del orden en que se presenten,
sino de su valor conjunto.

Un Transformer sobre este tipo de entrada no aportaría beneficio: el mecanismo de self-attention
está diseñado para capturar dependencias entre posiciones de una secuencia, y aquí no hay
secuencia — hay un perfil fijo. Un MLP es la herramienta adecuada para proyectar un vector
tabular a un espacio latente, exactamente como ya lo hace el `TabularEncoder` existente para
las features del producto.

---

## 3. Integración en la Arquitectura

### 3.1. Sistema Actual (sin personalización)

El modelo `BTRModel` procesa el producto en dos ramas independientes que se fusionan antes del
clasificador. En modo `cross`, el flujo es:

```
Paso 1:  TabularEncoder(x_num, x_cat)           →  e_tab ∈ ℝ^d_tab
Paso 2:  TextTransformerEncoder(input_ids, mask) →  H_text ∈ ℝ^(T × d_text)
                    ↑ internamente usa self-attention (cada token atiende a otros tokens)

Paso 3:  CrossAttentionFusion(H_text, e_tab)     →  e_text ∈ ℝ^d_text
                    ↑ Q = e_tab · W_Q,  K,V = H_text · W_K, H_text · W_V
                    ↑ El producto "pregunta" a su propio texto qué es relevante

Paso 4:  concat(e_text, e_tab)                   →  vector fusionado ∈ ℝ^(d_text + d_tab)
Paso 5:  ClassifierHead(vector fusionado)         →  logit de compra
```

El cross-attention del Paso 3 reemplaza al pooling (mean/cls/max): en vez de promediar todos los
tokens, usa `e_tab` como Query para hacer un promedio ponderado inteligente sobre la secuencia
de texto, donde las features tabulares del producto condicionan qué tokens reciben más peso.

### 3.2. Sistema Propuesto (con personalización)

Se introduce un `UserEncoder` (MLP) como tercera rama. La modificación clave es que el **Query
del cross-attention se enriquece con información del usuario**, de modo que la selección de
tokens relevantes del texto queda condicionada tanto por el perfil del producto como por las
preferencias del usuario.

```
Paso 1:  TabularEncoder(x_num, x_cat)           →  e_tab  ∈ ℝ^d_tab
Paso 2:  UserEncoder(x_user)                    →  e_user ∈ ℝ^d_user
Paso 3:  TextTransformerEncoder(input_ids, mask) →  H_text ∈ ℝ^(T × d_text)
                    ↑ no cambia nada internamente

Paso 4:  q = concat(e_tab, e_user)               →  Query combinado ∈ ℝ^(d_tab + d_user)
Paso 5:  CrossAttentionFusion(H_text, q)          →  e_text ∈ ℝ^d_text
                    ↑ Q = concat(e_tab, e_user) · W_Q,  K,V = H_text · W_K, H_text · W_V
                    ↑ Producto Y usuario "preguntan juntos" al texto

Paso 6:  concat(e_text, e_tab, e_user)            →  vector fusionado ∈ ℝ^(d_text + d_tab + d_user)
Paso 7:  ClassifierHead(vector fusionado)          →  logit de compra personalizado
```

### 3.3. ¿Qué cambia concretamente?

| Componente | Sistema Actual | Sistema Propuesto |
|:---|:---|:---|
| `TextTransformerEncoder` | Sin cambios | Sin cambios |
| `TabularEncoder` | Sin cambios | Sin cambios |
| `UserEncoder` | No existe | **Nuevo:** MLP análogo al `TabularEncoder` |
| `CrossAttentionFusion.q_proj` | `nn.Linear(d_tab, d_text)` | `nn.Linear(d_tab + d_user, d_text)` |
| `FusionConfig.fused_dim` | `d_text + d_tab` | `d_text + d_tab + d_user` |
| `ClassifierHead` | `input_dim = d_text + d_tab` | `input_dim = d_text + d_tab + d_user` |
| `BTRModel.forward()` | Recibe `input_ids, mask, x_num, x_cat` | Recibe adicionalmente `x_user` |

La arquitectura interna del Transformer (self-attention, positional encoding, feed-forward) no
se modifica. El cambio se concentra en el **módulo de fusión** y en el **clasificador final**.

### 3.4. Intuición del Efecto

Supongamos que el texto de un producto es:

> *"Cedar House Organic Almond Milk | Unsweetened plant-based milk. USDA certified organic.
> No artificial flavors. | Almonds, Water, Salt"*

- **Sin personalización** (sistema actual): el Query se construye solo con `e_tab` (precio,
  categoría, etc.). El cross-attention pondera los tokens según las características del producto.
  La salida es la misma para todos los usuarios.

- **Con personalización**: si el usuario tiene `organic_ratio = 0.85` y
  `preferred_category = Dairy`, su `e_user` codifica una fuerte preferencia por productos
  orgánicos y lácteos/sustitutos. El Query combinado `concat(e_tab, e_user)` genera pesos de
  atención que enfatizan los tokens *"Organic"*, *"plant-based"*, *"USDA"*, *"Almond Milk"*.
  El resumen `e_text` resultante captura los aspectos del producto que son relevantes **para
  este usuario en particular**, lo que produce una probabilidad de compra más alta.

  Para otro usuario con `organic_ratio = 0.05` y `preferred_category = Snacks`, los mismos
  tokens recibirían pesos bajos, y el `e_text` resultante sería más "tibio", reflejando menor
  afinidad con el producto.

---

## 4. Alternativa Considerada: Historial como Secuencia con Transformer

### 4.1. Idea

En lugar de representar al usuario con features agregadas, se podría modelar su **historial de
interacciones como una secuencia temporal** y procesarla con un segundo Transformer. Cada
"token" de la secuencia sería la representación de un producto con el que el usuario interactuó
previamente (visto, añadido al carrito o comprado), ordenado cronológicamente.

Es importante aclarar que este `UserHistoryTransformer` sería un **Transformer completamente
independiente** del `TextTransformerEncoder` existente. No se trata de agregar capas al
Transformer de texto ni de encadenar sus salidas, sino de una **tercera rama paralela** que
reemplaza al `UserEncoder` MLP propuesto en la sección 3. Desde la fusión en adelante, el
sistema no distingue si `e_user` provino de un MLP o de un Transformer: en ambos casos es un
vector de dimensión fija `d_user`.

Los dos Transformers procesan secuencias de naturaleza completamente distinta:

| | TextTransformerEncoder (existente) | UserHistoryTransformer (nuevo) |
|:---|:---|:---|
| **Secuencia** | Tokens de texto del producto actual (subwords BPE) | Productos pasados que el usuario compró |
| **Cada "token"** | Un subword: `"Organic"`, `"Milk"`, `"onds"` | Un producto entero: `prod₁`, `prod₂`, ... |
| **Largo típico** | ~128 subwords | ~últimas 20-50 compras |
| **Self-attention captura** | Relaciones entre palabras del texto | Relaciones entre compras del usuario |
| **Salida** | `H_text` → representación del producto actual | `e_user` → representación del usuario |

El siguiente diagrama muestra el flujo completo de las tres ramas independientes y cómo
convergen en la fusión:

```
RAMA 1 — Texto del producto actual (Transformer existente, no se modifica):
  ["Organic", "Almond", "Milk", ...]  →  TextTransformerEncoder  →  H_text ∈ ℝ^(T × d_text)

RAMA 2 — Tabulares del producto actual (MLP existente, no se modifica):
  [price=3.5, category=Dairy, ...]    →  TabularEncoder (MLP)    →  e_tab ∈ ℝ^d_tab

RAMA 3 — Historial del usuario (NUEVO Transformer, independiente):
  [prod₁, prod₂, prod₃, ..., prodₙ]  →  UserHistoryTransformer  →  Pooling  →  e_user ∈ ℝ^d_user
       ↑                                          ↑
       Cada producto es un embedding               Self-attention sobre la
       precomputado (ej. categoría +               secuencia de compras
       precio + marca proyectados)

FUSIÓN (igual que en la propuesta de la sección 3):
  q = concat(e_tab, e_user)
  CrossAttention(H_text, q)           →  e_text ∈ ℝ^d_text
  concat(e_text, e_tab, e_user)       →  ClassifierHead  →  P(compra | usuario, producto)
```

Cada producto del historial se representaría con un **vector precomputado** (por ejemplo, la
concatenación de su categoría embebida, su precio normalizado y su marca proyectada). No es
necesario pasar cada producto del historial por el `TextTransformerEncoder` del texto — eso
sería computacionalmente prohibitivo y no aportaría información nueva dado que el Transformer
de texto ya procesa el producto *actual* que se está evaluando.

### 4.2. Ventajas

- **Captura patrones temporales**: el self-attention aprende que compras recientes son más
  informativas que las antiguas, y que ciertas secuencias de compra son predictivas (ej.
  "compró leche y cereales → es probable que compre fruta").
- **No requiere feature engineering manual**: no es necesario diseñar features agregadas como
  `organic_ratio` o `preferred_category` — el modelo las descubre de manera implícita.
- **Evolución de gustos**: al operar sobre la secuencia temporal, captura naturalmente cambios
  en las preferencias del usuario (ej. un usuario que migra de comida rápida a productos
  saludables).
- **Estado del arte en recomendación**: esta arquitectura es la base de modelos como SASRec
  (Self-Attentive Sequential Recommendation) y BERT4Rec, que representan el estado del arte
  en sistemas de recomendación secuencial.

### 4.3. ¿Por qué no se elige esta alternativa?

1. **Complejidad de implementación**: requiere un segundo Transformer completo, con su propia
   gestión de secuencias, padding, y positional encoding. Duplica la complejidad arquitectónica
   del sistema.
2. **Costo computacional**: el entrenamiento requiere procesar dos Transformers por cada ejemplo
   (uno para el texto del producto, otro para el historial del usuario), lo que incrementa
   significativamente el tiempo de entrenamiento y los recursos necesarios.
3. **Requisitos de datos**: necesita un historial suficientemente largo por usuario para que el
   Transformer pueda aprender patrones secuenciales útiles. Usuarios con pocas interacciones
   generarían secuencias cortas y ruidosas.
4. **Cold start severo**: usuarios completamente nuevos no tendrían secuencia alguna, lo que
   obliga a implementar un fallback al modelo sin personalización.
5. **Scope del ejercicio**: el foco del TP es demostrar comprensión de la arquitectura Transformer
   y su integración. La propuesta del MLP con Query combinado ya logra ese objetivo de manera
   concisa y coherente con el sistema existente.

No obstante, para un sistema de producción a gran escala con millones de usuarios y
disponibilidad de datos de historial extenso, esta alternativa sería la más potente y
representaría una evolución natural del sistema propuesto.
