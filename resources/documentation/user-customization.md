# Personalización del Usuario en la Predicción de BTR

**73.69 Large Language Models — Trabajo Práctico 1**  
**Ejercicio 3: Personalización**  
**Documento de Diseño:** Incorporación del Factor de Usuario

---

## 1. Contexto y Motivación

El sistema actual de predicción de BTR se construye casi por completo sobre **atributos del
producto**: texto (título, descripción, ingredientes) y variables tabulares (precio, categoría,
marca, etc.). La única excepción es `price_span`, derivada del rango de precio que el usuario
filtró en su búsqueda, que sí captura una señal de **intención** en el momento de la consulta.

Pero esa señal es anónima y puntual: el modelo sabe qué rango de precio se filtró, no **quién**
lo filtró ni qué compró esa persona antes. En consecuencia, para un mismo producto el sistema
predice siempre la misma probabilidad de compra, independientemente de quién lo esté mirando.

En la realidad, la probabilidad de compra depende fuertemente del **usuario**. Un usuario vegano
tiene alta probabilidad de comprar leche de almendras; uno que prioriza precio buscará marcas
económicas. Incorporar este factor de personalización requiere:

1. Contar con **datos del usuario** (actualmente no disponibles en el dataset).
2. Definir **qué features del usuario** se utilizarán.
3. Decidir **dónde y cómo** integrar la representación del usuario en la arquitectura existente.

### El BTR deja de ser una propiedad del producto

La consecuencia de fondo excede lo arquitectónico y alcanza a la **definición misma de la
métrica**. Hoy el BTR es una propiedad del producto: un número por ítem del catálogo, que responde
a la pregunta *"¿qué tan probable es que este producto se compre?"*. Con personalización pasa a
ser una propiedad del par **(usuario, producto)**: `P(compra | usuario, producto)`.

Eso redefine también el objetivo de negocio planteado en la consigna. *"Identificar los mejores
productos y promocionarlos en otras áreas del e-commerce"* deja de resolverse con un único ranking
global del catálogo y pasa a requerir **un ranking por usuario**: el mejor producto para promocionar
ya no es el mismo para todos.

---

## 2. Features de Personalización del Usuario

> [!IMPORTANT]
> **Supuesto de partida.** `supermarket_products.csv` no contiene identificador de usuario: cada
> fila es una impresión anónima. Todo lo que sigue asume que el dataset se extiende con un campo
> `user_id` y con el historial de interacciones asociado a cada usuario. Ese es el insumo mínimo
> sin el cual ninguna forma de personalización es posible, y es el primer requisito que el
> sistema debería satisfacer antes de implementar lo que se describe a continuación.

Bajo ese supuesto, se propone representar al usuario mediante un conjunto de **features
agregadas** que resumen su comportamiento histórico en la plataforma. Estas variables se calculan
como estadísticos sobre el historial de interacciones del usuario, previo al momento de la
impresión que se está evaluando.

### 2.1. Criterio de Selección: Alineación con los Atributos del Producto

La personalización no surge de describir al usuario en abstracto, sino de **compararlo con el
producto que se está evaluando**. Un dato como "este usuario gasta $6 en promedio" no dice nada
por sí solo: adquiere significado únicamente frente al `price = 12` del producto actual. Por eso
el criterio de selección adoptado es que **cada feature del usuario viva sobre el mismo eje que
un atributo del producto**, de modo que el modelo pueda calcular una comparación entre ambos.

| Feature de usuario | Qué representa | Contraparte en el producto |
|:---|:---|:---|
| `avg_purchase_price` | Precio promedio de lo que compra: sensibilidad al precio absoluto | `price` |
| `avg_price_per_oz` | Precio promedio por onza: si busca rendimiento o conveniencia | `price_per_oz` |
| `avg_nutrition_score` | Puntuación nutricional promedio: preferencia por productos saludables | `nutrition_score` |
| `category_shares` | Proporción de compras en cada categoría: afinidad por categoría | `category` |
| `brand_shares` | Proporción de compras de cada marca: lealtad de marca | `brand` |
| `allergen_avoidance` | Proporción de compras que declaran cada alérgeno: restricción alimentaria | `allergens` |
| `preferred_storage_type` | Tipo de almacenamiento más frecuente: hábito de compra (fresco vs. despensa) | `storage_type` |
| `purchase_frequency` | Compras por semana: nivel de actividad en la plataforma | - |
| `days_since_last_purchase` | Días desde la última compra: recencia de la actividad | - |
| `cart_to_purchase_rate` | Proporción de carritos que terminan en compra: eficiencia de conversión | - |

Todas las contrapartes son variables que **ya existen en el `TabularPreprocessor` del producto**,
con sus vocabularios y sus escalas ya ajustados sobre el split de entrenamiento. No se introducen
categorías nuevas ni ejes que el sistema actual no maneje.

### 2.2. Cómo se Conecta Cada Feature con su Contraparte

La comparación toma tres formas distintas según el tipo de eje:

* **Ejes continuos** (`avg_purchase_price`, `avg_price_per_oz`, `avg_nutrition_score`): la
  comparación es de **posición relativa**. El modelo no evalúa si el producto es caro o sano en
  términos absolutos, sino si lo es *para este usuario*. Un producto de $12 es una anomalía para
  quien promedia $6 y una compra rutinaria para quien promedia $20, y el par de valores le da al
  clasificador todo lo necesario para distinguir ambos casos.

* **Ejes categóricos con distribución** (`category_shares`, `brand_shares`,
  `allergen_avoidance`): la comparación es una **búsqueda indexada**. El valor categórico del
  producto actual selecciona la componente correspondiente del vector del usuario, y esa
  componente es directamente la afinidad del usuario hacia *este* producto en particular. Si el
  producto es de categoría `Dairy`, lo que importa es la proporción de compras en `Dairy` de ese
  usuario, no cuál es su categoría favorita en general.

  Es la razón por la que estas tres se definen como **vectores de proporciones y no como el
  valor más frecuente (`argmax`)**. Un usuario que reparte sus compras 30% `Dairy` / 28%
  `Produce` / 25% `Bakery` quedaría representado igual que uno que compra 100% `Dairy`, y se
  perdería justamente la información que permite evaluar productos fuera de la categoría
  dominante. `allergen_avoidance` opera con el signo invertido: una proporción cercana a cero
  sostenida sobre un historial extenso es la señal de evitación (por ejemplo, un usuario celíaco
  frente a un producto con `allergens = Wheat`).

* **Ejes categóricos de baja cardinalidad** (`preferred_storage_type`, con 3 valores): la
  comparación se reduce a **coincidencia o no coincidencia** con el `storage_type` del producto.
  Con tan pocos valores el `argmax` no pierde información relevante y basta con One-Hot.

### 2.3. Preferencia vs. Propensión: Dos Roles Distintos

Las diez features no cumplen la misma función, y la distinción es la que ordena toda la
integración arquitectónica de la sección 3.

**Las siete primeras son features de preferencia.** Tienen contraparte en el producto, por lo que
**reordenan**: ante el mismo catálogo, dos usuarios distintos obtienen rankings distintos. Son las
que producen el efecto que pide la consigna — que el BTR deje de ser una propiedad del producto y
pase a ser una propiedad del par (usuario, producto).

**Las tres últimas son features de propensión.** No tienen contraparte, y esa ausencia no es un
descuido sino una consecuencia de lo que miden: `purchase_frequency`, `days_since_last_purchase` y
`cart_to_purchase_rate` describen *cuánto* compra el usuario, no *qué* compra. Su efecto es un
desplazamiento del nivel base de probabilidad: suben o bajan el score de **todos** los productos
por igual, sin alterar el orden entre ellos. Un usuario que hace tres compras por semana tiene más
probabilidad de comprar cualquier cosa que uno inactivo hace dos meses, pero eso no dice nada
sobre cuál de los productos del catálogo le conviene mostrarle.

`cart_to_purchase_rate` es la que mide el tramo final del embudo: de todo lo que el usuario mandó
al carrito, qué fracción terminó comprando. Distingue a quien decide rápido de quien acumula
carritos y abandona, y esa distinción es un factor multiplicativo sobre cualquier producto que se
le muestre, no una preferencia por alguno en particular.

> [!NOTE]
> **Sobre el uso de `cart`.** El campo se descarta en el sistema actual por *leakage* de embudo
> (`feature_planning.md`): para la impresión que se está prediciendo, el add-to-cart es posterior
> a la impresión y anterior a la compra. Esa objeción es sobre la **fila**, no sobre la columna: el
> `cart` de interacciones *anteriores* del mismo usuario es un hecho ya consumado, igual que el
> `bought` sobre el que se calculan las siete features de preferencia — que es literalmente el
> target y resulta admisible por el mismo corte temporal.

Se conservan porque el BTR es una probabilidad y calibrar su nivel por usuario es parte del
problema, no un agregado. Pero el rol diferenciado tiene una consecuencia concreta sobre dónde
inyectar cada bloque, que se desarrolla en la sección 3: **solo las features de preferencia
enriquecen el Query del cross-attention**, porque solo ellas pueden determinar qué tokens del
texto son relevantes; la propensión entra únicamente en la concatenación previa al clasificador.

### 2.4. Preprocesamiento

Las features reciben el mismo tratamiento que sus contrapartes en el `TabularEncoder` actual:

* **Continuas** (`avg_purchase_price`, `avg_price_per_oz`, `avg_nutrition_score`): estandarización
  z-score ajustada exclusivamente sobre el split de entrenamiento, con `log1p` previo en
  `avg_price_per_oz` por su asimetría, replicando el criterio ya aplicado a `price_per_oz`.
* **Propensión** (`purchase_frequency`, `days_since_last_purchase`): `log1p` seguido de z-score,
  por tratarse de distribuciones de cola larga. `cart_to_purchase_rate` queda exceptuada: es un
  cociente acotado en $[0, 1]$ e ingresa sin transformación, con imputación en la media del split
  de entrenamiento cuando el usuario no registra carritos previos y el cociente es indefinido.
* **Vectores de proporciones** (`category_shares`, `brand_shares`, `allergen_avoidance`): ya
  acotados en $[0, 1]$ y sumando 1 por construcción, ingresan sin transformación adicional. Sus
  dimensiones quedan fijadas por los vocabularios existentes del preprocesador (12 categorías,
  15 marcas, 8 alérgenos).
* **Categórica** (`preferred_storage_type`): One-Hot sobre el mismo vocabulario de 3 valores del
  producto.

### 2.5. ¿Por qué un MLP y no un Transformer?

Estas features son un **vector tabular de dimensión fija**: un conjunto de estadísticos
precomputados que no tienen estructura secuencial ni dependencias posicionales entre sí. La
relación entre `avg_purchase_price` y `avg_nutrition_score` no depende del orden en que se
presenten, sino de su valor conjunto.

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

Paso 3:  CrossAttentionFusion(H_text, e_tab)     →  e_cross ∈ ℝ^d_text
                    ↑ Q = e_tab · W_Q,  K,V = H_text · W_K, H_text · W_V
                    ↑ El producto "pregunta" a su propio texto qué es relevante

Paso 4:  concat(e_cross, e_tab)                  →  vector fusionado ∈ ℝ^(d_text + d_tab)
Paso 5:  ClassifierHead(vector fusionado)         →  logit de compra
```

El cross-attention del Paso 3 reemplaza al pooling (mean/cls/max): en vez de promediar todos los
tokens, usa `e_tab` como Query para hacer un promedio ponderado inteligente sobre la secuencia
de texto, donde las features tabulares del producto condicionan qué tokens reciben más peso.

### 3.2. Sistema Propuesto (con personalización)

Se introduce un `UserEncoder` (MLP) como tercera rama. Siguiendo la distinción de la sección 2.3,
**el encoder emite dos vectores en lugar de uno**: `e_pref`, con los siete ejes que tienen
contraparte en el producto, y `e_prop`, con las tres features de actividad. Cada uno se inyecta en
un punto distinto de la arquitectura.

```
Paso 1:  TabularEncoder(x_num, x_cat)           →  e_tab  ∈ ℝ^d_tab
Paso 2:  UserEncoder(x_user)                    →  e_pref ∈ ℝ^d_pref   (preferencia)
                                                →  e_prop ∈ ℝ^d_prop   (propensión)
Paso 3:  TextTransformerEncoder(input_ids, mask) →  H_text ∈ ℝ^(T × d_text)
                    ↑ no cambia nada internamente

Paso 4:  q = concat(e_tab, e_pref)               →  Query combinado ∈ ℝ^(d_tab + d_pref)
Paso 5:  CrossAttentionFusion(H_text, q)          →  e_cross ∈ ℝ^d_text
                    ↑ Q = concat(e_tab, e_pref) · W_Q,  K,V = H_text · W_K, H_text · W_V
                    ↑ Producto Y preferencias del usuario "preguntan juntos" al texto

Paso 6:  concat(e_cross, e_tab, e_pref, e_prop)   →  vector fusionado
                                                     ∈ ℝ^(d_text + d_tab + d_pref + d_prop)
Paso 7:  ClassifierHead(vector fusionado)          →  logit de compra personalizado
```

**Por qué la propensión no entra al Query.** El Query determina qué tokens del texto del producto
se leen con más peso. "Este usuario compra tres veces por semana" no puede responder esa
pregunta: no hay ningún token de la descripción que sea más o menos relevante según la frecuencia
de compra. En cambio sí desplaza el nivel base de probabilidad, y ese efecto se captura
correctamente en la concatenación del Paso 6. Inyectarla en el Query solo agregaría parámetros
sin señal aprovechable.

**Dimensionamiento y necesidad del MLP.** Las diez features ocupan **44 dimensiones crudas**
(12 categorías + 15 marcas + 8 alérgenos + 3 storage + 3 continuas + 3 de propensión). En la
configuración ganadora del Ejercicio 2 (`d_text = 8`, `d_tab = 66`), un `e_user` sin proyectar
sería más de cinco veces mayor que `e_text` y dominaría el vector fusionado, que pasaría de 74 a
118 dimensiones. Por eso el `UserEncoder` **sí** usa MLP proyector, a diferencia del
`TabularEncoder` ganador que corre con `use_mlp=False`: la decisión no es estética sino de
balance entre modalidades.

**Compatibilidad con el modo `late`.** El sistema soporta las dos estrategias de fusión y la
ablación del Ejercicio 2 las compara. En modo `late` no hay Query que enriquecer, de modo que
`e_pref` y `e_prop` entran únicamente en la concatenación final. La personalización queda así
disponible en ambos modos y el diseño sigue siendo conmutable, igual que el resto del sistema.

**Usuario sin perfil (cold start).** Un usuario nuevo no tiene historial del cual derivar las
features. El mecanismo ya existe en el código: el `TabularEncoder` reserva `UNKNOWN_INDEX = 0`
con `padding_idx` para categorías no vistas, lo que produce un vector nulo. La rama de usuario
replica ese criterio (proporciones en cero, continuas imputadas con la media poblacional del
split de entrenamiento), de modo que el sistema degrada de forma controlada al comportamiento no
personalizado en lugar de fallar.

### 3.3. Intuición del Efecto

Tomemos una fila real del dataset (`transformer_dataset_complete.csv`):

> *"Green Fork Organic Avocados | New Listing | Organic avocados in a 3 lb package for online
> grocery orders. Listed under produce and intended for ambient storage. Limited customer
> feedback so far. | Whole produce | Thailand | No Allergens"*
>
> `price = 2.96` · `category = Produce` · `nutrition_score = 99` · `price_per_oz = 0.059`

**Sin personalización** (sistema actual): el Query se construye solo con `e_tab`. La atención
pondera los tokens según las características del producto y la salida es idéntica para todos los
usuarios.

**Con personalización**, el efecto se produce por dos vías distintas que conviene no confundir:

**a) El Query cambia *qué se lee* del producto.** Para un usuario con
`category_shares[Produce] = 0.40` y `avg_nutrition_score = 88`, el Query combinado
`concat(e_tab, e_pref)` desplaza los pesos de atención hacia los tokens *"Organic"*, *"produce"*
y *"Whole produce"*. Para un usuario con `category_shares[Snacks] = 0.55` y
`avg_nutrition_score = 41`, esos mismos tokens reciben menos peso y el resumen se apoya en otras
partes de la secuencia. El resultado es que `e_cross` **es un vector distinto para cada usuario**:
el mismo producto se resume de forma diferente según quién lo mira.

**b) La concatenación final determina *cuánto le gusta*.** Es importante notar que la vía anterior
no sube ni baja el score por sí sola: la salida del cross-attention es una combinación convexa de
los vectores value, de modo que el Query puede cambiar qué tokens se leen pero no la escala del
resultado. El desplazamiento efectivo de la probabilidad ocurre en el Paso 6, donde el
clasificador compara directamente los ejes alineados de la sección 2: `nutrition_score = 99`
contra un `avg_nutrition_score` de 88 (afín) o de 41 (disonante), `price = 2.96` contra un
`avg_purchase_price` de 3.20 (dentro del rango habitual) o de 9.50 (fuera de él).

Sobre esas dos vías, `e_prop` aplica el último ajuste: un usuario activo esta semana recibe
scores más altos en **todos** los productos por igual, sin que cambie el orden entre ellos.

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
reemplaza al `UserEncoder` MLP propuesto en la sección 3. Concretamente sustituye al bloque de
**preferencia**: su salida ocupa el lugar de `e_pref`, mientras que `e_prop` se sigue calculando
igual (la frecuencia y la recencia no requieren un Transformer). Desde la fusión en adelante, el
sistema no distingue si `e_pref` provino de un MLP o de un Transformer: en ambos casos es un
vector de dimensión fija.

Los dos Transformers procesan secuencias de naturaleza completamente distinta:

| | TextTransformerEncoder (existente) | UserHistoryTransformer (nuevo) |
|:---|:---|:---|
| **Secuencia** | Tokens de texto del producto actual (subwords BPE) | Productos con los que el usuario interactuó previamente |
| **Cada "token"** | Un subword: `"Organic"`, `"Avoc"`, `"ados"` | Un producto entero: `prod₁`, `prod₂`, ... |
| **Largo típico** | ~128 subwords | ~últimas 20-50 compras |
| **Self-attention captura** | Relaciones entre palabras del texto | Relaciones entre compras del usuario |
| **Salida** | `H_text` → representación del producto actual | `e_pref` → representación del usuario |

El siguiente diagrama muestra el flujo completo de las tres ramas independientes y cómo
convergen en la fusión:

```
RAMA 1 — Texto del producto actual (Transformer existente, no se modifica):
  ["Organic", "Avoc", "ados", ...]   →  TextTransformerEncoder  →  H_text ∈ ℝ^(T × d_text)

RAMA 2 — Tabulares del producto actual (MLP existente, no se modifica):
  [price=2.96, category=Produce, ...] →  TabularEncoder (MLP)    →  e_tab ∈ ℝ^d_tab

RAMA 3 — Historial del usuario (NUEVO Transformer, independiente):
  [prod₁, prod₂, prod₃, ..., prodₙ]  →  UserHistoryTransformer  →  Pooling  →  e_pref
       ↑                                          ↑
       Cada producto es un embedding               Self-attention sobre la
       precomputado (ej. categoría +               secuencia de compras
       precio + marca proyectados)

  [purchase_frequency, days_since_last_purchase, cart_to_purchase_rate]  →  MLP  →  e_prop
       ↑ la propensión no requiere Transformer: se mantiene igual que en la sección 3

FUSIÓN (igual que en la propuesta de la sección 3):
  q = concat(e_tab, e_pref)
  CrossAttention(H_text, q)                 →  e_cross ∈ ℝ^d_text
  concat(e_cross, e_tab, e_pref, e_prop)    →  ClassifierHead  →  P(compra | usuario, producto)
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
  `category_shares` o `allergen_avoidance` — el modelo las descubre de manera implícita, ni
  decidir a mano qué ejes tienen contraparte en el producto.
- **Evolución de gustos**: al operar sobre la secuencia temporal, captura naturalmente cambios
  en las preferencias del usuario (ej. un usuario que migra de comida rápida a productos
  saludables).
- **Coherencia con el foco del TP**: la personalización pasaría a resolverse con el mismo
  mecanismo que el resto del sistema — self-attention sobre una secuencia — en lugar de con un
  MLP sobre features diseñadas a mano.

### 4.3. ¿Por qué no se elige esta alternativa?

1. **Complejidad de implementación**: requiere un segundo Transformer completo, con su propia
   gestión de secuencias, padding, y positional encoding. Duplica la complejidad arquitectónica
   del sistema.
2. **Costo de datos por ejemplo**: el peso no está tanto en el forward — un Transformer sobre 20-50
   ítems con `d_model` chico es incluso más barato que el de texto sobre 128 tokens — sino en que
   **cada ejemplo de entrenamiento arrastra el historial completo del usuario** en memoria y en
   I/O. Y como ese historial cambia en cada instante de tiempo, no puede precomputarse ni
   cachearse: hay que reconstruirlo para cada impresión.
3. **Requisitos de datos**: necesita un historial suficientemente largo por usuario para que el
   Transformer pueda aprender patrones secuenciales útiles. Usuarios con pocas interacciones
   generarían secuencias cortas y ruidosas.
4. **Cold start con umbral, no gradual**: ambas propuestas sufren cuando falta historial, pero de
   forma distinta. Las features agregadas **degradan suavemente**: con tres compras las
   proporciones ya son imperfectas pero informativas, y el vector existe. El Transformer de
   secuencia **degrada por umbral**: con uno o dos ítems la self-attention no tiene sobre qué
   atender y la salida es ruido, por bien entrenado que esté el modelo. Existe un largo mínimo de
   secuencia por debajo del cual la rama simplemente no aporta.
5. **Robustez frente a datos que aún no existen**: las features de la sección 2 son
   **especificables sin ver el historial**, porque cada una está anclada a una variable del
   producto que ya conocemos. El Transformer de historial exige, en cambio, decisiones que
   dependen de la distribución real de los datos — largo de secuencia, representación de cada
   ítem, estrategia de pooling — que sobre un dataset inexistente serían conjeturas.
6. **Scope del ejercicio**: el foco del TP es demostrar comprensión de la arquitectura Transformer
   y su integración. La propuesta del MLP con el Query enriquecido por la preferencia ya logra ese
   objetivo de manera concisa y coherente con el sistema existente.

No obstante, para un sistema de producción a gran escala con millones de usuarios y
disponibilidad de datos de historial extenso, esta alternativa sería la más potente y
representaría una evolución natural del sistema propuesto.
