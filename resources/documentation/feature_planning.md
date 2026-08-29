# Selección y Filtrado de Features

**73.69 Large Language Models - Trabajo Práctico 1**  
**Dataset:** `resources/datasets/supermarket_products.csv`

---

## 1. Decisiones sobre las 22 Variables Originales

| # | Variable Original | Decisión | Detalle / Acción |
|---|---|---|---|
| 1 | `title` | **TRANSFORMAR** | Se descarta el string crudo; se reemplaza por `title_clean` (sin badges ni sufijos de medida) y se extrae `title_tag`. |
| 2 | `description` | **CONSERVAR** | Descripción del producto. |
| 3 | `price` | **CONSERVAR y DERIVAR** | Precio del producto en USD. Se conserva y se usa para derivar `price_per_oz`. |
| 4 | `category` | **CONSERVAR** | Categoría del producto. |
| 5 | `timestamp` | **CONSERVAR y DERIVAR** | Se conserva para ordenar cronológicamente y realizar el split (Train, Val, Test), y se deriva `day_of_week`. |
| 6 | `query_id` | **DESCARTAR (NO)** | No se utiliza. |
| 7 | `filter_category` | **DESCARTAR (NO)** | No se utiliza. |
| 8 | `filter_price_min` | **DESCARTAR (NO)** | Se descarta el valor individual; solo se usa para calcular el `price_span`. |
| 9 | `filter_price_max` | **DESCARTAR (NO)** | Se descarta el valor individual; solo se usa para calcular el `price_span`. |
| 10 | `filter_storage_type` | **DESCARTAR (NO)** | No se utiliza. |
| 11 | `cart` | **DESCARTAR (NO)** | No se utiliza (data leakage de funnel). |
| 12 | `brand` | **CONSERVAR** | Marca del producto. |
| 13 | `package_size` | **DESCARTAR (NO)** | Redundante con `net_weight_oz` y `unit_of_measure`. |
| 14 | `unit_of_measure` | **CONSERVAR** | Unidad de medida (`oz`, `lb`, `fl oz`, `ct`, `gal`). |
| 15 | `net_weight_oz` | **CONSERVAR y DERIVAR** | Peso neto en onzas. Se conserva y se usa para derivar `price_per_oz`. |
| 16 | `dimensions_in` | **DESCARTAR (NO)** | Se descarta el string original y se reemplaza por el cálculo del volumen (`volume`). |
| 17 | `storage_type` | **CONSERVAR** | Tipo de almacenamiento requerido (`Ambient`, `Refrigerated`, `Frozen`). |
| 18 | `ingredients` | **CONSERVAR y DERIVAR** | Lista de ingredientes. Se conserva y se deriva `num_ingredients`. |
| 19 | `allergens` | **CONSERVAR y DERIVAR** | Alérgenos declarados. Los nulos (4.455 filas) se imputan con el centinela `No Allergens`, ya que `NaN` significa "sin alérgenos declarados" y no dato faltante. Se deriva `has_allergens`. |
| 20 | `nutrition_score` | **CONSERVAR** | Puntuación nutricional continua (0 a 100). |
| 21 | `country_of_origin` | **CONSERVAR** | País de origen del producto. |
| 22 | `bought` | **TARGET (AL FINAL)** | Variable objetivo (BTR). Se ubica al final para separarla como target en el entrenamiento. |

---

## 2. Resumen de Variables DESCARTADAS

1. **`title` (crudo)**: Reemplazado por `title_clean` y `title_tag`.
2. **`query_id`**: Descartada.
3. **`filter_category`**: Descartada.
4. **`filter_storage_type`**: Descartada.
5. **`filter_price_min`**: Descartada en forma cruda (solo se usa en el cálculo de `price_span`).
6. **`filter_price_max`**: Descartada en forma cruda (solo se usa en el cálculo de `price_span`).
7. **`cart`**: Descartada.
8. **`package_size`**: Descartada por redundancia.
9. **`dimensions_in`**: Descartada en string crudo (reemplazada por `volume`).

---

## 3. Variables Derivadas

1. **`title_clean`**: Título del producto limpio de forma completa: se elimina tanto el badge de reputación en paréntesis (`(Customer Favorite)`) como el sufijo de medida comercial (` - 10 oz`, ` - 6 ct`). Deja el nombre semántico puro del producto (ej. *"Cedar House Steamable Pepperoni Pizza"*). Reemplaza a `title`.
2. **`title_tag`**: Badge de reputación / social proof extraído de `title` entre paréntesis mediante regex `r'\((.*?)\)'` (ej. `Best Seller`, `Customer Favorite`, `No Tag`).
3. **`day_of_week`**: Día de la semana derivado a partir de `timestamp` (ej. `Monday`, `Tuesday`, etc., o 0 a 6).
4. **`price_span`**: Amplitud del rango de precio del filtro ($\text{filter\_price\_max} - \text{filter\_price\_min}$).
5. **`price_per_oz`**: Precio unitario por onza ($\frac{\text{price}}{\text{net\_weight\_oz}}$).
6. **`volume`**: Volumen físico calculado a partir de `dimensions_in` ($\text{largo} \times \text{ancho} \times \text{alto}$).
7. **`num_ingredients`**: Cantidad de ingredientes declarados (conteo a partir de `ingredients`).
8. **`has_allergens`**: Flag binaria ($1$ si contiene alérgenos declarados, $0$ si el valor original era `NaN`).

> [!IMPORTANT]
> **El centinela de imputación no puede ser `'None'`.** pandas incluye los strings `None`, `NA`,
> `NULL` y `NaN` en su lista de `na_values` por defecto, de modo que un `fillna('None')` se
> reinterpreta como nulo al releer el CSV y la imputación se pierde en cada round-trip. Por eso
> `allergens` usa `No Allergens`, siguiendo la misma convención que `No Tag` en `title_tag`.

---

## 4. Lista Final de Variables para el Dataset

A continuación se listan las 21 variables que conformarán el dataset procesado:

1. **`timestamp`**: Fecha y hora UTC del evento (para ordenar cronológicamente y particionar en Train, Val y Test).
2. **`title_clean`**: Nombre puro del producto (sin badges ni sufijos de medida; reemplaza a `title`).
3. **`title_tag`**: Badge de reputación extraído de `title`.
4. **`description`**: Descripción del producto.
5. **`price`**: Precio en USD.
6. **`price_span`**: Amplitud del rango de precio del filtro ($\text{filter\_price\_max} - \text{filter\_price\_min}$).
7. **`price_per_oz`**: Precio unitario por onza ($\frac{\text{price}}{\text{net\_weight\_oz}}$).
8. **`category`**: Categoría principal.
9. **`day_of_week`**: Día de la semana derivado de `timestamp`.
10. **`brand`**: Marca.
11. **`unit_of_measure`**: Unidad de medida comercial.
12. **`net_weight_oz`**: Peso neto en onzas.
13. **`volume`**: Volumen físico calculado ($\text{largo} \times \text{ancho} \times \text{alto}$).
14. **`storage_type`**: Tipo de almacenamiento.
15. **`ingredients`**: Ingredientes del producto.
16. **`num_ingredients`**: Cantidad de ingredientes declarados.
17. **`allergens`**: Alérgenos declarados (nulos imputados como `No Allergens`).
18. **`has_allergens`**: Flag binaria de presencia de alérgenos ($1$ o $0$).
19. **`nutrition_score`**: Puntuación nutricional.
20. **`country_of_origin`**: País de origen.
21. **`bought`**: Variable objetivo (Target BTR: `True`/`False` o `1`/`0`). Ubicada al final para su posterior separación en $X$ e $y$.

---

## 5. Ruteo de Features: Rama de Texto vs. Rama Tabular

`clean_dataset.csv` es un **feature store agnóstico**: conserva las 21 variables sin asumir qué
módulo las consume. La asignación a cada rama se decide en
`src/data_extraction/build_transformer_dataset.py` (constante `DEFAULT_TEXT_FIELDS`) y es
configurable vía `--text_fields`.

### 5.1. Composición de la secuencia de texto

Los campos se concatenan con el separador `" | "` en este orden:

```
title_clean | badge | description | ingredients | country_of_origin | allergens
```

### 5.2. Criterio de asignación

El criterio es **si el string ya está presente en la secuencia**, no si la variable tiene señal.
Un campo cuyo valor ya aparece literalmente en el texto no se agrega, porque duplicaría tokens
sin aportar información nueva (verificado empíricamente sobre las 10.000 filas):

| Variable | Presencia literal en el texto | Decisión |
| :--- | :---: | :--- |
| `brand` | **100%** (es el prefijo de `title_clean`) | Solo rama tabular. Ya entra al encoder vía el título. |
| `category` | **100%** (la descripción la enuncia) | Solo rama tabular. |
| `storage_type` | **100%** (la descripción la enuncia) | Solo rama tabular. |
| `unit_of_measure` | **100%** (la descripción la enuncia) | Solo rama tabular. |
| `allergens` | 35% de las filas que declaran alérgeno | **Se agrega al texto** + rama tabular. |
| `country_of_origin` | **0%** | **Se agrega al texto** + rama tabular. |

La verificación de `brand` está automatizada en
`src/data_analysis/brand_title_consistency.py`, que confirma la inclusión en el título en
10.000/10.000 filas bajo tres criterios (contenida en `title`, contenida en `title_clean`, y
`title_clean` empieza con la marca).

### 5.3. Redundancia deliberada entre ramas

Las variables presentes en ambas ramas **no son un error de diseño**. Cada representación aporta
algo distinto:

* **En el texto:** representación composicional y distribuida entre subpalabras, robusta ante
  valores no vistos (el BPE byte-level nunca emite `[UNK]`).
* **En la rama tabular:** identidad atómica y sin ambigüedad léxica, con un sesgo inductivo mucho
  más fuerte y por lo tanto más eficiente en cantidad de muestras.

Esa redundancia es lo que habilita el estudio de ablación: comparar el modelo con y sin la rama
tabular cuantifica cuánto de la señal categórica el Transformer ya extrae por su cuenta desde el
texto. Requisito para poder medirlo: **la variable debe estar en `clean_dataset.csv`**, aunque un
experimento puntual decida no usarla.
