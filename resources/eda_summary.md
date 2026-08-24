# Informe Completo de Exploratory Data Analysis (EDA) y Formulación del Problema

**Materia:** 73.69 Large Language Models - 2026  
**Trabajo Práctico 1:** Transformers & Buy Through Rate (BTR) Prediction  
**Dataset:** `resources/supermarket_products.csv` (10,000 filas, 49 columnas crudas)  

---

## 1. ¿Qué se predice? (Definición Formal de la Variable Objetivo)

### 1.1. Definición de Negocio del BTR
El **Buy Through Rate (BTR)** en e-commerce mide la efectividad con la que un producto exhibido en los resultados de búsqueda se convierte en una compra:
$$\text{BTR} = \frac{\text{Cantidad de Compras}}{\text{Cantidad de Impresiones}}$$

### 1.2. Formulación Matemática
Cada registro $k$ en el dataset representa una **impresión** de un producto $i$ mostrado ante una búsqueda $q$.
* **Variable Objetivo:** $Y \in \{0, 1\}$, donde $Y = 1 \iff \text{bought} = \text{True}$.
* **Función de Predicción:** El modelo estima la probabilidad condicionada de compra $p(x) = P(Y = 1 \mid X = x)$, donde $x$ es el vector que combina representaciones de texto, atributos numéricos y contexto de la query.
* **Tasa Base (Desbalance):** El dataset presenta un **BTR global del 13.01%** (1,301 compras sobre 10,000 impresiones).

### 1.3. Embudo de Conversión (Funnel)
El proceso de compra sigue una jerarquía estricta: $\text{Impresión} \to \text{Cart} \to \text{Bought}$:
* **Tasa Global de Carrito (`cart`):** **30.07%** (3,007 agregados).
* **Conversión Condicionada:**
  * $P(\text{bought} = \text{True} \mid \text{cart} = \text{True}) = 43.27\%$
  * $P(\text{bought} = \text{True} \mid \text{cart} = \text{False}) = 0.00\%$
* **Implicancia:** Ningún usuario compra un producto sin agregarlo previamente al carrito.

---

## 2. Características de la Información Provista (Data Quality & Distribuciones)

### 2.1. Calidad de Datos y Valores Faltantes
* **Registros totales:** 10,000 filas sin duplicados.
* **Valores nulos:** Únicamente presentes en la columna `allergens` (4,455 nulos = 44.55%), lo cual indica que el producto **no contiene alérgenos declarados**. Todas las demás 21 columnas tienen 0 valores nulos (100% completitud).
* **Formatos y parseo estructural:**
  * `dimensions_in`: Strings con formato `"L x W x H\""` parseados a largo, ancho, alto y volumen en pulgadas cúbicas ($in^3$).
  * `timestamp`: Rango de 2 años (desde 2024-07-08 hasta 2026-07-08).
  * `package_size`: Homogéneo con `unit_of_measure` y `net_weight_oz`.

### 2.2. Distribución de Variables Numéricas Continuas

| Variable | Media | Desv. Est. | Mínimo | Mediana | P95 | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `price` | 8.58 | 5.64 | 1.20 | 7.25 | 21.18 | 34.98 |
| `net_weight_oz` | 28.67 | 29.11 | 2.77 | 17.16 | 76.98 | 155.44 |
| `nutrition_score` | 51.31 | 26.17 | 0.00 | 55.00 | 89.00 | 99.00 |
| `volume_cu_in` | 243.17 | 328.79 | 7.50 | 109.45 | 924.54 | 3403.54 |
| `density_oz_per_cu_in` | 0.26 | 0.34 | 0.00 | 0.16 | 0.79 | 5.03 |

### 2.3. Señales Semánticas en Texto (Social Proof y Reputación)
El análisis reveló que las etiquetas semánticas dentro del título y la descripción actúan como el predictor más fuerte del BTR:

| Tag / Badge en Título | Impresiones | BTR (%) |
| :--- | :---: | :---: |
| Customer Favorite | 493 | 67.75% |
| Best Seller | 470 | 65.74% |
| Top Rated | 472 | 62.71% |
| #1 Pick | 496 | 62.50% |
| Well Reviewed | 477 | 3.77% |
| Shopper Favorite | 507 | 2.76% |
| Highly Rated | 520 | 2.12% |
| Popular Choice | 469 | 1.92% |
| Discontinuing Soon | 550 | 0.00% |
| Limited Feedback | 524 | 0.00% |
| Low Feedback | 466 | 0.00% |
| New Listing | 522 | 0.00% |
| Current Stock | 515 | 0.00% |
| Rarely Reordered | 500 | 0.00% |
| Recently Added | 481 | 0.00% |
| Regular Listing | 494 | 0.00% |
| Standard Listing | 506 | 0.00% |
| Clearance Listing | 510 | 0.00% |
| Unrated Listing | 517 | 0.00% |
| No Tag | 511 | 0.00% |

* **Conclusión:** La semántica del texto discrimina entre productos de conversión masiva (>60% BTR) y productos sin conversión (0% BTR).

### 2.4. Distribución por Categoría y Marca

<details>
<summary><b>Ver tabla de BTR por Categoría</b></summary>

| Categoría | Impresiones | BTR (%) |
| :--- | :---: | :---: |
| Baby | 209 | 19.14% |
| Bakery | 917 | 16.36% |
| Dairy | 1,003 | 14.86% |
| Frozen | 929 | 13.99% |
| Personal Care | 602 | 13.79% |
| Household | 642 | 13.40% |
| Produce | 1,148 | 12.98% |
| Meat | 709 | 12.98% |
| Snacks | 791 | 12.90% |
| Beverages | 1,076 | 11.80% |
| Pantry | 1,412 | 11.40% |
| Seafood | 562 | 5.69% |

</details>

<details>
<summary><b>Ver tabla de BTR por Marca</b></summary>

| Marca | Impresiones | BTR (%) |
| :--- | :---: | :---: |
| North Star Foods | 664 | 15.81% |
| Corner Market | 698 | 15.19% |
| Purely Good | 679 | 13.84% |
| Cedar House | 597 | 13.57% |
| Daily Table | 703 | 13.51% |
| Green Fork | 640 | 13.28% |
| Market Pantry Co. | 697 | 13.20% |
| Blue Cart | 660 | 12.73% |
| Harvest Lane | 661 | 12.71% |
| FreshField | 726 | 12.53% |
| Sunny Basket | 654 | 12.39% |
| Riverbend | 647 | 12.21% |
| Golden Acre | 683 | 12.15% |
| Oak & Grain | 674 | 10.98% |
| Valley Select | 617 | 10.86% |

</details>

### 2.5. Contexto de Búsqueda (Queries y Matching de Filtros)
* **Búsquedas únicas (`query_id`):** 2,012 queries con un promedio de 4.97 productos impresos por búsqueda.
* **Cumplimiento de Filtros:** El 100% de los productos exhibidos cumplen estrictamente con `filter_category == category`, `filter_storage_type == storage_type` y `filter_price_min <= price <= filter_price_max`.
* **Posición de Precio dentro de la Búsqueda:** Dentro de cada query, los productos en posiciones intermedias/altas de precio tienen mayor propensión de compra debido a que coinciden con los artículos mejor calificados.

---

## 3. Features Seleccionadas para el Modelo

1. **Features Textuales (Procesadas por Transformer):**
   * `title`: Nombre y badges del producto.
   * `description`: Detalle del producto y feedback de clientes.
   * `ingredients`: Ingredientes que componen el artículo.
   * `allergens`: Alérgenos declarados.
2. **Features Numéricas Continuas:**
   * `price`: Precio en USD.
   * `net_weight_oz`: Peso neto en onzas.
   * `nutrition_score`: Score nutricional normalizado.
   * `volume_cu_in`: Volumen calculado a partir de `dimensions_in`.
   * `price_pos_in_query`: Posición relativa del precio dentro de la lista de resultados de la búsqueda.
   * `price_rank_in_query`: Ranking ordinal de precio dentro de la query.
3. **Features Categóricas:**
   * `category`: Categoría del producto (12 categorías).
   * `brand`: Marca fabricante (15 marcas).
   * `storage_type`: Tipo de almacenamiento (Ambient, Refrigerated, Frozen).
   * `country_of_origin`: País de origen (10 países).
4. **Features de Matching / Indicadores:**
   * `has_allergens`: Indicador booleano de presencia de alérgenos.
   * `num_ingredients`: Cantidad de ingredientes.

---

## 4. Estrategia de Preprocesamiento por Feature

| Feature | Tipo | Técnica de Preprocesamiento | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Texto concatenado** (`title` + `description` + `ingredients` + `allergens`) | Texto libre | Tokenización de subpalabras (BPE/WordPiece) con `max_seq_length = 64`, padding y attention mask | Permite al Transformer aprender atención cruzada entre el título, la descripción y los atributos. |
| `category`, `storage_type`, `country_of_origin` | Categórica (Baja cardinalidad) | **One-Hot Encoding** | Número reducido de clases (<15); evita asumir ordinalidad artificial. |
| `brand` | Categórica (Media cardinalidad) | **Categorical Embeddings** (dimensión 8) o One-Hot | Permite proyectar las 15 marcas en un espacio denso continuo donde marcas similares quedan cercanas. |
| `price`, `net_weight_oz`, `volume_cu_in` | Numéricas continuas | **RobustScaler / StandardScaler** | Reduce la sensibilidad a valores extremos y acelera la convergencia del gradiente. |
| `nutrition_score` | Numérica acotada (0 a 100) | **Min-Max Scaling ($[0, 1]$)** | Normaliza la escala directamente al rango unitario sin alterar su distribución lineal. |
| `price_pos_in_query` | Numérica relativa | Cálculo en pipeline: $\frac{p - p_{\min}}{p_{\max} - p_{\min} + \epsilon}$ | Aporta la noción competitiva directa dentro de la búsqueda que vio el usuario. |
| `allergens` | Multilabel / Missing | Imputación `'None'` + flag binaria `has_allergens` | Trata el valor nulo como una categoría informativa real (libre de alérgenos). |

---

## 5. Estrategia de Partición de Datos (Data Splitting)

* **Método Seleccionado:** **Group Split por `query_id`** (70% Train, 15% Validation, 15% Test).
* **Justificación Anti-Leakage:** Una búsqueda devuelve un conjunto cerrado de productos que compiten entre sí. Si dividiéramos aleatoriamente a nivel fila, el modelo vería productos de la misma query en Train y Test, memorizando el contexto de búsqueda (*data leakage*).
* **Verificación de Split:** Cero intersección de `query_id` entre particiones y preservación del BTR (~13%) en cada conjunto.

---

## 6. Índice de Figuras Generadas (`results/figures/`)

| Archivo | Descripción del Gráfico |
| :--- | :--- |
| [`01_funnel_and_target_distribution.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/01_funnel_and_target_distribution.png) | Distribución de la variable objetivo y embudo de conversión $\text{Impresión} \to \text{Cart} \to \text{Bought}$. |
| [`02_btr_by_category_and_storage.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/02_btr_by_category_and_storage.png) | Tasa de compra por categoría y tipo de almacenamiento. |
| [`03_btr_by_brand.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/03_btr_by_brand.png) | BTR comparativo por marca vs promedio global del e-commerce. |
| [`04_btr_by_social_proof_tags.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/04_btr_by_social_proof_tags.png) | Impacto determinante de los badges de reputación del título en el BTR. |
| [`05_numerical_distributions_and_btr.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/05_numerical_distributions_and_btr.png) | Histogramas de densidad de precio, volumen, peso y score nutricional. |
| [`06_query_relative_dynamics.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/06_query_relative_dynamics.png) | Efecto del ranking de precio dentro de la búsqueda y tamaño de queries. |
| [`07_text_length_distributions.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/07_text_length_distributions.png) | Distribución de longitudes de secuencia para tokenización de texto. |
| [`08_temporal_trends.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/08_temporal_trends.png) | Evolución temporal del BTR mensual y por día de la semana. |
| [`09_correlation_matrix.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/09_correlation_matrix.png) | Matriz de correlaciones lineales de Pearson. |
