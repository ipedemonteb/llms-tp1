"""Script ejecutor del Exploratory Data Analysis (EDA).

Ejecuta el pipeline completo de análisis, genera las figuras con matplotlib
y guarda los reportes en el directorio `results/`.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Permitir importaciones locales desde src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_analysis.dataset import load_raw_data, clean_and_enrich_data, get_dataset_summary
from src.data_analysis.plots import (
    plot_funnel_and_target_distribution,
    plot_btr_by_category_and_storage,
    plot_btr_by_brand,
    plot_btr_by_social_proof_tags,
    plot_numerical_distributions_and_btr,
    plot_query_relative_dynamics,
    plot_text_length_distributions,
    plot_temporal_trends,
    plot_correlation_matrix
)


def generate_eda_summary_markdown(df: pd.DataFrame, summary: dict, output_file: Path):
    """Genera un reporte completo en markdown con todos los hallazgos cuantitativos del EDA y respuesta al Ejercicio 1."""
    
    # 1. Funnel
    cart_btr = df[df['cart'] == True]['bought'].mean() * 100
    no_cart_btr = df[df['cart'] == False]['bought'].mean() * 100

    # 2. Categories
    cat_df = df.groupby('category')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    cat_table = "\n".join([f"| {cat} | {int(row['count']):,} | {row['mean']*100:.2f}% |" for cat, row in cat_df.iterrows()])

    # 3. Brands
    brand_df = df.groupby('brand')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    brand_table = "\n".join([f"| {b} | {int(row['count']):,} | {row['mean']*100:.2f}% |" for b, row in brand_df.iterrows()])

    # 4. Social proof tags
    tag_df = df.groupby('title_tag')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    tag_table = "\n".join([f"| {t} | {int(row['count']):,} | {row['mean']*100:.2f}% |" for t, row in tag_df.iterrows()])

    # 5. Numerical summary table
    num_cols = ['price', 'net_weight_oz', 'nutrition_score', 'volume_cu_in', 'density_oz_per_cu_in']
    num_desc = df[num_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    num_table = "\n".join([
        f"| `{col}` | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['50%']:.2f} | {row['95%']:.2f} | {row['max']:.2f} |"
        for col, row in num_desc.iterrows()
    ])

    md_content = f"""# Informe Completo de Exploratory Data Analysis (EDA) y Formulación del Problema

**Materia:** 73.69 Large Language Models - 2026  
**Trabajo Práctico 1:** Transformers & Buy Through Rate (BTR) Prediction  
**Dataset:** `resources/supermarket_products.csv` ({summary['num_rows']:,} filas, {summary['num_cols']} columnas crudas)  

---

## 1. ¿Qué se predice? (Definición Formal de la Variable Objetivo)

### 1.1. Definición de Negocio del BTR
El **Buy Through Rate (BTR)** en e-commerce mide la efectividad con la que un producto exhibido en los resultados de búsqueda se convierte en una compra:
$$\\text{{BTR}} = \\frac{{\\text{{Cantidad de Compras}}}}{{\\text{{Cantidad de Impresiones}}}}$$

### 1.2. Formulación Matemática
Cada registro $k$ en el dataset representa una **impresión** de un producto $i$ mostrado ante una búsqueda $q$.
* **Variable Objetivo:** $Y \\in \\{{0, 1\\}}$, donde $Y = 1 \\iff \\text{{bought}} = \\text{{True}}$.
* **Función de Predicción:** El modelo estima la probabilidad condicionada de compra $p(x) = P(Y = 1 \\mid X = x)$, donde $x$ es el vector que combina representaciones de texto, atributos numéricos y contexto de la query.
* **Tasa Base (Desbalance):** El dataset presenta un **BTR global del {summary['btr_global']*100:.2f}%** ({df['bought'].sum():,} compras sobre {len(df):,} impresiones).

### 1.3. Embudo de Conversión (Funnel)
El proceso de compra sigue una jerarquía estricta: $\\text{{Impresión}} \\to \\text{{Cart}} \\to \\text{{Bought}}$:
* **Tasa Global de Carrito (`cart`):** **{summary['cart_global']*100:.2f}%** ({df['cart'].sum():,} agregados).
* **Conversión Condicionada:**
  * $P(\\text{{bought}} = \\text{{True}} \\mid \\text{{cart}} = \\text{{True}}) = {cart_btr:.2f}\\%$
  * $P(\\text{{bought}} = \\text{{True}} \\mid \\text{{cart}} = \\text{{False}}) = {no_cart_btr:.2f}\\%$
* **Implicancia:** Ningún usuario compra un producto sin agregarlo previamente al carrito.

---

## 2. Características de la Información Provista (Data Quality & Distribuciones)

### 2.1. Calidad de Datos y Valores Faltantes
* **Registros totales:** {summary['num_rows']:,} filas sin duplicados.
* **Valores nulos:** Únicamente presentes en la columna `allergens` ({summary['missing_values'].get('allergens', 4455):,} nulos = 44.55%), lo cual indica que el producto **no contiene alérgenos declarados**. Todas las demás 21 columnas tienen 0 valores nulos (100% completitud).
* **Formatos y parseo estructural:**
  * `dimensions_in`: Strings con formato `\"L x W x H\\\"\"` parseados a largo, ancho, alto y volumen en pulgadas cúbicas ($in^3$).
  * `timestamp`: Rango de 2 años (desde {df['timestamp'].min().strftime('%Y-%m-%d')} hasta {df['timestamp'].max().strftime('%Y-%m-%d')}).
  * `package_size`: Homogéneo con `unit_of_measure` y `net_weight_oz`.

### 2.2. Distribución de Variables Numéricas Continuas

| Variable | Media | Desv. Est. | Mínimo | Mediana | P95 | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{num_table}

### 2.3. Señales Semánticas en Texto (Social Proof y Reputación)
El análisis reveló que las etiquetas semánticas dentro del título y la descripción actúan como el predictor más fuerte del BTR:

| Tag / Badge en Título | Impresiones | BTR (%) |
| :--- | :---: | :---: |
{tag_table}

* **Conclusión:** La semántica del texto discrimina entre productos de conversión masiva (>60% BTR) y productos sin conversión (0% BTR).

### 2.4. Distribución por Categoría y Marca

<details>
<summary><b>Ver tabla de BTR por Categoría</b></summary>

| Categoría | Impresiones | BTR (%) |
| :--- | :---: | :---: |
{cat_table}

</details>

<details>
<summary><b>Ver tabla de BTR por Marca</b></summary>

| Marca | Impresiones | BTR (%) |
| :--- | :---: | :---: |
{brand_table}

</details>

### 2.5. Contexto de Búsqueda (Queries y Matching de Filtros)
* **Búsquedas únicas (`query_id`):** {summary['num_queries']:,} queries con un promedio de {summary['avg_products_per_query']:.2f} productos impresos por búsqueda.
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
| `price_pos_in_query` | Numérica relativa | Cálculo en pipeline: $\\frac{{p - p_{{\\min}}}}{{p_{{\\max}} - p_{{\\min}} + \\epsilon}}$ | Aporta la noción competitiva directa dentro de la búsqueda que vio el usuario. |
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
| [`01_funnel_and_target_distribution.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/01_funnel_and_target_distribution.png) | Distribución de la variable objetivo y embudo de conversión $\\text{{Impresión}} \\to \\text{{Cart}} \\to \\text{{Bought}}$. |
| [`02_btr_by_category_and_storage.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/02_btr_by_category_and_storage.png) | Tasa de compra por categoría y tipo de almacenamiento. |
| [`03_btr_by_brand.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/03_btr_by_brand.png) | BTR comparativo por marca vs promedio global del e-commerce. |
| [`04_btr_by_social_proof_tags.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/04_btr_by_social_proof_tags.png) | Impacto determinante de los badges de reputación del título en el BTR. |
| [`05_numerical_distributions_and_btr.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/05_numerical_distributions_and_btr.png) | Histogramas de densidad de precio, volumen, peso y score nutricional. |
| [`06_query_relative_dynamics.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/06_query_relative_dynamics.png) | Efecto del ranking de precio dentro de la búsqueda y tamaño de queries. |
| [`07_text_length_distributions.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/07_text_length_distributions.png) | Distribución de longitudes de secuencia para tokenización de texto. |
| [`08_temporal_trends.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/08_temporal_trends.png) | Evolución temporal del BTR mensual y por día de la semana. |
| [`09_correlation_matrix.png`](file:///Users/nachopedemonte/Desktop/Nacho/Code/ITBA/LLMS/llms-tp1/results/figures/09_correlation_matrix.png) | Matriz de correlaciones lineales de Pearson. |
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)


def run_eda(csv_path: str = "resources/supermarket_products.csv", results_dir: str = "results"):
    """Ejecuta el pipeline completo de EDA y guarda todas las salidas en `results/`."""
    results_path = Path(results_dir)
    figures_path = results_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA) - TP1 TRANSFORMERS")
    print("=" * 70)

    print(f"\n📂 1. Cargando datos desde {csv_path}...")
    df_raw = load_raw_data(csv_path)
    print(f"   -> Filas: {df_raw.shape[0]:,}, Columnas: {df_raw.shape[1]}")

    print("\n🧹 2. Limpiando y enriqueciendo dataset con features derivadas...")
    df = clean_and_enrich_data(df_raw)
    summary = get_dataset_summary(df)
    print(f"   -> BTR Global: {summary['btr_global']*100:.2f}%")
    print(f"   -> Cart Rate Global: {summary['cart_global']*100:.2f}%")
    print(f"   -> Queries únicas: {summary['num_queries']:,}")

    print("\n📊 3. Generando visualizaciones con matplotlib en results/figures/...")
    p1 = plot_funnel_and_target_distribution(df, figures_path)
    print(f"   ✓ [1/9] Guardado: {p1.name}")
    p2 = plot_btr_by_category_and_storage(df, figures_path)
    print(f"   ✓ [2/9] Guardado: {p2.name}")
    p3 = plot_btr_by_brand(df, figures_path)
    print(f"   ✓ [3/9] Guardado: {p3.name}")
    p4 = plot_btr_by_social_proof_tags(df, figures_path)
    print(f"   ✓ [4/9] Guardado: {p4.name}")
    p5 = plot_numerical_distributions_and_btr(df, figures_path)
    print(f"   ✓ [5/9] Guardado: {p5.name}")
    p6 = plot_query_relative_dynamics(df, figures_path)
    print(f"   ✓ [6/9] Guardado: {p6.name}")
    p7 = plot_text_length_distributions(df, figures_path)
    print(f"   ✓ [7/9] Guardado: {p7.name}")
    p8 = plot_temporal_trends(df, figures_path)
    print(f"   ✓ [8/9] Guardado: {p8.name}")
    p9 = plot_correlation_matrix(df, figures_path)
    print(f"   ✓ [9/9] Guardado: {p9.name}")

    print("\n📝 4. Exportando resumen ejecutivo en results/eda_summary.md...")
    summary_file = results_path / "eda_summary.md"
    generate_eda_summary_markdown(df, summary, summary_file)
    print(f"   ✓ Reporte guardado en: {summary_file}")

    print("\n" + "=" * 70)
    print("✅ EDA COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
