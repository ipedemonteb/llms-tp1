"""Módulo de carga, limpieza y enriquecimiento de datos para el análisis exploratorio y modelado.

Estructura de Datos:
El dataset modela eventos de impresión de productos en sesiones de búsqueda (queries).
- Entidad Query (query_id): Definida por una configuración de parámetros de búsqueda
  (filter_category, filter_storage_type, filter_price_min, filter_price_max).
- Entidad Producto (Item): Productos retornados por el motor de búsqueda para dicha query.
- Interacciones Query x Producto: Posición en filtro de precio, ranking de precio intra-query,
  y ratios relativos frente a los competidores de la pantalla.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def parse_dimensions(dim_str: str) -> Tuple[float, float, float, float]:
    """Parsea el string de dimensiones (e.g. '3.3 x 4.0 x 4.1\"') a largo, ancho, alto y volumen."""
    if not isinstance(dim_str, str) or not dim_str:
        return np.nan, np.nan, np.nan, np.nan
    try:
        clean = dim_str.replace('"', '').strip()
        parts = [float(p.strip()) for p in clean.split('x')]
        if len(parts) == 3:
            l, w, h = parts
            return l, w, h, l * w * h
        return np.nan, np.nan, np.nan, np.nan
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def load_raw_data(csv_path: str = "resources/supermarket_products.csv") -> pd.DataFrame:
    """Carga el dataset crudo desde el archivo CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path.resolve()}")
    return pd.read_csv(path)


def clean_and_enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza de tipos, parseo de estructuras complejas y generación de features derivadas.
    
    Genera features a nivel Producto, a nivel Query y de interacción Query x Producto.
    """
    df = df.copy()

    # 1. Parseo temporal y variables cíclicas
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['year_month'] = df['timestamp'].dt.tz_localize(None).dt.to_period('M').astype(str)

    # Transformaciones cíclicas senoidales/cosenoidales
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    # 2. Parseo de dimensiones físicas y volumen
    dims = df['dimensions_in'].apply(parse_dimensions)
    df['length_in'] = [d[0] for d in dims]
    df['width_in'] = [d[1] for d in dims]
    df['height_in'] = [d[2] for d in dims]
    df['volume_cu_in'] = [d[3] for d in dims]
    df['density_oz_per_cu_in'] = df['net_weight_oz'] / (df['volume_cu_in'] + 1e-6)

    # 3. Tratamiento de alérgenos e ingredientes
    df['has_allergens'] = df['allergens'].notna().astype(int)
    df['allergens_clean'] = df['allergens'].fillna('None')
    df['num_allergens'] = df['allergens'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['num_ingredients'] = df['ingredients'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

    # 4. Extracción de señales de texto (Social proof y reputación)
    # Títulos suelen incluir tags en paréntesis como (Best Seller), (Customer Favorite), etc.
    df['title_tag'] = df['title'].str.extract(r'\((.*?)\)')[0].fillna('No Tag')
    df['title_clean'] = df['title'].str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()

    # Última oración de la descripción (evaluación y feedback)
    def extract_desc_feedback(text: str) -> str:
        if not isinstance(text, str):
            return ''
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[-1] if sentences else ''

    df['desc_feedback'] = df['description'].apply(extract_desc_feedback)

    # Longitudes de texto
    df['title_word_count'] = df['title'].fillna('').apply(lambda x: len(x.split()))
    df['desc_word_count'] = df['description'].fillna('').apply(lambda x: len(x.split()))
    df['title_char_count'] = df['title'].fillna('').apply(len)
    df['desc_char_count'] = df['description'].fillna('').apply(len)

    # Template textual estructurado para Transformer
    df['text_full'] = (
        "Búsqueda: " + df['filter_category'] + " (" + df['filter_storage_type'] + 
        ", Max: $" + df['filter_price_max'].round(2).astype(str) + ") | " +
        "Producto: " + df['title_clean'] + " | " +
        "Badge: " + df['title_tag'] + " | " +
        "Marca: " + df['brand'] + " | " +
        "Precio: $" + df['price'].round(2).astype(str) + " | " +
        "Ingredientes: " + df['ingredients'] + " | " +
        "Alérgenos: " + df['allergens_clean'] + " | " +
        "Origen: " + df['country_of_origin'] + " | " +
        "Descripción: " + df['description']
    )

    # 5. Features de Contexto de la Query y Filtros
    df['filter_price_span'] = df['filter_price_max'] - df['filter_price_min']
    df['price_pos_in_filter'] = (df['price'] - df['filter_price_min']) / (df['filter_price_span'] + 1e-6)

    # 6. Dinámica relativa dentro de la búsqueda (query_id) - Competencia intra-query
    df['query_size'] = df.groupby('query_id')['price'].transform('count')
    df['price_rank_in_query'] = df.groupby('query_id')['price'].rank(ascending=True)
    query_mean_price = df.groupby('query_id')['price'].transform('mean')
    df['price_rel_to_query_mean'] = df['price'] / (query_mean_price + 1e-6)
    
    query_min_p = df.groupby('query_id')['price'].transform('min')
    query_max_p = df.groupby('query_id')['price'].transform('max')
    query_p_span = query_max_p - query_min_p
    df['price_pos_in_query'] = np.where(query_p_span > 0, (df['price'] - query_min_p) / (query_p_span + 1e-6), 0.5)

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas resumen del dataset para el reporte de calidad y formulación."""
    n_rows, n_cols = df.shape
    btr_global = float(df['bought'].mean())
    cart_global = float(df['cart'].mean())
    
    # Conversión condicional en el funnel
    cart_to_bought = float(df[df['cart'] == True]['bought'].mean()) if cart_global > 0 else 0.0
    no_cart_to_bought = float(df[df['cart'] == False]['bought'].mean()) if (1 - cart_global) > 0 else 0.0

    missing = df.isnull().sum().to_dict()

    # Análisis de configuraciones de query
    query_configs = df[['filter_category', 'filter_storage_type', 'filter_price_min', 'filter_price_max']].drop_duplicates()
    
    summary = {
        "num_rows": n_rows,
        "num_cols": n_cols,
        "btr_global": btr_global,
        "cart_global": cart_global,
        "funnel_cart_to_bought": cart_to_bought,
        "funnel_no_cart_to_bought": no_cart_to_bought,
        "num_queries": int(df['query_id'].nunique()),
        "unique_query_configurations": len(query_configs),
        "avg_products_per_query": float(df.groupby('query_id')['price'].count().mean()),
        "min_products_per_query": int(df.groupby('query_id')['price'].count().min()),
        "max_products_per_query": int(df.groupby('query_id')['price'].count().max()),
        "num_categories": int(df['category'].nunique()),
        "num_brands": int(df['brand'].nunique()),
        "num_countries": int(df['country_of_origin'].nunique()),
        "missing_values": {k: v for k, v in missing.items() if v > 0},
    }
    return summary
