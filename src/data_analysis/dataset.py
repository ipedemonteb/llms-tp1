"""Módulo de carga, limpieza y enriquecimiento de datos para el análisis exploratorio."""

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
    """Realiza la limpieza de tipos, parseo de estructuras complejas y generación de features derivadas."""
    df = df.copy()

    # 1. Parseo temporal
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['year_month'] = df['timestamp'].dt.tz_localize(None).dt.to_period('M').astype(str)

    # 2. Parseo de dimensiones físicas y volumen
    dims = df['dimensions_in'].apply(parse_dimensions)
    df['length_in'] = [d[0] for d in dims]
    df['width_in'] = [d[1] for d in dims]
    df['height_in'] = [d[2] for d in dims]
    df['volume_cu_in'] = [d[3] for d in dims]
    df['density_oz_per_cu_in'] = df['net_weight_oz'] / (df['volume_cu_in'] + 1e-6)

    # 3. Tratamiento de alérgenos e ingredientes
    df['has_allergens'] = df['allergens'].notna()
    df['allergens_filled'] = df['allergens'].fillna('None')
    df['num_allergens'] = df['allergens'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['num_ingredients'] = df['ingredients'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

    # 4. Extracción de señales de texto (Social proof y reputación)
    # Títulos suelen incluir tags en paréntesis como (Best Seller), (Shopper Favorite), etc.
    df['title_tag'] = df['title'].str.extract(r'\((.*?)\)')[0].fillna('No Tag')

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

    # 5. Matching Query vs Producto
    df['match_category'] = df['category'] == df['filter_category']
    df['match_storage'] = df['storage_type'] == df['filter_storage_type']
    df['price_in_filter_range'] = (df['price'] >= df['filter_price_min']) & (df['price'] <= df['filter_price_max'])

    range_span = df['filter_price_max'] - df['filter_price_min']
    df['price_pos_in_filter'] = (df['price'] - df['filter_price_min']) / (range_span + 1e-6)

    # 6. Dinámica relativa dentro de la búsqueda (query_id)
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
    
    summary = {
        "num_rows": n_rows,
        "num_cols": n_cols,
        "btr_global": btr_global,
        "cart_global": cart_global,
        "funnel_cart_to_bought": cart_to_bought,
        "funnel_no_cart_to_bought": no_cart_to_bought,
        "num_queries": int(df['query_id'].nunique()),
        "avg_products_per_query": float(df.groupby('query_id')['price'].count().mean()),
        "num_categories": int(df['category'].nunique()),
        "num_brands": int(df['brand'].nunique()),
        "num_countries": int(df['country_of_origin'].nunique()),
        "missing_values": {k: v for k, v in missing.items() if v > 0},
    }
    return summary
