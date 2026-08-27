"""Script de extracción, filtrado y preprocesamiento de datos para entrenamiento.

Toma el dataset crudo en `resources/supermarket_products.csv`, aplica el filtrado
y transformaciones acordadas en `resources/feature_planning.md`, y genera el
archivo `resources/clean_dataset.csv`.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def parse_volume(dim_str: str) -> float:
    """Parsea el string de dimensiones (ej. '3.3 x 4.0 x 4.1\"') y calcula el volumen."""
    if not isinstance(dim_str, str) or not dim_str:
        return np.nan
    try:
        clean = dim_str.replace('"', '').strip()
        parts = [float(p.strip()) for p in clean.split('x')]
        if len(parts) == 3:
            return round(parts[0] * parts[1] * parts[2], 2)
        return np.nan
    except Exception:
        return np.nan


def clean_title(title_series: pd.Series) -> pd.Series:
    """Limpia el título eliminando badges entre paréntesis y sufijos de medida comercial."""
    # 1. Quitar el badge de reputación en paréntesis: (Customer Favorite), (Well Reviewed), etc.
    no_tag = title_series.str.replace(r'\s*\(.*?\)', '', regex=True)
    # 2. Quitar el sufijo de medida comercial: - 10 oz, - 6 ct, - 1 lb, etc.
    pure_title = no_tag.str.replace(r'\s*-\s*[\d\.]+\s*(oz|fl oz|lb|ct|gal)\s*$', '', regex=True)
    return pure_title.str.strip()


def extract_title_tag(title_series: pd.Series) -> pd.Series:
    """Extrae el badge de reputación/social proof presente entre paréntesis en el título."""
    return title_series.str.extract(r'\((.*?)\)')[0].fillna('No Tag')


def build_clean_dataset(
    input_path: str = "resources/supermarket_products.csv",
    output_path: str = "resources/clean_dataset.csv"
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de filtrado y transformación de features."""
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada en: {in_file.resolve()}")

    print(f"📂 Cargando dataset original desde: {in_file}")
    df_raw = pd.read_csv(in_file)
    print(f"   -> Filas: {df_raw.shape[0]:,}, Columnas originales: {df_raw.shape[1]}")

    print("⚙️  Aplicando transformaciones y feature engineering...")
    ts = pd.to_datetime(df_raw['timestamp'], utc=True)

    df_clean = pd.DataFrame()

    # 1. timestamp (para ordenamiento temporal y splits)
    df_clean['timestamp'] = df_raw['timestamp']

    # 2. title_clean (nombre puro del producto sin tags ni medidas)
    df_clean['title_clean'] = clean_title(df_raw['title'])

    # 3. title_tag (badge de reputación extraído)
    df_clean['title_tag'] = extract_title_tag(df_raw['title'])

    # 4. description
    df_clean['description'] = df_raw['description']

    # 5. price
    df_clean['price'] = df_raw['price']

    # 6. price_span (amplitud del filtro de precio: max - min)
    df_clean['price_span'] = (df_raw['filter_price_max'] - df_raw['filter_price_min']).round(2)

    # 7. price_per_oz (precio unitario por onza)
    df_clean['price_per_oz'] = (df_raw['price'] / df_raw['net_weight_oz']).round(4)

    # 8. category
    df_clean['category'] = df_raw['category']

    # 9. day_of_week (día de la semana derivado de timestamp)
    df_clean['day_of_week'] = ts.dt.day_name()

    # 10. brand
    df_clean['brand'] = df_raw['brand']

    # 11. unit_of_measure
    df_clean['unit_of_measure'] = df_raw['unit_of_measure']

    # 12. net_weight_oz
    df_clean['net_weight_oz'] = df_raw['net_weight_oz']

    # 13. volume (volumen en pulgadas cúbicas calculado de dimensions_in)
    df_clean['volume'] = df_raw['dimensions_in'].apply(parse_volume)

    # 14. storage_type
    df_clean['storage_type'] = df_raw['storage_type']

    # 15. ingredients
    df_clean['ingredients'] = df_raw['ingredients']

    # 16. num_ingredients (cantidad de ingredientes declarados)
    df_clean['num_ingredients'] = df_raw['ingredients'].fillna('').apply(
        lambda x: len(x.split(',')) if x else 0
    )

    # 17. allergens (imputando NaN a 'None')
    df_clean['allergens'] = df_raw['allergens'].fillna('None')

    # 18. has_allergens (flag binaria de presencia de alérgenos: 1 o 0)
    df_clean['has_allergens'] = df_raw['allergens'].notna().astype(int)

    # 19. nutrition_score
    df_clean['nutrition_score'] = df_raw['nutrition_score']

    # 20. country_of_origin
    df_clean['country_of_origin'] = df_raw['country_of_origin']

    # 21. bought (variable objetivo BTR ubicada al final para separar en X e y)
    df_clean['bought'] = df_raw['bought']

    # Guardar archivo limpio
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)
    print(f"✅ Dataset limpio generado con éxito en: {out_file}")
    print(f"   -> Filas: {df_clean.shape[0]:,}, Columnas finales: {df_clean.shape[1]}")
    print(f"   -> Columnas: {list(df_clean.columns)}")

    return df_clean


if __name__ == "__main__":
    build_clean_dataset()
