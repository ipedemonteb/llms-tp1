"""Script de preparación y partición del dataset de texto para modelos Transformer.

Toma el dataset procesado (o crudo), construye la secuencia de texto concatenando
(title_clean, badge, description, ingredients) con separadores '|', convierte
la variable objetivo `bought` a binaria (0/1), ordena cronológicamente por `timestamp`
y genera particiones temporales estrictas (70% Train, 15% Val, 15% Test) para
evitar cualquier tipo de data leakage temporal.

Guarda los datasets generados en la carpeta `resources/datasets/`:
- `resources/datasets/transformer_train.csv` (70%)
- `resources/datasets/transformer_val.csv` (15%)
- `resources/datasets/transformer_test.csv` (15%)
- `resources/datasets/transformer_dataset_complete.csv` (100% con columna 'split')
"""

import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd


def clean_title(title_series: pd.Series) -> pd.Series:
    """Limpia el título eliminando badges entre paréntesis y sufijos de medida comercial."""
    no_tag = title_series.str.replace(r'\s*\(.*?\)', '', regex=True)
    pure_title = no_tag.str.replace(r'\s*-\s*[\d\.]+\s*(oz|fl oz|lb|ct|gal)\s*$', '', regex=True)
    return pure_title.str.strip()


def extract_title_tag(title_series: pd.Series) -> pd.Series:
    """Extrae el badge de reputación/social proof presente entre paréntesis en el título."""
    return title_series.str.extract(r'\((.*?)\)')[0].fillna('No Tag').str.strip()


def build_text_sequence(
    title_clean: pd.Series,
    badge: pd.Series,
    description: pd.Series,
    ingredients: pd.Series,
    separator: str = " | "
) -> pd.Series:
    """Concatena los campos textuales en una única secuencia formateada."""
    s_title = title_clean.fillna('').astype(str).str.strip()
    s_badge = badge.fillna('No Tag').astype(str).str.strip()
    s_desc = description.fillna('').astype(str).str.strip()
    s_ing = ingredients.fillna('None').astype(str).str.strip()

    return s_title + separator + s_badge + separator + s_desc + separator + s_ing


def load_and_preprocess(
    input_path: Path,
    separator: str = " | "
) -> pd.DataFrame:
    """Carga y estandariza los datos para el pipeline de texto del Transformer."""
    print(f"📂 Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   -> Filas cargadas: {len(df):,}, Columnas: {len(df.columns)}")

    # 1. Asegurar title_clean y badge (title_tag)
    if 'title_clean' in df.columns:
        title_clean = df['title_clean']
    elif 'title' in df.columns:
        title_clean = clean_title(df['title'])
    else:
        raise KeyError("No se encontró columna 'title_clean' ni 'title' en el dataset.")

    if 'title_tag' in df.columns:
        badge = df['title_tag']
    elif 'badge' in df.columns:
        badge = df['badge']
    elif 'title' in df.columns:
        badge = extract_title_tag(df['title'])
    else:
        badge = pd.Series('No Tag', index=df.index)

    # 2. Descripción e ingredientes
    description = df['description'] if 'description' in df.columns else pd.Series('', index=df.index)
    ingredients = df['ingredients'] if 'ingredients' in df.columns else pd.Series('None', index=df.index)

    # 3. Variable objetivo: bought a binario (0 o 1)
    if 'bought' not in df.columns:
        raise KeyError("No se encontró la columna objetivo 'bought' en el dataset.")
    
    # Maneja bools, strings ('true'/'false') y numéricos
    bought_series = df['bought'].replace({'True': 1, 'False': 0, 'true': 1, 'false': 0, True: 1, False: 0}).astype(int)

    # 4. Parseo de timestamp
    if 'timestamp' not in df.columns:
        raise KeyError("No se encontró la columna 'timestamp' requerida para el ordenamiento temporal.")
    
    timestamps = pd.to_datetime(df['timestamp'], utc=True)

    # 5. Generar secuencia de texto unificada
    text_sequence = build_text_sequence(
        title_clean=title_clean,
        badge=badge,
        description=description,
        ingredients=ingredients,
        separator=separator
    )

    # Construir DataFrame final ordenado
    df_result = pd.DataFrame({
        'timestamp': df['timestamp'],
        'timestamp_dt': timestamps,
        'title_clean': title_clean,
        'badge': badge,
        'description': description,
        'ingredients': ingredients,
        'text': text_sequence,
        'bought': bought_series
    })

    # Ordenar estrictamente por timestamp
    print("⏳ Ordenando cronológicamente por timestamp...")
    df_result = df_result.sort_values('timestamp_dt').reset_index(drop=True)
    df_result = df_result.drop(columns=['timestamp_dt'])

    return df_result


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide temporalmente el dataset y exporta los archivos CSV."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Las proporciones de split deben sumar 1.0"
    
    total_len = len(df)
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    # Crear columna identificadora de split en el dataset completo
    df_complete = df.copy()
    df_complete['split'] = 'train'
    df_complete.iloc[n_train:n_train + n_val, df_complete.columns.get_loc('split')] = 'val'
    df_complete.iloc[n_train + n_val:, df_complete.columns.get_loc('split')] = 'test'

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "transformer_train.csv"
    val_path = output_dir / "transformer_val.csv"
    test_path = output_dir / "transformer_test.csv"
    complete_path = output_dir / "transformer_dataset_complete.csv"

    # Exportar los splits
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_complete.to_csv(complete_path, index=False)

    print("\n" + "="*70)
    print("📊 RESUMEN DE PARTICIÓN TEMPORAL DEL DATASET TRANSFORMER")
    print("="*70)
    
    splits_info = [
        ("Train (70%)", df_train, train_path),
        ("Val   (15%)", df_val, val_path),
        ("Test  (15%)", df_test, test_path),
        ("Total (100%)", df_complete, complete_path)
    ]

    for name, split_df, path in splits_info:
        bought_1 = (split_df['bought'] == 1).sum()
        bought_0 = (split_df['bought'] == 0).sum()
        btr = split_df['bought'].mean() * 100
        t_min = split_df['timestamp'].min()
        t_max = split_df['timestamp'].max()
        print(f"\n🔹 {name}:")
        print(f"   - Archivo: {path}")
        print(f"   - Filas: {len(split_df):,} ({len(split_df)/total_len*100:.1f}%)")
        print(f"   - Rango Temporal: [{t_min}  ->  {t_max}]")
        print(f"   - Balance Target: 0 (No)={bought_0:,} | 1 (Bought)={bought_1:,} | BTR: {btr:.2f}%")

    print("\n" + "="*70)
    print("📝 EJEMPLO DE SECUENCIA GENERADA (Sample 1 de Train):")
    print("="*70)
    sample_row = df_train.iloc[0]
    print(f"Texto:  {sample_row['text']}")
    print(f"Target: bought = {sample_row['bought']}")
    print("="*70 + "\n")

    return df_train, df_val, df_test


def main():
    parser = argparse.ArgumentParser(description="Genera el dataset de texto particionado para Transformer.")
    parser.add_argument(
        "--input",
        type=str,
        default="resources/datasets/clean_dataset.csv",
        help="Ruta al archivo CSV de entrada (por defecto: resources/datasets/clean_dataset.csv o resources/datasets/supermarket_products.csv)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resources/datasets",
        help="Carpeta de destino para los archivos CSV generados (por defecto: resources/datasets)."
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" | ",
        help="Separador utilizado entre campos textuales (por defecto: ' | ')."
    )
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Proporción para Train (0.70).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proporción para Validation (0.15).")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proporción para Test (0.15).")

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        fallback = Path("resources/datasets/supermarket_products.csv")
        if fallback.exists():
            print(f"⚠️ No se encontró {in_path}. Usando fallback: {fallback}")
            in_path = fallback
        else:
            raise FileNotFoundError(f"No se encontró el dataset en {in_path} ni en {fallback}")

    out_dir = Path(args.output_dir)

    df_prepared = load_and_preprocess(input_path=in_path, separator=args.separator)
    split_and_save(
        df=df_prepared,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
