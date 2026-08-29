"""Script de preparación y partición del dataset para modelos Transformer.

Toma el dataset procesado (o crudo), construye la secuencia de texto concatenando
los campos indicados en `--text_fields` con separadores '|', convierte la variable
objetivo `bought` a binaria (0/1), ordena cronológicamente por `timestamp` y genera
particiones temporales estrictas (70% Train, 15% Val, 15% Test) para evitar
cualquier tipo de data leakage temporal.

Composición de la secuencia de texto
------------------------------------
Los campos que forman la columna `text` son configurables vía `--text_fields`.
El default incluye seis campos:

    title_clean | badge | description | ingredients | country_of_origin | allergens

Los dos últimos se incorporaron porque el EDA mostró que NO están representados en
la prosa del catálogo: `country_of_origin` no aparece en ninguna fila y `allergens`
solo en el 35% de las que declaran alérgeno. En cambio `brand`, `category`,
`storage_type` y `unit_of_measure` ya aparecen literalmente en el texto en el 100%
de las filas (ver `src/data_analysis/brand_title_consistency.py`), por lo que
agregarlos duplicaría tokens sin aportar información nueva.

Todas las columnas del dataset de entrada se conservan en la salida, de modo que la
rama tabular pueda consumirlas y que la composición del texto pueda recalcularse en
tiempo de entrenamiento sin regenerar las particiones.

Guarda los datasets generados en la carpeta `resources/datasets/`:
- `resources/datasets/transformer_train.csv` (70%)
- `resources/datasets/transformer_val.csv` (15%)
- `resources/datasets/transformer_test.csv` (15%)
- `resources/datasets/transformer_dataset_complete.csv` (100% con columna 'split')

Uso:
    # Default: seis campos de texto
    uv run python -m src.data_extraction.build_transformer_dataset

    # Ablación: agregar `brand` como séptimo campo
    uv run python -m src.data_extraction.build_transformer_dataset \
        --text_fields title_clean,badge,description,ingredients,country_of_origin,allergens,brand
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd


# Campos que componen la secuencia de texto por defecto (en este orden)
DEFAULT_TEXT_FIELDS: List[str] = [
    "title_clean",
    "badge",
    "description",
    "ingredients",
    "country_of_origin",
    "allergens",
]

# Valor de relleno para nulos, por campo. Se usa un centinela explícito en vez de string
# vacío para que la ausencia sea un token con significado y no un hueco en la secuencia.
# IMPORTANTE: ningún centinela puede ser 'None', 'NA', 'NULL' ni 'NaN' — pandas los incluye
# en su lista de `na_values` por defecto y los reinterpretaría como NaN al releer el CSV.
FILL_VALUES = {
    "title_clean": "",
    "badge": "No Tag",
    "title_tag": "No Tag",
    "description": "",
    "ingredients": "No Ingredients",
    "allergens": "No Allergens",
    "country_of_origin": "Unknown",
    "brand": "Unknown",
    "category": "Unknown",
    "storage_type": "Unknown",
    "unit_of_measure": "Unknown",
}


def clean_title(title_series: pd.Series) -> pd.Series:
    """Limpia el título eliminando badges entre paréntesis y sufijos de medida comercial."""
    no_tag = title_series.str.replace(r'\s*\(.*?\)', '', regex=True)
    pure_title = no_tag.str.replace(r'\s*-\s*[\d\.]+\s*(oz|fl oz|lb|ct|gal)\s*$', '', regex=True)
    return pure_title.str.strip()


def extract_title_tag(title_series: pd.Series) -> pd.Series:
    """Extrae el badge de reputación/social proof presente entre paréntesis en el título."""
    return title_series.str.extract(r'\((.*?)\)')[0].fillna('No Tag').str.strip()


def build_text_sequence(
    df: pd.DataFrame,
    text_fields: List[str],
    separator: str = " | "
) -> pd.Series:
    """Concatena los campos indicados en una única secuencia de texto formateada.

    Args:
        df: DataFrame que contiene (al menos) las columnas listadas en `text_fields`.
        text_fields: Nombres de columnas a concatenar, en el orden deseado.
        separator: Separador insertado entre campos.

    Returns:
        Serie de strings con la secuencia de texto por fila.
    """
    faltantes = [c for c in text_fields if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Campos de texto no encontrados en el dataset: {faltantes}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    partes = []
    for col in text_fields:
        relleno = FILL_VALUES.get(col, "")
        partes.append(df[col].fillna(relleno).astype(str).str.strip())

    secuencia = partes[0]
    for parte in partes[1:]:
        secuencia = secuencia + separator + parte
    return secuencia


def load_and_preprocess(
    input_path: Path,
    text_fields: List[str] = None,
    separator: str = " | "
) -> pd.DataFrame:
    """Carga el dataset, normaliza los campos derivados y construye la secuencia de texto.

    Conserva todas las columnas del archivo de entrada y agrega `text` (secuencia
    tokenizable) y `bought` binarizado, ordenando cronológicamente por `timestamp`.
    """
    if text_fields is None:
        text_fields = DEFAULT_TEXT_FIELDS

    print(f"📂 Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   -> Filas cargadas: {len(df):,}, Columnas: {len(df.columns)}")

    df_result = df.copy()

    # 1. Asegurar title_clean (derivándolo de `title` si el input es el CSV crudo)
    if 'title_clean' not in df_result.columns:
        if 'title' in df_result.columns:
            df_result['title_clean'] = clean_title(df_result['title'])
        else:
            raise KeyError("No se encontró columna 'title_clean' ni 'title' en el dataset.")

    # 2. Asegurar badge (alias de title_tag; se deriva de `title` si hace falta)
    if 'badge' not in df_result.columns:
        if 'title_tag' in df_result.columns:
            df_result['badge'] = df_result['title_tag']
        elif 'title' in df_result.columns:
            df_result['badge'] = extract_title_tag(df_result['title'])
        else:
            df_result['badge'] = 'No Tag'

    # 3. Normalizar alérgenos: NaN significa "sin alérgenos declarados", no dato faltante
    if 'allergens' in df_result.columns:
        df_result['allergens'] = df_result['allergens'].fillna(FILL_VALUES['allergens'])

    # 4. Variable objetivo: bought a binario (0 o 1). Maneja bools, strings y numéricos.
    if 'bought' not in df_result.columns:
        raise KeyError("No se encontró la columna objetivo 'bought' en el dataset.")
    df_result['bought'] = df_result['bought'].replace(
        {'True': 1, 'False': 0, 'true': 1, 'false': 0, True: 1, False: 0}
    ).astype(int)

    # 5. Parseo de timestamp para el ordenamiento cronológico
    if 'timestamp' not in df_result.columns:
        raise KeyError("No se encontró la columna 'timestamp' requerida para el ordenamiento temporal.")
    timestamps = pd.to_datetime(df_result['timestamp'], utc=True)

    # 6. Generar la secuencia de texto unificada
    print(f"🧩 Componiendo la secuencia de texto con {len(text_fields)} campos:")
    for i, campo in enumerate(text_fields, start=1):
        print(f"   {i}. {campo}")
    df_result['text'] = build_text_sequence(df_result, text_fields=text_fields, separator=separator)

    # 7. Ordenar estrictamente por timestamp
    print("⏳ Ordenando cronológicamente por timestamp...")
    df_result = df_result.assign(_ts=timestamps).sort_values('_ts').drop(columns=['_ts'])
    df_result = df_result.reset_index(drop=True)

    # 8. Reubicar `text` y `bought` al final para separar fácilmente X de y
    cols = [c for c in df_result.columns if c not in ('text', 'bought')] + ['text', 'bought']
    df_result = df_result[cols]

    print(f"   -> Columnas conservadas en la salida: {len(df_result.columns)}")

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

    print("\n" + "=" * 70)
    print("📊 RESUMEN DE PARTICIÓN TEMPORAL DEL DATASET TRANSFORMER")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("📝 EJEMPLO DE SECUENCIA GENERADA (Sample 1 de Train):")
    print("=" * 70)
    sample_row = df_train.iloc[0]
    print(f"Texto:  {sample_row['text']}")
    print(f"Target: bought = {sample_row['bought']}")
    print("=" * 70 + "\n")

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
        "--text_fields",
        type=str,
        default=",".join(DEFAULT_TEXT_FIELDS),
        help=(
            "Lista separada por comas de los campos que componen la secuencia de texto, en orden. "
            f"Por defecto: {','.join(DEFAULT_TEXT_FIELDS)}"
        )
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

    text_fields = [c.strip() for c in args.text_fields.split(",") if c.strip()]
    if not text_fields:
        raise ValueError("--text_fields no puede quedar vacío.")

    out_dir = Path(args.output_dir)

    df_prepared = load_and_preprocess(
        input_path=in_path,
        text_fields=text_fields,
        separator=args.separator
    )
    split_and_save(
        df=df_prepared,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
