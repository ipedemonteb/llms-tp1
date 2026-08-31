"""Serialización total del dataset crudo a texto plano para el Transformer "pelado".

Este script implementa la Fase 1 del plan (`src/raw_transformer/PLAN.md`): toma el CSV
crudo `resources/datasets/supermarket_products.csv` y convierte cada fila completa —
incluidas las variables numéricas y categóricas — en una única secuencia de texto con
formato `campo: valor | campo: valor | ...`.

Decisiones de diseño aplicadas (ver PLAN.md):
- **D1 (formato):** valores crudos, sin redondear ni bucketizar. `price: 8.3` se escribe
  tal cual viene del CSV, de modo que el modelo deba inferir la magnitud numérica a partir
  de los caracteres.
- **D2 (campos):** los 20 campos del CSV excluyendo `cart` (leakage de embudo) y `bought`
  (variable objetivo). Sin features derivadas.

Se ofrecen dos presets de campos:
- `all`: los 20 campos. Es el "pelado total", sin ninguna curaduría humana.
- `product_only`: excluye el contexto de búsqueda (`query_id`, `filter_*`) y `timestamp`,
  para que la comparación contra `hybrid_transformer` sea controlada (ese descarta esos
  campos en `feature_planning.md`).

El split temporal replica exactamente el de `src/data_extraction/build_transformer_dataset.py`
(orden cronológico estable + corte 70/15/15 por índice), de modo que ambos modelos vean
las mismas filas en train, val y test.

Salidas en `resources/datasets/`:
- `raw_train.csv` (70%) / `raw_val.csv` (15%) / `raw_test.csv` (15%)
- `raw_dataset_complete.csv` (100%, con columna `split`)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

# Campos excluidos de forma permanente del input del modelo.
EXCLUDED_FIELDS = {
    "cart",    # Leakage de embudo: bought=1 implica cart=1 siempre (verificado en el EDA).
    "bought",  # Variable objetivo.
}

# Preset "all": los 20 campos del CSV crudo, en el orden en que se definen en el enunciado.
ALL_FIELDS: List[str] = [
    "title",
    "description",
    "price",
    "category",
    "timestamp",
    "query_id",
    "filter_category",
    "filter_price_min",
    "filter_price_max",
    "filter_storage_type",
    "brand",
    "package_size",
    "unit_of_measure",
    "net_weight_oz",
    "dimensions_in",
    "storage_type",
    "ingredients",
    "allergens",
    "nutrition_score",
    "country_of_origin",
]

# Preset "product_only": solo atributos del producto, sin contexto de búsqueda ni tiempo.
# Es el conjunto comparable con el que consume `hybrid_transformer`.
SEARCH_CONTEXT_FIELDS = {
    "timestamp",
    "query_id",
    "filter_category",
    "filter_price_min",
    "filter_price_max",
    "filter_storage_type",
}
PRODUCT_ONLY_FIELDS: List[str] = [f for f in ALL_FIELDS if f not in SEARCH_CONTEXT_FIELDS]

FIELD_PRESETS = {
    "all": ALL_FIELDS,
    "product_only": PRODUCT_ONLY_FIELDS,
}

NULL_PLACEHOLDER = "None"


def format_value(value) -> str:
    """Convierte un valor de celda a su representación textual cruda.

    No se redondea, no se fija cantidad de decimales y no se bucketiza: el objetivo del
    experimento es que el modelo reciba el valor sin ninguna ayuda. Los nulos se escriben
    explícitamente como `None` en lugar de omitir el campo, para que la secuencia mantenga
    una estructura constante en todas las filas.
    """
    if pd.isna(value):
        return NULL_PLACEHOLDER
    # Los booleanos de pandas se escriben en minúscula, igual que en el CSV original.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def serialize_row(row: pd.Series, fields: Sequence[str], separator: str = " | ") -> str:
    """Serializa una fila completa como `campo: valor | campo: valor | ...`."""
    return separator.join(f"{field}: {format_value(row[field])}" for field in fields)


def serialize_dataframe(
    df: pd.DataFrame,
    fields: Sequence[str],
    separator: str = " | ",
) -> pd.Series:
    """Aplica la serialización a todas las filas del DataFrame."""
    return df.apply(lambda row: serialize_row(row, fields, separator), axis=1)


def load_and_serialize(
    input_path: Path,
    fields: Sequence[str],
    separator: str = " | ",
) -> pd.DataFrame:
    """Carga el CSV crudo, valida los campos, serializa y ordena cronológicamente."""
    print(f"📂 Cargando dataset crudo desde: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   -> Filas: {len(df):,}, Columnas: {len(df.columns)}")

    # 1. Validaciones de integridad
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise KeyError(f"Campos no encontrados en el CSV: {missing}")

    leaked = EXCLUDED_FIELDS.intersection(fields)
    if leaked:
        raise ValueError(
            f"Los campos {sorted(leaked)} nunca deben formar parte del input del modelo "
            f"(leakage de embudo / variable objetivo)."
        )

    if "bought" not in df.columns:
        raise KeyError("No se encontró la columna objetivo 'bought' en el dataset.")
    if "timestamp" not in df.columns:
        raise KeyError("No se encontró la columna 'timestamp', requerida para el split temporal.")

    # 2. Serializar la fila completa a una única secuencia de texto
    print(f"⚙️  Serializando {len(fields)} campos por fila...")
    text_sequence = serialize_dataframe(df, fields, separator)

    # 3. Variable objetivo a binario (0/1), tolerando bools, strings y numéricos
    bought = df["bought"].replace(
        {"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0}
    ).astype(int)

    result = pd.DataFrame({
        "timestamp": df["timestamp"],
        "text": text_sequence,
        "bought": bought,
    })

    # 4. Orden cronológico estable — replica el split de build_transformer_dataset.py
    #    para que ambos modelos vean exactamente las mismas filas en cada partición.
    print("⏳ Ordenando cronológicamente por timestamp...")
    order = pd.to_datetime(df["timestamp"], utc=True)
    result = result.iloc[order.sort_values(kind="stable").index].reset_index(drop=True)

    return result


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "raw",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide temporalmente el dataset serializado y exporta los CSV resultantes."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Las proporciones de split deben sumar 1.0"

    total = len(df)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    df_complete = df.copy()
    df_complete["split"] = "train"
    split_col = df_complete.columns.get_loc("split")
    df_complete.iloc[n_train:n_train + n_val, split_col] = "val"
    df_complete.iloc[n_train + n_val:, split_col] = "test"

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / f"{prefix}_train.csv",
        "val": output_dir / f"{prefix}_val.csv",
        "test": output_dir / f"{prefix}_test.csv",
        "complete": output_dir / f"{prefix}_dataset_complete.csv",
    }

    df_train.to_csv(paths["train"], index=False)
    df_val.to_csv(paths["val"], index=False)
    df_test.to_csv(paths["test"], index=False)
    df_complete.to_csv(paths["complete"], index=False)

    print("\n" + "=" * 70)
    print("📊 PARTICIÓN TEMPORAL DEL DATASET SERIALIZADO (RAW)")
    print("=" * 70)

    for name, split_df, path in [
        ("Train (70%)", df_train, paths["train"]),
        ("Val   (15%)", df_val, paths["val"]),
        ("Test  (15%)", df_test, paths["test"]),
        ("Total (100%)", df_complete, paths["complete"]),
    ]:
        n1 = int((split_df["bought"] == 1).sum())
        n0 = int((split_df["bought"] == 0).sum())
        btr = split_df["bought"].mean() * 100
        print(f"\n🔹 {name}:")
        print(f"   - Archivo: {path}")
        print(f"   - Filas: {len(split_df):,} ({len(split_df) / total * 100:.1f}%)")
        print(f"   - Rango temporal: [{split_df['timestamp'].min()} -> {split_df['timestamp'].max()}]")
        print(f"   - Balance: 0 (No)={n0:,} | 1 (Bought)={n1:,} | BTR: {btr:.2f}%")

    return df_train, df_val, df_test


def report_lengths(df: pd.DataFrame) -> None:
    """Reporta la distribución de longitudes de las secuencias serializadas.

    Es un insumo para la decisión D4 del plan (elección de `max_seq_len`): la longitud en
    tokens del BPE será mayor que la longitud en palabras, y el costo de la atención crece
    de forma cuadrática con la longitud de secuencia.
    """
    chars = df["text"].str.len()
    words = df["text"].str.split().str.len()

    print("\n" + "=" * 70)
    print("📏 LONGITUD DE LAS SECUENCIAS SERIALIZADAS")
    print("=" * 70)
    print(f"  Caracteres -> media: {chars.mean():.0f} | p50: {chars.median():.0f} | "
          f"p95: {chars.quantile(0.95):.0f} | max: {chars.max()}")
    print(f"  Palabras   -> media: {words.mean():.0f} | p50: {words.median():.0f} | "
          f"p95: {words.quantile(0.95):.0f} | max: {words.max()}")
    print("  ⚠️  La longitud en tokens BPE será mayor. Se define en la Fase 2 del plan.")


def print_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Imprime ejemplos de secuencias serializadas para inspección manual (checkpoint Fase 1)."""
    print("\n" + "=" * 70)
    print(f"📝 {n} EJEMPLOS DE SECUENCIAS SERIALIZADAS")
    print("=" * 70)
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        print(f"\n--- Ejemplo {i + 1}  (bought = {row['bought']}) ---")
        print(row["text"])


def build_raw_dataset(
    input_path: str = "resources/datasets/supermarket_products.csv",
    output_dir: str = "resources/datasets",
    field_preset: str = "all",
    separator: str = " | ",
    prefix: str = "raw",
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de serialización y partición."""
    if field_preset not in FIELD_PRESETS:
        raise ValueError(
            f"Preset '{field_preset}' inválido. Opciones: {sorted(FIELD_PRESETS)}"
        )
    fields = FIELD_PRESETS[field_preset]

    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada en: {in_file.resolve()}")

    print(f"🧩 Preset de campos: '{field_preset}' ({len(fields)} campos)")
    print(f"   {fields}")

    df = load_and_serialize(in_file, fields, separator)
    report_lengths(df)
    print_examples(df, n=5)
    split_and_save(df, Path(output_dir), prefix=prefix)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Serializa el dataset crudo a texto plano para el Transformer pelado."
    )
    parser.add_argument(
        "--input", type=str, default="resources/datasets/supermarket_products.csv",
        help="Ruta al CSV crudo de entrada.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="resources/datasets",
        help="Carpeta de destino para los CSV generados.",
    )
    parser.add_argument(
        "--field_preset", type=str, default="all", choices=sorted(FIELD_PRESETS),
        help="'all' = los 20 campos del CSV; 'product_only' = sin contexto de búsqueda ni timestamp.",
    )
    parser.add_argument(
        "--separator", type=str, default=" | ",
        help="Separador entre campos serializados.",
    )
    parser.add_argument(
        "--prefix", type=str, default="raw",
        help="Prefijo de los archivos de salida (ej. 'raw' -> raw_train.csv).",
    )
    args = parser.parse_args()

    build_raw_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        field_preset=args.field_preset,
        separator=args.separator,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
