"""Entrenamiento y análisis del tokenizador BPE para el Transformer "pelado".

Implementa la Fase 2 del plan (`src/raw_transformer/PLAN.md`).

El tokenizador del `hybrid_transformer` (`resources/tokenizer/bpe_tokenizer.json`) se
entrenó únicamente sobre campos textuales: nunca vio dígitos, nombres de campo ni valores
categóricos escritos como palabras. Reutilizarlo aquí produciría una fragmentación pésima
de los números. Por eso se entrena un BPE propio sobre el corpus serializado y se guarda
por separado, sin pisar el del hybrid.

El entrenamiento usa **exclusivamente el split de train**: el vocabulario es un parámetro
aprendido del modelo, y entrenarlo con val/test sería leakage.

Además del entrenamiento, el script produce los dos análisis que alimentan la decisión D4
y la presentación:
1. Distribución de longitudes en tokens → elección de `max_seq_len` y costo de atención.
2. Inspección de cómo el BPE fragmenta los valores numéricos → evidencia empírica de por
   qué un Transformer "pelado" tiene dificultad con la magnitud numérica.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from src.tokenizer import ByteLevelBPETokenizer

# Fragmentos representativos del corpus serializado, para inspeccionar la tokenización.
NUMERIC_PROBES: List[str] = [
    "price: 8.3",
    "price: 8.25",
    "price: 13.33",
    "price: 2.68",
    "nutrition_score: 0",
    "nutrition_score: 61",
    "nutrition_score: 69",
    "net_weight_oz: 10.56",
    "net_weight_oz: 73.86",
    "dimensions_in: 4.9 x 10.1 x 6.1\"",
]

CATEGORICAL_PROBES: List[str] = [
    "category: Frozen",
    "category: Household",
    "brand: Harvest Lane",
    "storage_type: Ambient",
    "allergens: None",
]


def token_lengths(tokenizer: ByteLevelBPETokenizer, texts: Sequence[str]) -> pd.Series:
    """Cuenta tokens por texto SIN truncar ni padear, para medir la longitud real."""
    tokenizer.tokenizer.no_truncation()
    tokenizer.tokenizer.no_padding()
    encodings = tokenizer.tokenizer.encode_batch(list(texts), add_special_tokens=True)
    return pd.Series([len(enc.ids) for enc in encodings])


def report_length_distribution(
    tokenizer: ByteLevelBPETokenizer,
    splits: dict[str, pd.DataFrame],
) -> pd.Series:
    """Reporta la distribución de longitudes en tokens y sugiere `max_seq_len` (decisión D4)."""
    print("\n" + "=" * 78)
    print("📏 DISTRIBUCIÓN DE LONGITUD EN TOKENS BPE")
    print("=" * 78)

    train_lengths = None
    for name, df in splits.items():
        lengths = token_lengths(tokenizer, df["text"].tolist())
        if name == "train":
            train_lengths = lengths
        print(f"\n🔹 {name} ({len(df):,} filas)")
        print(f"   media: {lengths.mean():.0f} | p50: {lengths.median():.0f} | "
              f"p90: {lengths.quantile(0.90):.0f} | p95: {lengths.quantile(0.95):.0f} | "
              f"p99: {lengths.quantile(0.99):.0f} | max: {lengths.max()}")

    # Sugerencia de max_seq_len: potencias de 2 y cobertura resultante
    print("\n" + "-" * 78)
    print("  Cobertura según max_seq_len (sobre train) y costo relativo de la atención:")
    print("-" * 78)
    print(f"  {'max_seq_len':>12} | {'% secuencias completas':>22} | {'costo O(N²) relativo':>21}")
    print(f"  {'-'*12} | {'-'*22} | {'-'*21}")
    base = 128
    for n in [128, 192, 256, 384, 512]:
        coverage = (train_lengths <= n).mean() * 100
        cost = (n / base) ** 2
        print(f"  {n:>12} | {coverage:>21.1f}% | {cost:>20.1f}x")

    return train_lengths


def report_numeric_tokenization(tokenizer: ByteLevelBPETokenizer) -> None:
    """Muestra cómo el BPE fragmenta números y categóricas — el análisis central del experimento."""
    print("\n" + "=" * 78)
    print("🔬 CÓMO EL BPE FRAGMENTA LOS VALORES")
    print("=" * 78)

    print("\n--- NUMÉRICOS (el modelo debe inferir la magnitud desde estos fragmentos) ---")
    for probe in NUMERIC_PROBES:
        tokens = tokenizer.encode(probe, add_special_tokens=False).tokens
        print(f"  {probe!r:34} -> {len(tokens):>2} tokens  {tokens}")

    print("\n--- CATEGÓRICOS (comparación: acá la fragmentación importa mucho menos) ---")
    for probe in CATEGORICAL_PROBES:
        tokens = tokenizer.encode(probe, add_special_tokens=False).tokens
        print(f"  {probe!r:34} -> {len(tokens):>2} tokens  {tokens}")

    # Cuánto del presupuesto de secuencia se gasta solo en nombres de campo y separadores
    overhead = "title: | description: | price: | category: | timestamp: | query_id: | " \
               "filter_category: | filter_price_min: | filter_price_max: | " \
               "filter_storage_type: | brand: | package_size: | unit_of_measure: | " \
               "net_weight_oz: | dimensions_in: | storage_type: | ingredients: | " \
               "allergens: | nutrition_score: | country_of_origin:"
    n_overhead = len(tokenizer.encode(overhead, add_special_tokens=False).tokens)
    print(f"\n  ℹ️  Andamiaje (nombres de campo + separadores, sin ningún valor): "
          f"{n_overhead} tokens por secuencia.")


def train_raw_tokenizer(
    train_path: str = "resources/datasets/raw_train.csv",
    val_path: str = "resources/datasets/raw_val.csv",
    test_path: str = "resources/datasets/raw_test.csv",
    save_path: str = "resources/tokenizer/bpe_tokenizer_raw.json",
    vocab_size: int = 2048,
    min_frequency: int = 2,
    max_length: int = 128,
) -> ByteLevelBPETokenizer:
    """Entrena el BPE sobre el corpus serializado de train y ejecuta los análisis de la Fase 2."""
    paths = {"train": Path(train_path), "val": Path(val_path), "test": Path(test_path)}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan los datasets serializados: {missing}. "
            f"Corré primero: python -m src.raw_transformer.serialize"
        )

    splits = {name: pd.read_csv(p) for name, p in paths.items()}

    # 1. Entrenar EXCLUSIVAMENTE sobre train (el vocabulario es un parámetro aprendido)
    print("=" * 78)
    print("🔤 ENTRENAMIENTO DEL BPE SOBRE EL CORPUS SERIALIZADO")
    print("=" * 78)
    print(f"  Corpus: {train_path} ({len(splits['train']):,} secuencias)")
    print(f"  vocab_size: {vocab_size} | min_frequency: {min_frequency}")
    print("  ⚠️  Se entrena solo con train: usar val/test sería leakage.")

    tokenizer = ByteLevelBPETokenizer(max_length=max_length)
    tokenizer.train_from_iterator(
        iterator=splits["train"]["text"].dropna().astype(str).tolist(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    tokenizer.save(save_path)
    print(f"\n✅ Tokenizador guardado en: {save_path}")
    print(f"   Vocabulario efectivo: {tokenizer.vocab_size} tokens")

    # 2. Análisis para la decisión D4 y para la presentación
    report_length_distribution(tokenizer, splits)
    report_numeric_tokenization(tokenizer)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el BPE del Transformer pelado sobre el corpus serializado."
    )
    parser.add_argument("--train_path", type=str, default="resources/datasets/raw_train.csv")
    parser.add_argument("--val_path", type=str, default="resources/datasets/raw_val.csv")
    parser.add_argument("--test_path", type=str, default="resources/datasets/raw_test.csv")
    parser.add_argument(
        "--save_path", type=str, default="resources/tokenizer/bpe_tokenizer_raw.json",
        help="Se guarda aparte para no pisar el tokenizador del hybrid_transformer.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=2048,
        help="Igual al del hybrid (2048) para que la comparación sea justa.",
    )
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    train_raw_tokenizer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        save_path=args.save_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
