"""Fixtures compartidas por la suite de tests."""

import numpy as np
import pandas as pd
import pytest

RUTA_TRAIN = "resources/datasets/transformer_train.csv"
RUTA_TOKENIZER = "resources/tokenizer/bpe_tokenizer.json"

# Los tests que dependen de los artefactos generados se saltean si no existen todavía
from pathlib import Path

requiere_datos = pytest.mark.skipif(
    not Path(RUTA_TRAIN).exists(),
    reason="Faltan los splits. Ejecutar: uv run python -m src.data_extraction.build_transformer_dataset",
)
requiere_tokenizer = pytest.mark.skipif(
    not Path(RUTA_TOKENIZER).exists(),
    reason="Falta el tokenizador. Ejecutar: uv run python -m src.tokenizer.bpe",
)


@pytest.fixture(scope="session")
def df_sintetico() -> pd.DataFrame:
    """DataFrame mínimo con el esquema del dataset real, para tests sin dependencias externas."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "title_clean": [f"Marca{i % 3} Producto {i}" for i in range(n)],
        "badge": rng.choice(["Best Seller", "No Tag"], n),
        "title_tag": rng.choice(["Best Seller", "No Tag"], n),
        "description": [f"Descripción del producto {i}." for i in range(n)],
        "ingredients": rng.choice(["Harina, Sal", "Agua"], n),
        "country_of_origin": rng.choice(["Chile", "Peru"], n),
        "allergens": rng.choice(["Wheat", "No Allergens"], n),
        "price": rng.uniform(1, 30, n),
        "price_span": rng.uniform(5, 40, n),
        "price_per_oz": rng.uniform(0.01, 8, n),
        "net_weight_oz": rng.uniform(3, 150, n),
        "volume": rng.uniform(8, 3000, n),
        "num_ingredients": rng.integers(1, 6, n),
        "nutrition_score": rng.integers(0, 100, n),
        "category": rng.choice(["Dairy", "Frozen", "Produce"], n),
        "day_of_week": rng.choice(["Monday", "Friday"], n),
        "brand": rng.choice(["Marca0", "Marca1", "Marca2"], n),
        "unit_of_measure": rng.choice(["oz", "lb"], n),
        "storage_type": rng.choice(["Ambient", "Frozen"], n),
        "has_allergens": rng.integers(0, 2, n),
        "bought": rng.integers(0, 2, n),
    })


@pytest.fixture(scope="session")
def df_train() -> pd.DataFrame:
    """Split de entrenamiento real, si está disponible."""
    return pd.read_csv(RUTA_TRAIN)
