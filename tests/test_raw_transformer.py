"""Tests del módulo raw_transformer: serialización, dataset, modelo e integración con el Trainer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.raw_transformer.dataset import RawSerializedDataset, split_files
from src.raw_transformer.model import RawTransformerClassifier, RawTransformerConfig
from src.raw_transformer.serialize import (
    ALL_FIELDS,
    EXCLUDED_FIELDS,
    PRODUCT_ONLY_FIELDS,
    SEARCH_CONTEXT_FIELDS,
    format_value,
    load_and_serialize,
    serialize_row,
    split_and_save,
)
from src.tokenizer import ByteLevelBPETokenizer
from src.training.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _df_crudo(n: int = 40) -> pd.DataFrame:
    """DataFrame mínimo con el esquema del CSV crudo (los 20 campos + cart + bought)."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({campo: [f"{campo}_{i}" for i in range(n)] for campo in ALL_FIELDS})
    base["price"] = rng.uniform(1, 30, n).round(2)
    base["net_weight_oz"] = rng.uniform(3, 150, n).round(2)
    base["nutrition_score"] = rng.integers(0, 100, n)
    base["allergens"] = [None if i % 3 == 0 else "Wheat" for i in range(n)]
    # Timestamps desordenados a propósito, para verificar el ordenamiento cronológico
    fechas = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    base["timestamp"] = [f.isoformat() for f in fechas[rng.permutation(n)]]
    base["cart"] = rng.integers(0, 2, n)
    base["bought"] = [bool(i % 4 == 0) for i in range(n)]
    return base


@pytest.fixture(scope="module")
def tokenizer_chico() -> ByteLevelBPETokenizer:
    """BPE mínimo entrenado sobre textos con la pinta del corpus serializado."""
    textos = [
        "title: Producto uno | price: 8.3 | category: Frozen | nutrition_score: 61",
        "title: Producto dos | price: 13.25 | category: Dairy | nutrition_score: 9",
        "title: Producto tres | price: 2.68 | category: Frozen | allergens: None",
    ]
    return ByteLevelBPETokenizer().train_from_iterator(
        textos, vocab_size=300, min_frequency=1, show_progress=False
    )


@pytest.fixture()
def csv_serializado(tmp_path) -> Path:
    """CSV con la forma de la salida de serialize.py (text + bought), con ambas clases."""
    n = 48
    df = pd.DataFrame({
        "text": [f"title: Producto {i} | price: {i}.5 | category: Frozen" for i in range(n)],
        "bought": [1 if i % 4 == 0 else 0 for i in range(n)],
    })
    ruta = tmp_path / "raw_train.csv"
    df.to_csv(ruta, index=False)
    return ruta


def _config_chica(**kwargs) -> RawTransformerConfig:
    defaults = dict(vocab_size=300, max_seq_len=32, d_model=16, n_heads=2,
                    d_ff=32, num_layers=1, dropout=0.0)
    defaults.update(kwargs)
    return RawTransformerConfig(**defaults)


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

def test_format_value_maneja_nulos_bools_y_espacios():
    assert format_value(float("nan")) == "None"
    assert format_value(None) == "None"
    assert format_value(True) == "true"
    assert format_value(False) == "false"
    assert format_value("  Frozen  ") == "Frozen"
    assert format_value(8.3) == "8.3"


def test_serialize_row_formato_campo_valor():
    fila = pd.Series({"price": 8.3, "category": "Frozen"})
    assert serialize_row(fila, ["price", "category"]) == "price: 8.3 | category: Frozen"


def test_presets_excluyen_target_y_leakage():
    for preset in (ALL_FIELDS, PRODUCT_ONLY_FIELDS):
        assert not EXCLUDED_FIELDS.intersection(preset)
    # product_only además no puede ver el contexto de búsqueda
    assert not SEARCH_CONTEXT_FIELDS.intersection(PRODUCT_ONLY_FIELDS)
    assert SEARCH_CONTEXT_FIELDS.issubset(ALL_FIELDS)


def test_load_and_serialize_ordena_y_binariza(tmp_path):
    ruta = tmp_path / "crudo.csv"
    _df_crudo(40).to_csv(ruta, index=False)

    resultado = load_and_serialize(ruta, ALL_FIELDS)

    assert set(resultado.columns) == {"timestamp", "text", "bought"}
    assert resultado["bought"].isin([0, 1]).all()
    tiempos = pd.to_datetime(resultado["timestamp"], utc=True)
    assert tiempos.is_monotonic_increasing
    # Cada secuencia contiene todos los nombres de campo del preset
    assert all(f"{campo}:" in resultado["text"].iloc[0] for campo in ALL_FIELDS)
    # Los nulos se escriben como el centinela explícito
    assert "allergens: None" in " ".join(resultado["text"].tolist())


def test_load_and_serialize_rechaza_campos_excluidos(tmp_path):
    ruta = tmp_path / "crudo.csv"
    _df_crudo(10).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_and_serialize(ruta, ALL_FIELDS + ["bought"])


def test_split_and_save_proporciones_y_prefijo(tmp_path):
    ruta = tmp_path / "crudo.csv"
    _df_crudo(40).to_csv(ruta, index=False)
    df = load_and_serialize(ruta, PRODUCT_ONLY_FIELDS)

    train, val, test = split_and_save(df, tmp_path, prefix="raw_po")

    assert (len(train), len(val), len(test)) == (28, 6, 6)
    for nombre in ("raw_po_train.csv", "raw_po_val.csv", "raw_po_test.csv", "raw_po_dataset_complete.csv"):
        assert (tmp_path / nombre).exists()
    # El split es temporal: train termina antes de que empiece val, y val antes que test
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_split_files_construye_nombres_desde_el_prefijo():
    assert split_files() == {"train": "raw_train.csv", "val": "raw_val.csv", "test": "raw_test.csv"}
    assert split_files("raw_po")["val"] == "raw_po_val.csv"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_emite_batches_compatibles_con_el_trainer(csv_serializado, tokenizer_chico):
    ds = RawSerializedDataset(csv_serializado, tokenizer_chico, max_length=32)

    assert len(ds) == 48
    item = ds[0]
    # Las claves deben ser las que espera Trainer._forward_batch (decisión D6)
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["labels"].dtype == torch.float32

    # Padding a longitud fija: todas las filas comparten shape
    assert ds.input_ids.shape == (48, 32)
    assert ds.n_truncated == 0


def test_dataset_positive_rate_y_pos_weight(csv_serializado, tokenizer_chico):
    ds = RawSerializedDataset(csv_serializado, tokenizer_chico, max_length=32)
    assert ds.positive_rate == pytest.approx(12 / 48)
    assert ds.pos_weight().item() == pytest.approx(36 / 12)


def test_dataset_archivo_faltante_lanza_error(tmp_path, tokenizer_chico):
    with pytest.raises(FileNotFoundError, match="serialize"):
        RawSerializedDataset(tmp_path / "no_existe.csv", tokenizer_chico)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def test_modelo_forward_shapes_y_probabilidades():
    model = RawTransformerClassifier(_config_chica())
    model.eval()
    input_ids = torch.randint(1, 300, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=mask)

    assert logits.shape == (4,)
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_modelo_ignora_las_posiciones_de_padding():
    model = RawTransformerClassifier(_config_chica())
    model.eval()
    input_ids = torch.randint(1, 300, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    mask[:, 20:] = 0
    input_ids = input_ids * mask

    alterado = input_ids.clone()
    alterado[:, 20:] = torch.randint(1, 300, (2, 12))

    with torch.no_grad():
        delta = (model(input_ids, attention_mask=mask) - model(alterado, attention_mask=mask)).abs().max()
    assert delta.item() < 1e-4


def test_modelo_gradientes_llegan_a_los_embeddings():
    model = RawTransformerClassifier(_config_chica())
    model.train()
    input_ids = torch.randint(1, 300, (4, 32))
    logits = model(input_ids, attention_mask=torch.ones(4, 32, dtype=torch.long))
    torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0, 1.0, 0.0])).backward()

    grad = model.encoder.embedding.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_modelo_rechaza_pooling_none():
    with pytest.raises(ValueError, match="pooling"):
        _config_chica(pooling_mode="none")


def test_config_se_proyecta_sobre_el_encoder():
    config = _config_chica()
    encoder_config = config.to_encoder_config()
    for campo in ("vocab_size", "max_seq_len", "d_model", "n_heads", "d_ff", "num_layers"):
        assert getattr(encoder_config, campo) == getattr(config, campo)


# ---------------------------------------------------------------------------
# Integración con el Trainer común
# ---------------------------------------------------------------------------

def test_entrenamiento_end_to_end_con_el_trainer_comun(csv_serializado, tokenizer_chico):
    ds = RawSerializedDataset(csv_serializado, tokenizer_chico, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    model = RawTransformerClassifier(_config_chica(vocab_size=tokenizer_chico.vocab_size))
    trainer = Trainer(model, TrainerConfig(epochs=1, patience=None, verbose=False, seed=0))
    historia = trainer.fit(loader, loader)

    assert len(historia) == 1
    assert {"train_loss", "val_loss", "val_pr_auc", "val_roc_auc"}.issubset(historia[0])

    metricas = trainer.evaluate(loader, prefix="test_")
    assert 0.0 <= metricas["test_pr_auc"] <= 1.0
    assert 0.0 <= metricas["test_roc_auc"] <= 1.0
