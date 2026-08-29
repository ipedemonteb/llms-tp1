"""Tests del Dataset, los DataLoaders y el loop de entrenamiento.

El test más importante es `test_el_modelo_puede_sobreajustar_un_lote_chico`: si el modelo no
logra memorizar 32 ejemplos, hay un error en el cableado (gradientes que no fluyen, etiquetas
desalineadas, optimizador mal conectado) que ninguna verificación de dimensiones detecta.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.hybrid_transformer.fusion import BTRModel, FusionConfig
from src.hybrid_transformer.tabular_encoder import (
    TabularEncoder,
    TabularEncoderConfig,
    TabularPreprocessor,
)
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder
from src.training.dataset import SupermarketDataset, build_dataloaders
from src.training.trainer import Trainer, TrainerConfig, set_seed

from .conftest import requiere_datos, requiere_tokenizer


# --- Dataset ---

def test_dataset_solo_tabular(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, tokenizer=None, preprocessor=pre)
    assert len(ds) == len(df_sintetico)
    item = ds[0]
    assert set(item) == {"labels", "x_num", "x_cat"}
    assert item["labels"].dtype == torch.float32
    assert item["x_cat"].dtype == torch.int64


def test_dataset_sin_ninguna_rama_lanza_error(df_sintetico):
    with pytest.raises(ValueError):
        SupermarketDataset(df_sintetico, tokenizer=None, preprocessor=None)


def test_dataset_sin_target_lanza_error(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    with pytest.raises(KeyError):
        SupermarketDataset(df_sintetico.drop(columns=["bought"]), preprocessor=pre)


def test_labels_alineadas_con_el_dataframe(df_sintetico):
    """Un desalineamiento entre features y etiquetas es un error silencioso y fatal."""
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    esperado = torch.tensor(df_sintetico["bought"].to_numpy(), dtype=torch.float32)
    assert torch.equal(ds.labels, esperado)


def test_positive_rate_coincide_con_la_media(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    assert ds.positive_rate == pytest.approx(df_sintetico["bought"].mean())


@requiere_datos
@requiere_tokenizer
def test_dataset_con_texto_produce_tensores_correctos(df_train):
    from src.tokenizer.bpe import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer.from_file("resources/tokenizer/bpe_tokenizer.json")
    ds = SupermarketDataset(df_train.head(20), tokenizer=tok, max_length=64)
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    # El primer token debe ser [CLS] y la máscara debe marcar tokens reales
    assert item["input_ids"][0].item() == tok.cls_token_id
    assert item["attention_mask"].sum() > 0


@requiere_datos
@requiere_tokenizer
def test_build_dataloaders_no_filtra_informacion_entre_splits():
    """El preprocesador debe ajustarse solo con train: val no puede quedar centrado en 0."""
    loaders, art = build_dataloaders(batch_size=32, preprocessor_path=None)
    assert set(loaders) == {"train", "val", "test"}
    assert art["sizes"]["train"] == 7000

    val_x_num = loaders["val"].dataset.x_num
    assert not torch.allclose(val_x_num.mean(dim=0), torch.zeros(val_x_num.shape[1]), atol=1e-6)


@requiere_datos
@requiere_tokenizer
def test_los_splits_no_se_solapan():
    loaders, art = build_dataloaders(batch_size=32, preprocessor_path=None)
    total = sum(art["sizes"].values())
    assert total == 10000


# --- Loop de entrenamiento ---

def _modelo_chico(cards, num_numeric):
    return BTRModel(
        text_encoder=None,
        tabular_encoder=TabularEncoder(TabularEncoderConfig(
            num_numeric=num_numeric, cardinalities=cards, d_tab=16, dropout=0.0,
        )),
        fusion_config=FusionConfig(dropout=0.0),
    )


def test_el_modelo_puede_sobreajustar_un_lote_chico(df_sintetico):
    """Sanity check central: con 32 ejemplos y sin regularización, la loss debe bajar mucho."""
    set_seed(0)
    df = df_sintetico.head(32).copy()
    pre = TabularPreprocessor().fit(df)
    ds = SupermarketDataset(df, preprocessor=pre)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(epochs=150, lr=0.05, weight_decay=0.0,
                                            patience=None, verbose=False, device="cpu"))
    loss_inicial = trainer.evaluate(loader, prefix="x_")["x_loss"]
    for _ in range(150):
        trainer.train_epoch(loader)
    loss_final = trainer.evaluate(loader, prefix="x_")["x_loss"]

    assert loss_final < loss_inicial * 0.5, f"La loss no bajó lo suficiente: {loss_inicial} -> {loss_final}"
    assert trainer.evaluate(loader, prefix="x_")["x_pr_auc"] > 0.9


def test_early_stopping_corta_antes_del_maximo(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(epochs=50, patience=2, verbose=False, device="cpu"))
    historial = trainer.fit(loader, loader)
    assert len(historial) < 50
    assert trainer.best_epoch >= 1


def test_se_restaura_el_mejor_checkpoint(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(epochs=6, patience=None, verbose=False, device="cpu"))
    trainer.fit(loader, loader)

    mejor = max(trainer.history, key=lambda h: h["val_pr_auc"])
    assert trainer.best_val_pr_auc == pytest.approx(mejor["val_pr_auc"])
    # Los pesos restaurados deben reproducir la métrica del mejor epoch
    assert trainer.evaluate(loader)["val_pr_auc"] == pytest.approx(mejor["val_pr_auc"], abs=1e-6)


def test_el_historial_registra_una_entrada_por_epoca(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(epochs=3, patience=None, verbose=False, device="cpu"))
    historial = trainer.fit(loader, loader)
    assert len(historial) == 3
    assert [h["epoch"] for h in historial] == [1, 2, 3]
    assert {"train_loss", "val_loss", "train_pr_auc", "val_pr_auc", "val_roc_auc"} <= set(historial[0])


def test_evaluate_no_modifica_los_pesos(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(verbose=False, device="cpu"))
    antes = [p.clone() for p in modelo.parameters()]
    trainer.evaluate(loader)
    assert all(torch.equal(a, b) for a, b in zip(antes, modelo.parameters()))


def test_misma_semilla_produce_el_mismo_resultado(df_sintetico):
    def correr():
        set_seed(123)
        pre = TabularPreprocessor().fit(df_sintetico)
        ds = SupermarketDataset(df_sintetico, preprocessor=pre)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
        trainer = Trainer(modelo, TrainerConfig(epochs=2, patience=None, verbose=False,
                                                device="cpu", seed=123))
        return trainer.fit(loader, loader)[-1]["val_loss"]

    assert correr() == pytest.approx(correr(), rel=1e-9)


def test_predict_devuelve_logits_y_labels_alineados(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(verbose=False, device="cpu"))
    logits, labels = trainer.predict(loader)
    assert logits.shape == labels.shape == (len(ds),)
    assert np.array_equal(labels, ds.labels.numpy())


def test_checkpoint_se_guarda_con_historial(df_sintetico, tmp_path):
    pre = TabularPreprocessor().fit(df_sintetico)
    ds = SupermarketDataset(df_sintetico, preprocessor=pre)
    loader = DataLoader(ds, batch_size=16)
    modelo = _modelo_chico([pre.cardinalities[c] for c in pre.categorical_fields], len(pre.numeric_fields))
    trainer = Trainer(modelo, TrainerConfig(epochs=2, patience=None, verbose=False, device="cpu"))
    trainer.fit(loader, loader)

    ruta = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    guardado = torch.load(ruta, weights_only=False)
    assert set(guardado) == {"model_state_dict", "best_epoch", "best_val_pr_auc", "history"}
    assert len(guardado["history"]) == 2
