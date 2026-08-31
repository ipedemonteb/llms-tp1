"""Tests del modelo: encoders, fusión y cabeza clasificadora.

Verifican las propiedades estructurales que deben cumplirse siempre: dimensiones de salida,
flujo de gradientes, efecto real de la máscara de padding y equivalencia entre configuraciones
que deberían comportarse igual.
"""

import pytest
import torch

from src.hybrid_transformer.fusion import (
    BTRModel,
    ClassifierHead,
    CrossAttentionFusion,
    FusionConfig,
)
from src.hybrid_transformer.tabular_encoder import TabularEncoder, TabularEncoderConfig
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder

VOCAB, D_MODEL, D_TAB = 128, 32, 16
B, T = 6, 12


def _texto(pooling="mean"):
    return TextTransformerEncoder(TextTransformerConfig(
        vocab_size=VOCAB, max_seq_len=T, d_model=D_MODEL, n_heads=4,
        d_ff=64, num_layers=2, pooling_mode=pooling,
    ))


def _tabular(use_mlp: bool = False):
    return TabularEncoder(TabularEncoderConfig(
        num_numeric=4, num_direct=1, embedding_cardinalities=[3, 4],
        onehot_cardinalities=[5, 6], d_tab=D_TAB, use_mlp=use_mlp,
    ))


@pytest.fixture
def lote():
    torch.manual_seed(0)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 8:] = 0
    mask[1, 5:] = 0
    cards = [3, 4, 5, 6]
    return {
        "input_ids": torch.randint(1, VOCAB, (B, T)) * mask,
        "attention_mask": mask,
        "x_num": torch.randn(B, 5),
        "x_cat": torch.stack([torch.randint(1, c + 1, (B,)) for c in cards], dim=1),
    }


# --- Cabeza clasificadora ---

def test_cabeza_devuelve_un_logit_por_muestra():
    salida = ClassifierHead(input_dim=48)(torch.randn(B, 48))
    assert salida.shape == (B,)


def test_cabeza_no_aplica_sigmoide():
    """La salida debe ser un logit sin acotar: la sigmoide vive en BCEWithLogitsLoss."""
    cabeza = ClassifierHead(input_dim=8)
    with torch.no_grad():
        salida = cabeza(torch.randn(200, 8) * 50)
    assert salida.min() < 0.0 and salida.max() > 0.0
    assert not torch.all((salida >= 0) & (salida <= 1))


# --- Cross-attention ---

def test_cross_attention_devuelve_dimension_de_texto(lote):
    fusion = CrossAttentionFusion(D_MODEL, D_TAB, n_heads=4)
    salida = fusion(torch.randn(B, T, D_MODEL), torch.randn(B, D_TAB), lote["attention_mask"])
    assert salida.shape == (B, D_MODEL)


def test_pesos_de_atencion_suman_uno_y_respetan_la_mascara(lote):
    fusion = CrossAttentionFusion(D_MODEL, D_TAB, n_heads=4, dropout=0.0)
    fusion.eval()
    with torch.no_grad():
        _, pesos = fusion(
            torch.randn(B, T, D_MODEL), torch.randn(B, D_TAB), lote["attention_mask"],
            return_attention=True,
        )
    assert pesos.shape == (B, 4, T)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(B, 4), atol=1e-5)
    # Los tokens enmascarados no deben recibir peso
    assert pesos[0, :, 8:].max() < 1e-6
    assert pesos[1, :, 5:].max() < 1e-6


def test_cross_attention_exige_d_text_divisible_por_heads():
    with pytest.raises(ValueError):
        CrossAttentionFusion(d_text=30, d_tab=16, n_heads=4)


# --- Modelo completo ---

def test_late_fusion_produce_logits(lote):
    modelo = BTRModel(_texto(), _tabular(), FusionConfig(mode="late"))
    assert modelo(**lote).shape == (B,)


def test_cross_fusion_produce_logits(lote):
    modelo = BTRModel(_texto(), _tabular(), FusionConfig(mode="cross"))
    assert modelo(**lote).shape == (B,)


def test_modo_cross_fuerza_pooling_none():
    """El cross-attention necesita la secuencia sin colapsar."""
    encoder = _texto(pooling="mean")
    BTRModel(encoder, _tabular(), FusionConfig(mode="cross"))
    assert encoder.config.pooling_mode == "none"


def test_baseline_solo_texto(lote):
    modelo = BTRModel(text_encoder=_texto(), tabular_encoder=None)
    assert modelo.uses_text and not modelo.uses_tabular
    salida = modelo(input_ids=lote["input_ids"], attention_mask=lote["attention_mask"])
    assert salida.shape == (B,)


def test_baseline_solo_tabular(lote):
    modelo = BTRModel(text_encoder=None, tabular_encoder=_tabular())
    assert modelo.uses_tabular and not modelo.uses_text
    assert modelo(x_num=lote["x_num"], x_cat=lote["x_cat"]).shape == (B,)


def test_modelo_sin_ramas_lanza_error():
    with pytest.raises(ValueError):
        BTRModel(None, None)


def test_cross_sin_ambas_ramas_lanza_error():
    with pytest.raises(ValueError):
        FusionConfig(d_text=64, d_tab=0, mode="cross")


def test_gradientes_llegan_a_ambas_ramas(lote):
    modelo = BTRModel(_texto(), _tabular(use_mlp=False), FusionConfig(mode="late"))
    modelo(**lote).sum().backward()
    grad_texto = modelo.text_encoder.embedding.weight.grad
    grad_tabular = modelo.tabular_encoder.embeddings[0].weight.grad
    assert grad_texto is not None and not torch.isnan(grad_texto).any()
    assert grad_tabular is not None and not torch.isnan(grad_tabular).any()
    assert grad_tabular.abs().sum() > 0


def test_gradientes_con_tabular_mlp(lote):
    modelo = BTRModel(_texto(), _tabular(use_mlp=True), FusionConfig(mode="late"))
    modelo(**lote).sum().backward()
    grad_mlp = modelo.tabular_encoder.mlp[0].weight.grad
    assert grad_mlp is not None and not torch.isnan(grad_mlp).any()
    assert grad_mlp.abs().sum() > 0


def test_la_mascara_de_padding_cambia_el_resultado(lote):
    """Si la máscara no tuviera efecto, los tokens [PAD] contaminarían la representación."""
    modelo = BTRModel(_texto(), None)
    modelo.eval()
    with torch.no_grad():
        con_mascara = modelo(input_ids=lote["input_ids"], attention_mask=lote["attention_mask"])
        sin_mascara = modelo(input_ids=lote["input_ids"], attention_mask=torch.ones(B, T, dtype=torch.long))
    assert not torch.allclose(con_mascara[:2], sin_mascara[:2])


def test_las_filas_sin_padding_no_se_ven_afectadas(lote):
    """Las muestras 2..5 no tienen padding: enmascarar o no debe darles el mismo resultado."""
    modelo = BTRModel(_texto(), None)
    modelo.eval()
    with torch.no_grad():
        a = modelo(input_ids=lote["input_ids"], attention_mask=lote["attention_mask"])
        b = modelo(input_ids=lote["input_ids"], attention_mask=torch.ones(B, T, dtype=torch.long))
    assert torch.allclose(a[2:], b[2:], atol=1e-5)


def test_desglose_de_parametros_suma_el_total(lote):
    modelo = BTRModel(_texto(), _tabular(), FusionConfig(mode="cross"))
    d = modelo.param_breakdown()
    assert d["texto"] + d["tabular"] + d["cross_attention"] + d["cabeza"] == d["total"]


def test_tabular_y_modelo_producen_la_forma_correcta(lote):
    tabular = _tabular(use_mlp=False)
    assert tabular(lote["x_num"], lote["x_cat"]).shape == (B, tabular.output_dim)
    modelo = BTRModel(_texto(), tabular)
    assert modelo(**lote).shape == (B,)

    tabular_mlp = _tabular(use_mlp=True)
    assert tabular_mlp(lote["x_num"], lote["x_cat"]).shape == (B, D_TAB)


def test_muestras_independientes_en_el_batch(lote):
    """Modificar una fila del lote no debe alterar la predicción de las demás."""
    modelo = BTRModel(_texto(), _tabular())
    modelo.eval()
    with torch.no_grad():
        original = modelo(**lote)
        modificado = dict(lote)
        modificado["x_num"] = lote["x_num"].clone()
        modificado["x_num"][0] += 10.0
        nuevo = modelo(**modificado)
    assert torch.allclose(original[1:], nuevo[1:], atol=1e-5)
