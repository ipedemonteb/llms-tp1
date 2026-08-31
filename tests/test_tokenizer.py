"""Tests unitarios del tokenizador ByteLevelBPETokenizer y su módulo de benchmark."""

from pathlib import Path
import pytest
import torch

from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.tokenizer.benchmark import (
    cargar_textos,
    plot_vocab_size_sweep,
    plot_sequence_length_distribution,
    plot_subword_fertility,
    plot_min_frequency_impact,
    plot_token_rank_frequency_zipf,
    plot_tradeoff_memory_vs_computation,
)
from .conftest import requiere_datos, requiere_tokenizer


def test_tokenizer_entrenamiento_desde_iterador():
    textos = [
        "Galletitas de chocolate con chips | Deliciosas galletitas dulces | Harina, Azúcar, Cacao",
        "Leche descremada pasteurizada | Leche fluida reducida en grasa | Leche, Vitamina A, Vitamina D",
        "Aceite de girasol puro | Aceite vegetal para cocinar | Aceite de girasol",
    ]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(textos, vocab_size=300, min_frequency=1, show_progress=False)
    assert tok.vocab_size > 256

    res = tok.encode("Galletitas dulces", add_special_tokens=True)
    assert res.ids[0] == tok.cls_token_id
    assert res.ids[-1] == tok.sep_token_id
    assert len(res.ids) >= 3


def test_tokenizer_batch_encode_retorna_tensores_pytorch():
    textos = ["Producto uno", "Producto dos con mas palabras"]
    tok = ByteLevelBPETokenizer().train_from_iterator(textos, vocab_size=300, min_frequency=1, show_progress=False)
    salida = tok.encode_batch(textos, max_length=16, return_tensors="pt")
    assert "input_ids" in salida and "attention_mask" in salida
    assert salida["input_ids"].shape == (2, 16)
    assert salida["attention_mask"].shape == (2, 16)
    assert isinstance(salida["input_ids"], torch.Tensor)


@requiere_datos
def test_benchmark_plots_se_generan_correctamente(tmp_path):
    textos = [
        "MarcaA Producto uno | Descripcion corta | Ingrediente1, Ingrediente2",
        "MarcaB Producto dos | Descripcion algo mas larga | Ingrediente3",
        "MarcaC Producto tres | Otra descripcion | Ingrediente1, Ingrediente4",
    ]
    p1 = plot_vocab_size_sweep(textos, textos, vocab_sizes=[270, 300], output_dir=tmp_path)
    p2 = plot_sequence_length_distribution(textos, textos, output_dir=tmp_path)
    p3 = plot_subword_fertility(textos, textos, vocab_sizes=[270, 300], output_dir=tmp_path)
    p4 = plot_min_frequency_impact(textos, textos, min_frequencies=[1, 2], output_dir=tmp_path)
    p5 = plot_token_rank_frequency_zipf(textos, textos, output_dir=tmp_path)
    p6 = plot_tradeoff_memory_vs_computation(output_dir=tmp_path)

    for p in (p1, p2, p3, p4, p5, p6):
        assert p.exists()
        assert p.stat().st_size > 1000
