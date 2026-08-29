"""Tests de las métricas de evaluación.

Se validan contra valores conocidos analíticamente: un clasificador perfecto, uno invertido y
uno sin señal tienen AUC exactas que no dependen de la implementación.
"""

import numpy as np
import pytest

from src.training.metrics import (
    binary_cross_entropy,
    compute_metrics,
    lift_over_baseline,
    sigmoid,
)


def test_sigmoid_valores_conocidos():
    assert sigmoid([0.0])[0] == pytest.approx(0.5)
    assert sigmoid([100.0])[0] == pytest.approx(1.0)
    assert sigmoid([-100.0])[0] == pytest.approx(0.0)


def test_sigmoid_no_desborda_en_extremos():
    """La implementación por ramas debe evitar overflow con logits muy grandes."""
    salida = sigmoid([1e4, -1e4])
    assert np.all(np.isfinite(salida))


def test_clasificador_perfecto_alcanza_auc_uno():
    y = np.array([0, 0, 1, 1])
    logits = np.array([-5.0, -4.0, 4.0, 5.0])
    m = compute_metrics(y, logits)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_clasificador_invertido_alcanza_roc_auc_cero():
    y = np.array([0, 0, 1, 1])
    logits = np.array([5.0, 4.0, -4.0, -5.0])
    m = compute_metrics(y, logits)
    assert m["roc_auc"] == pytest.approx(0.0)


def test_sin_señal_da_roc_auc_de_media():
    """Con todos los logits iguales no hay ranking posible: la ROC-AUC debe ser 0.5."""
    y = np.array([0, 1, 0, 1, 0, 1])
    m = compute_metrics(y, np.zeros(6))
    assert m["roc_auc"] == pytest.approx(0.5)


def test_pr_auc_baseline_es_la_prevalencia():
    y = np.array([0] * 87 + [1] * 13)
    m = compute_metrics(y, np.zeros(100))
    assert m["pr_auc_baseline"] == pytest.approx(0.13)
    # Sin señal la PR-AUC converge a la prevalencia
    assert m["pr_auc"] == pytest.approx(0.13, abs=0.01)


def test_bce_coincide_con_torch():
    """La BCE calculada desde logits debe igualar a BCEWithLogitsLoss de PyTorch."""
    import torch

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50).astype(float)
    z = rng.normal(0, 3, 50)
    esperado = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(z), torch.tensor(y)
    ).item()
    assert binary_cross_entropy(y, z) == pytest.approx(esperado, rel=1e-9)


def test_bce_es_menor_cuando_las_predicciones_son_correctas():
    y = np.array([0.0, 1.0])
    assert binary_cross_entropy(y, np.array([-5.0, 5.0])) < binary_cross_entropy(y, np.array([5.0, -5.0]))


def test_una_sola_clase_devuelve_nan_sin_lanzar():
    """Con una sola clase presente las AUC no están definidas; deben ser NaN, no una excepción."""
    m = compute_metrics(np.zeros(10), np.random.randn(10))
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])
    assert np.isfinite(m["bce"])


def test_dimensiones_incompatibles_lanzan_error():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_prefijo_se_aplica_a_todas_las_claves():
    m = compute_metrics(np.array([0, 1]), np.array([-1.0, 1.0]), prefix="val_")
    assert all(k.startswith("val_") for k in m)


def test_lift_sobre_baseline():
    m = {"test_pr_auc": 0.26, "test_pr_auc_baseline": 0.13}
    assert lift_over_baseline(m, prefix="test_") == pytest.approx(2.0)
