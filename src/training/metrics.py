"""Métricas de evaluación para la predicción de BTR.

La consigna pide evaluar con PR-AUC y ROC-AUC, más el monitoreo de la BCE para detectar
overfitting/underfitting. No se define umbral de decisión: las tres métricas operan sobre las
probabilidades y evalúan la calidad del ranking, no de una clasificación binaria.

Por qué PR-AUC es la métrica primaria: con un BTR global de ~13%, un clasificador que predice
siempre la clase negativa alcanza 87% de accuracy sin ser útil. La ROC-AUC también resulta
optimista con clases desbalanceadas porque premia el manejo de los negativos, abundantes. La
PR-AUC se concentra en la clase positiva y su línea base es directamente la prevalencia, lo que
la vuelve interpretable: un valor de 0,13 equivale a no haber aprendido nada.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Normaliza tensores, listas o arrays a un ndarray 1-D de float64."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def sigmoid(logits: ArrayLike) -> np.ndarray:
    """Sigmoide numéricamente estable para convertir logits en probabilidades."""
    z = _to_numpy(logits)
    salida = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    salida[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    salida[neg] = exp_z / (1.0 + exp_z)
    return salida


def binary_cross_entropy(y_true: ArrayLike, logits: ArrayLike) -> float:
    """BCE media calculada desde logits, sin materializar probabilidades saturadas."""
    y = _to_numpy(y_true)
    z = _to_numpy(logits)
    # log(1+exp(-|z|)) + max(z,0) - z*y  es la forma estable de la BCE con logits
    return float(np.mean(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - z * y))


def compute_metrics(
    y_true: ArrayLike,
    logits: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """Calcula PR-AUC, ROC-AUC, BCE y la línea base de PR-AUC.

    Las AUC quedan en NaN si el split tiene una sola clase presente, situación en la que no
    están definidas. Devolver NaN en lugar de lanzar excepción permite que el entrenamiento
    continúe y que el problema quede visible en el log.

    Args:
        y_true: Etiquetas binarias (0/1).
        logits: Salida cruda del modelo, sin sigmoide.
        prefix: Prefijo opcional para las claves (ej. 'val_').

    Returns:
        Diccionario con 'pr_auc', 'roc_auc', 'bce', 'pr_auc_baseline' y 'positive_rate'.
    """
    y = _to_numpy(y_true)
    z = _to_numpy(logits)
    if y.shape != z.shape:
        raise ValueError(f"Dimensiones incompatibles: y_true {y.shape} vs logits {z.shape}")

    probas = sigmoid(z)
    tasa_positivos = float(np.mean(y)) if y.size else float("nan")
    una_sola_clase = y.size == 0 or len(np.unique(y)) < 2

    metricas = {
        "pr_auc": float("nan") if una_sola_clase else float(average_precision_score(y, probas)),
        "roc_auc": float("nan") if una_sola_clase else float(roc_auc_score(y, probas)),
        "bce": binary_cross_entropy(y, z),
        "pr_auc_baseline": tasa_positivos,
        "positive_rate": tasa_positivos,
    }
    return {f"{prefix}{k}": v for k, v in metricas.items()}


def format_metrics(metricas: Dict[str, float], separator: str = " | ") -> str:
    """Formatea un diccionario de métricas para el log de consola."""
    orden = ["bce", "pr_auc", "roc_auc"]
    partes = []
    for clave in orden:
        coincidencias = [k for k in metricas if k.endswith(clave)]
        for k in coincidencias:
            partes.append(f"{k}={metricas[k]:.4f}")
    return separator.join(partes)


def lift_over_baseline(metricas: Dict[str, float], prefix: str = "") -> Optional[float]:
    """Cociente entre la PR-AUC obtenida y la línea base (prevalencia de la clase positiva).

    Un valor de 1.0 significa que el modelo no aporta nada sobre predecir al azar respetando
    la prevalencia. Valores por debajo de 1.0 indican que empeora el ranking aleatorio.
    """
    pr = metricas.get(f"{prefix}pr_auc")
    base = metricas.get(f"{prefix}pr_auc_baseline")
    if pr is None or base is None or not base or np.isnan(pr):
        return None
    return float(pr / base)


def compute_extended_metrics(
    y_true: ArrayLike,
    logits: ArrayLike,
    prefix: str = "",
    top_k_pcts: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict[str, float]:
    """Calcula la batería completa de métricas exigidas y complementarias de negocio.

    Incluye:
    - Métricas de Ranking y Discriminación: PR-AUC, ROC-AUC, BCE Loss, Lift over Baseline.
    - Métricas de Calibración: Brier Score (mean squared error de probabilidades).
    - Punto de Operación Óptimo: Best F1-Score y su Umbral Óptimo correspondiente.
    - Métricas de Negocio / Ranking Top-K%: Precision@K%, Recall@K% y Lift@K%.
    """
    y = _to_numpy(y_true)
    z = _to_numpy(logits)
    base_metrics = compute_metrics(y, z, prefix=prefix)
    probas = sigmoid(z)
    N = len(y)
    total_pos = int(np.sum(y))

    if N == 0 or total_pos == 0 or total_pos == N:
        return base_metrics

    # 1. Brier score (calibración)
    brier = float(brier_score_loss(y, probas))
    base_metrics[f"{prefix}brier_score"] = brier

    # 2. Lift global
    lift = lift_over_baseline(base_metrics, prefix=prefix)
    if lift is not None:
        base_metrics[f"{prefix}lift"] = lift

    # 3. Best F1 y Umbral óptimo sobre la curva PR
    precision, recall, thresholds = precision_recall_curve(y, probas)
    # precision y recall tienen N+1 elementos; thresholds tiene N
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1] + 1e-12,
    )
    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    base_metrics[f"{prefix}best_f1"] = float(f1_scores[best_idx]) if len(f1_scores) > 0 else float("nan")
    base_metrics[f"{prefix}best_threshold"] = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    # 4. Métricas de Negocio Top-K%
    order = np.argsort(-probas)
    y_sorted = y[order]
    prevalencia = float(np.mean(y))

    for pct in top_k_pcts:
        k = max(1, int(round(N * pct)))
        pos_in_top_k = int(np.sum(y_sorted[:k]))
        p_at_k = pos_in_top_k / k
        r_at_k = pos_in_top_k / total_pos
        lift_at_k = p_at_k / (prevalencia + 1e-12)
        pct_tag = int(pct * 100)
        base_metrics[f"{prefix}precision_top{pct_tag}pct"] = float(p_at_k)
        base_metrics[f"{prefix}recall_top{pct_tag}pct"] = float(r_at_k)
        base_metrics[f"{prefix}lift_top{pct_tag}pct"] = float(lift_at_k)

    return base_metrics

