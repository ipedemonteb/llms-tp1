"""Módulo de entrenamiento, evaluación y métricas del sistema de predicción de BTR."""

from src.training.dataset import SupermarketDataset, build_dataloaders
from src.training.metrics import (
    binary_cross_entropy,
    compute_metrics,
    format_metrics,
    lift_over_baseline,
    sigmoid,
)
from src.training.trainer import Trainer, TrainerConfig, get_device, set_seed

__all__ = [
    "SupermarketDataset",
    "build_dataloaders",
    "compute_metrics",
    "binary_cross_entropy",
    "format_metrics",
    "lift_over_baseline",
    "sigmoid",
    "Trainer",
    "TrainerConfig",
    "get_device",
    "set_seed",
]
