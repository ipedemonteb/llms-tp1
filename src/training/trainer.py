"""Loop de entrenamiento y evaluación del modelo de BTR.

Implementa el ciclo estándar con `BCEWithLogitsLoss` + `AdamW`, early stopping sobre la PR-AUC
de validación y registro del historial por época para diagnosticar overfitting/underfitting.

Decisiones de diseño:
- **`BCEWithLogitsLoss` en lugar de `BCELoss` + sigmoide**: aplica el truco log-sum-exp y evita
  la pérdida de precisión cuando las probabilidades saturan cerca de 0 o 1.
- **AdamW en vez de Adam**: desacopla el weight decay del paso adaptativo, que es la forma
  correcta de regularizar con optimizadores adaptativos.
- **Early stopping sobre PR-AUC de validación**, no sobre la loss: la métrica que reporta el
  trabajo es la que debe decidir el mejor checkpoint.
- **Selección del mejor modelo por validación**: el test se evalúa una sola vez, al final, con
  los pesos del mejor epoch de validación.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics, format_metrics


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de random, numpy y torch para hacer reproducible el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferencia: Optional[str] = None) -> torch.device:
    """Selecciona el dispositivo disponible: CUDA, MPS o CPU."""
    if preferencia is not None:
        return torch.device(preferencia)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    """Hiperparámetros del entrenamiento.

    Atributos:
        epochs: Máximo de épocas.
        lr: Learning rate de AdamW.
        weight_decay: Regularización L2 desacoplada.
        patience: Épocas sin mejora en la PR-AUC de validación antes de cortar. None desactiva.
        grad_clip: Norma máxima del gradiente. None desactiva el clipping.
        pos_weight: Peso de la clase positiva en la BCE. None la deja sin ponderar.
        device: 'cuda', 'mps', 'cpu' o None para autodetectar.
        seed: Semilla de reproducibilidad.
        verbose: Si True, imprime el progreso por época.
    """

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.01
    patience: Optional[int] = 5
    grad_clip: Optional[float] = 1.0
    pos_weight: Optional[float] = None
    device: Optional[str] = None
    seed: int = 42
    verbose: bool = True


class Trainer:
    """Encapsula el entrenamiento, la evaluación y la selección del mejor checkpoint."""

    def __init__(self, model: nn.Module, config: Optional[TrainerConfig] = None) -> None:
        self.config = config if config is not None else TrainerConfig()
        set_seed(self.config.seed)

        self.device = get_device(self.config.device)
        self.model = model.to(self.device)

        pos_weight = (
            torch.tensor([self.config.pos_weight], device=self.device)
            if self.config.pos_weight is not None else None
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: int = -1
        self.best_val_pr_auc: float = -float("inf")

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mueve un lote al dispositivo y ejecuta el forward. Devuelve (logits, labels)."""
        entradas = {
            clave: batch[clave].to(self.device)
            for clave in ("input_ids", "attention_mask", "x_num", "x_cat")
            if clave in batch
        }
        labels = batch["labels"].to(self.device)
        return self.model(**entradas), labels

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Ejecuta una época de entrenamiento y devuelve las métricas sobre train."""
        self.model.train()
        perdida_total, n_muestras = 0.0, 0
        todos_logits, todas_labels = [], []

        for batch in loader:
            logits, labels = self._forward_batch(batch)
            perdida = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            perdida.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            perdida_total += perdida.item() * labels.size(0)
            n_muestras += labels.size(0)
            todos_logits.append(logits.detach().cpu())
            todas_labels.append(labels.detach().cpu())

        metricas = compute_metrics(torch.cat(todas_labels), torch.cat(todos_logits), prefix="train_")
        metricas["train_loss"] = perdida_total / max(n_muestras, 1)
        return metricas

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val_") -> Dict[str, float]:
        """Evalúa el modelo sobre un loader sin actualizar pesos."""
        self.model.eval()
        perdida_total, n_muestras = 0.0, 0
        todos_logits, todas_labels = [], []

        for batch in loader:
            logits, labels = self._forward_batch(batch)
            perdida = self.criterion(logits, labels)
            perdida_total += perdida.item() * labels.size(0)
            n_muestras += labels.size(0)
            todos_logits.append(logits.cpu())
            todas_labels.append(labels.cpu())

        metricas = compute_metrics(torch.cat(todas_labels), torch.cat(todos_logits), prefix=prefix)
        metricas[f"{prefix}loss"] = perdida_total / max(n_muestras, 1)
        return metricas

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (logits, labels) como arrays de numpy para análisis posterior."""
        self.model.eval()
        logits, labels = [], []
        for batch in loader:
            lote_logits, lote_labels = self._forward_batch(batch)
            logits.append(lote_logits.cpu())
            labels.append(lote_labels.cpu())
        return torch.cat(logits).numpy(), torch.cat(labels).numpy()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, float]]:
        """Entrena con early stopping y restaura el mejor checkpoint según PR-AUC de validación."""
        if self.config.verbose:
            print(f"🖥️  Dispositivo: {self.device} | Parámetros: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"⚙️  lr={self.config.lr} | weight_decay={self.config.weight_decay} | "
                  f"epochs={self.config.epochs} | patience={self.config.patience}\n")

        epocas_sin_mejora = 0
        for epoca in range(1, self.config.epochs + 1):
            inicio = time.time()
            metricas = self.train_epoch(train_loader)
            metricas.update(self.evaluate(val_loader, prefix="val_"))
            metricas["epoch"] = epoca
            metricas["seconds"] = time.time() - inicio
            self.history.append(metricas)

            val_pr_auc = metricas["val_pr_auc"]
            mejoro = val_pr_auc > self.best_val_pr_auc
            if mejoro:
                self.best_val_pr_auc = val_pr_auc
                self.best_epoch = epoca
                self.best_state = copy.deepcopy(self.model.state_dict())
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1

            if self.config.verbose:
                marca = " ⭐" if mejoro else ""
                print(f"  época {epoca:>3}/{self.config.epochs} "
                      f"[{metricas['seconds']:.1f}s]  "
                      f"train: loss={metricas['train_loss']:.4f} pr_auc={metricas['train_pr_auc']:.4f}  |  "
                      f"val: loss={metricas['val_loss']:.4f} pr_auc={val_pr_auc:.4f} "
                      f"roc_auc={metricas['val_roc_auc']:.4f}{marca}")

            if self.config.patience is not None and epocas_sin_mejora >= self.config.patience:
                if self.config.verbose:
                    print(f"\n⏹️  Early stopping en la época {epoca} "
                          f"({self.config.patience} épocas sin mejora).")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            if self.config.verbose:
                print(f"↩️  Restaurado el mejor checkpoint: época {self.best_epoch} "
                      f"(val PR-AUC = {self.best_val_pr_auc:.4f})")

        return self.history

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Guarda los pesos del mejor checkpoint junto con el historial de entrenamiento."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.best_state if self.best_state is not None else self.model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_pr_auc": self.best_val_pr_auc,
                "history": self.history,
            },
            destino,
        )
        return destino

    def history_dataframe(self):
        """Devuelve el historial como DataFrame para graficar las curvas de aprendizaje."""
        import pandas as pd
        return pd.DataFrame(self.history)
