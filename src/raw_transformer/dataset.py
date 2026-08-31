"""Dataset y DataLoaders de PyTorch para el Transformer "pelado".

Implementa la Fase 4 del plan (`src/raw_transformer/PLAN.md`).

Toma los CSV serializados por `serialize.py`, los tokeniza con el BPE entrenado por
`train_tokenizer.py` y los expone como `torch.utils.data.Dataset`.

**Tokenización anticipada (eager):** el corpus completo se tokeniza una única vez en el
constructor, en lugar de hacerlo fila por fila dentro de `__getitem__`. Con 10.000 filas
el costo de memoria es despreciable (~10.000 × 256 × 8 bytes ≈ 20 MB por tensor) y evita
repetir el mismo trabajo de tokenización en cada época. En un corpus grande convendría
lo contrario.

**Padding a longitud fija (256)** en lugar de padding dinámico por batch: garantiza que
todas las secuencias entren completas (ver decisión D4) y que las shapes sean constantes,
lo que simplifica la comparación contra el `hybrid_transformer`.

**Compatibilidad con el Trainer común (decisión D6):** los batches usan las mismas claves
que `src.training.dataset.SupermarketDataset` (`input_ids`, `attention_mask`, `labels`),
de modo que `src.training.trainer.Trainer` los consume sin ninguna adaptación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import ByteLevelBPETokenizer

DEFAULT_TOKENIZER_PATH = "resources/tokenizer/bpe_tokenizer_raw.json"
DEFAULT_DATA_DIR = "resources/datasets"
DEFAULT_PREFIX = "raw"


def split_files(prefix: str = DEFAULT_PREFIX) -> Dict[str, str]:
    """Nombres de archivo de los tres splits para un prefijo de serialización dado.

    El prefijo distingue los presets de campos de `serialize.py`: 'raw' para el preset
    `all` y, por ejemplo, 'raw_po' para `product_only` (la comparación controlada contra
    el hybrid, sin contexto de búsqueda).
    """
    return {name: f"{prefix}_{name}.csv" for name in ("train", "val", "test")}


class RawSerializedDataset(Dataset):
    """Dataset de filas serializadas: `(input_ids, attention_mask, labels)`."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: ByteLevelBPETokenizer,
        max_length: int = 256,
        text_column: str = "text",
        label_column: str = "bought",
    ) -> None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Corré primero: "
                f"python -m src.raw_transformer.serialize"
            )

        df = pd.read_csv(path)
        for col in (text_column, label_column):
            if col not in df.columns:
                raise KeyError(f"Columna '{col}' no encontrada en {path}.")

        self.path = path
        self.texts = df[text_column].astype(str).tolist()
        self.labels = torch.tensor(df[label_column].astype(float).values, dtype=torch.float32)

        # Tokenización anticipada de todo el split, con padding a longitud fija.
        encoded = tokenizer.encode_batch(
            self.texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids: torch.Tensor = encoded["input_ids"]
        self.attention_mask: torch.Tensor = encoded["attention_mask"]

        # Cuántas secuencias fueron truncadas: con max_length=256 debería ser 0.
        self.n_truncated = int((self.attention_mask.sum(dim=1) == max_length).sum().item())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    @property
    def positive_rate(self) -> float:
        """Proporción de la clase positiva — el BTR empírico del split."""
        return float(self.labels.mean().item())

    def pos_weight(self) -> torch.Tensor:
        """Peso de la clase positiva para `BCEWithLogitsLoss` = n_negativos / n_positivos.

        Compensa el desbalance del dataset (BTR ≈ 13%): sin este peso, la loss está dominada
        por los negativos y el modelo tiende a predecir probabilidades bajas para todo.
        Debe calcularse **solo sobre train**.
        """
        n_pos = float(self.labels.sum().item())
        n_neg = float(len(self.labels) - n_pos)
        if n_pos == 0:
            raise ValueError("El split no tiene ejemplos positivos; pos_weight no está definido.")
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def load_tokenizer(
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    max_length: int = 256,
) -> ByteLevelBPETokenizer:
    """Carga el BPE entrenado sobre el corpus serializado."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el tokenizador en {path}. Corré primero: "
            f"python -m src.raw_transformer.train_tokenizer"
        )
    return ByteLevelBPETokenizer.from_file(path, max_length=max_length)


def build_datasets(
    data_dir: str = DEFAULT_DATA_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    max_length: int = 256,
    prefix: str = DEFAULT_PREFIX,
) -> Dict[str, RawSerializedDataset]:
    """Construye los tres datasets (train/val/test) con un tokenizador compartido."""
    tokenizer = load_tokenizer(tokenizer_path, max_length=max_length)
    base = Path(data_dir)
    return {
        name: RawSerializedDataset(base / filename, tokenizer, max_length=max_length)
        for name, filename in split_files(prefix).items()
    }


def build_dataloaders(
    data_dir: str = DEFAULT_DATA_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: Optional[int] = 42,
    prefix: str = DEFAULT_PREFIX,
) -> Tuple[Dict[str, DataLoader], Dict[str, RawSerializedDataset]]:
    """Construye los DataLoaders de PyTorch.

    Solo `train` se mezcla (`shuffle=True`); val y test mantienen el orden cronológico
    para que las métricas sean reproducibles entre corridas.
    """
    datasets = build_datasets(data_dir, tokenizer_path, max_length, prefix=prefix)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            generator=generator if name == "train" else None,
            drop_last=False,
        )
        for name, ds in datasets.items()
    }
    return loaders, datasets


def run_checkpoint() -> None:
    """Checkpoint de la Fase 4: verifica shapes, truncamiento, balance y un batch real."""
    print("=" * 78)
    print("🧪 CHECKPOINT FASE 4 — Dataset y DataLoader")
    print("=" * 78)

    loaders, datasets = build_dataloaders(batch_size=32)

    print(f"\n{'split':>6} | {'filas':>6} | {'BTR':>7} | {'truncadas':>10} | {'batches':>8}")
    print(f"{'-'*6} | {'-'*6} | {'-'*7} | {'-'*10} | {'-'*8}")
    for name, ds in datasets.items():
        print(f"{name:>6} | {len(ds):>6,} | {ds.positive_rate*100:>6.2f}% | "
              f"{ds.n_truncated:>10,} | {len(loaders[name]):>8}")

    train_ds = datasets["train"]
    print(f"\n⚖️  pos_weight (calculado solo sobre train): {train_ds.pos_weight().item():.3f}")
    print(f"   -> cada ejemplo positivo pesa {train_ds.pos_weight().item():.1f}x un negativo "
          f"en la BCE, compensando el {train_ds.positive_rate*100:.1f}% de tasa base.")

    # Inspección de un batch real
    batch = next(iter(loaders["train"]))
    print(f"\n📦 Batch de train:")
    print(f"   input_ids:      {tuple(batch['input_ids'].shape)}  dtype={batch['input_ids'].dtype}")
    print(f"   attention_mask: {tuple(batch['attention_mask'].shape)}  dtype={batch['attention_mask'].dtype}")
    print(f"   labels:         {tuple(batch['labels'].shape)}  dtype={batch['labels'].dtype}")
    print(f"   positivos en el batch: {int(batch['labels'].sum().item())}/{len(batch['labels'])}")

    lengths = batch["attention_mask"].sum(dim=1)
    print(f"   tokens reales por fila: min={lengths.min().item()}, "
          f"media={lengths.float().mean().item():.0f}, max={lengths.max().item()}")

    # Verificación de integridad end-to-end: el modelo consume el batch sin romper
    from src.raw_transformer.model import RawTransformerClassifier, RawTransformerConfig

    model = RawTransformerClassifier(RawTransformerConfig(max_seq_len=256))
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    print(f"\n✅ El modelo consume el batch -> logits {tuple(logits.shape)}")
    print(f"   BTR estimado (sin entrenar): media {torch.sigmoid(logits).mean().item():.4f}")

    assert all(ds.n_truncated == 0 for ds in datasets.values()), \
        "Hay secuencias truncadas: max_length es insuficiente."
    print("\n✅ Ninguna secuencia truncada en ningún split.")

    print("\n" + "=" * 78)
    print("🎉 CHECKPOINT FASE 4 SUPERADO")
    print("=" * 78)


if __name__ == "__main__":
    run_checkpoint()
