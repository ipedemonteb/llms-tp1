"""Dataset y DataLoaders para el entrenamiento del modelo de BTR.

Conecta los CSV particionados con los tensores que consumen las dos ramas del modelo:

    transformer_{train,val,test}.csv
        │
        ├── columna `text` (o recompuesta desde `text_fields`) -> tokenizador BPE -> input_ids, attention_mask
        └── columnas numéricas y categóricas -> TabularPreprocessor -> x_num, x_cat

La composición de la secuencia de texto se decide **acá**, en tiempo de entrenamiento, y no al
generar el CSV. Eso permite variar los campos entre experimentos manteniendo idéntica la
partición train/val/test, condición necesaria para que la ablación sea comparable.

Prevención de data leakage: el `TabularPreprocessor` se ajusta exclusivamente con el split de
train (`build_dataloaders` lo hace automáticamente) y val/test solo se transforman.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_extraction.build_transformer_dataset import DEFAULT_TEXT_FIELDS, build_text_sequence
from src.hybrid_transformer.tabular_encoder import TabularPreprocessor
from src.tokenizer.bpe import ByteLevelBPETokenizer


class SupermarketDataset(Dataset):
    """Dataset de productos de supermercado con target binario `bought`.

    La tokenización se resuelve una sola vez en el constructor: con 10.000 filas el costo de
    memoria es despreciable y evita repetir el trabajo en cada época.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[ByteLevelBPETokenizer] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        text_fields: Optional[List[str]] = None,
        max_length: int = 128,
        target_column: str = "bought",
    ) -> None:
        if tokenizer is None and preprocessor is None:
            raise ValueError("Se requiere al menos un tokenizador o un preprocesador tabular.")
        if target_column not in df.columns:
            raise KeyError(f"No se encontró la columna objetivo '{target_column}'.")

        self.df = df.reset_index(drop=True)
        self.text_fields = list(text_fields) if text_fields is not None else list(DEFAULT_TEXT_FIELDS)
        self.max_length = max_length
        self.labels = torch.tensor(self.df[target_column].astype(int).to_numpy(), dtype=torch.float32)

        self.uses_text = tokenizer is not None
        self.uses_tabular = preprocessor is not None

        if self.uses_text:
            textos = build_text_sequence(self.df, text_fields=self.text_fields).tolist()
            codificado = tokenizer.encode_batch(textos, max_length=max_length, return_tensors="pt")
            self.input_ids = codificado["input_ids"]
            self.attention_mask = codificado["attention_mask"]

        if self.uses_tabular:
            self.x_num, self.x_cat = preprocessor.transform(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {"labels": self.labels[idx]}
        if self.uses_text:
            item["input_ids"] = self.input_ids[idx]
            item["attention_mask"] = self.attention_mask[idx]
        if self.uses_tabular:
            item["x_num"] = self.x_num[idx]
            item["x_cat"] = self.x_cat[idx]
        return item

    @property
    def positive_rate(self) -> float:
        """Proporción de la clase positiva (BTR del split)."""
        return float(self.labels.mean())


def build_dataloaders(
    data_dir: Union[str, Path] = "resources/datasets",
    tokenizer_path: Union[str, Path] = "resources/tokenizer/bpe_tokenizer.json",
    preprocessor_path: Optional[Union[str, Path]] = "resources/preprocessor/tabular_preprocessor.json",
    text_fields: Optional[List[str]] = None,
    max_length: int = 128,
    batch_size: int = 64,
    use_text: bool = True,
    use_tabular: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    """Construye los DataLoaders de train, val y test a partir de los CSV particionados.

    El `TabularPreprocessor` se ajusta con train y se guarda en `preprocessor_path`; val y test
    solo se transforman con los parámetros aprendidos.

    Returns:
        Tupla (loaders, artefactos) donde `loaders` tiene las claves 'train', 'val' y 'test',
        y `artefactos` contiene el tokenizador, el preprocesador y metadatos del modelo.
    """
    if not use_text and not use_tabular:
        raise ValueError("Al menos una de las dos ramas debe estar activa.")

    data_dir = Path(data_dir)
    splits = {}
    for nombre in ("train", "val", "test"):
        ruta = data_dir / f"transformer_{nombre}.csv"
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}. Ejecutar primero:\n"
                "  uv run python -m src.data_extraction.build_transformer_dataset"
            )
        splits[nombre] = pd.read_csv(ruta)

    tokenizer = None
    if use_text:
        tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path, max_length=max_length)

    preprocessor = None
    if use_tabular:
        preprocessor = TabularPreprocessor().fit(splits["train"])
        if preprocessor_path is not None:
            preprocessor.save(preprocessor_path)

    datasets = {
        nombre: SupermarketDataset(
            df=df,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            text_fields=text_fields,
            max_length=max_length,
        )
        for nombre, df in splits.items()
    }

    generador = torch.Generator().manual_seed(seed)
    loaders = {
        nombre: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(nombre == "train"),
            num_workers=num_workers,
            generator=generador if nombre == "train" else None,
            # BatchNorm falla con lotes de tamaño 1; se descarta la cola solo en train
            drop_last=(nombre == "train"),
        )
        for nombre, ds in datasets.items()
    }

    artefactos = {
        "tokenizer": tokenizer,
        "preprocessor": preprocessor,
        "vocab_size": tokenizer.vocab_size if tokenizer is not None else 0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer is not None else 0,
        "num_numeric": len(preprocessor.numeric_fields) if preprocessor is not None else 0,
        "num_direct": len(preprocessor.direct_fields) if preprocessor is not None else 0,
        "embedding_cardinalities": (
            [preprocessor.embedding_cardinalities[c] for c in preprocessor.embedding_fields]
            if preprocessor is not None else []
        ),
        "onehot_cardinalities": (
            [preprocessor.onehot_cardinalities[c] for c in preprocessor.onehot_fields]
            if preprocessor is not None else []
        ),
        "cardinalities": (
            [preprocessor.cardinalities[c] for c in preprocessor.categorical_fields]
            if preprocessor is not None else []
        ),
        "positive_rate": {n: ds.positive_rate for n, ds in datasets.items()},
        "sizes": {n: len(ds) for n, ds in datasets.items()},
    }

    return loaders, artefactos
