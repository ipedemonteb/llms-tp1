"""Módulo de Transformer "pelado" (raw) sobre serialización total del dataset.

A diferencia de `src.hybrid_transformer`, que separa el problema en una rama textual
(Transformer) y una rama tabular (MLP) para fusionarlas al final, este módulo convierte
TODAS las variables a texto plano y las procesa con un único Transformer, sin ningún
sesgo inductivo sobre la estructura tabular.

Ver `src/raw_transformer/PLAN.md` para el diseño completo del experimento.
"""

from src.raw_transformer.model import (
    ClassificationHead,
    RawTransformerClassifier,
    RawTransformerConfig,
)
from src.raw_transformer.serialize import (
    ALL_FIELDS,
    PRODUCT_ONLY_FIELDS,
    build_raw_dataset,
    serialize_dataframe,
)

__all__ = [
    "ALL_FIELDS",
    "PRODUCT_ONLY_FIELDS",
    "ClassificationHead",
    "RawTransformerClassifier",
    "RawTransformerConfig",
    "build_raw_dataset",
    "serialize_dataframe",
]
