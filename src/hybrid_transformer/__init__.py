"""Módulo de arquitecturas Transformer e híbridas para predicción de BTR."""

from src.hybrid_transformer.text_encoder import (
    MultiHeadSelfAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    TextTransformerConfig,
    TextTransformerEncoder,
    TransformerEncoderBlock,
)
from src.hybrid_transformer.fusion import (
    BTRModel,
    ClassifierHead,
    CrossAttentionFusion,
    FusionConfig,
)
from src.hybrid_transformer.tabular_encoder import (
    DEFAULT_CATEGORICAL_FIELDS,
    DEFAULT_LOG1P_FIELDS,
    DEFAULT_NUMERIC_FIELDS,
    TabularEncoder,
    TabularEncoderConfig,
    TabularPreprocessor,
)

__all__ = [
    # Rama de texto
    "MultiHeadSelfAttention",
    "PositionwiseFeedForward",
    "PositionalEncoding",
    "TextTransformerConfig",
    "TextTransformerEncoder",
    "TransformerEncoderBlock",
    # Rama tabular
    "TabularEncoder",
    "TabularEncoderConfig",
    "TabularPreprocessor",
    "DEFAULT_NUMERIC_FIELDS",
    "DEFAULT_CATEGORICAL_FIELDS",
    "DEFAULT_LOG1P_FIELDS",
    # Fusión y clasificador
    "BTRModel",
    "ClassifierHead",
    "CrossAttentionFusion",
    "FusionConfig",
]
