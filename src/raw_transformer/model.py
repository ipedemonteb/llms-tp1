"""Clasificador Transformer "pelado" para la predicción de BTR.

Implementa la Fase 3 del plan (`src/raw_transformer/PLAN.md`).

La arquitectura es deliberadamente mínima: una única rama que procesa la secuencia
serializada completa, sin rama tabular, sin embeddings de categóricas y sin fusión.

    input_ids (B, T)
          │
    TextTransformerEncoder      ← importado de src.hybrid_transformer, sin modificar
          │
    e_seq (B, d_model)          ← pooling ('mean' por defecto, igual que el hybrid)
          │
    Cabeza de clasificación     ← LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear
          │
    logit (B,)

**El encoder se reutiliza tal cual, no se duplica.** La única diferencia arquitectónica
frente al `hybrid_transformer` es lo que NO está: no hay `nn.Embedding` de categorías,
no hay `BatchNorm` de numéricas y no hay módulo de fusión. Toda la información entra
por `input_ids`.

El forward devuelve **logits crudos**, no probabilidades. Es intencional: permite usar
`nn.BCEWithLogitsLoss`, que es numéricamente estable (aplica el log-sum-exp trick en vez
de componer `sigmoid` con `log`) y soporta `pos_weight` para compensar el desbalance de
clases del dataset (BTR ≈ 13%). Para obtener el BTR estimado se aplica `sigmoid` al logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder


@dataclass
class RawTransformerConfig:
    """Hiperparámetros del clasificador pelado.

    Los valores por defecto replican los del `hybrid_transformer` para que la comparación
    sea justa (mismo d_model, n_heads, num_layers, dropout, vocab_size y pooling). La única
    excepción es `max_seq_len`, que sube de 128 a 256 porque las secuencias serializadas
    miden ~198 tokens y con 128 no entraría ninguna completa (decisión D4 del plan).

    Atributos:
        vocab_size: Tamaño del vocabulario del BPE entrenado sobre el corpus serializado.
        max_seq_len: Longitud máxima de secuencia. 256 cubre el 100% del corpus.
        d_model: Dimensión de embeddings y estados ocultos (< 100 según la consigna).
        n_heads: Cabezales de auto-atención.
        d_ff: Dimensión interna de la red feed-forward.
        num_layers: Cantidad de bloques encoder apilados.
        dropout: Dropout del encoder.
        activation: Activación de la FFN ('gelu' o 'relu').
        norm_first: Pre-LN (True, más estable) o Post-LN (False, Vaswani original).
        pos_encoding_type: 'sinusoidal', 'learned' o 'none' (para ablación).
        pooling_mode: Cómo se colapsa la secuencia a un vector ('mean', 'cls' o 'max').
        pad_token_id: ID del token [PAD].
        head_hidden_dim: Ancho de la capa oculta de la cabeza de clasificación.
        head_dropout: Dropout de la cabeza de clasificación.
    """

    vocab_size: int = 2048
    max_seq_len: int = 256
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True
    pos_encoding_type: str = "sinusoidal"
    pooling_mode: str = "mean"
    pad_token_id: int = 0
    head_hidden_dim: int = 64
    head_dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.pooling_mode == "none":
            raise ValueError(
                "pooling_mode='none' devuelve la secuencia sin colapsar y no sirve para "
                "clasificación. Usá 'mean', 'cls' o 'max'."
            )

    def to_encoder_config(self) -> TextTransformerConfig:
        """Proyecta esta configuración sobre la del encoder reutilizado."""
        return TextTransformerConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            pos_encoding_type=self.pos_encoding_type,
            pooling_mode=self.pooling_mode,
            pad_token_id=self.pad_token_id,
        )


class ClassificationHead(nn.Module):
    """MLP de dos capas que mapea el vector de secuencia a un logit escalar.

    La `LayerNorm` inicial estabiliza la escala del vector que llega del pooling, que puede
    variar bastante entre batches al principio del entrenamiento.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_model) -> (B,) logits."""
        return self.net(x).squeeze(-1)


class RawTransformerClassifier(nn.Module):
    """Transformer pelado: encoder sobre la fila serializada + cabeza de clasificación."""

    def __init__(self, config: Optional[RawTransformerConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = RawTransformerConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.encoder = TextTransformerEncoder(config.to_encoder_config())
        self.head = ClassificationHead(
            d_model=config.d_model,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Paso forward.

        Args:
            input_ids: (B, T) IDs de tokens de la fila serializada.
            attention_mask: (B, T) con 1 en tokens reales y 0 en [PAD].
            return_attention: Si True, devuelve además los mapas de atención por capa,
                útiles para visualizar qué campos de la fila mira el modelo (XAI).

        Returns:
            Logits de forma (B,). Aplicar `sigmoid` para obtener el BTR estimado.
        """
        if return_attention:
            pooled, attentions = self.encoder(
                input_ids, attention_mask=attention_mask, return_attention=True
            )
            return self.head(pooled), attentions

        pooled = self.encoder(input_ids, attention_mask=attention_mask)
        return self.head(pooled)

    @torch.no_grad()
    def predict_btr(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Devuelve el BTR estimado en [0, 1] — la probabilidad de compra por impresión."""
        self.eval()
        return torch.sigmoid(self(input_ids, attention_mask=attention_mask))

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Parámetros entrenables totales; si `non_embedding`, descuenta la tabla de embeddings."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.encoder.embedding.weight.numel()
        return total


def run_smoke_tests() -> None:
    """Verifica shapes, máscara de padding, rango de salida y flujo de gradientes."""
    print("=" * 75)
    print("🧪 SMOKE TESTS — RawTransformerClassifier")
    print("=" * 75)

    config = RawTransformerConfig()
    model = RawTransformerClassifier(config)

    total = model.get_num_params()
    non_emb = model.get_num_params(non_embedding=True)
    head = sum(p.numel() for p in model.head.parameters())
    print(f"📊 Parámetros totales:      {total:,}")
    print(f"   - Embeddings:            {model.encoder.embedding.weight.numel():,}")
    print(f"   - Encoder (sin emb.):    {non_emb - head:,}")
    print(f"   - Cabeza clasificación:  {head:,}")

    # Batch sintético con padding real al final de cada secuencia
    batch_size, seq_len = 4, 200
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, 150:] = 0
    attention_mask[1, 180:] = 0
    input_ids = input_ids * attention_mask

    # 1. Forward: shapes correctas
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (batch_size,), f"Se esperaba ({batch_size},), se obtuvo {logits.shape}"
    print(f"\n✅ Forward -> logits {tuple(logits.shape)}: {logits.tolist()}")

    # 2. Las probabilidades deben caer dentro de [0, 1]
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilidades fuera de [0, 1]"
    print(f"✅ BTR estimado en [0,1]: {[round(p, 4) for p in probs.tolist()]}")

    # 3. El padding no debe influir: cambiar tokens enmascarados no cambia la salida
    tampered = input_ids.clone()
    tampered[0, 150:] = torch.randint(1, config.vocab_size, (seq_len - 150,))
    with torch.no_grad():
        logits_tampered = model(tampered, attention_mask=attention_mask)
    delta = (logits - logits_tampered).abs().max().item()
    assert delta < 1e-4, f"El padding está afectando la salida (delta={delta})"
    print(f"✅ Máscara de padding efectiva (delta máx = {delta:.2e})")

    # 4. Los gradientes deben llegar hasta la tabla de embeddings
    model.train()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = nn.BCEWithLogitsLoss()(model(input_ids, attention_mask=attention_mask), labels)
    loss.backward()
    emb_grad = model.encoder.embedding.weight.grad
    assert emb_grad is not None and emb_grad.abs().sum() > 0, "No llegan gradientes al embedding"
    print(f"✅ Backward OK — loss={loss.item():.4f}, grad embeddings={emb_grad.abs().sum():.4f}")

    # 5. Los mapas de atención se exponen para el análisis de explicabilidad
    model.eval()
    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask=attention_mask, return_attention=True)
    print(f"✅ Atención: {len(attentions)} capas, shape por capa {tuple(attentions[0].shape)}")

    print("\n" + "=" * 75)
    print("🎉 TODOS LOS SMOKE TESTS PASARON")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_tests()
