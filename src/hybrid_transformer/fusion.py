"""Fusión multimodal y cabeza clasificadora para la predicción de BTR.

Integra el vector de la rama de texto (`e_text`, Transformer) con el de la rama tabular
(`e_tab`, MLP) y produce el logit de compra. Soporta dos estrategias de fusión conmutables
para el estudio de ablación:

- `late`  : concatenación estática de ambos vectores tras el pooling (Alternativa 1).
- `cross` : cross-attention donde el vector tabular actúa como Query sobre la secuencia
            completa de tokens, que actúa como Keys y Values (Alternativa 2).

`BTRModel` ensambla ambas ramas y admite desactivar cualquiera de las dos, lo que habilita
los baselines de texto puro y tabular puro sin código adicional.

La salida son **logits sin activar**: la sigmoide vive dentro de `BCEWithLogitsLoss` por
estabilidad numérica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hybrid_transformer.tabular_encoder import TabularEncoder
from src.hybrid_transformer.text_encoder import TextTransformerEncoder


@dataclass
class FusionConfig:
    """Configuración del módulo de fusión y de la cabeza clasificadora.

    Atributos:
        d_text: Dimensión de `e_text`. Usar 0 para desactivar la rama de texto.
        d_tab: Dimensión de `e_tab`. Usar 0 para desactivar la rama tabular.
        mode: 'late' (concatenación) o 'cross' (cross-attention).
        n_heads: Cabezales de cross-attention. Debe dividir a `d_text`.
        hidden_dims: Capas ocultas del MLP clasificador.
        dropout: Dropout en la cabeza y en la atención cruzada.
        activation: 'gelu' o 'relu'.
    """

    d_text: int = 64
    d_tab: int = 32
    mode: str = "late"
    n_heads: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.mode not in {"late", "cross"}:
            raise ValueError(f"mode inválido '{self.mode}'. Opciones: 'late', 'cross'")
        if self.activation.lower() not in {"gelu", "relu"}:
            raise ValueError(f"activation inválida '{self.activation}'. Opciones: 'gelu', 'relu'")
        if self.d_text == 0 and self.d_tab == 0:
            raise ValueError("Al menos una de las dos ramas debe estar activa.")
        if self.mode == "cross":
            if self.d_text == 0 or self.d_tab == 0:
                raise ValueError("El modo 'cross' requiere ambas ramas activas.")
            if self.d_text % self.n_heads != 0:
                raise ValueError(f"d_text ({self.d_text}) debe ser divisible por n_heads ({self.n_heads}).")

    @property
    def fused_dim(self) -> int:
        """Dimensión del vector fusionado que entra al MLP clasificador."""
        return self.d_text + self.d_tab


class CrossAttentionFusion(nn.Module):
    """Cross-attention multi-cabezal con el vector tabular como Query.

    A diferencia del late fusion, no colapsa la secuencia de texto antes de interactuar con
    lo tabular: el perfil del producto condiciona dinámicamente qué tokens son relevantes.

        Q = e_tab W_Q  (1 posición)      K, V = H_text W_K, H_text W_V  (T posiciones)
        e_cross = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_text: int, d_tab: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if d_text % n_heads != 0:
            raise ValueError(f"d_text ({d_text}) debe ser divisible por n_heads ({n_heads}).")

        self.d_text = d_text
        self.n_heads = n_heads
        self.d_k = d_text // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_proj = nn.Linear(d_tab, d_text)
        self.k_proj = nn.Linear(d_text, d_text)
        self.v_proj = nn.Linear(d_text, d_text)
        self.out_proj = nn.Linear(d_text, d_text)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_text)

    def forward(
        self,
        h_text: torch.Tensor,
        e_tab: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calcula el vector de texto condicionado por las variables tabulares.

        Args:
            h_text: (B, T, d_text) secuencia de estados ocultos del Transformer.
            e_tab: (B, d_tab) vector de la rama tabular.
            attention_mask: (B, T) con 1 en tokens válidos y 0 en [PAD].
            return_attention: Si True, devuelve también los pesos (B, n_heads, T).

        Returns:
            (B, d_text), o la tupla (salida, pesos_de_atencion).
        """
        B, T, _ = h_text.shape

        q = self.q_proj(e_tab).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(h_text).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(h_text).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 1, T)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(self.attn_dropout(attn), v)  # (B, H, 1, d_k)
        context = context.transpose(1, 2).reshape(B, self.d_text)
        salida = self.norm(self.out_proj(context))

        if return_attention:
            return salida, attn.squeeze(2)
        return salida


class ClassifierHead(nn.Module):
    """MLP que proyecta el vector fusionado a un único logit de compra."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims if hidden_dims is not None else [64]
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        capas: List[nn.Module] = []
        dim_previa = input_dim
        for dim in hidden_dims:
            capas.extend([nn.Linear(dim_previa, dim), act, nn.Dropout(dropout)])
            dim_previa = dim
        capas.append(nn.Linear(dim_previa, 1))
        self.mlp = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits de forma (B,)."""
        return self.mlp(x).squeeze(-1)


class BTRModel(nn.Module):
    """Modelo completo de predicción de BTR: ramas de texto y tabular, fusión y clasificador.

    Cualquiera de las dos ramas puede omitirse (pasando None), lo que produce directamente los
    baselines de texto puro y tabular puro sin necesidad de otra clase.
    """

    def __init__(
        self,
        text_encoder: Optional[TextTransformerEncoder] = None,
        tabular_encoder: Optional[TabularEncoder] = None,
        fusion_config: Optional[FusionConfig] = None,
    ) -> None:
        super().__init__()
        if text_encoder is None and tabular_encoder is None:
            raise ValueError("Se requiere al menos una de las dos ramas.")

        self.text_encoder = text_encoder
        self.tabular_encoder = tabular_encoder

        d_text = text_encoder.config.d_model if text_encoder is not None else 0
        d_tab = tabular_encoder.config.d_tab if tabular_encoder is not None else 0

        if fusion_config is None:
            fusion_config = FusionConfig(d_text=d_text, d_tab=d_tab)
        else:
            fusion_config.d_text, fusion_config.d_tab = d_text, d_tab
            fusion_config.__post_init__()
        self.config = fusion_config

        # El modo 'cross' necesita la secuencia sin colapsar
        if text_encoder is not None:
            esperado = "none" if fusion_config.mode == "cross" else text_encoder.config.pooling_mode
            if fusion_config.mode == "cross" and text_encoder.config.pooling_mode != "none":
                text_encoder.config.pooling_mode = "none"
            self._pooling_esperado = esperado

        self.cross_attention = (
            CrossAttentionFusion(d_text, d_tab, fusion_config.n_heads, fusion_config.dropout)
            if fusion_config.mode == "cross"
            else None
        )

        self.head = ClassifierHead(
            input_dim=fusion_config.fused_dim,
            hidden_dims=fusion_config.hidden_dims,
            dropout=fusion_config.dropout,
            activation=fusion_config.activation,
        )

    @property
    def uses_text(self) -> bool:
        return self.text_encoder is not None

    @property
    def uses_tabular(self) -> bool:
        return self.tabular_encoder is not None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Devuelve los logits (B,) de probabilidad de compra."""
        partes: List[torch.Tensor] = []
        pesos_atencion = None

        e_tab = self.tabular_encoder(x_num, x_cat) if self.uses_tabular else None

        if self.uses_text:
            salida_texto = self.text_encoder(input_ids, attention_mask=attention_mask)
            if self.config.mode == "cross":
                if return_attention:
                    e_text, pesos_atencion = self.cross_attention(
                        salida_texto, e_tab, attention_mask, return_attention=True
                    )
                else:
                    e_text = self.cross_attention(salida_texto, e_tab, attention_mask)
            else:
                e_text = salida_texto
            partes.append(e_text)

        if e_tab is not None:
            partes.append(e_tab)

        logits = self.head(torch.cat(partes, dim=1))

        if return_attention:
            return logits, pesos_atencion
        return logits

    def get_num_params(self) -> int:
        """Cantidad total de parámetros entrenables del sistema completo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        """Desglose de parámetros por componente, útil para el informe."""
        def contar(modulo: Optional[nn.Module]) -> int:
            if modulo is None:
                return 0
            return sum(p.numel() for p in modulo.parameters() if p.requires_grad)

        return {
            "texto": contar(self.text_encoder),
            "tabular": contar(self.tabular_encoder),
            "cross_attention": contar(self.cross_attention),
            "cabeza": contar(self.head),
            "total": self.get_num_params(),
        }
