"""Implementación modular del Transformer Encoder para la rama de texto.

Este módulo define la arquitectura del Transformer Encoder (Encoder-Only) diseñado para
procesar las secuencias tokenizadas de productos (título, tags, descripción, ingredientes)
y generar representaciones densas (embeddings contextuales) listas para fusionarse con
variables tabulares o pasar por un clasificador.

Componentes incluidos:
- `PositionalEncoding`: Codificación posicional sinusoidal (Vaswani et al., 2017),
  aprendible (learned embeddings) o desactivada (none para estudio de ablación).
- `MultiHeadSelfAttention`: Mecanismo de auto-atención multi-cabezal con máscara de padding
  y soporte para retornar mapas de atención para explicabilidad (XAI).
- `PositionwiseFeedForward`: Red feed-forward de dos capas lineales con activación
  configurable (GELU o ReLU) y regularización por Dropout.
- `TransformerEncoderBlock`: Bloque encoder individual con soporte para Pre-LN (moderno/estable)
  y Post-LN (Vaswani original) y conexiones residuales.
- `TextTransformerEncoder`: Modelo completo que integra nn.Embedding, Positional Encoding,
  stack de L bloques encoder y estrategias configurables de Pooling ('mean', 'cls', 'max', 'none').
- `TextTransformerConfig`: Dataclass para configurar e hiperparametrizar la arquitectura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextTransformerConfig:
    """Configuración de hiperparámetros para el Transformer Encoder de Texto.

    Atributos:
        vocab_size: Tamaño del vocabulario de tokens (coherente con el BPE Tokenizer).
        max_seq_len: Longitud máxima de secuencia en tokens (padding / truncamiento).
        d_model: Dimensión del espacio de embedding y estados ocultos (< 100 según consigna).
        n_heads: Cantidad de cabezales de atención (debe dividir a d_model).
        d_ff: Dimensión de la capa intermedia en la red Feed-Forward (típicamente 4 * d_model).
        num_layers: Cantidad de capas / bloques TransformerEncoderBlock apilados.
        dropout: Probabilidad de dropout aplicada en embeddings, atención y FFN.
        activation: Función de activación no lineal en FFN ('gelu' o 'relu').
        norm_first: Si True, usa Pre-LayerNorm (mayor estabilidad de gradientes); si False, Post-LN.
        pos_encoding_type: Tipo de codificación posicional ('sinusoidal', 'learned', 'none').
        pooling_mode: Estrategia de reducción temporal para obtener el vector e_text
            ('mean', 'cls', 'max', 'none'). 'none' devuelve la secuencia completa (B, T, d_model)
            para arquitecturas con Cross-Attention.
        pad_token_id: ID del token especial [PAD] (usado en nn.Embedding para gradiente cero).
        scale_embedding: Si True, multiplica los embeddings por sqrt(d_model) antes de sumar posiciones.
    """

    vocab_size: int = 2048
    max_seq_len: int = 128
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
    scale_embedding: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible exactamente por n_heads ({self.n_heads})."
            )
        valid_pos = {"sinusoidal", "learned", "none"}
        if self.pos_encoding_type not in valid_pos:
            raise ValueError(f"pos_encoding_type inválido '{self.pos_encoding_type}'. Opciones: {valid_pos}")

        valid_pooling = {"mean", "cls", "max", "none"}
        if self.pooling_mode not in valid_pooling:
            raise ValueError(f"pooling_mode inválido '{self.pooling_mode}'. Opciones: {valid_pooling}")

        valid_acts = {"gelu", "relu"}
        if self.activation.lower() not in valid_acts:
            raise ValueError(f"activation inválida '{self.activation}'. Opciones: {valid_acts}")


class PositionalEncoding(nn.Module):
    """Módulo de codificación posicional configurable.

    Soporta:
    - 'sinusoidal': Funciones sinusoidales fijas (Vaswani et al., 2017) almacenadas en un buffer no entrenable.
    - 'learned': Matriz de embeddings entrenable nn.Embedding(max_seq_len, d_model).
    - 'none': Sin información de posición (identidad), útil para estudios de ablación.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512,
        encoding_type: str = "sinusoidal",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.encoding_type = encoding_type.lower()
        self.dropout = nn.Dropout(p=dropout)

        if self.encoding_type == "sinusoidal":
            # Calcular matriz de posiciones sinusoidales fija: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
            self.register_buffer("pe", pe, persistent=False)
            self.learned_pe = None
        elif self.encoding_type == "learned":
            self.learned_pe = nn.Embedding(max_seq_len, d_model)
            self.register_buffer("pe", None, persistent=False)
        elif self.encoding_type == "none":
            self.learned_pe = None
            self.register_buffer("pe", None, persistent=False)
        else:
            raise ValueError(f"Tipo de codificación posicional desconocido: {encoding_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Suma la codificación posicional al tensor de entrada x de dimensión (B, T, d_model)."""
        seq_len = x.size(1)

        if self.encoding_type == "sinusoidal":
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Longitud de secuencia ({seq_len}) excede max_seq_len ({self.max_seq_len})."
                )
            x = x + self.pe[:, :seq_len, :]
        elif self.encoding_type == "learned":
            positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            x = x + self.learned_pe(positions)
        elif self.encoding_type == "none":
            pass  # Ablación: sin información posicional

        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Mecanismo de Auto-Atención Multi-Cabezal (Multi-Head Self-Attention).

    Permite a cada posición de la secuencia atender conjuntamente a la información de
    diferentes subespacios de representación en diferentes posiciones.

    Fórmula:
        Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) debe ser múltiplo de n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Proyecciones lineales para Query, Key, Value y Output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(p=dropout)
        self.resid_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calcula el Multi-Head Self-Attention.

        Args:
            x: Tensor de entrada de forma (batch_size, seq_len, d_model).
            attention_mask: Tensor binario de máscara (batch_size, seq_len) donde 1 indica token válido
                y 0 indica token de padding ([PAD]).
            return_attention: Si True, retorna también la matriz de pesos de atención (B, H, T, T).

        Returns:
            Tensor de salida (batch_size, seq_len, d_model) o tupla (salida, attention_weights).
        """
        B, T, _ = x.shape

        # 1. Proyecciones lineales y reorganización a múltiples cabezales: (B, T, d_model) -> (B, H, T, d_k)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Aplicar máscara de padding si se provee
        if attention_mask is not None:
            # Expandir máscara: (B, T) -> (B, 1, 1, T) para broadcast sobre cabezales y consultas
            # Los tokens con mask == 0 reciben un valor fuertemente negativo (-1e9) para que softmax -> 0
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)

        # 4. Softmax sobre la última dimensión (claves atendidas)
        attn_weights = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        # 5. Ponderar los Valores V: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        context = torch.matmul(attn_probs, v)

        # 6. Concatenar cabezales y proyectar a la salida: (B, H, T, d_k) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.resid_dropout(self.out_proj(context))

        if return_attention:
            return output, attn_weights
        return output


class PositionwiseFeedForward(nn.Module):
    """Red Feed-Forward de dos capas lineales aplicada posición por posición.

    FFN(x) = Activation(x W_1 + b_1) W_2 + b_2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        act = activation.lower()
        if act == "gelu":
            self.activation = nn.GELU()
        elif act == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activación '{activation}' no soportada. Usar 'gelu' o 'relu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la proyección lineal con activación y dropout."""
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class TransformerEncoderBlock(nn.Module):
    """Bloque individual del Transformer Encoder.

    Integra:
    - Subcapa 1: Multi-Head Self-Attention con conexión residual y LayerNorm.
    - Subcapa 2: Position-wise Feed-Forward Network con conexión residual y LayerNorm.
    - Soporte para 'Pre-LN' (norm_first=True, estándar moderno) y 'Post-LN' (norm_first=False, Vaswani 2017).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Ejecuta el bloque encoder respetando la topología Pre-LN o Post-LN."""
        attn_weights = None

        if self.norm_first:
            # Pre-LayerNorm: LayerNorm -> Subcapa -> Residual Addition (Mayor estabilidad)
            norm_x = self.norm1(x)
            if return_attention:
                attn_out, attn_weights = self.self_attn(
                    norm_x, attention_mask=attention_mask, return_attention=True
                )
            else:
                attn_out = self.self_attn(
                    norm_x, attention_mask=attention_mask, return_attention=False
                )
            x = x + attn_out

            ff_out = self.feed_forward(self.norm2(x))
            x = x + ff_out
        else:
            # Post-LayerNorm: Subcapa -> Residual Addition -> LayerNorm (Vaswani et al. 2017)
            if return_attention:
                attn_out, attn_weights = self.self_attn(
                    x, attention_mask=attention_mask, return_attention=True
                )
            else:
                attn_out = self.self_attn(
                    x, attention_mask=attention_mask, return_attention=False
                )
            x = self.norm1(x + attn_out)

            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)

        if return_attention:
            return x, attn_weights
        return x


class TextTransformerEncoder(nn.Module):
    """Transformer Encoder completo para procesamiento y extracción semántica de texto.

    Flujo de ejecución:
    1. Input IDs -> nn.Embedding simple estándar de PyTorch.
    2. Escalamiento opcional por sqrt(d_model).
    3. Suma de Positional Encoding (sinusoidal / learned / none).
    4. Pasaje por L bloques TransformerEncoderBlock (con masking de padding).
    5. Normalización final (en Pre-LN).
    6. Mecanismo de Pooling configurable:
       - 'mean': Promedio de tokens válidos ponderado por attention_mask.
       - 'cls': Extracción del primer token [CLS] (índice 0).
       - 'max': Máximo sobre tokens válidos.
       - 'none': Retorna el tensor secuencial completo (B, T, d_model) para Cross-Attention.
    """

    def __init__(self, config: Optional[TextTransformerConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TextTransformerConfig(**kwargs)
        elif kwargs:
            # Permitir sobrescribir campos del config con kwargs
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config

        # 1. Capa de Embedding simple estándar de PyTorch (nn.Embedding)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # 2. Codificación Posicional
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            encoding_type=config.pos_encoding_type,
            dropout=config.dropout,
        )

        # 3. Stack de capas / bloques del Encoder
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm_first=config.norm_first,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 4. LayerNorm final (requerida cuando norm_first=True)
        self.final_norm = nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización de pesos con Xavier/Normal para convergencia estable."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def pool_sequence(
        self, sequence_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aplica la estrategia de pooling configurada sobre la secuencia (B, T, d_model).

        Args:
            sequence_output: Tensor con representaciones ocultas (batch_size, seq_len, d_model).
            attention_mask: Tensor binario (batch_size, seq_len) con 1 para tokens reales y 0 para [PAD].

        Returns:
            Tensor pooled de forma (batch_size, d_model).
        """
        mode = self.config.pooling_mode.lower()

        if mode == "cls":
            # Extraer la posición 0 donde se ubica el token [CLS]
            return sequence_output[:, 0, :]

        elif mode == "mean":
            if attention_mask is not None:
                # Expandir máscara a (B, T, 1) y convertir a float
                mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)  # (B, d_model)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, 1)
                return sum_embeddings / sum_mask
            else:
                return sequence_output.mean(dim=1)

        elif mode == "max":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).bool()  # (B, T, 1)
                masked_seq = sequence_output.masked_fill(~mask_expanded, -1e9)
                return torch.max(masked_seq, dim=1).values
            else:
                return torch.max(sequence_output, dim=1).values

        elif mode == "none":
            # Retorna el tensor secuencial completo sin colapsar (para Cross-Attention)
            return sequence_output

        else:
            raise ValueError(f"Modo de pooling desconocido: {mode}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Paso forward del Transformer Encoder.

        Args:
            input_ids: Tensor de IDs de tokens de forma (batch_size, seq_len).
            attention_mask: Tensor de máscara de forma (batch_size, seq_len). 1 para tokens válidos, 0 para PAD.
            return_attention: Si True, devuelve una tupla (output, lista_de_atenciones_por_capa).

        Returns:
            - Si pooling_mode != 'none': Tensor e_text de forma (batch_size, d_model).
            - Si pooling_mode == 'none': Tensor secuencial de forma (batch_size, seq_len, d_model).
            - Si return_attention == True: Tupla (output, [attn_layer_1, attn_layer_2, ...]).
        """
        # 1. Lookup de tokens en la tabla de embeddings
        x = self.embedding(input_ids)  # (B, T, d_model)

        # 2. Escalar embedding por sqrt(d_model) si está activo
        if self.config.scale_embedding:
            x = x * math.sqrt(self.config.d_model)

        # 3. Sumar codificación posicional + dropout
        x = self.pos_encoding(x)

        # 4. Pasar secuencialmente a través de las capas del Encoder
        all_attentions = [] if return_attention else None

        for layer in self.layers:
            if return_attention:
                x, layer_attn = layer(
                    x, attention_mask=attention_mask, return_attention=True
                )
                all_attentions.append(layer_attn)
            else:
                x = layer(x, attention_mask=attention_mask, return_attention=False)

        # 5. Normalización final
        x = self.final_norm(x)

        # 6. Pooling para generar e_text
        output = self.pool_sequence(x, attention_mask=attention_mask)

        if return_attention:
            return output, all_attentions
        return output

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calcula el número total de parámetros entrenables del modelo.

        Args:
            non_embedding: Si True, descuenta los parámetros de la matriz de embedding de vocabulario.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            emb_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
            total -= emb_params
        return total


def run_smoke_tests() -> None:
    """Ejecuta una batería completa de pruebas unitarias y de integración."""
    print("=" * 75)
    print("🧪 INICIANDO BATERÍA DE SMOKE TESTS — TextTransformerEncoder")
    print("=" * 75)

    # 1. Configuración base de prueba
    config = TextTransformerConfig(
        vocab_size=2048,
        max_seq_len=128,
        d_model=64,
        n_heads=4,
        d_ff=256,
        num_layers=2,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
        pos_encoding_type="sinusoidal",
        pooling_mode="mean",
        pad_token_id=0,
    )

    model = TextTransformerEncoder(config)
    model.eval()

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"📊 Parámetros Totales:       {total_params:,}")
    print(f"   - Parámetros Embedding:   {model.embedding.weight.numel():,}")
    print(f"   - Parámetros Transformer: {non_emb_params:,}")

    # 2. Datos sintéticos de prueba
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(low=1, high=config.vocab_size, size=(batch_size, seq_len))
    # Simular padding al final de las secuencias
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, 10:] = 0
    attention_mask[1, 8:] = 0
    attention_mask[2, 12:] = 0
    input_ids = input_ids * attention_mask  # PAD tokens tienen ID 0

    # 3. Test Forward Básico con Mean Pooling
    with torch.no_grad():
        out_mean = model(input_ids, attention_mask=attention_mask)
    assert out_mean.shape == (batch_size, config.d_model), (
        f"Shape incorrecto para mean pooling: {out_mean.shape} != {(batch_size, config.d_model)}"
    )
    print(f"✅ [1/6] Forward con 'mean' pooling exitoso: Shape {out_mean.shape}")

    # 4. Test con CLS Pooling y Max Pooling
    model.config.pooling_mode = "cls"
    with torch.no_grad():
        out_cls = model(input_ids, attention_mask=attention_mask)
    assert out_cls.shape == (batch_size, config.d_model)
    print(f"✅ [2/6] Forward con 'cls' pooling exitoso: Shape {out_cls.shape}")

    model.config.pooling_mode = "max"
    with torch.no_grad():
        out_max = model(input_ids, attention_mask=attention_mask)
    assert out_max.shape == (batch_size, config.d_model)
    print(f"✅ [3/6] Forward con 'max' pooling exitoso: Shape {out_max.shape}")

    # 5. Test con Sequence Mode ('none' para Cross-Attention)
    model.config.pooling_mode = "none"
    with torch.no_grad():
        out_seq = model(input_ids, attention_mask=attention_mask)
    assert out_seq.shape == (batch_size, seq_len, config.d_model)
    print(f"✅ [4/6] Forward con 'none' pooling (Secuencia Completa) exitoso: Shape {out_seq.shape}")

    # 6. Test Extracción de Mapas de Atención (XAI)
    out_pooled, attn_maps = model(input_ids, attention_mask=attention_mask, return_attention=True)
    assert len(attn_maps) == config.num_layers, f"Esperadas {config.num_layers} capas de atención."
    assert attn_maps[0].shape == (batch_size, config.n_heads, seq_len, seq_len)
    print(f"✅ [5/6] Extracción de mapas de atención exitosa: {len(attn_maps)} capas, Shape {attn_maps[0].shape}")

    # 7. Test de Retropropagación y Flujo de Gradientes
    model.train()
    model.config.pooling_mode = "mean"
    out_train = model(input_ids, attention_mask=attention_mask)
    loss = out_train.sum()
    loss.backward()

    has_grads = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grads, "Falló el cálculo o flujo de gradientes."
    print("✅ [6/6] Flujo de gradientes y backward pass verificado sin NaN.")

    # 8. Test con Tokenizador Real
    try:
        from src.tokenizer.bpe import ByteLevelBPETokenizer

        tok_path = "resources/tokenizer/bpe_tokenizer.json"
        tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
        sample_texts = [
            "Cedar House Steamable Pepperoni Pizza | "
            "Steamable pepperoni pizza in a 10 oz package for online grocery orders. "
            "Listed under frozen and intended for frozen storage. A dependable pick according to reviews. | "
            "Prepared ingredients, Spices, Salt",
            "Sunny Basket Ready To Heat Waffles | "
            "Ready to heat waffles in a 6 ct package for online grocery orders. "
            "Listed under frozen and intended for frozen storage. Well liked by regular shoppers. | "
            "Flour, Sugar, Eggs",
        ]
        encoded = tokenizer.encode_batch(sample_texts, max_length=128, return_tensors="pt")

        real_config = TextTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=128,
            d_model=64,
            n_heads=4,
            d_ff=256,
            num_layers=2,
            pad_token_id=tokenizer.pad_token_id or 0,
            pooling_mode="mean",
        )
        real_model = TextTransformerEncoder(real_config)
        real_model.eval()

        with torch.no_grad():
            real_out = real_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

        print("\n" + "-" * 75)
        print("🔗 INTEGRACIÓN TOKENIZADOR BPE -> TEXT TRANSFORMER ENCODER")
        print("-" * 75)
        print(f"Textos de entrada:           {len(sample_texts)}")
        print(f"Vocab Size del Tokenizador:  {tokenizer.vocab_size}")
        print(f"Input IDs Shape:             {encoded['input_ids'].shape}")
        print(f"Attention Mask Shape:        {encoded['attention_mask'].shape}")
        print(f"Vector de Salida (e_text):   {real_out.shape} (dtype: {real_out.dtype})")
        print(f"Norma L2 del vector e_text:  {torch.norm(real_out, dim=-1).tolist()}")
        print("-" * 75)
    except Exception as e:
        print(f"ℹ️ Nota sobre integración real: {e}")

    print("\n🎉 TODOS LOS SMOKE TESTS PASARON EXITOSAMENTE.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_smoke_tests()
