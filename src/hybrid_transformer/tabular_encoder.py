"""Implementación modular de la rama tabular para variables numéricas, directas y categóricas.

Distribución de features en la rama tabular:
- Entity Embedding : brand, category, country_of_origin, allergens
- One Hot          : storage_type, unit_of_measure, title_tag, day_of_week
- Numéricas        : price, price_span, price_per_oz, net_weight_oz, volume, num_ingredients, nutrition_score
                     (con log1p en price_per_oz, net_weight_oz, volume y estandarización z-score)
- Directa          : has_allergens (passthrough 0/1 float)

Componentes incluidos:
- `TabularPreprocessor`: ajusta y aplica las transformaciones aprendidas de los datos
  (log1p, estandarización z-score, vocabularios categóricos y passthrough directo).
  Se ajusta ESTRICTAMENTE sobre el split de entrenamiento y se serializa a JSON.
- `TabularEncoder`: red que proyecta las variables preprocesadas a `e_tab` aplicando
  Entity Embeddings a las variables designadas, One-Hot a las de baja cardinalidad,
  normalización a las continuas y concatenando el valor directo antes de pasar por el MLP.
- `TabularEncoderConfig`: dataclass de hiperparámetros.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Variables numéricas continuas que reciben estandarización Z-score
DEFAULT_NUMERIC_FIELDS: List[str] = [
    "price",
    "price_span",
    "price_per_oz",
    "net_weight_oz",
    "volume",
    "num_ingredients",
    "nutrition_score",
]

# Subconjunto de las numéricas con asimetría que requiere compresión log1p previa
DEFAULT_LOG1P_FIELDS: List[str] = [
    "price_per_oz",
    "net_weight_oz",
    "volume",
]

# Variables directas (passthrough numérico sin Z-score ni normalización)
DEFAULT_DIRECT_FIELDS: List[str] = [
    "has_allergens",
]

# Variables categóricas que se procesan con Entity Embeddings
DEFAULT_EMBEDDING_FIELDS: List[str] = [
    "brand",
    "category",
    "country_of_origin",
    "allergens",
]

# Variables categóricas que se procesan con One-Hot Encoding
DEFAULT_ONEHOT_FIELDS: List[str] = [
    "storage_type",
    "unit_of_measure",
    "title_tag",
    "day_of_week",
]

# Todas las variables categóricas combinadas (embedding + one-hot)
DEFAULT_CATEGORICAL_FIELDS: List[str] = DEFAULT_EMBEDDING_FIELDS + DEFAULT_ONEHOT_FIELDS

# Índice reservado para categorías no vistas en entrenamiento (OOV)
UNKNOWN_INDEX = 0


class TabularPreprocessor:
    """Ajusta y aplica el preprocesamiento de variables numéricas, directas y categóricas.

    Aprende de los datos exclusivamente del split de entrenamiento:
    - Por cada numérica: media y desvío estándar (calculados tras log1p si corresponde).
    - Por cada categórica (embeddings y one-hot): vocabulario ordenado de valores únicos.
    - Directas: se validan y se conservan en escala original.
    """

    def __init__(
        self,
        numeric_fields: Optional[List[str]] = None,
        log1p_fields: Optional[List[str]] = None,
        direct_fields: Optional[List[str]] = None,
        embedding_fields: Optional[List[str]] = None,
        onehot_fields: Optional[List[str]] = None,
    ) -> None:
        self.numeric_fields = list(numeric_fields if numeric_fields is not None else DEFAULT_NUMERIC_FIELDS)
        self.log1p_fields = list(log1p_fields if log1p_fields is not None else DEFAULT_LOG1P_FIELDS)
        self.direct_fields = list(direct_fields if direct_fields is not None else DEFAULT_DIRECT_FIELDS)
        self.embedding_fields = list(embedding_fields if embedding_fields is not None else DEFAULT_EMBEDDING_FIELDS)
        self.onehot_fields = list(onehot_fields if onehot_fields is not None else DEFAULT_ONEHOT_FIELDS)

        desconocidos = [c for c in self.log1p_fields if c not in self.numeric_fields]
        if desconocidos:
            raise ValueError(
                f"log1p_fields debe ser un subconjunto de numeric_fields. Sobran: {desconocidos}"
            )

        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.embedding_categories: Dict[str, List[str]] = {}
        self.onehot_categories: Dict[str, List[str]] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Indica si el preprocesador ya fue ajustado sobre un conjunto de entrenamiento."""
        return self._fitted

    @property
    def all_fields(self) -> List[str]:
        """Lista de todos los campos requeridos en el DataFrame."""
        return self.numeric_fields + self.direct_fields + self.embedding_fields + self.onehot_fields

    @property
    def categorical_fields(self) -> List[str]:
        """Lista de todas las variables categóricas (embedding + one-hot)."""
        return self.embedding_fields + self.onehot_fields

    @property
    def embedding_cardinalities(self) -> Dict[str, int]:
        """Cardinalidad de las variables que van a Entity Embeddings."""
        return {c: len(v) for c, v in self.embedding_categories.items()}

    @property
    def onehot_cardinalities(self) -> Dict[str, int]:
        """Cardinalidad de las variables que van a One-Hot."""
        return {c: len(v) for c, v in self.onehot_categories.items()}

    @property
    def cardinalities(self) -> Dict[str, int]:
        """Diccionario combinado de cardinalidades para todas las categóricas."""
        return {**self.embedding_cardinalities, **self.onehot_cardinalities}

    @property
    def categories(self) -> Dict[str, List[str]]:
        """Diccionario de vocabularios de todas las variables categóricas."""
        return {**self.embedding_categories, **self.onehot_categories}

    def _apply_log1p(self, serie: pd.Series, columna: str) -> np.ndarray:
        """Aplica log1p si la columna está marcada como asimétrica."""
        valores = serie.astype(float).to_numpy()
        if columna in self.log1p_fields:
            if np.any(valores < 0):
                raise ValueError(
                    f"La columna '{columna}' contiene valores negativos y no admite log1p."
                )
            valores = np.log1p(valores)
        return valores

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        """Aprende medias, desvíos y vocabularios categóricos desde el split de entrenamiento."""
        faltantes = [c for c in self.all_fields if c not in df.columns]
        if faltantes:
            raise KeyError(f"Columnas ausentes en el DataFrame: {faltantes}")

        for columna in self.numeric_fields:
            valores = self._apply_log1p(df[columna], columna)
            media = float(np.mean(valores))
            desvio = float(np.std(valores))
            self.means[columna] = media
            self.stds[columna] = desvio if desvio > 1e-8 else 1.0

        for columna in self.embedding_fields:
            valores = sorted(df[columna].dropna().astype(str).unique().tolist())
            self.embedding_categories[columna] = valores

        for columna in self.onehot_fields:
            valores = sorted(df[columna].dropna().astype(str).unique().tolist())
            self.onehot_categories[columna] = valores

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforma un DataFrame a los tensores requeridos por `TabularEncoder`.

        Returns:
            Tupla (x_num, x_cat):
            - x_num: FloatTensor (B, n_num + n_dir) con numéricas estandarizadas y directas.
            - x_cat: LongTensor (B, n_emb + n_oh) con los índices categóricos (0 = unknown).
        """
        if not self._fitted:
            raise RuntimeError("El preprocesador no fue ajustado. Llamar a fit() sobre el split de train.")

        # 1. Variables numéricas estandarizadas
        columnas_num = []
        for columna in self.numeric_fields:
            valores = self._apply_log1p(df[columna], columna)
            columnas_num.append((valores - self.means[columna]) / self.stds[columna])

        # 2. Variables directas (passthrough en escala natural)
        for columna in self.direct_fields:
            columnas_num.append(df[columna].astype(float).to_numpy())

        x_num = torch.tensor(np.stack(columnas_num, axis=1), dtype=torch.float32)

        # 3. Variables categóricas para Entity Embeddings
        columnas_cat = []
        for columna in self.embedding_fields:
            mapeo = {valor: i + 1 for i, valor in enumerate(self.embedding_categories[columna])}
            indices = df[columna].astype(str).map(mapeo).fillna(UNKNOWN_INDEX).astype(int)
            columnas_cat.append(indices.to_numpy())

        # 4. Variables categóricas para One-Hot
        for columna in self.onehot_fields:
            mapeo = {valor: i + 1 for i, valor in enumerate(self.onehot_categories[columna])}
            indices = df[columna].astype(str).map(mapeo).fillna(UNKNOWN_INDEX).astype(int)
            columnas_cat.append(indices.to_numpy())

        x_cat = torch.tensor(np.stack(columnas_cat, axis=1), dtype=torch.long)
        return x_num, x_cat

    def fit_transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ajusta sobre `df` y devuelve su transformación. Usar SOLO con train."""
        return self.fit(df).transform(df)

    def count_unknowns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Cuenta, por variable categórica, cuántas filas traen un valor no visto en train."""
        if not self._fitted:
            raise RuntimeError("El preprocesador no fue ajustado.")
        resultado = {}
        for columna in self.embedding_fields:
            conocidos = set(self.embedding_categories[columna])
            resultado[columna] = int((~df[columna].astype(str).isin(conocidos)).sum())
        for columna in self.onehot_fields:
            conocidos = set(self.onehot_categories[columna])
            resultado[columna] = int((~df[columna].astype(str).isin(conocidos)).sum())
        return resultado

    def save(self, path: Union[str, Path]) -> Path:
        """Serializa los parámetros aprendidos a un archivo JSON versionable."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "numeric_fields": self.numeric_fields,
            "log1p_fields": self.log1p_fields,
            "direct_fields": self.direct_fields,
            "embedding_fields": self.embedding_fields,
            "onehot_fields": self.onehot_fields,
            "means": self.means,
            "stds": self.stds,
            "embedding_categories": self.embedding_categories,
            "onehot_categories": self.onehot_categories,
        }
        destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destino

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> TabularPreprocessor:
        """Carga un preprocesador previamente ajustado desde su archivo JSON."""
        origen = Path(path)
        if not origen.exists():
            raise FileNotFoundError(f"No se encontró el preprocesador en: {origen}")
        payload = json.loads(origen.read_text(encoding="utf-8"))

        instancia = cls(
            numeric_fields=payload.get("numeric_fields", DEFAULT_NUMERIC_FIELDS),
            log1p_fields=payload.get("log1p_fields", DEFAULT_LOG1P_FIELDS),
            direct_fields=payload.get("direct_fields", DEFAULT_DIRECT_FIELDS),
            embedding_fields=payload.get("embedding_fields", DEFAULT_EMBEDDING_FIELDS),
            onehot_fields=payload.get("onehot_fields", DEFAULT_ONEHOT_FIELDS),
        )
        instancia.means = payload["means"]
        instancia.stds = payload["stds"]
        instancia.embedding_categories = payload.get("embedding_categories", {})
        instancia.onehot_categories = payload.get("onehot_categories", {})
        instancia._fitted = True
        return instancia


@dataclass
class TabularEncoderConfig:
    """Configuración de hiperparámetros de la rama tabular.

    Atributos:
        num_numeric: Cantidad de variables numéricas con Z-score (default: 7).
        num_direct: Cantidad de variables directas passthrough (default: 1, has_allergens).
        embedding_cardinalities: Cardinalidades de las variables de Entity Embeddings.
        embedding_dims: Dimensiones para cada embedding (si es None aplica min(50, ceil(card / 2))).
        onehot_cardinalities: Cardinalidades de las variables One-Hot.
        hidden_dims: Dimensiones de las capas ocultas del MLP tabular.
        d_tab: Dimensión del vector de salida e_tab.
        dropout: Probabilidad de dropout en las capas ocultas.
        activation: Activación no lineal ('gelu' o 'relu').
        use_batchnorm: Si True, aplica BatchNorm1d a las numéricas antes de concatenar.
    """

    num_numeric: int = 7
    num_direct: int = 1
    embedding_cardinalities: List[int] = field(default_factory=list)
    embedding_dims: Optional[List[int]] = None
    onehot_cardinalities: List[int] = field(default_factory=list)
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    d_tab: int = 32
    dropout: float = 0.1
    activation: str = "gelu"
    use_batchnorm: bool = True

    def __post_init__(self) -> None:
        if self.activation.lower() not in {"gelu", "relu"}:
            raise ValueError(f"activation inválida '{self.activation}'. Opciones: gelu, relu")
        if self.num_numeric < 0 or self.num_direct < 0:
            raise ValueError("num_numeric y num_direct no pueden ser negativos.")

        if self.embedding_dims is None:
            self.embedding_dims = [min(50, math.ceil(c / 2)) for c in self.embedding_cardinalities]
        elif len(self.embedding_dims) != len(self.embedding_cardinalities):
            raise ValueError(
                f"embedding_dims ({len(self.embedding_dims)}) debe coincidir con embedding_cardinalities ({len(self.embedding_cardinalities)})."
            )

    @property
    def embedding_output_dim(self) -> int:
        """Dimensión total del bloque de embeddings concatenados."""
        return sum(self.embedding_dims or [])

    @property
    def onehot_output_dim(self) -> int:
        """Dimensión total del bloque One-Hot."""
        return sum(self.onehot_cardinalities or [])

    @property
    def input_dim(self) -> int:
        """Dimensión total que ingresa a la primera capa del MLP tabular."""
        return self.num_numeric + self.num_direct + self.embedding_output_dim + self.onehot_output_dim


class TabularEncoder(nn.Module):
    """Codificador tabular hacia el vector denso `e_tab`.

    Flujo:
    1. Numéricas Z-score -> BatchNorm1d opcional.
    2. Directas -> passthrough.
    3. Categóricas Embedding -> nn.Embedding por variable -> concatenación.
    4. Categóricas One-Hot -> F.one_hot por variable -> concatenación.
    5. Concatenación total -> MLP -> e_tab (B, d_tab).
    """

    def __init__(self, config: TabularEncoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_batchnorm and config.num_numeric > 0:
            self.numeric_norm: nn.Module = nn.BatchNorm1d(config.num_numeric)
        else:
            self.numeric_norm = nn.Identity()

        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim, padding_idx=UNKNOWN_INDEX)
            for card, dim in zip(config.embedding_cardinalities, config.embedding_dims or [])
        ])

        act = nn.GELU() if config.activation.lower() == "gelu" else nn.ReLU()
        capas: List[nn.Module] = []
        dim_previa = config.input_dim
        for dim_oculta in config.hidden_dims:
            capas.append(nn.Linear(dim_previa, dim_oculta))
            if config.use_batchnorm:
                capas.append(nn.BatchNorm1d(dim_oculta))
            capas.append(act)
            capas.append(nn.Dropout(config.dropout))
            dim_previa = dim_oculta
        capas.append(nn.Linear(dim_previa, config.d_tab))
        self.mlp = nn.Sequential(*capas)

        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Proyecta las variables tabulares al vector e_tab.

        Args:
            x_num: FloatTensor (B, n_num + n_dir).
            x_cat: LongTensor (B, n_emb + n_oh).

        Returns:
            FloatTensor (B, d_tab).
        """
        partes = []

        # 1. Numéricas Z-score
        if self.config.num_numeric > 0:
            partes.append(self.numeric_norm(x_num[:, :self.config.num_numeric]))

        # 2. Directas (passthrough)
        if self.config.num_direct > 0:
            partes.append(x_num[:, self.config.num_numeric:])

        # 3. Entity Embeddings
        n_emb = len(self.config.embedding_cardinalities)
        if n_emb > 0:
            emb_vectors = [self.embeddings[i](x_cat[:, i]) for i in range(n_emb)]
            partes.append(torch.cat(emb_vectors, dim=1))

        # 4. One-Hot
        n_oh = len(self.config.onehot_cardinalities)
        if n_oh > 0:
            oh_blocks = []
            for j, card in enumerate(self.config.onehot_cardinalities):
                col_idx = n_emb + j
                oh = F.one_hot(x_cat[:, col_idx], num_classes=card + 1)[:, 1:]
                oh_blocks.append(oh.float())
            partes.append(torch.cat(oh_blocks, dim=1))

        x = torch.cat(partes, dim=1)
        return self.mlp(x)

    def get_num_params(self) -> int:
        """Cantidad total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_smoke_tests() -> None:
    """Batería de verificación del preprocesador y el encoder sobre los splits reales."""
    print("=" * 78)
    print("🧪 SMOKE TESTS — TabularPreprocessor + TabularEncoder")
    print("=" * 78)

    train_path = Path("resources/datasets/transformer_train.csv")
    val_path = Path("resources/datasets/transformer_val.csv")
    if not train_path.exists():
        print(f"⚠️  No se encontró {train_path}. Ejecutar primero:")
        print("    uv run python -m src.data_extraction.build_transformer_dataset")
        return

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # 1. Ajuste del preprocesador SOLO sobre train
    pre = TabularPreprocessor()
    x_num_tr, x_cat_tr = pre.fit_transform(df_train)
    x_num_va, x_cat_va = pre.transform(df_val)

    print(f"\n📊 Variables: {len(pre.numeric_fields)} numéricas, {len(pre.direct_fields)} directas, "
          f"{len(pre.embedding_fields)} embedding, {len(pre.onehot_fields)} one-hot")
    print(f"   log1p aplicado a: {pre.log1p_fields}")
    print(f"   x_num train {tuple(x_num_tr.shape)} | x_cat train {tuple(x_cat_tr.shape)}")
    print(f"   x_num val   {tuple(x_num_va.shape)} | x_cat val   {tuple(x_cat_va.shape)}")

    # 2. Verificar la estandarización sobre las numéricas de train
    n_num = len(pre.numeric_fields)
    medias = x_num_tr[:, :n_num].mean(dim=0)
    desvios = x_num_tr[:, :n_num].std(dim=0)
    assert torch.allclose(medias, torch.zeros_like(medias), atol=1e-5), "Las medias de train no son 0."
    assert torch.allclose(desvios, torch.ones_like(desvios), atol=1e-2), "Los desvíos de train no son 1."
    print(f"\n✅ [1/5] Estandarización correcta en numéricas de train (media≈{medias.abs().max():.2e})")

    # 3. Directas en escala 0/1
    assert torch.all((x_num_tr[:, n_num:] == 0.0) | (x_num_tr[:, n_num:] == 1.0))
    print("✅ [2/5] Variable directa has_allergens preservada como 0/1.")

    # 4. Forward del TabularEncoder
    cfg = TabularEncoderConfig(
        num_numeric=len(pre.numeric_fields),
        num_direct=len(pre.direct_fields),
        embedding_cardinalities=[pre.embedding_cardinalities[c] for c in pre.embedding_fields],
        onehot_cardinalities=[pre.onehot_cardinalities[c] for c in pre.onehot_fields],
    )
    enc = TabularEncoder(cfg)
    enc.eval()
    with torch.no_grad():
        e_tab = enc(x_num_tr[:8], x_cat_tr[:8])
    assert e_tab.shape == (8, cfg.d_tab)
    print(f"✅ [3/5] Forward TabularEncoder: entrada {cfg.input_dim} dims -> e_tab {tuple(e_tab.shape)} | {enc.get_num_params():,} params")

    # 5. Backward pass
    enc.train()
    salida = enc(x_num_tr[:32], x_cat_tr[:32])
    salida.sum().backward()
    sin_nan = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in enc.parameters() if p.requires_grad
    )
    assert sin_nan, "Falló el flujo de gradientes."
    print("✅ [4/5] Backward pass verificado sin NaN.")

    # 6. Round-trip del artefacto JSON
    tmp = Path("resources/preprocessor/tabular_preprocessor.json")
    pre.save(tmp)
    pre2 = TabularPreprocessor.from_file(tmp)
    x_num2, x_cat2 = pre2.transform(df_val)
    assert torch.allclose(x_num_va, x_num2) and torch.equal(x_cat_va, x_cat2)
    print(f"✅ [5/5] Round-trip del artefacto JSON verificado.")

    print("\n🎉 TODOS LOS SMOKE TESTS PASARON EXITOSAMENTE.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_smoke_tests()

