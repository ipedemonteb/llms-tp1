"""Implementación modular de la rama tabular para variables numéricas y categóricas.

Este módulo cubre las 13 variables de `clean_dataset.csv` que la rama de texto no procesa,
más las dos que se incluyen de forma deliberadamente redundante (`allergens`,
`country_of_origin`). Produce el vector denso `e_tab` que se fusiona con `e_text` del
Transformer antes de la cabeza clasificadora.

Componentes incluidos:
- `TabularPreprocessor`: ajusta y aplica las transformaciones aprendidas de los datos
  (log1p, estandarización z-score y vocabularios categóricos). Se ajusta ESTRICTAMENTE
  sobre el split de entrenamiento y se serializa a JSON para garantizar reproducibilidad
  y evitar data leakage, con el mismo criterio que el tokenizador BPE.
- `TabularEncoder`: red que proyecta las variables preprocesadas a `e_tab`, con
  codificación categórica conmutable ('onehot' o 'embedding') para el estudio de ablación.
- `TabularEncoderConfig`: dataclass de hiperparámetros.

Justificación del preprocesamiento (medido sobre el dataset, ver `feature_planning.md`):
- `log1p` en las variables muy asimétricas (`price_per_oz` skew 4.09, `volume` 2.97,
  `net_weight_oz` 2.76). Reduce la asimetría de `volume` de 2.97 a 0.44, impidiendo que los
  outliers dominen el gradiente.
- Estandarización z-score en todas las numéricas, porque conviven escalas incomparables
  (`volume` en cientos vs `num_ingredients` entre 1 y 5) y el optimizador usa un único
  learning rate para todos los pesos.
- One-hot en las categóricas: con cardinalidad máxima de 15, son solo 62 columnas y evita
  imponer un orden falso entre categorías sin relación ordinal.
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


# Variables numéricas continuas que no procesa la rama de texto
DEFAULT_NUMERIC_FIELDS: List[str] = [
    "price",
    "price_span",
    "price_per_oz",
    "net_weight_oz",
    "volume",
    "num_ingredients",
    "nutrition_score",
]

# Subconjunto de las numéricas con asimetría > 1.5 que requiere compresión logarítmica
DEFAULT_LOG1P_FIELDS: List[str] = [
    "price_per_oz",
    "net_weight_oz",
    "volume",
]

# Variables categóricas. `allergens` y `country_of_origin` se incluyen pese a estar también
# en la secuencia de texto: la redundancia entre modalidades es deliberada (el texto aporta
# composicionalidad, el one-hot identidad atómica y barata de aprender).
DEFAULT_CATEGORICAL_FIELDS: List[str] = [
    "category",
    "day_of_week",
    "brand",
    "unit_of_measure",
    "storage_type",
    "has_allergens",
    "allergens",
    "country_of_origin",
]

# Índice reservado para categorías no vistas en entrenamiento
UNKNOWN_INDEX = 0


class TabularPreprocessor:
    """Ajusta y aplica el preprocesamiento de variables numéricas y categóricas.

    Aprende de los datos:
    - Por cada numérica: la media y el desvío estándar (calculados DESPUÉS del log1p si
      corresponde), para la estandarización z-score.
    - Por cada categórica: el vocabulario ordenado de valores posibles.

    Debe ajustarse únicamente sobre el split de entrenamiento. Val y test solo se transforman
    con los parámetros ya aprendidos, replicando el comportamiento en inferencia real.
    """

    def __init__(
        self,
        numeric_fields: Optional[List[str]] = None,
        categorical_fields: Optional[List[str]] = None,
        log1p_fields: Optional[List[str]] = None,
    ) -> None:
        self.numeric_fields = list(numeric_fields if numeric_fields is not None else DEFAULT_NUMERIC_FIELDS)
        self.categorical_fields = list(
            categorical_fields if categorical_fields is not None else DEFAULT_CATEGORICAL_FIELDS
        )
        self.log1p_fields = list(log1p_fields if log1p_fields is not None else DEFAULT_LOG1P_FIELDS)

        desconocidos = [c for c in self.log1p_fields if c not in self.numeric_fields]
        if desconocidos:
            raise ValueError(
                f"log1p_fields debe ser un subconjunto de numeric_fields. Sobran: {desconocidos}"
            )

        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.categories: Dict[str, List[str]] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Indica si el preprocesador ya fue ajustado sobre un conjunto de entrenamiento."""
        return self._fitted

    @property
    def cardinalities(self) -> Dict[str, int]:
        """Cantidad de valores distintos vistos en entrenamiento, por variable categórica."""
        return {c: len(v) for c, v in self.categories.items()}

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
        faltantes = [c for c in self.numeric_fields + self.categorical_fields if c not in df.columns]
        if faltantes:
            raise KeyError(f"Columnas ausentes en el DataFrame: {faltantes}")

        for columna in self.numeric_fields:
            valores = self._apply_log1p(df[columna], columna)
            media = float(np.mean(valores))
            desvio = float(np.std(valores))
            # Guarda contra columnas constantes: evita la división por cero
            self.means[columna] = media
            self.stds[columna] = desvio if desvio > 1e-8 else 1.0

        for columna in self.categorical_fields:
            valores = sorted(df[columna].dropna().astype(str).unique().tolist())
            self.categories[columna] = valores

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforma un DataFrame a los tensores que consume `TabularEncoder`.

        Returns:
            Tupla (x_num, x_cat):
            - x_num: FloatTensor (B, n_numericas) con los valores estandarizados.
            - x_cat: LongTensor (B, n_categoricas) con los índices de categoría.
              El índice 0 queda reservado para valores no vistos en entrenamiento.
        """
        if not self._fitted:
            raise RuntimeError("El preprocesador no fue ajustado. Llamar a fit() sobre el split de train.")

        columnas_num = []
        for columna in self.numeric_fields:
            valores = self._apply_log1p(df[columna], columna)
            columnas_num.append((valores - self.means[columna]) / self.stds[columna])
        x_num = torch.tensor(np.stack(columnas_num, axis=1), dtype=torch.float32)

        columnas_cat = []
        for columna in self.categorical_fields:
            # Los índices arrancan en 1; el 0 se reserva para categorías desconocidas
            mapeo = {valor: i + 1 for i, valor in enumerate(self.categories[columna])}
            indices = df[columna].astype(str).map(mapeo).fillna(UNKNOWN_INDEX).astype(int)
            columnas_cat.append(indices.to_numpy())
        x_cat = torch.tensor(np.stack(columnas_cat, axis=1), dtype=torch.long)

        return x_num, x_cat

    def fit_transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ajusta sobre `df` y devuelve su transformación. Usar SOLO con el split de train."""
        return self.fit(df).transform(df)

    def count_unknowns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Cuenta, por variable categórica, cuántas filas traen un valor no visto en train."""
        if not self._fitted:
            raise RuntimeError("El preprocesador no fue ajustado.")
        resultado = {}
        for columna in self.categorical_fields:
            conocidos = set(self.categories[columna])
            resultado[columna] = int((~df[columna].astype(str).isin(conocidos)).sum())
        return resultado

    def save(self, path: Union[str, Path]) -> Path:
        """Serializa los parámetros aprendidos a un archivo JSON versionable."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "numeric_fields": self.numeric_fields,
            "log1p_fields": self.log1p_fields,
            "categorical_fields": self.categorical_fields,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
        }
        destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"💾 Preprocesador tabular guardado en: {destino.resolve()}")
        return destino

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> TabularPreprocessor:
        """Carga un preprocesador previamente ajustado desde su archivo JSON."""
        origen = Path(path)
        if not origen.exists():
            raise FileNotFoundError(f"No se encontró el preprocesador en: {origen}")
        payload = json.loads(origen.read_text(encoding="utf-8"))

        instancia = cls(
            numeric_fields=payload["numeric_fields"],
            categorical_fields=payload["categorical_fields"],
            log1p_fields=payload["log1p_fields"],
        )
        instancia.means = payload["means"]
        instancia.stds = payload["stds"]
        instancia.categories = payload["categories"]
        instancia._fitted = True
        return instancia


@dataclass
class TabularEncoderConfig:
    """Configuración de hiperparámetros de la rama tabular.

    Atributos:
        num_numeric: Cantidad de variables numéricas de entrada.
        cardinalities: Valores distintos por variable categórica, en el mismo orden de columnas
            que produce `TabularPreprocessor.transform`.
        categorical_encoding: 'onehot' (baseline) o 'embedding' (entity embeddings, ablación).
        embedding_dims: Dimensión por variable categórica cuando se usa 'embedding'. Si es None
            se aplica la heurística min(50, ceil(cardinalidad / 2)).
        hidden_dims: Dimensiones de las capas ocultas del MLP tabular.
        d_tab: Dimensión del vector de salida e_tab.
        dropout: Probabilidad de dropout en las capas ocultas.
        activation: Activación no lineal ('gelu' o 'relu').
        use_batchnorm: Si True, aplica BatchNorm1d a las numéricas antes de concatenar.
    """

    num_numeric: int
    cardinalities: List[int]
    categorical_encoding: str = "onehot"
    embedding_dims: Optional[List[int]] = None
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    d_tab: int = 32
    dropout: float = 0.1
    activation: str = "gelu"
    use_batchnorm: bool = True

    def __post_init__(self) -> None:
        valid_enc = {"onehot", "embedding"}
        if self.categorical_encoding not in valid_enc:
            raise ValueError(
                f"categorical_encoding inválido '{self.categorical_encoding}'. Opciones: {valid_enc}"
            )
        if self.activation.lower() not in {"gelu", "relu"}:
            raise ValueError(f"activation inválida '{self.activation}'. Opciones: gelu, relu")
        if self.num_numeric < 0:
            raise ValueError("num_numeric no puede ser negativo.")

        if self.categorical_encoding == "embedding":
            if self.embedding_dims is None:
                # Heurística estándar para entity embeddings
                self.embedding_dims = [min(50, math.ceil(c / 2)) for c in self.cardinalities]
            elif len(self.embedding_dims) != len(self.cardinalities):
                raise ValueError(
                    f"embedding_dims ({len(self.embedding_dims)}) debe tener un valor por cada "
                    f"variable categórica ({len(self.cardinalities)})."
                )

    @property
    def categorical_output_dim(self) -> int:
        """Dimensión del bloque categórico tras la codificación."""
        if self.categorical_encoding == "onehot":
            return sum(self.cardinalities)
        return sum(self.embedding_dims or [])

    @property
    def input_dim(self) -> int:
        """Dimensión total de entrada al MLP tabular."""
        return self.num_numeric + self.categorical_output_dim


class TabularEncoder(nn.Module):
    """Codificador de variables numéricas y categóricas hacia el vector denso `e_tab`.

    Flujo de ejecución:
    1. Numéricas (ya estandarizadas por el preprocesador) -> BatchNorm1d opcional.
    2. Categóricas -> one-hot o entity embeddings, según configuración.
    3. Concatenación de ambos bloques.
    4. MLP con activación, BatchNorm y Dropout -> e_tab (B, d_tab).
    """

    def __init__(self, config: TabularEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Normalización de las numéricas. Redundante con el z-score del preprocesador, pero
        # estabiliza las activaciones a lo largo del entrenamiento (batch statistics).
        if config.use_batchnorm and config.num_numeric > 0:
            self.numeric_norm: nn.Module = nn.BatchNorm1d(config.num_numeric)
        else:
            self.numeric_norm = nn.Identity()

        # 2. Embeddings por variable categórica (solo en modo 'embedding').
        # +1 en num_embeddings para reservar el índice 0 a categorías desconocidas.
        if config.categorical_encoding == "embedding":
            self.embeddings = nn.ModuleList([
                nn.Embedding(card + 1, dim, padding_idx=UNKNOWN_INDEX)
                for card, dim in zip(config.cardinalities, config.embedding_dims or [])
            ])
        else:
            self.embeddings = nn.ModuleList()

        # 3. MLP tabular
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
        """Inicialización Xavier en capas lineales y normal acotada en embeddings."""
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

    def encode_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Codifica los índices categóricos según el modo configurado.

        Args:
            x_cat: LongTensor (B, n_categoricas) con índices; 0 = categoría desconocida.

        Returns:
            FloatTensor (B, categorical_output_dim).
        """
        if x_cat.numel() == 0 or len(self.config.cardinalities) == 0:
            return x_cat.new_zeros((x_cat.shape[0], 0), dtype=torch.float32)

        if self.config.categorical_encoding == "embedding":
            vectores = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            return torch.cat(vectores, dim=1)

        # One-hot: se genera con cardinalidad+1 y se descarta la columna 0, de modo que una
        # categoría desconocida quede representada como un vector de ceros (equivalente al
        # handle_unknown='ignore' de scikit-learn) sin gastar una columna muerta.
        bloques = []
        for i, card in enumerate(self.config.cardinalities):
            oh = F.one_hot(x_cat[:, i], num_classes=card + 1)[:, 1:]
            bloques.append(oh.float())
        return torch.cat(bloques, dim=1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Proyecta las variables tabulares al vector e_tab.

        Args:
            x_num: FloatTensor (B, n_numericas) ya estandarizado por `TabularPreprocessor`.
            x_cat: LongTensor (B, n_categoricas) con índices de categoría.

        Returns:
            FloatTensor (B, d_tab).
        """
        partes = []
        if self.config.num_numeric > 0:
            partes.append(self.numeric_norm(x_num))
        cat_encoded = self.encode_categorical(x_cat)
        if cat_encoded.shape[1] > 0:
            partes.append(cat_encoded)

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

    print(f"\n📊 Variables: {len(pre.numeric_fields)} numéricas, {len(pre.categorical_fields)} categóricas")
    print(f"   log1p aplicado a: {pre.log1p_fields}")
    print(f"   x_num train {tuple(x_num_tr.shape)} | x_cat train {tuple(x_cat_tr.shape)}")
    print(f"   x_num val   {tuple(x_num_va.shape)} | x_cat val   {tuple(x_cat_va.shape)}")

    # 2. Verificar la estandarización sobre train
    medias = x_num_tr.mean(dim=0)
    desvios = x_num_tr.std(dim=0)
    assert torch.allclose(medias, torch.zeros_like(medias), atol=1e-5), "Las medias de train no son 0."
    assert torch.allclose(desvios, torch.ones_like(desvios), atol=1e-2), "Los desvíos de train no son 1."
    print(f"\n✅ [1/6] Estandarización correcta en train (media≈{medias.abs().max():.2e}, desvío≈{desvios.mean():.4f})")

    # 3. Val NO debe estar perfectamente estandarizado: usa los parámetros de train
    print(f"✅ [2/6] Val transformado con parámetros de train "
          f"(media={x_num_va.mean():.4f}, desvío={x_num_va.std():.4f}) — desvío de 0/1 esperado")

    # 4. Cardinalidades y categorías desconocidas
    cards = [pre.cardinalities[c] for c in pre.categorical_fields]
    desconocidas = pre.count_unknowns(df_val)
    total_desc = sum(desconocidas.values())
    print(f"✅ [3/6] Cardinalidades: {dict(zip(pre.categorical_fields, cards))}")
    print(f"         Categorías no vistas en val: {total_desc} "
          f"({'ninguna, conjunto cerrado' if total_desc == 0 else desconocidas})")

    # 5. Forward en modo one-hot
    cfg_oh = TabularEncoderConfig(num_numeric=len(pre.numeric_fields), cardinalities=cards)
    enc_oh = TabularEncoder(cfg_oh)
    enc_oh.eval()
    with torch.no_grad():
        e_tab_oh = enc_oh(x_num_tr[:8], x_cat_tr[:8])
    assert e_tab_oh.shape == (8, cfg_oh.d_tab)
    print(f"✅ [4/6] Forward 'onehot': entrada {cfg_oh.input_dim} dims "
          f"({cfg_oh.num_numeric} num + {cfg_oh.categorical_output_dim} one-hot) "
          f"-> e_tab {tuple(e_tab_oh.shape)} | {enc_oh.get_num_params():,} params")

    # 6. Forward en modo embedding
    cfg_emb = TabularEncoderConfig(
        num_numeric=len(pre.numeric_fields), cardinalities=cards, categorical_encoding="embedding"
    )
    enc_emb = TabularEncoder(cfg_emb)
    enc_emb.eval()
    with torch.no_grad():
        e_tab_emb = enc_emb(x_num_tr[:8], x_cat_tr[:8])
    assert e_tab_emb.shape == (8, cfg_emb.d_tab)
    print(f"✅ [5/6] Forward 'embedding': dims por variable {cfg_emb.embedding_dims} "
          f"-> entrada {cfg_emb.input_dim} dims -> e_tab {tuple(e_tab_emb.shape)} "
          f"| {enc_emb.get_num_params():,} params")

    # 7. Backward
    enc_oh.train()
    salida = enc_oh(x_num_tr[:32], x_cat_tr[:32])
    salida.sum().backward()
    sin_nan = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in enc_oh.parameters() if p.requires_grad
    )
    assert sin_nan, "Falló el flujo de gradientes."
    print("✅ [6/6] Backward pass verificado sin NaN.")

    # 8. Round-trip del artefacto serializado
    tmp = Path("resources/preprocessor/tabular_preprocessor.json")
    pre.save(tmp)
    pre2 = TabularPreprocessor.from_file(tmp)
    x_num2, x_cat2 = pre2.transform(df_val)
    assert torch.allclose(x_num_va, x_num2) and torch.equal(x_cat_va, x_cat2)
    print(f"✅ Round-trip del artefacto JSON verificado: transformaciones idénticas.")

    print("\n🎉 TODOS LOS SMOKE TESTS PASARON EXITOSAMENTE.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_smoke_tests()
