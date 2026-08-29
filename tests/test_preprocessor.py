"""Tests del preprocesador tabular.

El foco está en las propiedades que pueden romperse en silencio: la estandarización correcta,
la ausencia de data leakage entre splits, el manejo de categorías no vistas y la fidelidad del
artefacto serializado.
"""

import json

import numpy as np
import pytest
import torch

from src.hybrid_transformer.tabular_encoder import (
    DEFAULT_LOG1P_FIELDS,
    DEFAULT_NUMERIC_FIELDS,
    UNKNOWN_INDEX,
    TabularPreprocessor,
)


def test_estandarizacion_deja_media_cero_y_desvio_uno(df_sintetico):
    pre = TabularPreprocessor()
    x_num, _ = pre.fit_transform(df_sintetico)
    assert torch.allclose(x_num.mean(dim=0), torch.zeros(x_num.shape[1]), atol=1e-5)
    assert torch.allclose(x_num.std(dim=0), torch.ones(x_num.shape[1]), atol=1e-2)


def test_log1p_se_aplica_solo_a_los_campos_marcados(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    for campo in DEFAULT_LOG1P_FIELDS:
        esperado = float(np.mean(np.log1p(df_sintetico[campo].to_numpy())))
        assert pre.means[campo] == pytest.approx(esperado, rel=1e-6)
    for campo in set(DEFAULT_NUMERIC_FIELDS) - set(DEFAULT_LOG1P_FIELDS):
        esperado = float(np.mean(df_sintetico[campo].to_numpy()))
        assert pre.means[campo] == pytest.approx(esperado, rel=1e-6)


def test_log1p_reduce_la_asimetria(df_sintetico):
    """Verifica el motivo por el que se aplica log1p: comprimir la cola derecha."""
    from scipy.stats import skew

    crudo = df_sintetico["volume"].to_numpy()
    assert abs(skew(np.log1p(crudo))) < abs(skew(crudo)) or abs(skew(crudo)) < 0.5


def test_transform_no_reajusta_los_parametros(df_sintetico):
    """Transformar un segundo DataFrame no debe modificar medias ni desvíos aprendidos."""
    pre = TabularPreprocessor().fit(df_sintetico.iloc[:60])
    medias_antes = dict(pre.means)
    pre.transform(df_sintetico.iloc[60:])
    assert pre.means == medias_antes


def test_split_distinto_no_queda_perfectamente_estandarizado(df_sintetico):
    """Si val quedara con media 0 y desvío 1 exactos, sería señal de leakage."""
    pre = TabularPreprocessor().fit(df_sintetico.iloc[:60])
    x_val, _ = pre.transform(df_sintetico.iloc[60:])
    assert not torch.allclose(x_val.mean(dim=0), torch.zeros(x_val.shape[1]), atol=1e-6)


def test_categoria_no_vista_va_al_indice_desconocido(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    df_nuevo = df_sintetico.head(3).copy()
    df_nuevo["brand"] = "MarcaInexistente"
    _, x_cat = pre.transform(df_nuevo)
    idx_brand = pre.categorical_fields.index("brand")
    assert torch.all(x_cat[:, idx_brand] == UNKNOWN_INDEX)


def test_count_unknowns_detecta_valores_nuevos(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    df_nuevo = df_sintetico.head(5).copy()
    df_nuevo["category"] = "CategoriaNueva"
    assert pre.count_unknowns(df_nuevo)["category"] == 5


def test_transform_antes_de_fit_lanza_error(df_sintetico):
    with pytest.raises(RuntimeError):
        TabularPreprocessor().transform(df_sintetico)


def test_columna_faltante_lanza_error(df_sintetico):
    with pytest.raises(KeyError):
        TabularPreprocessor().fit(df_sintetico.drop(columns=["price"]))


def test_log1p_fuera_de_numeric_fields_lanza_error():
    with pytest.raises(ValueError):
        TabularPreprocessor(numeric_fields=["price"], log1p_fields=["volume"])


def test_columna_constante_no_divide_por_cero(df_sintetico):
    df = df_sintetico.copy()
    df["price"] = 5.0
    pre = TabularPreprocessor().fit(df)
    x_num, _ = pre.transform(df)
    assert torch.all(torch.isfinite(x_num))


def test_round_trip_del_artefacto_json(df_sintetico, tmp_path):
    pre = TabularPreprocessor().fit(df_sintetico)
    x_num, x_cat = pre.transform(df_sintetico)

    ruta = tmp_path / "pre.json"
    pre.save(ruta)
    recuperado = TabularPreprocessor.from_file(ruta)
    x_num2, x_cat2 = recuperado.transform(df_sintetico)

    assert torch.allclose(x_num, x_num2)
    assert torch.equal(x_cat, x_cat2)


def test_artefacto_es_json_legible(df_sintetico, tmp_path):
    """El artefacto debe ser inspeccionable y versionable, no un pickle opaco."""
    ruta = tmp_path / "pre.json"
    TabularPreprocessor().fit(df_sintetico).save(ruta)
    payload = json.loads(ruta.read_text())
    assert set(payload) == {
        "numeric_fields", "log1p_fields", "categorical_fields", "means", "stds", "categories"
    }


def test_transform_es_determinista(df_sintetico):
    pre = TabularPreprocessor().fit(df_sintetico)
    a_num, a_cat = pre.transform(df_sintetico)
    b_num, b_cat = pre.transform(df_sintetico)
    assert torch.equal(a_num, b_num) and torch.equal(a_cat, b_cat)
