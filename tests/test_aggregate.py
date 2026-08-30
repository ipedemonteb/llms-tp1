"""Tests del agregador de corridas multi-semilla.

Los `summary.json` se fabrican en `tmp_path` con el mismo esquema que escribe `train.main`, de modo
que la suite no depende de haber entrenado nada.

El test más importante es `test_el_pareado_aparea_por_semilla_y_no_por_orden`: aparear contra el
índice de la lista en vez de contra la semilla da un resultado plausible pero incorrecto, y ninguna
verificación de dimensiones lo detecta.
"""

import json

import numpy as np
import pytest

from src.training.aggregate import (
    Corrida,
    GrillaInvalida,
    agrupar,
    cargar_corridas,
    comparacion_pareada,
    elegir_referencia,
    firma_arquitectura,
    main,
    resumen_por_config,
    tabla_corridas,
    validar,
)


ARGS_BASE = {
    "config": "late_fusion", "seed": 42, "use_text": True, "use_tabular": True,
    "fusion": "late", "d_model": 64, "n_heads": 4, "d_ff": 256, "num_layers": 2,
    "pooling": "mean", "pos_encoding": "sinusoidal", "d_tab": 32,
    "epochs": 20, "patience": 5, "dropout": 0.1, "lr": 0.001, "weight_decay": 0.01,
    "batch_size": 64, "max_length": 128,
}


def escribir_corrida(base, config, seed, pr_auc, val_pr_auc=None, params=223151,
                     best_epoch=5, roc_auc=0.97, **sobrescritos):
    """Fabrica un summary.json con el esquema real de `train.main`."""
    args = {**ARGS_BASE, "config": config, "seed": seed, **sobrescritos}
    nombre = f"{config}_s{seed}" if config else f"suelta_s{seed}"
    destino = base / nombre
    destino.mkdir(parents=True, exist_ok=True)
    resumen = {
        "run_name": nombre,
        "args": args,
        "param_breakdown": {"total": params},
        "best_epoch": best_epoch,
        "best_val_pr_auc": val_pr_auc if val_pr_auc is not None else pr_auc - 0.02,
        "test_metrics": {
            "test_pr_auc": pr_auc, "test_roc_auc": roc_auc, "test_bce": 0.14,
            "test_pr_auc_baseline": 0.1313, "test_positive_rate": 0.1313,
        },
        "test_lift": pr_auc / 0.1313,
    }
    (destino / "summary.json").write_text(json.dumps(resumen), encoding="utf-8")
    return destino


@pytest.fixture
def grilla(tmp_path):
    """Grilla completa: dos configuraciones × tres semillas."""
    for seed, (late, cross) in zip([1, 2, 3], [(0.70, 0.68), (0.74, 0.75), (0.72, 0.70)]):
        escribir_corrida(tmp_path, "late_fusion", seed, late)
        escribir_corrida(tmp_path, "cross_attention", seed, cross,
                         fusion="cross", params=237871)
    return tmp_path


# --- Firma de arquitectura ---

def test_la_firma_ignora_la_semilla():
    a = firma_arquitectura({**ARGS_BASE, "seed": 42})
    b = firma_arquitectura({**ARGS_BASE, "seed": 7})
    assert a == b


def test_la_firma_distingue_arquitecturas():
    late = firma_arquitectura({**ARGS_BASE, "fusion": "late"})
    cross = firma_arquitectura({**ARGS_BASE, "fusion": "cross"})
    chico = firma_arquitectura({**ARGS_BASE, "d_model": 32})
    assert len({late, cross, chico}) == 3


def test_la_firma_completa_las_claves_ausentes_con_los_defaults():
    completo = firma_arquitectura(ARGS_BASE)
    parcial = firma_arquitectura({"config": "late_fusion", "seed": 42})
    assert completo == parcial


def test_la_firma_delata_una_corrida_mal_nombrada():
    tabular = firma_arquitectura({**ARGS_BASE, "config": "late_fusion", "use_text": False})
    assert tabular.startswith("tab")


# --- Carga y agrupamiento ---

def test_carga_todas_las_corridas(grilla):
    assert len(cargar_corridas(grilla)) == 6


def test_las_corridas_sin_config_quedan_aparte(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, None, 9, 0.55)
    grupos, sueltas = agrupar(cargar_corridas(tmp_path))
    assert set(grupos) == {"late_fusion"}
    assert [c.seed for c in sueltas] == [9]


def test_directorio_vacio_lanza_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--results_dir", str(tmp_path)])


# --- Validación de la grilla ---

def test_la_grilla_completa_no_reporta_problemas(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    assert validar(grupos) == ([], [])


def test_semilla_faltante_es_advertencia(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74)
    escribir_corrida(tmp_path, "cross_attention", 1, 0.68, fusion="cross")
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    fatales, advertencias = validar(grupos)
    assert fatales == []
    assert len(advertencias) == 1 and "cross_attention" in advertencias[0] and "2" in advertencias[0]


def test_arquitecturas_mezcladas_es_fatal(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74, d_model=32)
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    fatales, _ = validar(grupos)
    assert len(fatales) == 1 and "mezcla arquitecturas" in fatales[0]


def test_presupuestos_mezclados_es_fatal(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70, epochs=20)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74, epochs=2)
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    fatales, _ = validar(grupos)
    assert len(fatales) == 1 and "presupuestos" in fatales[0]


def test_semilla_duplicada_es_fatal(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    ruta = escribir_corrida(tmp_path, "late_fusion", 1, 0.74)
    (ruta.parent / "late_fusion_s1_bis").mkdir()
    (ruta.parent / "late_fusion_s1_bis" / "summary.json").write_text(
        (ruta / "summary.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    fatales, _ = validar(grupos)
    assert any("duplicadas" in f for f in fatales)


def test_main_aborta_con_grilla_invalida(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74, d_model=32)
    with pytest.raises(GrillaInvalida):
        main(["--results_dir", str(tmp_path), "--sin_figura",
              "--output_dir", str(tmp_path / "out")])


def test_forzar_permite_continuar_con_grilla_invalida(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74, d_model=32)
    salida = main(["--results_dir", str(tmp_path), "--sin_figura", "--forzar",
                   "--output_dir", str(tmp_path / "out")])
    assert len(salida["resumen"]) == 1


def test_main_aborta_si_ninguna_corrida_tiene_config(tmp_path):
    escribir_corrida(tmp_path, None, 1, 0.70)
    with pytest.raises(GrillaInvalida):
        main(["--results_dir", str(tmp_path), "--sin_figura",
              "--output_dir", str(tmp_path / "out")])


# --- Descriptiva ---

def test_media_y_desvio_muestral(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    resumen = resumen_por_config(grupos).set_index("config")
    valores = np.array([0.70, 0.74, 0.72])
    assert resumen.loc["late_fusion", "media"] == pytest.approx(valores.mean())
    assert resumen.loc["late_fusion", "desvio"] == pytest.approx(valores.std(ddof=1))
    assert resumen.loc["late_fusion", "min"] == pytest.approx(0.70)
    assert resumen.loc["late_fusion", "max"] == pytest.approx(0.74)


def test_una_sola_semilla_deja_el_desvio_en_nan(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    resumen = resumen_por_config(grupos)
    assert resumen.loc[0, "n"] == 1
    assert np.isnan(resumen.loc[0, "desvio"])


def test_el_resumen_queda_ordenado_por_media(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    medias = resumen_por_config(grupos)["media"].tolist()
    assert medias == sorted(medias, reverse=True)


def test_metrica_inexistente_lanza_error(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    with pytest.raises(KeyError):
        resumen_por_config(grupos, metrica="test_inventada")


def test_tabla_tidy_tiene_una_fila_por_corrida(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    tidy = tabla_corridas(grupos)
    assert len(tidy) == 6
    assert set(tidy.columns) >= {"config", "seed", "test_pr_auc", "val_pr_auc"}


# --- Comparación pareada ---

def test_el_pareado_aparea_por_semilla_y_no_por_orden(tmp_path):
    """Las semillas se escriben en orden distinto en cada config: aparear por índice daría otro Δ."""
    for seed, valor in [(1, 0.70), (2, 0.74), (3, 0.72)]:
        escribir_corrida(tmp_path, "late_fusion", seed, valor)
    for seed, valor in [(3, 0.60), (1, 0.65), (2, 0.69)]:
        escribir_corrida(tmp_path, "cross_attention", seed, valor, fusion="cross")

    grupos, _ = agrupar(cargar_corridas(tmp_path))
    pareado = comparacion_pareada(grupos, "late_fusion").set_index("config")

    esperado = np.mean([0.70 - 0.65, 0.74 - 0.69, 0.72 - 0.60])
    assert pareado.loc["cross_attention", "delta_media"] == pytest.approx(esperado)
    assert pareado.loc["cross_attention", "n"] == 3
    assert pareado.loc["cross_attention", "semillas_a_favor"] == 3


def test_el_pareado_usa_solo_las_semillas_compartidas(tmp_path):
    escribir_corrida(tmp_path, "late_fusion", 1, 0.70)
    escribir_corrida(tmp_path, "late_fusion", 2, 0.74)
    escribir_corrida(tmp_path, "cross_attention", 2, 0.69, fusion="cross")
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    pareado = comparacion_pareada(grupos, "late_fusion").set_index("config")
    assert pareado.loc["cross_attention", "n"] == 2 - 1
    assert pareado.loc["cross_attention", "delta_media"] == pytest.approx(0.74 - 0.69)


def test_el_desvio_pareado_es_menor_que_el_de_cada_config(tmp_path):
    """Corridas correlacionadas: la diferencia es estable aunque cada config varíe mucho."""
    for seed, base in [(1, 0.60), (2, 0.75), (3, 0.68)]:
        escribir_corrida(tmp_path, "late_fusion", seed, base + 0.03)
        escribir_corrida(tmp_path, "cross_attention", seed, base, fusion="cross")

    grupos, _ = agrupar(cargar_corridas(tmp_path))
    resumen = resumen_por_config(grupos).set_index("config")
    pareado = comparacion_pareada(grupos, "late_fusion").set_index("config")

    assert pareado.loc["cross_attention", "delta_desvio"] < resumen.loc["late_fusion", "desvio"]
    assert pareado.loc["cross_attention", "delta_media"] == pytest.approx(0.03)


def test_el_intervalo_de_confianza_contiene_la_media(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    fila = comparacion_pareada(grupos, "late_fusion").iloc[0]
    assert fila["ic_bajo"] < fila["delta_media"] < fila["ic_alto"]


def test_referencia_inexistente_lanza_error(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    with pytest.raises(KeyError):
        comparacion_pareada(grupos, "no_existe")


def test_la_referencia_no_se_compara_contra_si_misma(grilla):
    grupos, _ = agrupar(cargar_corridas(grilla))
    pareado = comparacion_pareada(grupos, "late_fusion")
    assert "late_fusion" not in pareado["config"].tolist()


def test_la_referencia_por_defecto_sale_de_validacion(tmp_path):
    escribir_corrida(tmp_path, "gana_en_test", 1, pr_auc=0.90, val_pr_auc=0.50)
    escribir_corrida(tmp_path, "gana_en_val", 1, pr_auc=0.60, val_pr_auc=0.80, d_model=32)
    grupos, _ = agrupar(cargar_corridas(tmp_path))
    assert elegir_referencia(grupos) == "gana_en_val"


# --- Salidas ---

def test_main_escribe_las_tres_tablas(grilla, tmp_path):
    salida = tmp_path / "agregado"
    main(["--results_dir", str(grilla), "--output_dir", str(salida), "--sin_figura"])
    for archivo in ("corridas.csv", "resumen.csv", "pareado.csv"):
        assert (salida / archivo).exists(), f"Falta {archivo}"


def test_main_genera_la_figura(grilla, tmp_path):
    main(["--results_dir", str(grilla), "--output_dir", str(tmp_path / "agregado"),
          "--figures_dir", str(tmp_path / "figuras")])
    assert (tmp_path / "figuras" / "05_comparacion_agregada.png").exists()


def test_filtrar_configuraciones(grilla, tmp_path):
    salida = main(["--results_dir", str(grilla), "--output_dir", str(tmp_path / "out"),
                   "--sin_figura", "--configs", "late_fusion"])
    assert salida["resumen"]["config"].tolist() == ["late_fusion"]


def test_configuracion_inexistente_lanza_error(grilla, tmp_path):
    with pytest.raises(KeyError):
        main(["--results_dir", str(grilla), "--output_dir", str(tmp_path / "out"),
              "--sin_figura", "--configs", "no_existe"])
