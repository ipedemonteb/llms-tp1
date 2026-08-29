"""Tests de la carga de configuraciones de experimento.

El foco está en la precedencia (los flags explícitos deben ganarle al archivo) y en que los
errores del usuario —config inexistente, JSON roto, clave mal escrita— fallen de forma ruidosa
en lugar de ignorarse en silencio.
"""

import json
from pathlib import Path

import pytest

from src.training.config import (
    build_run_name,
    apply_config,
    available_configs,
    explicit_cli_keys,
    load_config,
    resolve_config_path,
)
from src.training.train import build_parser

CLAVES = {a.dest for a in build_parser()._actions} - {"help", "config"}


@pytest.fixture
def dir_config(tmp_path) -> Path:
    (tmp_path / "prueba.json").write_text(json.dumps({"d_model": 32, "num_layers": 1, "seed": 7}))
    (tmp_path / "otra.json").write_text(json.dumps({"lr": 0.005}))
    return tmp_path


# --- Resolución de rutas ---

def test_nombre_suelto_se_resuelve_al_directorio_config():
    assert resolve_config_path("late_fusion") == Path("config/late_fusion.json")


def test_ruta_explicita_se_respeta():
    assert resolve_config_path("experimentos/x.json") == Path("experimentos/x.json")


def test_nombre_con_extension_se_respeta():
    assert resolve_config_path("x.json") == Path("x.json")


def test_available_configs_lista_los_archivos(dir_config):
    assert available_configs(dir_config) == ["otra", "prueba"]


def test_available_configs_no_falla_si_no_existe_el_directorio(tmp_path):
    assert available_configs(tmp_path / "inexistente") == []


# --- Carga y validación ---

def test_carga_los_valores_del_archivo(dir_config):
    assert load_config("prueba", CLAVES, dir_config) == {"d_model": 32, "num_layers": 1, "seed": 7}


def test_las_claves_con_guion_bajo_se_ignoran(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"_descripcion": "una nota", "d_model": 32}))
    assert load_config("c", CLAVES, tmp_path) == {"d_model": 32}


def test_config_inexistente_lanza_error_con_las_disponibles(dir_config):
    with pytest.raises(FileNotFoundError, match="prueba"):
        load_config("no_existe", CLAVES, dir_config)


def test_json_invalido_lanza_error(tmp_path):
    (tmp_path / "roto.json").write_text("{ esto no es json }")
    with pytest.raises(ValueError, match="no es JSON válido"):
        load_config("roto", CLAVES, tmp_path)


def test_json_que_no_es_objeto_lanza_error(tmp_path):
    (tmp_path / "lista.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="objeto JSON"):
        load_config("lista", CLAVES, tmp_path)


def test_clave_desconocida_lanza_error(tmp_path):
    """Una clave mal escrita debe fallar, no ignorarse en silencio."""
    (tmp_path / "c.json").write_text(json.dumps({"d_modell": 32}))
    with pytest.raises(ValueError, match="d_modell"):
        load_config("c", CLAVES, tmp_path)


# --- Detección de flags explícitos ---

def test_detecta_solo_los_flags_escritos():
    assert explicit_cli_keys(build_parser, ["--d_model", "32"]) == {"d_model"}


def test_sin_flags_no_detecta_ninguno():
    assert explicit_cli_keys(build_parser, []) == set()


def test_detecta_los_flags_booleanos():
    assert explicit_cli_keys(build_parser, ["--no_text"]) == {"use_text"}


# --- Precedencia ---

def test_la_config_pisa_los_defaults(dir_config):
    argv = ["--config", "prueba"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, dir_config)
    assert args.d_model == 32 and args.num_layers == 1 and args.seed == 7


def test_el_flag_explicito_le_gana_a_la_config(dir_config):
    argv = ["--config", "prueba", "--d_model", "96"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, dir_config)
    assert args.d_model == 96      # gana la CLI
    assert args.num_layers == 1    # sigue viniendo de la config


def test_los_valores_no_definidos_conservan_el_default(dir_config):
    argv = ["--config", "prueba"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, dir_config)
    assert args.d_ff == 256 and args.pooling == "mean"


def test_run_name_prefija_la_config_a_la_autodescripcion(dir_config):
    argv = ["--config", "prueba"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, dir_config)
    assert args.run_name == "prueba_hyb_late_d32_L1_H4_dt32_s7"


def test_pisar_una_config_cambia_el_directorio(dir_config):
    """Sin esto, `--config X --seed N` escribiría siempre en el mismo directorio."""
    base = ["--config", "prueba"]
    variante = ["--config", "prueba", "--seed", "99"]
    a = apply_config(build_parser().parse_args(base), build_parser, base, dir_config).run_name
    b = apply_config(build_parser().parse_args(variante), build_parser, variante, dir_config).run_name
    assert a != b
    assert a.startswith("prueba_") and b.startswith("prueba_")
    assert b.endswith("_s99")


def test_run_name_explicito_le_gana_al_nombre_del_archivo(dir_config):
    argv = ["--config", "prueba", "--run_name", "mi_corrida"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, dir_config)
    assert args.run_name == "mi_corrida"


def test_run_name_de_la_config_le_gana_al_nombre_del_archivo(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"run_name": "desde_json"}))
    argv = ["--config", "c"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, tmp_path)
    assert args.run_name == "desde_json"


def test_sin_config_los_flags_se_conservan():
    argv = ["--d_model", "96"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv)
    assert args.d_model == 96


# --- Nombre autodescriptivo de la corrida ---

def _nombre(argv):
    return apply_config(build_parser().parse_args(argv), build_parser, argv).run_name


def test_nombre_por_defecto_describe_el_modelo_hibrido():
    assert _nombre([]) == "hyb_late_d64_L2_H4_dt32_s42"


def test_nombre_refleja_la_rama_activa():
    assert _nombre(["--no_tabular"]).startswith("txt_")
    assert _nombre(["--no_text"]).startswith("tab_")


def test_nombre_refleja_el_modo_de_fusion():
    assert "cross" in _nombre(["--fusion", "cross"])


def test_nombre_refleja_las_dimensiones_principales():
    n = _nombre(["--d_model", "32", "--num_layers", "1", "--n_heads", "2"])
    assert "d32" in n and "L1" in n and "H2" in n


def test_los_hiperparametros_por_defecto_no_ensucian_el_nombre():
    """Solo los valores que difieren del default deben aparecer."""
    n = _nombre([])
    assert "do" not in n and "lr" not in n and "wd" not in n


def test_los_hiperparametros_modificados_aparecen_en_el_nombre():
    n = _nombre(["--dropout", "0.3", "--weight_decay", "0.05"])
    assert "do0.3" in n and "wd0.05" in n


def test_la_semilla_siempre_aparece():
    assert _nombre(["--seed", "7"]).endswith("_s7")


def test_configuraciones_distintas_dan_nombres_distintos():
    """Es la propiedad que evita que una corrida pise a otra sin querer."""
    nombres = {
        _nombre([]),
        _nombre(["--d_model", "32"]),
        _nombre(["--dropout", "0.3"]),
        _nombre(["--seed", "7"]),
        _nombre(["--no_tabular"]),
        _nombre(["--fusion", "cross"]),
    }
    assert len(nombres) == 6


def test_el_nombre_es_valido_como_directorio():
    n = _nombre(["--dropout", "0.25", "--lr", "0.0005"])
    assert "/" not in n and " " not in n


def test_booleanos_de_la_config_se_aplican(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"use_tabular": False}))
    argv = ["--config", "c"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, tmp_path)
    assert args.use_tabular is False and args.use_text is True


def test_text_fields_acepta_lista_en_json(tmp_path):
    campos = ["title_clean", "description"]
    (tmp_path / "c.json").write_text(json.dumps({"text_fields": campos}))
    argv = ["--config", "c"]
    args = apply_config(build_parser().parse_args(argv), build_parser, argv, tmp_path)
    assert args.text_fields == campos


# --- Configuraciones del repositorio ---

def test_las_configs_del_repo_son_validas():
    """Todas las configs versionadas deben cargarse sin errores de clave."""
    directorio = Path("config")
    nombres = available_configs(directorio)
    assert nombres, "No hay configuraciones en config/"
    for nombre in nombres:
        load_config(nombre, CLAVES, directorio)
