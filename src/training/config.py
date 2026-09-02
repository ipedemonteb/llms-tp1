"""Carga de configuraciones de experimento desde archivos JSON.

Un archivo de configuración agrupa los hiperparámetros de una corrida, evitando comandos con
veinte flags y dejando cada experimento versionado y reproducible.

Resolución de la referencia pasada a `--config`:
- Un nombre suelto (`late_fusion`) se resuelve como `config/late_fusion.json`.
- Una ruta explícita (`experimentos/prueba.json`) se usa tal cual.

Precedencia de valores, de menor a mayor prioridad:

    defaults del parser  <  archivo de configuración  <  flags explícitos de la CLI

Es decir que una config puede reutilizarse variando un único hiperparámetro desde la terminal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

CONFIG_DIR = Path("config")


def resolve_config_path(referencia: str, config_dir: Path = CONFIG_DIR) -> Path:
    """Convierte la referencia de `--config` en una ruta concreta."""
    ruta = Path(referencia)
    if ruta.suffix == ".json" or len(ruta.parts) > 1:
        return ruta
    return config_dir / f"{referencia}.json"


def available_configs(config_dir: Path = CONFIG_DIR) -> List[str]:
    """Nombres de las configuraciones disponibles en el directorio."""
    if not config_dir.exists():
        return []
    return sorted(p.stem for p in config_dir.glob("*.json"))


def load_config(referencia: str, claves_validas: Set[str], config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Lee y valida un archivo de configuración.

    Args:
        referencia: Nombre o ruta pasada a `--config`.
        claves_validas: Hiperparámetros aceptados, tomados del parser de la CLI.
        config_dir: Directorio donde buscar las configuraciones por nombre.

    Returns:
        Diccionario de hiperparámetros.

    Raises:
        FileNotFoundError: Si el archivo no existe. El mensaje lista las configs disponibles.
        ValueError: Si el JSON es inválido o contiene claves no reconocidas.
    """
    ruta = resolve_config_path(referencia, config_dir)
    if not ruta.exists():
        disponibles = available_configs(config_dir)
        detalle = ", ".join(disponibles) if disponibles else "(ninguna)"
        raise FileNotFoundError(
            f"No se encontró la configuración '{referencia}' en {ruta}.\n"
            f"Configuraciones disponibles en {config_dir}/: {detalle}"
        )

    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"El archivo {ruta} no es JSON válido: {e}") from e

    if not isinstance(datos, dict):
        raise ValueError(f"El archivo {ruta} debe contener un objeto JSON, no {type(datos).__name__}.")

    # Las claves que empiezan con '_' se ignoran, para permitir notas dentro del archivo
    datos = {k: v for k, v in datos.items() if not k.startswith("_")}

    desconocidas = set(datos) - claves_validas
    if desconocidas:
        raise ValueError(
            f"Claves no reconocidas en {ruta}: {sorted(desconocidas)}\n"
            f"Claves válidas: {sorted(claves_validas)}"
        )

    return datos


def explicit_cli_keys(parser_factory, argv: List[str]) -> Set[str]:
    """Determina qué argumentos fueron pasados explícitamente por línea de comandos.

    Recorre los tokens de `argv` y los mapea a su `dest`. No se puede usar
    `argument_default=SUPPRESS` porque cada argumento declara su propio `default`, que tiene
    prioridad y haría que todos aparezcan como explícitos.

    Soporta las dos formas de escritura: `--flag valor` y `--flag=valor`.
    """
    mapa = {
        opcion: accion.dest
        for accion in parser_factory()._actions
        for opcion in accion.option_strings
    }
    return {mapa[t.split("=", 1)[0]] for t in argv if t.split("=", 1)[0] in mapa}


def _formato_corto(valor: Any) -> str:
    """Representación compacta de un valor para usarla dentro del nombre de una corrida."""
    if isinstance(valor, bool):
        return "si" if valor else "no"
    if isinstance(valor, float):
        return f"{valor:g}"
    return str(valor)


def build_run_name(args: argparse.Namespace, parser_factory) -> str:
    """Genera un nombre autodescriptivo a partir de los hiperparámetros de la corrida.

    Incluye siempre las ramas activas, el modo de fusión, las dimensiones principales y la
    semilla. Los hiperparámetros secundarios aparecen **solo cuando difieren del default**, de
    modo que el nombre se mantenga legible y a la vez distinga corridas distintas.

    Ejemplos:
        hyb_late_d64_L2_H4_dt32_s42
        txt_d32_L1_H2_ff128_s7
        hyb_late_d64_L2_H4_dt32_do0.3_wd0.05_s42
    """
    defaults = vars(parser_factory().parse_args([]))
    valores = vars(args)
    partes: List[str] = []

    usa_texto, usa_tabular = valores["use_text"], valores["use_tabular"]
    if usa_texto and usa_tabular:
        partes += ["hyb", valores["fusion"]]
    elif usa_texto:
        partes.append("txt")
    else:
        partes.append("tab")

    if usa_texto:
        partes += [f"d{valores['d_model']}", f"L{valores['num_layers']}", f"H{valores['n_heads']}"]
    if usa_tabular:
        partes.append(f"dt{valores['d_tab']}")
        if valores.get("embedding_dim") is not None:
            partes.append(f"ed{valores['embedding_dim']}")

    opcionales = [("dropout", "do"), ("lr", "lr"), ("weight_decay", "wd"), ("batch_size", "bs")]
    if usa_texto:
        opcionales = [("d_ff", "ff"), ("pooling", "pool"), ("pos_encoding", "pos"),
                      ("max_length", "len")] + opcionales

    for clave, prefijo in opcionales:
        if valores.get(clave) != defaults.get(clave):
            partes.append(f"{prefijo}{_formato_corto(valores[clave])}")

    partes.append(f"s{valores['seed']}")
    return "_".join(partes)


def apply_config(
    args: argparse.Namespace,
    parser_factory,
    argv: List[str],
    config_dir: Path = CONFIG_DIR,
) -> argparse.Namespace:
    """Aplica el archivo de `--config` sobre los argumentos y resuelve el `run_name`.

    Los flags escritos explícitamente en la terminal tienen prioridad sobre el archivo.

    El `run_name` se resuelve en dos niveles, de mayor a menor prioridad:
    1. El valor explícito de `--run_name` o el definido dentro de la configuración.
    2. Un nombre autodescriptivo derivado de los hiperparámetros, prefijado con el nombre del
       archivo de configuración cuando se usó `--config`.

    El prefijo mantiene agrupadas las corridas de una misma configuración, y el sufijo
    autodescriptivo evita que los flags que la pisan terminen escribiendo en el mismo directorio.
    """
    explicitos = explicit_cli_keys(parser_factory, argv)
    configuracion: Dict[str, Any] = {}

    if getattr(args, "config", None):
        claves_validas = {a.dest for a in parser_factory()._actions} - {"help", "config"}
        configuracion = load_config(args.config, claves_validas, config_dir)
        for clave, valor in configuracion.items():
            if clave not in explicitos:
                setattr(args, clave, valor)

    if "run_name" not in explicitos and "run_name" not in configuracion:
        autodescripcion = build_run_name(args, parser_factory)
        if getattr(args, "config", None):
            prefijo = resolve_config_path(args.config, config_dir).stem
            args.run_name = f"{prefijo}_{autodescripcion}"
        else:
            args.run_name = autodescripcion

    return args
