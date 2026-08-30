"""Agregación de corridas multi-semilla para el estudio de ablación.

Lee los `summary.json` que deja cada corrida en `results/runs/`, los agrupa por configuración
y produce las dos tablas que van al informe:

1. Descriptiva por configuración: media, desvío, rango y mejor época sobre las N semillas.
2. Comparación **pareada** contra una configuración de referencia: la diferencia se calcula
   semilla por semilla, de modo que se cancela la componente de suerte compartida (mismo split,
   mismo orden de shuffle). El desvío de las diferencias es mucho menor que la diferencia de los
   desvíos, y es lo que permite concluir algo con pocas corridas.

Agrupamiento
------------
La clave primaria es `args["config"]`, la intención declarada al lanzar la corrida. Se verifica
contra una **firma de arquitectura** derivada de los hiperparámetros con `build_run_name`, que
detecta el caso en que dos corridas dicen ser la misma configuración pero no lo son. Las corridas
lanzadas solo con flags (sin `--config`) no tienen con qué agruparse y se reportan aparte.

Uso:
    uv run python -m src.training.aggregate
    uv run python -m src.training.aggregate --baseline late_fusion
    uv run python -m src.training.aggregate --configs late_fusion cross_attention --metric test_roc_auc
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.training.config import build_run_name
from src.training.train import build_parser


RESULTS_DIR = Path("results/runs")
OUTPUT_DIR = Path("results/aggregate")
FIGURES_DIR = Path("results/figures/training")

COLUMNAS_PAREADO = [
    "config", "n", "delta_media", "delta_desvio", "ic_bajo", "ic_alto",
    "semillas_a_favor", "p_valor",
]


class GrillaInvalida(Exception):
    """La grilla de corridas no admite un agregado con sentido."""


@dataclass(frozen=True)
class Corrida:
    """Una corrida individual, leída de su `summary.json`."""

    nombre: str
    config: Optional[str]
    seed: int
    firma: str
    epochs: Optional[int]
    patience: Optional[int]
    params: int
    best_epoch: int
    metricas: Dict[str, float] = field(default_factory=dict)


def firma_arquitectura(args: Dict) -> str:
    """Deriva una firma que identifica la arquitectura de una corrida, sin la semilla.

    Reutiliza `build_run_name`, que ya codifica ramas activas, modo de fusión y dimensiones, y le
    quita el sufijo `s{seed}`. Los hiperparámetros ausentes en un `summary.json` viejo se completan
    con los defaults del parser, que es el valor con el que esa corrida efectivamente entrenó.
    """
    defaults = vars(build_parser().parse_args([]))
    completos = {**defaults, **{k: v for k, v in args.items() if k in defaults}}
    nombre = build_run_name(argparse.Namespace(**completos), build_parser)
    return "_".join(nombre.split("_")[:-1])


def cargar_corridas(results_dir: Path = RESULTS_DIR) -> List[Corrida]:
    """Lee todos los `summary.json` bajo `results_dir`."""
    corridas: List[Corrida] = []
    for ruta in sorted(Path(results_dir).glob("*/summary.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        args = datos.get("args", {})

        metricas = {k: v for k, v in datos.get("test_metrics", {}).items() if v is not None}
        if datos.get("best_val_pr_auc") is not None:
            metricas["val_pr_auc"] = float(datos["best_val_pr_auc"])
        if datos.get("test_lift") is not None:
            metricas["test_lift"] = float(datos["test_lift"])

        corridas.append(Corrida(
            nombre=datos.get("run_name", ruta.parent.name),
            config=args.get("config"),
            seed=int(args.get("seed", -1)),
            firma=firma_arquitectura(args),
            epochs=args.get("epochs"),
            patience=args.get("patience"),
            params=int(datos.get("param_breakdown", {}).get("total", 0)),
            best_epoch=int(datos.get("best_epoch", -1)),
            metricas=metricas,
        ))
    return corridas


def agrupar(corridas: Sequence[Corrida]) -> Tuple[Dict[str, List[Corrida]], List[Corrida]]:
    """Separa las corridas agrupables por `config` de las que se lanzaron con flags sueltos."""
    grupos: Dict[str, List[Corrida]] = {}
    sueltas: List[Corrida] = []
    for corrida in corridas:
        if corrida.config is None:
            sueltas.append(corrida)
        else:
            grupos.setdefault(corrida.config, []).append(corrida)
    return grupos, sueltas


def validar(grupos: Dict[str, List[Corrida]]) -> Tuple[List[str], List[str]]:
    """Chequea que la grilla config × semilla admita un agregado con sentido.

    Returns:
        Tupla (fatales, advertencias). Los problemas fatales invalidan el promedio: mezclar dos
        arquitecturas bajo un mismo nombre, o tener dos corridas con la misma semilla, produce un
        número que no significa nada. La grilla incompleta es una advertencia porque puede ser un
        experimento a medio correr, pero rompe el análisis pareado y hay que verla.
    """
    fatales: List[str] = []
    advertencias: List[str] = []

    for nombre, corridas in sorted(grupos.items()):
        firmas = sorted({c.firma for c in corridas})
        if len(firmas) > 1:
            fatales.append(f"'{nombre}' mezcla arquitecturas distintas: {firmas}")

        presupuestos = sorted({(c.epochs, c.patience) for c in corridas}, key=str)
        if len(presupuestos) > 1:
            fatales.append(
                f"'{nombre}' mezcla presupuestos de entrenamiento (epochs, patience): {presupuestos}"
            )

        semillas = [c.seed for c in corridas]
        duplicadas = sorted({s for s in semillas if semillas.count(s) > 1})
        if duplicadas:
            fatales.append(f"'{nombre}' tiene corridas duplicadas para las semillas {duplicadas}")

    universo = sorted({c.seed for corridas in grupos.values() for c in corridas})
    for nombre, corridas in sorted(grupos.items()):
        faltantes = sorted(set(universo) - {c.seed for c in corridas})
        if faltantes:
            advertencias.append(f"'{nombre}' no tiene corridas para las semillas {faltantes}")

    return fatales, advertencias


def tabla_corridas(grupos: Dict[str, List[Corrida]]) -> pd.DataFrame:
    """DataFrame tidy con una fila por corrida, para rehacer cualquier análisis sin reentrenar."""
    filas = []
    for nombre, corridas in sorted(grupos.items()):
        for c in sorted(corridas, key=lambda x: x.seed):
            filas.append({
                "config": nombre, "seed": c.seed, "run_name": c.nombre, "firma": c.firma,
                "params": c.params, "best_epoch": c.best_epoch, **c.metricas,
            })
    return pd.DataFrame(filas)


def resumen_por_config(
    grupos: Dict[str, List[Corrida]],
    metrica: str = "test_pr_auc",
) -> pd.DataFrame:
    """Media, desvío muestral, rango y mejor época por configuración."""
    filas = []
    for nombre, corridas in sorted(grupos.items()):
        faltan = [c.nombre for c in corridas if metrica not in c.metricas]
        if faltan:
            raise KeyError(f"La métrica '{metrica}' no está en las corridas: {faltan}")

        valores = np.array([c.metricas[metrica] for c in corridas], dtype=float)
        n = len(valores)
        fila = {
            "config": nombre,
            "n": n,
            "params": corridas[0].params,
            "media": float(valores.mean()),
            "desvio": float(valores.std(ddof=1)) if n > 1 else float("nan"),
            "min": float(valores.min()),
            "max": float(valores.max()),
            "mejor_epoca_mediana": float(np.median([c.best_epoch for c in corridas])),
        }
        if all("test_roc_auc" in c.metricas for c in corridas):
            fila["roc_auc_media"] = float(np.mean([c.metricas["test_roc_auc"] for c in corridas]))
        filas.append(fila)

    return pd.DataFrame(filas).sort_values("media", ascending=False).reset_index(drop=True)


def comparacion_pareada(
    grupos: Dict[str, List[Corrida]],
    referencia: str,
    metrica: str = "test_pr_auc",
    confianza: float = 0.95,
) -> pd.DataFrame:
    """Diferencias semilla a semilla contra una configuración de referencia.

    Solo se usan las semillas presentes en ambas configuraciones, y la columna `n` deja constancia
    de cuántas fueron: aparear contra el índice de la lista en vez de contra la semilla es el error
    silencioso que este diseño evita.
    """
    if referencia not in grupos:
        raise KeyError(
            f"La configuración de referencia '{referencia}' no está entre las disponibles: "
            f"{sorted(grupos)}"
        )

    base = {c.seed: c.metricas[metrica] for c in grupos[referencia]}
    filas = []

    for nombre, corridas in sorted(grupos.items()):
        if nombre == referencia:
            continue
        otra = {c.seed: c.metricas[metrica] for c in corridas}
        semillas = sorted(set(base) & set(otra))
        if not semillas:
            continue

        diferencias = np.array([base[s] - otra[s] for s in semillas], dtype=float)
        n = len(diferencias)
        media = float(diferencias.mean())
        desvio = float(diferencias.std(ddof=1)) if n > 1 else float("nan")

        margen, p_valor = float("nan"), float("nan")
        if n > 1 and desvio > 0:
            t_critico = float(stats.t.ppf(0.5 + confianza / 2.0, n - 1))
            margen = t_critico * desvio / math.sqrt(n)
            p_valor = float(stats.ttest_1samp(diferencias, 0.0).pvalue)

        filas.append({
            "config": nombre,
            "n": n,
            "delta_media": media,
            "delta_desvio": desvio,
            "ic_bajo": media - margen,
            "ic_alto": media + margen,
            "semillas_a_favor": int(np.sum(diferencias > 0)),
            "p_valor": p_valor,
        })

    pareado = pd.DataFrame(filas, columns=COLUMNAS_PAREADO)
    return pareado.sort_values("delta_media", ascending=False).reset_index(drop=True)


def _num(valor: float, decimales: int = 4) -> str:
    """Formatea con coma decimal, para pegar directo en el informe."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return f"{valor:.{decimales}f}".replace(".", ",")


def _markdown(encabezados: Sequence[str], filas: Sequence[Sequence[str]], alineacion: Sequence[str]) -> str:
    """Arma una tabla markdown sin depender de `tabulate`."""
    separadores = {"izq": ":---", "der": "---:"}
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "|" + "|".join(separadores[a] for a in alineacion) + "|",
    ]
    for fila in filas:
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas)


def tabla_resumen_markdown(resumen: pd.DataFrame, metrica: str) -> str:
    """Renderiza la tabla descriptiva."""
    tiene_roc = "roc_auc_media" in resumen.columns
    encabezados = ["Configuración", "n", "Params", f"{metrica} (media ± σ)", "min–max"]
    alineacion = ["izq", "der", "der", "der", "der"]
    if tiene_roc:
        encabezados.append("ROC-AUC")
        alineacion.append("der")
    encabezados.append("Mejor época")
    alineacion.append("der")

    filas = []
    for f in resumen.itertuples():
        fila = [
            f.config, str(f.n), f"{f.params:,}".replace(",", "."),
            f"{_num(f.media)} ± {_num(f.desvio)}",
            f"{_num(f.min)}–{_num(f.max)}",
        ]
        if tiene_roc:
            fila.append(_num(f.roc_auc_media))
        fila.append(f"{f.mejor_epoca_mediana:g}")
        filas.append(fila)

    return _markdown(encabezados, filas, alineacion)


def tabla_pareada_markdown(pareado: pd.DataFrame, referencia: str, confianza: float = 0.95) -> str:
    """Renderiza la tabla de diferencias pareadas."""
    pct = int(round(confianza * 100))
    encabezados = [f"{referencia} vs.", "n", "Δ media ± σ", f"IC{pct}%", "Semillas a favor", "p"]
    alineacion = ["izq", "der", "der", "der", "der", "der"]

    filas = []
    for f in pareado.itertuples():
        filas.append([
            f.config, str(f.n),
            f"{_num(f.delta_media, 4)} ± {_num(f.delta_desvio, 4)}",
            f"[{_num(f.ic_bajo, 4)}, {_num(f.ic_alto, 4)}]",
            f"{f.semillas_a_favor}/{f.n}",
            _num(f.p_valor, 3),
        ])

    return _markdown(encabezados, filas, alineacion)


def plot_comparacion_agregada(
    grupos: Dict[str, List[Corrida]],
    resumen: pd.DataFrame,
    metrica: str = "test_pr_auc",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Barras de la media por configuración, con desvío y los puntos de cada semilla encima.

    Los puntos individuales evitan la lectura engañosa de la barra sola: dejan ver si el desvío
    viene de dispersión genuina o de una única corrida atípica.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.training.plots import (
        GRILLA, MUTE, SERIE_1, SERIE_2, SUPERFICIE, TINTA, TINTA_2, aplicar_estilo,
    )

    aplicar_estilo()
    datos = resumen.sort_values("media").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 0.95 * len(datos) + 2.6))
    y = np.arange(len(datos))

    ax.barh(y, datos["media"], height=0.4, color=SERIE_1, zorder=3)

    for i, fila in enumerate(datos.itertuples()):
        if not math.isnan(fila.desvio):
            ax.plot([fila.media - fila.desvio, fila.media + fila.desvio], [i, i],
                    color=TINTA_2, linewidth=1.4, solid_capstyle="butt", zorder=5)
        valores = [c.metricas[metrica] for c in grupos[fila.config]]
        ax.plot(valores, [i] * len(valores), "o", color=TINTA_2, markersize=6,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, linestyle="none", zorder=6)

    linea_base = None
    if metrica == "test_pr_auc":
        prevalencias = [
            c.metricas["test_pr_auc_baseline"]
            for corridas in grupos.values() for c in corridas
            if "test_pr_auc_baseline" in c.metricas
        ]
        if prevalencias:
            linea_base = float(np.mean(prevalencias))
            ax.axvline(linea_base, color=SERIE_2, linewidth=1.6, zorder=4)
            ax.annotate(f"línea base {_num(linea_base, 3)}",
                        xy=(linea_base + 0.012, len(datos) - 0.40),
                        color=SERIE_2, fontsize=10, va="center")

    for i, fila in enumerate(datos.itertuples()):
        ax.text(fila.media + 0.010, i + 0.24, f"{_num(fila.media)}", va="center",
                fontsize=10, color=TINTA)
        ax.text(fila.media + 0.010, i - 0.24, f"n={fila.n} · {fila.params:,} par.".replace(",", "."),
                va="center", fontsize=9, color=MUTE)

    ax.set_yticks(y)
    ax.set_yticklabels(datos["config"], fontsize=10, color=TINTA)
    ax.set_xlabel(f"{metrica} sobre test")
    tope = max(float(datos["media"].max()) * 1.34, (linea_base or 0.0) * 1.5)
    ax.set_xlim(0, tope)
    ax.set_ylim(-0.7, len(datos) - 0.15)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    ax.set_title("Comparación de arquitecturas sobre varias semillas", loc="left", pad=26,
                 fontsize=15, color=TINTA)
    ax.annotate("barra = media · línea = ± un desvío · punto = una semilla",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=9.5, color=MUTE, va="bottom")

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / "05_comparacion_agregada.png"
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


def elegir_referencia(grupos: Dict[str, List[Corrida]]) -> str:
    """Elige la configuración de referencia por PR-AUC de **validación**, no de test.

    La selección de arquitectura se hace con validación; test se mira una sola vez al final. Elegir
    la referencia por su resultado en test sería seleccionar sobre el mismo conjunto que después se
    reporta, y el número quedaría optimista.
    """
    puntajes = {}
    for nombre, corridas in grupos.items():
        valores = [c.metricas["val_pr_auc"] for c in corridas if "val_pr_auc" in c.metricas]
        if valores:
            puntajes[nombre] = float(np.mean(valores))
    if not puntajes:
        return sorted(grupos)[0]
    return max(puntajes, key=puntajes.get)


def build_parser_agregado() -> argparse.ArgumentParser:
    """Parser de la CLI del agregador."""
    parser = argparse.ArgumentParser(
        description="Agrega las corridas multi-semilla y arma las tablas del informe."
    )
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--figures_dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--metric", type=str, default="test_pr_auc",
                        help="Métrica a agregar (test_pr_auc, test_roc_auc, val_pr_auc, ...).")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Config de referencia del pareado. Por defecto, la mejor en validación.")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Restringe el agregado a estas configuraciones.")
    parser.add_argument("--confianza", type=float, default=0.95)
    parser.add_argument("--forzar", action="store_true",
                        help="Degrada los problemas fatales de la grilla a advertencia.")
    parser.add_argument("--sin_figura", action="store_true")
    return parser


def main(argv: Optional[list] = None) -> Dict[str, pd.DataFrame]:
    args = build_parser_agregado().parse_args(argv)

    corridas = cargar_corridas(Path(args.results_dir))
    if not corridas:
        raise FileNotFoundError(
            f"No se encontraron corridas con summary.json en {args.results_dir}/.\n"
            "Ejecutar primero: uv run python -m src.training.train --config <nombre> --seed <n>"
        )

    grupos, sueltas = agrupar(corridas)
    if args.configs:
        desconocidas = set(args.configs) - set(grupos)
        if desconocidas:
            raise KeyError(f"Configuraciones no encontradas: {sorted(desconocidas)}. "
                           f"Disponibles: {sorted(grupos)}")
        grupos = {k: v for k, v in grupos.items() if k in args.configs}

    print("=" * 88)
    print(f"  AGREGADO DE CORRIDAS — {len(corridas)} summary.json en {args.results_dir}/")
    print("=" * 88)

    if sueltas:
        print(f"\n⏭️  {len(sueltas)} corrida(s) sin `--config`, no se agregan a ningún grupo:")
        for c in sueltas:
            print(f"      {c.nombre:<48} firma={c.firma}")

    if not grupos:
        raise GrillaInvalida(
            "Ninguna corrida tiene `--config`, así que no hay nada que agrupar. "
            "Relanzar el experimento con `--config <nombre> --seed <n>`."
        )

    fatales, advertencias = validar(grupos)
    if advertencias:
        print("\n⚠️  Grilla incompleta:")
        for problema in advertencias:
            print(f"      {problema}")
    if fatales:
        print("\n❌ Problemas que invalidan el promedio:")
        for problema in fatales:
            print(f"      {problema}")
        if not args.forzar:
            raise GrillaInvalida(
                "La grilla no admite un agregado con sentido. Corregir los problemas de arriba, "
                "o repetir con --forzar si sabés lo que estás haciendo."
            )
        print("      (--forzar activo: se continúa igual)")

    referencia = args.baseline or elegir_referencia(grupos)
    resumen = resumen_por_config(grupos, metrica=args.metric)
    pareado = comparacion_pareada(grupos, referencia, metrica=args.metric, confianza=args.confianza)
    tidy = tabla_corridas(grupos)

    print(f"\n\n📊 DESCRIPTIVA POR CONFIGURACIÓN — {args.metric}\n")
    print(tabla_resumen_markdown(resumen, args.metric))

    if not pareado.empty:
        print(f"\n\n🔬 COMPARACIÓN PAREADA — referencia: {referencia}")
        if args.baseline is None:
            print("   (referencia elegida por PR-AUC de validación, no de test)\n")
        else:
            print()
        print(tabla_pareada_markdown(pareado, referencia, args.confianza))
        print("\n   Δ > 0 favorece a la referencia. Un IC que cruza el cero significa que la")
        print("   diferencia no se distingue de la variabilidad entre semillas.")

    destino = Path(args.output_dir)
    destino.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(destino / "corridas.csv", index=False)
    resumen.to_csv(destino / "resumen.csv", index=False)
    pareado.to_csv(destino / "pareado.csv", index=False)
    print(f"\n\n💾 Tablas guardadas en {destino}/ (corridas.csv, resumen.csv, pareado.csv)")

    if not args.sin_figura:
        figura = plot_comparacion_agregada(grupos, resumen, args.metric, Path(args.figures_dir))
        print(f"📈 Figura guardada en {figura}")

    print()
    return {"resumen": resumen, "pareado": pareado, "corridas": tidy}


if __name__ == "__main__":
    main()
