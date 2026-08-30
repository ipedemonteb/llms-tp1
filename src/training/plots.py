"""Figuras de resultados para la presentación del trabajo práctico.

Genera cuatro visualizaciones a partir de los artefactos que deja cada corrida en
`results/runs/<run_name>/`:

1. `plot_learning_curves`  — loss y PR-AUC de train vs val por época. Diagnóstico de
   overfitting / underfitting exigido por la consigna.
2. `plot_pr_roc_curves`    — curvas Precision-Recall y ROC sobre test, con sus líneas base.
3. `plot_model_comparison` — comparativa de PR-AUC entre corridas.
4. `plot_top_n_curve`      — precisión y recall según cuántos productos se promocionan,
   que traduce la métrica a la decisión de negocio.

Las figuras 2 y 4 necesitan las predicciones sobre test: se recalculan cargando el
checkpoint de la corrida, sin reentrenar.

Uso:
    uv run python -m src.training.plots                       # todas las corridas
    uv run python -m src.training.plots --runs late_fusion baseline_texto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

# Paleta validada con scripts/validate_palette.js (modo light, superficie #fcfcfb):
# separación CVD ΔE 9.2 y visión normal ΔE 27.6 en el peor par adyacente.
SERIE_1 = "#2a78d6"   # azul  — train / precisión
SERIE_2 = "#eb6834"   # naranja — validación / recall
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTE = "#8a8985"
GRILLA = "#e6e5e1"
SUPERFICIE = "#fcfcfb"

RESULTS_DIR = Path("results/runs")
FIGURES_DIR = Path("results/figures/training")


def aplicar_estilo() -> None:
    """Estilo común: marcas finas, grilla hairline sólida y ejes recesivos."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.size": 11,
        "figure.dpi": 150,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def _limpiar_ejes(ax) -> None:
    """Quita los bordes superior y derecho y deja la grilla solo en el eje Y."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def cargar_resumen(run_dir: Path) -> dict:
    """Lee el summary.json de una corrida."""
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def cargar_predicciones(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruye el modelo desde el checkpoint y devuelve (probabilidades, etiquetas) de test."""
    from src.training.dataset import build_dataloaders
    from src.training.train import build_model
    from src.training.trainer import Trainer, TrainerConfig

    resumen = cargar_resumen(run_dir)
    args = argparse.Namespace(**resumen["args"])

    campos = args.text_fields
    if isinstance(campos, str):
        campos = [c.strip() for c in campos.split(",") if c.strip()]

    loaders, artefactos = build_dataloaders(
        text_fields=campos, max_length=args.max_length, batch_size=args.batch_size,
        use_text=args.use_text, use_tabular=args.use_tabular, preprocessor_path=None,
    )
    modelo = build_model(args, artefactos)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    modelo.load_state_dict(checkpoint["model_state_dict"])

    trainer = Trainer(modelo, TrainerConfig(verbose=False, device="cpu"))
    logits, etiquetas = trainer.predict(loaders["test"])
    return 1.0 / (1.0 + np.exp(-logits)), etiquetas


# --- Figura 1: curvas de aprendizaje -------------------------------------------------

def plot_learning_curves(run_dir: Path, output_dir: Path = FIGURES_DIR) -> Path:
    """Loss y PR-AUC de train vs validación por época, con la mejor época señalada."""
    aplicar_estilo()
    historial = pd.read_csv(run_dir / "history.csv")
    resumen = cargar_resumen(run_dir)
    mejor = resumen["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    paneles = [
        (axes[0], "train_loss", "val_loss", "Binary Cross-Entropy", "menor es mejor"),
        (axes[1], "train_pr_auc", "val_pr_auc", "PR-AUC", "mayor es mejor"),
    ]

    for ax, col_train, col_val, titulo, nota in paneles:
        epocas = historial["epoch"]
        # La zona posterior a la mejor época de validación es donde el modelo sobreajusta
        if mejor < epocas.max():
            ax.axvspan(mejor, epocas.max(), color=MUTE, alpha=0.07, zorder=0)
        ax.plot(epocas, historial[col_train], color=SERIE_1, label="train")
        ax.plot(epocas, historial[col_val], color=SERIE_2, label="validación")
        ax.axvline(mejor, color=MUTE, linewidth=1.0, zorder=1)

        y_mejor = historial.loc[historial["epoch"] == mejor, col_val].iloc[0]
        ax.plot([mejor], [y_mejor], "o", color=SERIE_2, markersize=8,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=5)

        ax.set_title(f"{titulo}   ({nota})", loc="left", pad=12)
        ax.set_xlabel("época")
        _limpiar_ejes(ax)
        ax.legend(loc="best")

    # La banda sombreada necesita explicarse: sin rótulo el lector no sabe qué marca
    ultima = int(historial["epoch"].max())
    if mejor < ultima:
        for ax in axes:
            ax.annotate("zona de overfitting", xy=((mejor + ultima) / 2, 0.97),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=9.5, color=MUTE)

    ax = axes[1]
    final = historial.loc[historial["epoch"] == ultima]
    brecha_final = float(final["train_pr_auc"].iloc[0] - final["val_pr_auc"].iloc[0])
    ax.annotate(
        f"mejor época: {mejor}\nbrecha final train−val: {brecha_final:+.3f}",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=10, color=TINTA_2,
    )

    fig.suptitle(f"Curvas de aprendizaje — {resumen['run_name']}",
                 x=0.02, ha="left", fontsize=15, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / f"01_curvas_aprendizaje_{resumen['run_name']}.png"
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


# --- Figura 2: curvas PR y ROC --------------------------------------------------------

def plot_pr_roc_curves(run_dir: Path, output_dir: Path = FIGURES_DIR) -> Path:
    """Curvas Precision-Recall y ROC sobre test, cada una con su línea base."""
    aplicar_estilo()
    resumen = cargar_resumen(run_dir)
    probas, y = cargar_predicciones(run_dir)
    prevalencia = float(np.mean(y))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    precision, recall, _ = precision_recall_curve(y, probas)
    ax = axes[0]
    ax.plot(recall, precision, color=SERIE_1)
    ax.axhline(prevalencia, color=MUTE, linewidth=1.2)
    ax.annotate(f"línea base = {prevalencia:.3f}\n(un modelo sin señal)",
                xy=(0.52, prevalencia), xytext=(0.52, prevalencia + 0.06),
                fontsize=10, color=TINTA_2)
    ax.set_xlabel("recall — fracción de las compras capturada")
    ax.set_ylabel("precisión")
    ax.set_title(f"Precision-Recall   ·   PR-AUC = {resumen['test_metrics']['test_pr_auc']:.4f}",
                 loc="left", pad=12)
    ax.set_ylim(0, 1.02)
    _limpiar_ejes(ax)

    fpr, tpr, _ = roc_curve(y, probas)
    ax = axes[1]
    ax.plot(fpr, tpr, color=SERIE_1)
    ax.plot([0, 1], [0, 1], color=MUTE, linewidth=1.2)
    ax.annotate("línea base = 0.5\n(azar)", xy=(0.55, 0.48), fontsize=10, color=TINTA_2)
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos (recall)")
    ax.set_title(f"ROC   ·   ROC-AUC = {resumen['test_metrics']['test_roc_auc']:.4f}",
                 loc="left", pad=12)
    ax.set_ylim(0, 1.02)
    _limpiar_ejes(ax)

    fig.suptitle(f"Rendimiento sobre test — {resumen['run_name']}",
                 x=0.02, ha="left", fontsize=15, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / f"02_curvas_pr_roc_{resumen['run_name']}.png"
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


# --- Figura 3: comparativa entre modelos ----------------------------------------------

def plot_model_comparison(run_dirs: List[Path], output_dir: Path = FIGURES_DIR) -> Optional[Path]:
    """Barras horizontales de PR-AUC por corrida, con la línea base de prevalencia."""
    aplicar_estilo()
    filas = []
    for d in run_dirs:
        if not (d / "summary.json").exists():
            continue
        r = cargar_resumen(d)
        filas.append({
            "nombre": r["run_name"],
            "pr_auc": r["test_metrics"]["test_pr_auc"],
            "params": r["param_breakdown"]["total"],
            "base": r["test_metrics"]["test_pr_auc_baseline"],
        })
    if not filas:
        return None

    datos = pd.DataFrame(filas).sort_values("pr_auc")
    prevalencia = float(datos["base"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 0.82 * len(datos) + 2.2))
    y = np.arange(len(datos))
    # Una sola serie -> un solo color; el largo de la barra ya codifica la magnitud
    ax.barh(y, datos["pr_auc"], height=0.4, color=SERIE_1, zorder=3)
    ax.axvline(prevalencia, color=SERIE_2, linewidth=1.6, zorder=4)
    ax.annotate(f"línea base {prevalencia:.3f}", xy=(prevalencia + 0.012, len(datos) - 0.42),
                color=SERIE_2, fontsize=10, va="center")

    # El conteo de parámetros va a continuación del valor, no del lado del nombre,
    # para no colisionar con las etiquetas del eje
    for i, fila in enumerate(datos.itertuples()):
        ax.text(fila.pr_auc + 0.008, i, f"{fila.pr_auc:.4f}", va="center",
                fontsize=10, color=TINTA)
        ax.text(fila.pr_auc + 0.075, i, f"{fila.params:,} par.", va="center",
                fontsize=9, color=MUTE)

    ax.set_yticks(y)
    ax.set_yticklabels(datos["nombre"], fontsize=10, color=TINTA)
    ax.set_xlabel("PR-AUC sobre test")
    ax.set_xlim(0, max(datos["pr_auc"].max() * 1.32, prevalencia * 1.5))
    ax.set_ylim(-0.6, len(datos) - 0.15)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    ax.set_title("Comparación de arquitecturas", loc="left", pad=14,
                 fontsize=15, color=TINTA)

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / "03_comparacion_modelos.png"
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


# --- Figura 4: curva top-N ------------------------------------------------------------

def plot_top_n_curve(run_dir: Path, output_dir: Path = FIGURES_DIR) -> Path:
    """Precisión y recall en función de cuántos productos se promocionan."""
    aplicar_estilo()
    resumen = cargar_resumen(run_dir)
    probas, y = cargar_predicciones(run_dir)

    orden = np.argsort(-probas)
    y_ordenado = y[orden]
    acumulado = np.cumsum(y_ordenado)
    n = np.arange(1, len(y) + 1)
    precision = acumulado / n
    recall = acumulado / y.sum()
    prevalencia = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(n, precision, color=SERIE_1, label="precisión")
    ax.plot(n, recall, color=SERIE_2, label="recall")
    ax.axhline(prevalencia, color=MUTE, linewidth=1.2)
    ax.annotate(f"precisión sin modelo = {prevalencia:.3f}",
                xy=(len(y) * 0.55, prevalencia + 0.02), fontsize=10, color=TINTA_2)

    # Un único punto de operación como ejemplo de lectura. Se usa una guía vertical en lugar
    # de una flecha: cualquier trazo hacia el punto cruzaría la curva de recall.
    destacado = 150
    if destacado < len(y):
        ax.axvline(destacado, color=MUTE, linewidth=1.0, zorder=1)
        for valores, color in ((precision, SERIE_1), (recall, SERIE_2)):
            ax.plot([destacado], [valores[destacado - 1]], "o", color=color, markersize=9,
                    markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=5)
        ax.annotate(
            f"promocionando los {destacado} mejores:\n"
            f"{precision[destacado-1]:.0%} de acierto\n"
            f"capturo el {recall[destacado-1]:.0%} de las compras",
            xy=(len(y) * 0.42, 0.60), fontsize=10.5, color=TINTA, va="center",
        )

    ax.set_xlabel("cantidad de productos promocionados (ordenados por probabilidad)")
    ax.set_ylabel("proporción")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, len(y))
    ax.set_title(f"Lectura de negocio — {resumen['run_name']}", loc="left", pad=14,
                 fontsize=15, color=TINTA)
    _limpiar_ejes(ax)
    ax.legend(loc="center right")

    output_dir.mkdir(parents=True, exist_ok=True)
    salida = output_dir / f"04_curva_top_n_{resumen['run_name']}.png"
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera las figuras de resultados para la presentación.")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Nombres de corridas en results/runs/. Por defecto, todas.")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--skip_predictions", action="store_true",
                        help="Omite las figuras que requieren recargar el modelo (2 y 4).")
    args = parser.parse_args()

    base = Path(args.results_dir)
    salida = Path(args.output_dir)
    nombres = args.runs if args.runs else sorted(p.name for p in base.iterdir() if p.is_dir())
    directorios = [base / n for n in nombres if (base / n / "summary.json").exists()]

    if not directorios:
        print(f"⚠️  No se encontraron corridas con summary.json en {base}/")
        return

    print(f"📊 Generando figuras para {len(directorios)} corrida(s) en {salida}/\n")

    for d in directorios:
        print(f"  {d.name}")
        print(f"    ✓ {plot_learning_curves(d, salida).name}")
        if not args.skip_predictions:
            print(f"    ✓ {plot_pr_roc_curves(d, salida).name}")
            print(f"    ✓ {plot_top_n_curve(d, salida).name}")

    comparativa = plot_model_comparison(directorios, salida)
    if comparativa:
        print(f"\n  ✓ {comparativa.name}")

    print(f"\n✅ Figuras guardadas en {salida}/\n")


if __name__ == "__main__":
    main()
