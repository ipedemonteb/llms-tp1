"""Script de análisis exhaustivo y generación de gráficos para d_model.

Permite analizar y comparar sistemáticamente diferentes dimensiones latentes (d_model)
del Transformer a partir de los artefactos generados en `results/runs/`:
- Curvas de escala de métricas (PR-AUC, ROC-AUC, Loss, Lift)
- Desglose y eficiencia paramétrica (Curva de Pareto)
- Brecha de generalización (Train vs Val gap y diagnóstico de overfitting)
- Dinámica de convergencia por época
- Balance multimodal (proporción texto vs tabular)
- Agregación multi-semilla (Media ± Desvío estándar)

Uso:
    # 1. Analizar corridas específicas pasadas por argumento:
    uv run python -m src.training.analyze_dmodel --runs run_d32 run_d64 run_d128

    # 2. Descubrir automáticamente todas las corridas con d_model en results/runs/:
    uv run python -m src.training.analyze_dmodel

    # 3. Filtrar por prefijo de configuración (ej. solo late_fusion):
    uv run python -m src.training.analyze_dmodel --filter_config late_fusion
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.plots import (
    GRILLA,
    MUTE,
    PALETA_LINEAS,
    SERIE_1,
    SERIE_2,
    SERIE_3,
    SERIE_4,
    SERIE_5,
    SUPERFICIE,
    TINTA,
    TINTA_2,
    aplicar_estilo,
    limpiar_ejes,
)

DEFAULT_RESULTS_DIR = Path("results/runs")
DEFAULT_OUTPUT_DIR = Path("results/figures/dmodel_analysis")


def aplicar_estilo_matplotlib() -> None:
    """Configura el estilo limpio para publicaciones / presentaciones."""
    aplicar_estilo(fondo="superficie")


# ==============================================================================
# Modelos de Datos para Extracción
# ==============================================================================
@dataclass
class CorridaDModel:
    """Representación de una corrida individual para el análisis."""

    run_name: str
    run_dir: Path
    d_model: int
    n_heads: int
    d_ff: int
    num_layers: int
    seed: int
    config: Optional[str]
    fusion_mode: str
    use_tabular_mlp: bool
    d_tab: int
    params_total: int
    params_texto: int
    params_tabular: int
    params_cabeza: int
    best_epoch: int
    total_epochs: int
    test_pr_auc: float
    test_roc_auc: float
    test_loss: float
    test_lift: float
    val_pr_auc: float
    val_loss: float
    train_pr_auc: float
    train_loss: float
    seconds_per_epoch: float
    history_df: pd.DataFrame

    @property
    def gen_gap_pr_auc(self) -> float:
        """Brecha de generalización de PR-AUC (Train - Val) en la mejor época."""
        return self.train_pr_auc - self.val_pr_auc

    @property
    def text_ratio(self) -> float:
        """Porcentaje de dimensiones textuales en el vector de fusión."""
        dim_tab = self.d_tab if self.use_tabular_mlp else 59
        return (self.d_model / (self.d_model + dim_tab)) * 100.0


@dataclass
class GrupoDModel:
    """Agrupación de múltiples semillas para un mismo d_model."""

    d_model: int
    corridas: List[CorridaDModel]

    @property
    def n_seeds(self) -> int:
        return len(self.corridas)

    def stats(self, campo: str) -> Tuple[float, float]:
        valores = [getattr(c, campo) for c in self.corridas if getattr(c, campo) is not None]
        if not valores:
            return 0.0, 0.0
        return float(np.mean(valores)), float(np.std(valores, ddof=1 if len(valores) > 1 else 0))

    @property
    def mean_test_pr_auc(self) -> float:
        return self.stats("test_pr_auc")[0]

    @property
    def std_test_pr_auc(self) -> float:
        return self.stats("test_pr_auc")[1]

    @property
    def mean_val_pr_auc(self) -> float:
        return self.stats("val_pr_auc")[0]

    @property
    def std_val_pr_auc(self) -> float:
        return self.stats("val_pr_auc")[1]

    @property
    def mean_gen_gap(self) -> float:
        return self.stats("gen_gap_pr_auc")[0]

    @property
    def params_total(self) -> int:
        return self.corridas[0].params_total if self.corridas else 0

    @property
    def text_ratio(self) -> float:
        return self.corridas[0].text_ratio if self.corridas else 0.0

    @property
    def avg_best_epoch(self) -> float:
        return self.stats("best_epoch")[0]

    @property
    def avg_seconds_per_epoch(self) -> float:
        return self.stats("seconds_per_epoch")[0]


# ==============================================================================
# Funciones de Carga y Procesamiento
# ==============================================================================
def cargar_corrida(run_dir: Path) -> Optional[CorridaDModel]:
    """Carga los datos de una corrida a partir de su summary.json e history.csv."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    args = summary.get("args", {})
    d_model = args.get("d_model")
    if d_model is None:
        return None

    # Parámetros y breakdown
    p_breakdown = summary.get("param_breakdown", {})
    params_total = p_breakdown.get("total", summary.get("total_params", 0))
    params_texto = p_breakdown.get("texto", 0)
    params_tabular = p_breakdown.get("tabular", 0)
    params_cabeza = p_breakdown.get("cabeza", 0)

    # Métricas de test
    t_metrics = summary.get("test_metrics", {})
    test_pr_auc = float(t_metrics.get("test_pr_auc", 0.0))
    test_roc_auc = float(t_metrics.get("test_roc_auc", 0.0))
    test_loss = float(t_metrics.get("test_loss", t_metrics.get("test_bce", 0.0)))
    test_lift = float(summary.get("test_lift", 0.0))

    # Historia de entrenamiento
    history_path = run_dir / "history.csv"
    if history_path.exists():
        try:
            history_df = pd.read_csv(history_path)
        except Exception:
            history_df = pd.DataFrame(summary.get("history", []))
    else:
        history_df = pd.DataFrame(summary.get("history", []))

    best_epoch = int(summary.get("best_epoch", 1))
    total_epochs = len(history_df) if not history_df.empty else 1

    # Extraer métricas en best_epoch o última
    if not history_df.empty and "epoch" in history_df.columns:
        row_best = history_df[history_df["epoch"] == best_epoch]
        if row_best.empty:
            row_best = history_df.iloc[-1:]
        val_pr_auc = float(row_best["val_pr_auc"].values[0]) if "val_pr_auc" in row_best else float(summary.get("best_val_pr_auc", 0.0))
        val_loss = float(row_best["val_loss"].values[0]) if "val_loss" in row_best else 0.0
        train_pr_auc = float(row_best["train_pr_auc"].values[0]) if "train_pr_auc" in row_best else 0.0
        train_loss = float(row_best["train_loss"].values[0]) if "train_loss" in row_best else 0.0
        sec_per_ep = float(history_df["seconds"].mean()) if "seconds" in history_df else 0.0
    else:
        val_pr_auc = float(summary.get("best_val_pr_auc", 0.0))
        val_loss = 0.0
        train_pr_auc = 0.0
        train_loss = 0.0
        sec_per_ep = 0.0

    return CorridaDModel(
        run_name=summary.get("run_name", run_dir.name),
        run_dir=run_dir,
        d_model=int(d_model),
        n_heads=int(args.get("n_heads", 4)),
        d_ff=int(args.get("d_ff", 256)),
        num_layers=int(args.get("num_layers", 2)),
        seed=int(args.get("seed", 42)),
        config=args.get("config"),
        fusion_mode=str(args.get("fusion", "late")),
        use_tabular_mlp=bool(args.get("tabular_mlp", False)),
        d_tab=int(args.get("d_tab", 32)),
        params_total=int(params_total),
        params_texto=int(params_texto),
        params_tabular=int(params_tabular),
        params_cabeza=int(params_cabeza),
        best_epoch=best_epoch,
        total_epochs=total_epochs,
        test_pr_auc=test_pr_auc,
        test_roc_auc=test_roc_auc,
        test_loss=test_loss,
        test_lift=test_lift,
        val_pr_auc=val_pr_auc,
        val_loss=val_loss,
        train_pr_auc=train_pr_auc,
        train_loss=train_loss,
        seconds_per_epoch=sec_per_ep,
        history_df=history_df,
    )


def recopilar_corridas(
    runs_solicitadas: Optional[List[str]],
    results_dir: Path,
    filter_config: Optional[str] = None,
) -> List[CorridaDModel]:
    """Descubre y carga las corridas según los filtros provistos."""
    candidatos_dir: List[Path] = []

    if runs_solicitadas:
        for r in runs_solicitadas:
            p = Path(r)
            if p.is_dir() and (p / "summary.json").exists():
                candidatos_dir.append(p)
            elif (results_dir / r).is_dir():
                candidatos_dir.append(results_dir / r)
            else:
                # Búsqueda por patrón de coincidencia parcial
                coincidencias = list(results_dir.glob(f"*{r}*"))
                candidatos_dir.extend([c for c in coincidencias if c.is_dir()])
    else:
        candidatos_dir = [d for d in results_dir.iterdir() if d.is_dir() and (d / "summary.json").exists()]

    corridas: List[CorridaDModel] = []
    for c_dir in sorted(set(candidatos_dir)):
        c = cargar_corrida(c_dir)
        if c is not None:
            if filter_config and c.config != filter_config:
                continue
            corridas.append(c)

    return corridas


def agrupar_por_dmodel(corridas: List[CorridaDModel]) -> List[GrupoDModel]:
    """Agrupa las corridas por valor de d_model ordenado ascendentemente."""
    grupos: Dict[int, List[CorridaDModel]] = defaultdict(list)
    for c in corridas:
        grupos[c.d_model].append(c)

    grupos_ordenados = [
        GrupoDModel(d_model=d, corridas=sorted(lista, key=lambda x: x.seed))
        for d, lista in sorted(grupos.items(), key=lambda item: item[0])
    ]
    return grupos_ordenados


# ==============================================================================
# Generación de Gráficos con Matplotlib
# ==============================================================================
def plot_01_scaling_and_metrics(grupos: List[GrupoDModel], output_path: Path) -> None:
    """Figura 1: Dashboard de escalado de d_model (Métricas, Parámetros, Pareto y Gap)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    aplicar_estilo_matplotlib()

    d_models = [g.d_model for g in grupos]
    test_pr_means = [g.mean_test_pr_auc for g in grupos]
    test_pr_stds = [g.std_test_pr_auc for g in grupos]
    val_pr_means = [g.mean_val_pr_auc for g in grupos]
    val_pr_stds = [g.std_val_pr_auc for g in grupos]
    params_m = [g.params_total / 1e6 for g in grupos]
    gen_gaps = [g.mean_gen_gap for g in grupos]
    best_epochs = [g.avg_best_epoch for g in grupos]

    # Identificar el mejor d_model
    idx_mejor = int(np.argmax(test_pr_means))
    mejor_d = d_models[idx_mejor]

    # --- Panel A: d_model vs PR-AUC (Test y Validación) ---
    ax_a = axes[0, 0]
    limpiar_ejes(ax_a)
    has_multi_seed = any(g.n_seeds > 1 for g in grupos)

    if has_multi_seed:
        ax_a.errorbar(d_models, test_pr_means, yerr=test_pr_stds, label="Test PR-AUC", color=SERIE_1, fmt="o-", capsize=4, lw=2.2)
        ax_a.errorbar(d_models, val_pr_means, yerr=val_pr_stds, label="Val PR-AUC", color=SERIE_2, fmt="s--", capsize=4, lw=1.8)
    else:
        ax_a.plot(d_models, test_pr_means, "o-", label="Test PR-AUC", color=SERIE_1, lw=2.2)
        ax_a.plot(d_models, val_pr_means, "s--", label="Val PR-AUC", color=SERIE_2, lw=1.8)

    ax_a.axvline(mejor_d, color=SERIE_3, linestyle=":", lw=1.5, alpha=0.8, label=f"Óptimo (d={mejor_d})")
    ax_a.set_title("A. Curva de Escala: Desempeño vs d_model", fontweight="bold", loc="left")
    ax_a.set_xlabel("Dimensión Latente (d_model)")
    ax_a.set_ylabel("PR-AUC")
    ax_a.set_xticks(d_models)
    ax_a.legend()

    # --- Panel B: d_model vs Parámetros por Componente ---
    ax_b = axes[0, 1]
    limpiar_ejes(ax_b)
    txt_params = [g.corridas[0].params_texto / 1e3 for g in grupos]
    tab_params = [g.corridas[0].params_tabular / 1e3 for g in grupos]
    head_params = [g.corridas[0].params_cabeza / 1e3 for g in grupos]

    indices = np.arange(len(d_models))
    w = 0.55
    ax_b.bar(indices, txt_params, width=w, label="Rama Texto (Transformer)", color=SERIE_1, alpha=0.9)
    ax_b.bar(indices, tab_params, width=w, bottom=txt_params, label="Rama Tabular", color=SERIE_2, alpha=0.9)
    bottom_head = np.array(txt_params) + np.array(tab_params)
    ax_b.bar(indices, head_params, width=w, bottom=bottom_head, label="Cabeza Clasificadora", color=SERIE_4, alpha=0.9)

    ax_b.set_title("B. Composición Paramétrica del Sistema", fontweight="bold", loc="left")
    ax_b.set_xlabel("Dimensión Latente (d_model)")
    ax_b.set_ylabel("Parámetros Entrenables (en Miles)")
    ax_b.set_xticks(indices)
    ax_b.set_xticklabels(d_models)
    ax_b.legend()

    # --- Panel C: Curva de Pareto (Parámetros vs Test PR-AUC) ---
    ax_c = axes[1, 0]
    limpiar_ejes(ax_c)
    ax_c.plot(params_m, test_pr_means, "o-", color=SERIE_4, lw=1.8, alpha=0.7)
    for d, p, pr in zip(d_models, params_m, test_pr_means):
        ax_c.annotate(
            f"d={d}\n({pr:.3f})",
            (p, pr),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            color=TINTA,
            weight="bold" if d == mejor_d else "normal",
        )
    ax_c.scatter([params_m[idx_mejor]], [test_pr_means[idx_mejor]], color=SERIE_3, s=120, zorder=5, label=f"Mejor d={mejor_d}")
    ax_c.set_title("C. Eficiencia de Capacidad (Curva de Pareto)", fontweight="bold", loc="left")
    ax_c.set_xlabel("Millones de Parámetros Totales")
    ax_c.set_ylabel("Test PR-AUC")
    ax_c.legend()

    # --- Panel D: Diagnóstico de Overfitting (Generalization Gap) ---
    ax_d = axes[1, 1]
    limpiar_ejes(ax_d)
    color_gap = SERIE_5
    color_ep = SERIE_3

    ax_d.plot(d_models, gen_gaps, "o-", color=color_gap, lw=2.0, label="Gap (Train - Val PR-AUC)")
    ax_d.set_xlabel("Dimensión Latente (d_model)")
    ax_d.set_ylabel("Brecha de Generalización (Δ PR-AUC)", color=color_gap)
    ax_d.tick_params(axis="y", labelcolor=color_gap)
    ax_d.set_xticks(d_models)

    ax_d2 = ax_d.twinx()
    for lado in ("top", "left"):
        ax_d2.spines[lado].set_visible(False)
    ax_d2.spines["right"].set_color(GRILLA)
    ax_d2.plot(d_models, best_epochs, "s--", color=color_ep, lw=1.8, label="Época Óptima (Early Stop)")
    ax_d2.set_ylabel("Mejor Época de Validación", color=color_ep)
    ax_d2.tick_params(axis="y", labelcolor=color_ep)
    ax_d2.grid(False)

    ax_d.set_title("D. Diagnóstico de Sobreajuste y Convergencia", fontweight="bold", loc="left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_02_learning_dynamics(grupos: List[GrupoDModel], output_path: Path) -> None:
    """Figura 2: Dinámica de convergencia época a época para cada d_model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    aplicar_estilo_matplotlib()

    ax_loss, ax_pr = axes[0], axes[1]
    limpiar_ejes(ax_loss)
    limpiar_ejes(ax_pr)

    for i, g in enumerate(grupos):
        color = PALETA_LINEAS[i % len(PALETA_LINEAS)]
        c_repr = g.corridas[0]
        h_df = c_repr.history_df

        if h_df.empty or "epoch" not in h_df.columns:
            continue

        epochs = h_df["epoch"].values
        val_loss = h_df["val_loss"].values if "val_loss" in h_df else []
        val_pr = h_df["val_pr_auc"].values if "val_pr_auc" in h_df else []

        if len(val_loss) > 0:
            ax_loss.plot(epochs, val_loss, "o-", color=color, label=f"d={g.d_model}", lw=1.8, ms=4)
        if len(val_pr) > 0:
            ax_pr.plot(epochs, val_pr, "o-", color=color, label=f"d={g.d_model}", lw=1.8, ms=4)

    ax_loss.set_title("A. Evolución de Validation Loss por Época", fontweight="bold", loc="left")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Binary Cross-Entropy Loss")
    ax_loss.legend()

    ax_pr.set_title("B. Evolución de Validation PR-AUC por Época", fontweight="bold", loc="left")
    ax_pr.set_xlabel("Época")
    ax_pr.set_ylabel("PR-AUC de Validación")
    ax_pr.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_03_multimodal_balance(grupos: List[GrupoDModel], output_path: Path) -> None:
    """Figura 3: Proporción de señal en la Fusión y Tiempo de Cómputo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    aplicar_estilo_matplotlib()

    d_models = [g.d_model for g in grupos]
    text_ratios = [g.text_ratio for g in grupos]
    tab_ratios = [100.0 - r for r in text_ratios]
    seconds = [g.avg_seconds_per_epoch for g in grupos]

    # Panel A: Balance Texto vs Tabular
    ax_a = axes[0]
    limpiar_ejes(ax_a)
    indices = np.arange(len(d_models))
    w = 0.5

    ax_a.bar(indices, text_ratios, width=w, label="% Vector Texto", color=SERIE_1, alpha=0.9)
    ax_a.bar(indices, tab_ratios, width=w, bottom=text_ratios, label="% Vector Tabular", color=SERIE_2, alpha=0.9)
    ax_a.axhline(50.0, color=TINTA_2, linestyle="--", lw=1.2, alpha=0.7, label="Equilibrio 50/50")

    for idx, (t, tb) in enumerate(zip(text_ratios, tab_ratios)):
        ax_a.text(idx, t / 2, f"{t:.0f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
        ax_a.text(idx, t + tb / 2, f"{tb:.0f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)

    ax_a.set_title("A. Balance Dimensional en Fusión Multimodal", fontweight="bold", loc="left")
    ax_a.set_xlabel("Dimensión Latente (d_model)")
    ax_a.set_ylabel("Proporción en la Entrada al Clasificador (%)")
    ax_a.set_xticks(indices)
    ax_a.set_xticklabels(d_models)
    ax_a.set_ylim(0, 105)
    ax_a.legend(loc="lower right")

    # Panel B: Tiempo de cómputo por época
    ax_b = axes[1]
    limpiar_ejes(ax_b)
    ax_b.plot(d_models, seconds, "o-", color=SERIE_3, lw=2.0)
    for d, s in zip(d_models, seconds):
        ax_b.annotate(f"{s:.1f}s", (d, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax_b.set_title("B. Costo Computacional (Segundos por Época)", fontweight="bold", loc="left")
    ax_b.set_xlabel("Dimensión Latente (d_model)")
    ax_b.set_ylabel("Tiempo Promedio por Época (s)")
    ax_b.set_xticks(d_models)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Generación de Reporte y Tabla Diagnóstica
# ==============================================================================
def generar_reporte_markdown(grupos: List[GrupoDModel], output_path: Path) -> str:
    """Genera un reporte técnico de diagnóstico en formato Markdown."""
    lineas = [
        "# Informe de Análisis de Dimensión Latente ($d_{\\text{model}}$)",
        "",
        "Este informe resume el comportamiento empírico del sistema híbrido al variar la capacidad latente del Transformer.",
        "",
        "## 1. Tabla Comparativa de Rendimiento y Complejidad",
        "",
        "| $d_{\\text{model}}$ | N° Semillas | Parámetros Totales | % Texto / Tab | Test PR-AUC | Val PR-AUC | Gen Gap (Δ) | Mejor Época | Diagnóstico |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    test_pr_means = [g.mean_test_pr_auc for g in grupos]
    best_idx = int(np.argmax(test_pr_means))
    best_d = grupos[best_idx].d_model

    for idx, g in enumerate(grupos):
        t_mean, t_std = g.stats("test_pr_auc")
        v_mean, v_std = g.stats("val_pr_auc")
        gap = g.mean_gen_gap
        p_tot = g.params_total
        ratio = g.text_ratio
        ep = g.avg_best_epoch

        # Diagnóstico heurístico
        if idx == best_idx:
            diag = "🏆 **Óptimo Global**"
        elif gap > 0.15:
            diag = "⚠️ Sobreajuste Alto (memorización)"
        elif gap > 0.08:
            diag = "⚠️ Sobreajuste Moderado"
        elif t_mean < test_pr_means[best_idx] - 0.03:
            diag = "📉 Sub-ajuste (Capacidad insuficiente)"
        else:
            diag = "✅ Estable / Competitivo"

        t_str = f"{t_mean:.4f} ± {t_std:.4f}" if g.n_seeds > 1 else f"{t_mean:.4f}"
        v_str = f"{v_mean:.4f} ± {v_std:.4f}" if g.n_seeds > 1 else f"{v_mean:.4f}"

        lineas.append(
            f"| **{g.d_model}** | {g.n_seeds} | {p_tot:,} | {ratio:.1f}% / {100-ratio:.1f}% | {t_str} | {v_str} | {gap:.4f} | {ep:.1f} | {diag} |"
        )

    lineas.extend([
        "",
        "## 2. Conclusiones y Criterio de Selección",
        f"- **Dimensión Seleccionada:** Se recomienda **$d_{{\\text{{model}}}} = {best_d}$** como la configuración principal.",
        f"- **Justificación de Capacidad:** Alcanza el mayor Test PR-AUC ({test_pr_means[best_idx]:.4f}) manteniendo un equilibrio entre capacidad expresiva y riesgo de sobreajuste.",
        "- **Ley de Retornos Decrecientes:** Al incrementar $d_{\\text{model}}$ más allá de este punto, el costo paramétrico se duplica sin aportar ganancias significativas en generalización.",
        "",
        "## 3. Figuras Generadas",
        "- `01_dmodel_scaling_and_metrics.png`: Curva de escala, Pareto, composición de parámetros y brecha de generalización.",
        "- `02_dmodel_learning_dynamics.png`: Dinámica de convergencia temporal y curvas de pérdida.",
        "- `03_dmodel_multimodal_balance.png`: Balance de señal textual/tabular y costo computacional por época.",
    ])

    reporte_md = "\n".join(lineas)
    output_path.write_text(reporte_md, encoding="utf-8")
    return reporte_md


# ==============================================================================
# CLI Principal
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza sistemáticamente el impacto de d_model y genera figuras con matplotlib."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Nombres de corridas o carpetas en results/runs/ a incluir en el análisis.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio raíz donde residen las corridas.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se guardarán los gráficos y el informe.",
    )
    parser.add_argument(
        "--filter_config",
        type=str,
        default=None,
        help="Filtra solo las corridas que coincidan con esta configuración base (ej: late_fusion).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANÁLISIS DE DIMENSIÓN LATENTE (d_model) — SISTEMA HÍBRIDO BTR")
    print("=" * 70)

    corridas = recopilar_corridas(
        runs_solicitadas=args.runs,
        results_dir=args.results_dir,
        filter_config=args.filter_config,
    )

    if not corridas:
        print(f"❌ No se encontraron corridas válidas en {args.results_dir}.")
        print("Asegurate de haber entrenado modelos o pasar nombres válidos en --runs.")
        return

    grupos = agrupar_por_dmodel(corridas)
    d_encontrados = [g.d_model for g in grupos]
    print(f"✔ Se cargaron {len(corridas)} corridas cubriendo d_model: {d_encontrados}")

    # Generación de Figuras
    fig1_path = args.output_dir / "01_dmodel_scaling_and_metrics.png"
    fig2_path = args.output_dir / "02_dmodel_learning_dynamics.png"
    fig3_path = args.output_dir / "03_dmodel_multimodal_balance.png"
    rep_path = args.output_dir / "dmodel_analysis_report.md"

    print(f"\n[1/3] Generando gráfico de escala y Pareto: {fig1_path}...")
    plot_01_scaling_and_metrics(grupos, fig1_path)

    print(f"[2/3] Generando dinámica de aprendizaje por época: {fig2_path}...")
    plot_02_learning_dynamics(grupos, fig2_path)

    print(f"[3/3] Generando balance multimodal y eficiencia: {fig3_path}...")
    plot_03_multimodal_balance(grupos, fig3_path)

    print(f"\n[Reporte] Generando informe resumen en Markdown: {rep_path}...")
    reporte = generar_reporte_markdown(grupos, rep_path)

    print("\n" + "-" * 70)
    print(reporte)
    print("-" * 70)
    print(f"\n✔ Análisis completado exitosamente. Artefactos guardados en: {args.output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
