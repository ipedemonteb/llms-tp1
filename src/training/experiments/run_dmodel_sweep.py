"""Script de barrido de d_model para el modelo híbrido simple.

Ejecuta d_model=16 y d_model=64 de forma secuencial aislando los resultados en:
- results/aggregate/hybrid_baseline/dmodel_16/ y results/figures/hybrid_baseline/dmodel_16/
- results/aggregate/hybrid_baseline/dmodel_64/ y results/figures/hybrid_baseline/dmodel_64/
- Genera además un gráfico comparativo consolidado (d_model ∈ [16, 32, 64, 96]) en results/figures/hybrid_baseline/00_dmodel_overfitting_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.experiments.hybrid_evaluation import run_hybrid_experiment
from src.training.plots import SERIE_1, SERIE_2, SERIE_3, SERIE_5, aplicar_estilo_cientifico

SEEDS = [7, 42, 123, 456, 999]


def run_experiment(
    exp_name: str,
    d_model: int,
    d_ff: int,
    seeds: Sequence[int] = SEEDS,
) -> None:
    print("\n" + "=" * 80)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}] (d_model={d_model}, d_ff={d_ff})")
    print("=" * 80)
    run_hybrid_experiment(
        exp_name=exp_name,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=1,
        n_heads=1,
        pos_encoding="sinusoidal",
        pooling="mean",
        epochs=15,
        no_early_stopping=True,
        seeds=seeds,
    )


def generate_consolidated_comparison(
    base_agg: Path = Path("results/aggregate/hybrid_baseline"),
    base_fig: Path = Path("results/figures/hybrid_baseline"),
) -> None:
    """Consolida d_model=16, 32, 64, 96 y genera el gráfico comparativo de overfitting."""
    exp_dirs = {
        16: base_agg / "dmodel_16",
        32: base_agg / "dmodel_32",
        64: base_agg / "dmodel_64",
        96: base_agg,  # el original d_model=96
    }

    resumenes = []
    for d, p in exp_dirs.items():
        summary_p = p / "hybrid_evaluation_summary.csv"
        if summary_p.exists():
            df = pd.read_csv(summary_p)
            df["d_model_eval"] = d
            resumenes.append(df.iloc[0])

    if len(resumenes) < 2:
        print("⚠️  No hay suficientes experimentos ejecutados para consolidar la comparativa de d_model.")
        return

    df_comp = pd.DataFrame(resumenes).sort_values("d_model_eval")
    base_agg.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(base_agg / "hybrid_dmodel_sweep_comparison.csv", index=False)

    aplicar_estilo_cientifico()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(df_comp))
    labels = [f"d={int(d)}\n({int(p):,} params)" for d, p in zip(df_comp["d_model_eval"], df_comp["params_total"])]

    # Panel 1: PR-AUC (Train vs Val vs Test) y Brecha de Overfitting
    ax1.errorbar(x, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
                 fmt="s--", color=SERIE_1, linewidth=2, capsize=4, label="Train PR-AUC")
    ax1.errorbar(x, df_comp["val_pr_auc_mean"], yerr=df_comp["val_pr_auc_std"],
                 fmt="o-", color=SERIE_2, linewidth=2, capsize=4, label="Val PR-AUC")
    ax1.errorbar(x, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
                 fmt="^-", color=SERIE_3, linewidth=2.2, capsize=4, label="Test PR-AUC")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Dimensión de Texto ($d_{\\text{model}}$)")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("PR-AUC y Brecha de Generalización (5 Semillas)")
    ax1.legend(loc="lower right")

    # Panel 2: BCE Loss y ROC-AUC en Test
    ax2.errorbar(x, df_comp["test_bce_mean"], yerr=df_comp["test_bce_std"],
                 fmt="s-", color=SERIE_5, linewidth=2, capsize=4, label="Test BCE Loss (eje izq)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Dimensión de Texto ($d_{\\text{model}}$)")
    ax2.set_ylabel("Test BCE Loss", color=SERIE_5)
    ax2.tick_params(axis="y", labelcolor=SERIE_5)

    ax2_twin = ax2.twinx()
    ax2_twin.errorbar(x, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                      fmt="o--", color=SERIE_3, linewidth=2, capsize=4, label="Test ROC-AUC (eje der)")
    ax2_twin.set_ylabel("Test ROC-AUC", color=SERIE_3)
    ax2_twin.tick_params(axis="y", labelcolor=SERIE_3)
    ax2_twin.grid(False)
    ax2.set_title("Test BCE Loss y Test ROC-AUC")

    fig.suptitle("Impacto de $d_{\\text{model}}$ en el Sobreajuste del Modelo Híbrido (15 Épocas)", fontsize=14)
    fig.tight_layout()

    out_fig = base_fig / "00_dmodel_overfitting_comparison.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo global guardado en: {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido de d_model para el modelo híbrido.")
    parser.add_argument("--summary_only", action="store_true", help="Solo genera el gráfico comparativo consolidado.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if not args.summary_only:
        run_experiment("dmodel_16", 16, 64, seeds=args.seeds)
        run_experiment("dmodel_64", 64, 256, seeds=args.seeds)

    generate_consolidated_comparison()


if __name__ == "__main__":
    main()
