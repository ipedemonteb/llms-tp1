"""Script de barrido de regularización anti-overfitting para el modelo híbrido.

Prueba configuraciones con mayor regularización (dropout=0.3/0.4, lr=3e-4, weight_decay=0.05/0.10)
y genera gráficos aislados y una comparativa directa con el modelo sin regularizar.
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
    dropout: float,
    lr: float,
    weight_decay: float,
    seeds: Sequence[int] = SEEDS,
) -> None:
    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO REGULARIZACIÓN [{exp_name}] (dropout={dropout}, lr={lr}, wd={weight_decay})")
    print("=" * 88)
    run_hybrid_experiment(
        exp_name=exp_name,
        d_model=32,
        d_ff=128,
        num_layers=1,
        n_heads=1,
        pos_encoding="sinusoidal",
        pooling="mean",
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=15,
        no_early_stopping=True,
        seeds=seeds,
    )


def generate_regularization_comparison(
    base_agg: Path = Path("results/aggregate/hybrid_baseline"),
    base_fig: Path = Path("results/figures/hybrid_baseline"),
) -> None:
    """Genera una figura comparativa entre el modelo base y los modelos regularizados."""
    configs = {
        "Base (Sin Regularizar)\ndr=0.1, lr=1e-3, wd=0.01": base_agg / "dmodel_32",
        "Regularización Media\ndr=0.3, lr=3e-4, wd=0.05": base_agg / "reg_dropout03_lr3e4",
        "Regularización Fuerte\ndr=0.4, lr=3e-4, wd=0.10": base_agg / "reg_dropout04_lr3e4",
    }

    resumenes = []
    histories_by_config = {}

    for nombre, p in configs.items():
        summary_p = p / "hybrid_evaluation_summary.csv"
        hist_p = p / "hybrid_evaluation_histories.json"
        if summary_p.exists():
            df = pd.read_csv(summary_p)
            df["config_label"] = nombre
            resumenes.append(df.iloc[0])
        if hist_p.exists():
            histories_by_config[nombre] = json.loads(hist_p.read_text(encoding="utf-8"))

    if len(resumenes) < 2:
        print("⚠️  No hay suficientes experimentos ejecutados para consolidar la comparativa de regularización.")
        return

    df_comp = pd.DataFrame(resumenes)
    base_agg.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(base_agg / "hybrid_regularization_comparison.csv", index=False)

    aplicar_estilo_cientifico()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Curva de Dinámica de Pérdida en Validación (Val Loss por Época)
    colors = [SERIE_5, SERIE_1, SERIE_3]
    for (nombre, all_hists), color in zip(histories_by_config.items(), colors):
        filas = []
        for s_idx, h in enumerate(all_hists):
            for row in h:
                filas.append({"epoch": row["epoch"], "val_loss": row["val_loss"]})
        df_ep = pd.DataFrame(filas)
        agg = df_ep.groupby("epoch")["val_loss"].agg(["mean", "std"]).reset_index()
        ep = agg["epoch"]
        vl_mean = agg["mean"]
        vl_std = agg["std"].fillna(0)

        short_name = nombre.split("\n")[0]
        ax1.plot(ep, vl_mean, "o-", color=color, linewidth=2.0, label=short_name)
        ax1.fill_between(ep, vl_mean - vl_std, vl_mean + vl_std, color=color, alpha=0.15)

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Validation BCE Loss")
    ax1.set_title("Evolución de Val Loss (Media ± $\\sigma$)")
    ax1.legend(loc="upper left")

    # Panel 2: Brecha de Generalización PR-AUC (Train vs Test)
    x = np.arange(len(df_comp))
    w = 0.30
    ax2.bar(x - w / 2, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
            width=w, capsize=4, color=SERIE_1, label="Train PR-AUC (Memorización)")
    ax2.bar(x + w / 2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Generalización)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["config_label"])
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Brecha de Generalización $\\Delta(\\text{Train} - \\text{Test})$")
    ax2.legend(loc="lower right")

    # Panel 3: Test ROC-AUC y Test BCE Loss
    ax3.errorbar(x, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                 fmt="s-", color=SERIE_3, linewidth=2.2, capsize=4, label="Test ROC-AUC")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comp["config_label"])
    ax3.set_ylabel("Test ROC-AUC", color=SERIE_3)
    ax3.tick_params(axis="y", labelcolor=SERIE_3)
    ax3.set_ylim([0.95, 1.0])

    ax3_twin = ax3.twinx()
    ax3_twin.errorbar(x, df_comp["test_bce_mean"], yerr=df_comp["test_bce_std"],
                      fmt="o--", color=SERIE_5, linewidth=2.2, capsize=4, label="Test BCE Loss")
    ax3_twin.set_ylabel("Test BCE Loss", color=SERIE_5)
    ax3_twin.tick_params(axis="y", labelcolor=SERIE_5)
    ax3_twin.grid(False)
    ax3.set_title("Test ROC-AUC y Test BCE Loss")

    fig.suptitle("Impacto de la Regularización Anti-Overfitting en el Modelo Híbrido ($d_{\\text{model}}=32$)", fontsize=14)
    fig.tight_layout()

    out_fig = base_fig / "00_regularization_overfitting_comparison.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo de regularización guardado en: {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido de regularización para el modelo híbrido.")
    parser.add_argument("--summary_only", action="store_true", help="Solo genera el gráfico comparativo consolidado.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if not args.summary_only:
        run_experiment("reg_dropout03_lr3e4", dropout=0.3, lr=3e-4, weight_decay=0.05, seeds=args.seeds)
        run_experiment("reg_dropout04_lr3e4", dropout=0.4, lr=3e-4, weight_decay=0.10, seeds=args.seeds)

    generate_regularization_comparison()


if __name__ == "__main__":
    main()
