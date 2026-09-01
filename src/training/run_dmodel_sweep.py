"""Script de barrido de d_model para el modelo híbrido simple.

Ejecuta d_model=16 y d_model=64 de forma secuencial aislando los resultados en:
- results/aggregate/hybrid_baseline/dmodel_16/ y results/figures/hybrid_baseline/dmodel_16/
- results/aggregate/hybrid_baseline/dmodel_64/ y results/figures/hybrid_baseline/dmodel_64/
- Genera además un gráfico comparativo consolidado (d_model ∈ [16, 32, 64, 96]) en results/figures/hybrid_baseline/00_dmodel_overfitting_comparison.png
"""

import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [7, 42, 123, 456, 999]

def run_experiment(exp_name: str, d_model: int, d_ff: int) -> None:
    print("\n" + "=" * 80)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}] (d_model={d_model}, d_ff={d_ff})")
    print("=" * 80)
    cmd = [
        "uv", "run", "python", "-m", "src.training.hybrid_evaluation",
        "--exp_name", exp_name,
        "--d_model", str(d_model),
        "--d_ff", str(d_ff),
        "--num_layers", "1",
        "--n_heads", "1",
        "--pos_encoding", "sinusoidal",
        "--pooling", "mean",
        "--epochs", "15",
        "--no_early_stopping",
        "--seeds", *[str(s) for s in SEEDS]
    ]
    subprocess.run(cmd, check=True)

def generate_consolidated_comparison() -> None:
    """Consolida d_model=16, 32, 64, 96 y genera el gráfico comparativo de overfitting."""
    base_agg = Path("results/aggregate/hybrid_baseline")
    exp_dirs = {
        16: base_agg / "dmodel_16",
        32: base_agg / "dmodel_32",
        64: base_agg / "dmodel_64",
        96: base_agg, # el original d_model=96
    }

    resumenes = []
    for d, p in exp_dirs.items():
        summary_p = p / "hybrid_evaluation_summary.csv"
        if summary_p.exists():
            df = pd.read_csv(summary_p)
            df["d_model_eval"] = d
            resumenes.append(df.iloc[0])

    if len(resumenes) < 2:
        return

    df_comp = pd.DataFrame(resumenes).sort_values("d_model_eval")
    df_comp.to_csv(base_agg / "hybrid_dmodel_sweep_comparison.csv", index=False)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "font.size": 11,
        "figure.dpi": 150,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(df_comp))
    labels = [f"d={int(d)}\n({int(p):,} params)" for d, p in zip(df_comp["d_model_eval"], df_comp["params_total"])]

    # Panel 1: PR-AUC (Train vs Val vs Test) y Brecha de Overfitting
    ax1.errorbar(x, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
                 fmt="s--", color="#1f77b4", linewidth=2, capsize=4, label="Train PR-AUC")
    ax1.errorbar(x, df_comp["val_pr_auc_mean"], yerr=df_comp["val_pr_auc_std"],
                 fmt="o-", color="#ff7f0e", linewidth=2, capsize=4, label="Val PR-AUC")
    ax1.errorbar(x, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
                 fmt="^-", color="#2ca02c", linewidth=2.2, capsize=4, label="Test PR-AUC")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Dimensión de Texto ($d_{\\text{model}}$)")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("PR-AUC y Brecha de Generalización (5 Semillas)")
    ax1.legend(loc="lower right")

    # Panel 2: BCE Loss y ROC-AUC en Test
    ax2.errorbar(x, df_comp["test_bce_mean"], yerr=df_comp["test_bce_std"],
                 fmt="s-", color="#d62728", linewidth=2, capsize=4, label="Test BCE Loss (eje izq)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Dimensión de Texto ($d_{\\text{model}}$)")
    ax2.set_ylabel("Test BCE Loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax2_twin = ax2.twinx()
    ax2_twin.errorbar(x, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                      fmt="o--", color="#2ca02c", linewidth=2, capsize=4, label="Test ROC-AUC (eje der)")
    ax2_twin.set_ylabel("Test ROC-AUC", color="#2ca02c")
    ax2_twin.tick_params(axis="y", labelcolor="#2ca02c")
    ax2_twin.grid(False)
    ax2.set_title("Test BCE Loss y Test ROC-AUC")

    fig.suptitle("Impacto de $d_{\\text{model}}$ en el Sobreajuste del Modelo Híbrido (15 Épocas)", fontsize=14)
    fig.tight_layout()

    out_fig = Path("results/figures/hybrid_baseline/00_dmodel_overfitting_comparison.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo global guardado en: {out_fig}")

if __name__ == "__main__":
    run_experiment("dmodel_16", 16, 64)
    run_experiment("dmodel_64", 64, 256)
    generate_consolidated_comparison()
