"""Script de barrido de regularización anti-overfitting para el modelo híbrido.

Prueba configuraciones con mayor regularización (dropout=0.3/0.4, lr=3e-4, weight_decay=0.05/0.10)
y genera gráficos aislados y una comparativa directa con el modelo sin regularizar.
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

def run_experiment(exp_name: str, dropout: float, lr: float, weight_decay: float) -> None:
    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO REGULARIZACIÓN [{exp_name}] (dropout={dropout}, lr={lr}, wd={weight_decay})")
    print("=" * 88)
    cmd = [
        "uv", "run", "python", "-m", "src.training.hybrid_evaluation",
        "--exp_name", exp_name,
        "--d_model", "32",
        "--d_ff", "128",
        "--num_layers", "1",
        "--n_heads", "1",
        "--pos_encoding", "sinusoidal",
        "--pooling", "mean",
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--epochs", "15",
        "--no_early_stopping",
        "--seeds", *[str(s) for s in SEEDS]
    ]
    subprocess.run(cmd, check=True)

def generate_regularization_comparison() -> None:
    """Genera una figura comparativa entre el modelo base y los modelos regularizados."""
    base_agg = Path("results/aggregate/hybrid_baseline")
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
        return

    df_comp = pd.DataFrame(resumenes)
    df_comp.to_csv(base_agg / "hybrid_regularization_comparison.csv", index=False)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "font.size": 11,
        "figure.dpi": 150,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Curva de Dinámica de Pérdida en Validación (Val Loss por Época)
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
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
    ax2.bar(x - w/2, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
            width=w, capsize=4, color="#1f77b4", label="Train PR-AUC (Memorización)")
    ax2.bar(x + w/2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color="#2ca02c", label="Test PR-AUC (Generalización)")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["config_label"])
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Brecha de Generalización $\\Delta(\\text{Train} - \\text{Test})$")
    ax2.legend(loc="lower right")

    # Panel 3: Test ROC-AUC y Test BCE Loss
    ax3.errorbar(x, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                 fmt="s-", color="#2ca02c", linewidth=2.2, capsize=4, label="Test ROC-AUC")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comp["config_label"])
    ax3.set_ylabel("Test ROC-AUC", color="#2ca02c")
    ax3.tick_params(axis="y", labelcolor="#2ca02c")
    ax3.set_ylim([0.95, 1.0])

    ax3_twin = ax3.twinx()
    ax3_twin.errorbar(x, df_comp["test_bce_mean"], yerr=df_comp["test_bce_std"],
                      fmt="o--", color="#d62728", linewidth=2.2, capsize=4, label="Test BCE Loss")
    ax3_twin.set_ylabel("Test BCE Loss", color="#d62728")
    ax3_twin.tick_params(axis="y", labelcolor="#d62728")
    ax3_twin.grid(False)
    ax3.set_title("Test ROC-AUC y Test BCE Loss")

    fig.suptitle("Impacto de la Regularización Anti-Overfitting en el Modelo Híbrido ($d_{\\text{model}}=32$)", fontsize=14)
    fig.tight_layout()

    out_fig = Path("results/figures/hybrid_baseline/00_regularization_overfitting_comparison.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo de regularización guardado en: {out_fig}")

if __name__ == "__main__":
    run_experiment("reg_dropout03_lr3e4", dropout=0.3, lr=3e-4, weight_decay=0.05)
    run_experiment("reg_dropout04_lr3e4", dropout=0.4, lr=3e-4, weight_decay=0.10)
    generate_regularization_comparison()
