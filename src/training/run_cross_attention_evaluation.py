"""Script de evaluación exhaustiva de Cross-Attention vs Late Fusion.

Ejecuta el modelo con Cross-Attention donde el vector tabular actúa como Query
sobre la secuencia completa de tokens de texto (eliminando el mean pooling).

Compara:
1. Late Fusion (Mean Pooling) Base
2. Late Fusion (Mean Pooling) Regularizado (dr=0.3, lr=3e-4, wd=0.05)
3. Cross-Attention (Sin Pooling) Base (dr=0.1, lr=1e-3, wd=0.01)
4. Cross-Attention (Sin Pooling) Regularizado (dr=0.3, lr=3e-4, wd=0.05)

Genera figuras aisladas de 3 paneles y una comparativa global consolidada.
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

def run_experiment(exp_name: str, fusion_mode: str, dropout: float, lr: float, weight_decay: float) -> None:
    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}] (fusion={fusion_mode}, dropout={dropout}, lr={lr}, wd={weight_decay})")
    print("=" * 88)
    cmd = [
        "uv", "run", "python", "-m", "src.training.hybrid_evaluation",
        "--exp_name", exp_name,
        "--fusion_mode", fusion_mode,
        "--fusion_heads", "1",
        "--d_model", "32",
        "--d_ff", "128",
        "--num_layers", "1",
        "--n_heads", "1",
        "--pos_encoding", "sinusoidal",
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--epochs", "15",
        "--no_early_stopping",
        "--seeds", *[str(s) for s in SEEDS]
    ]
    subprocess.run(cmd, check=True)

def generate_comparative_summary() -> None:
    """Genera la figura de comparación directa entre Late Fusion y Cross-Attention."""
    base_agg = Path("results/aggregate/hybrid_baseline")
    configs = {
        "Late Fusion (Mean Pooling)\nBase (dr=0.1, lr=1e-3)": base_agg / "dmodel_32",
        "Late Fusion (Mean Pooling)\nReg (dr=0.3, lr=3e-4)": base_agg / "reg_dropout03_lr3e4",
        "Cross-Attention (Sin Pooling)\nBase (dr=0.1, lr=1e-3)": base_agg / "cross_attention",
        "Cross-Attention (Sin Pooling)\nReg (dr=0.3, lr=3e-4)": base_agg / "cross_attention_reg",
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
    df_comp.to_csv(base_agg / "cross_attention_vs_late_fusion_comparison.csv", index=False)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "font.size": 11,
        "figure.dpi": 150,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.2))

    # Panel 1: Dinámica de Validación (Val BCE Loss a lo largo de las 15 épocas)
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
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
        
        short_name = nombre.split("\n")[0] + " - " + nombre.split("\n")[1].split(" ")[0]
        ax1.plot(ep, vl_mean, "o-", color=color, linewidth=2.0, label=short_name)
        ax1.fill_between(ep, vl_mean - vl_std, vl_mean + vl_std, color=color, alpha=0.12)

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Validation BCE Loss")
    ax1.set_title("Dinámica de Pérdida en Val (Media ± $\\sigma$)")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: PR-AUC (Train vs Test) y Brecha de Generalización
    x = np.arange(len(df_comp))
    w = 0.35
    ax2.bar(x - w/2, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
            width=w, capsize=4, color="#1f77b4", label="Train PR-AUC (Memorización)")
    ax2.bar(x + w/2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color="#2ca02c", label="Test PR-AUC (Generalización)")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["config_label"], fontsize=9)
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Brecha de Generalización $\\Delta(\\text{Train} - \\text{Test})$")
    ax2.legend(loc="lower right")

    # Panel 3: Test PR-AUC y Test ROC-AUC
    ax3.bar(x - w/2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color="#2ca02c", label="Test PR-AUC (Primaria)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comp["config_label"], fontsize=9)
    ax3.set_ylabel("Test PR-AUC", color="#2ca02c")
    ax3.set_ylim([0.65, 0.76])

    ax3_twin = ax3.twinx()
    ax3_twin.errorbar(x + w/2, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                      fmt="s--", color="#1f77b4", linewidth=2.2, capsize=4, label="Test ROC-AUC (Secundaria)")
    ax3_twin.set_ylabel("Test ROC-AUC", color="#1f77b4")
    ax3_twin.set_ylim([0.95, 1.0])
    ax3_twin.grid(False)
    ax3.set_title("Rendimiento Final en Test (PR-AUC y ROC-AUC)")

    fig.suptitle("Comparativa Exhaustiva: Late Fusion (Mean Pooling) vs. Cross-Attention (Sin Pooling)", fontsize=14)
    fig.tight_layout()

    out_fig = Path("results/figures/hybrid_baseline/00_cross_attention_vs_late_fusion_comparison.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo de Cross-Attention guardado en: {out_fig}")

if __name__ == "__main__":
    # 1. Cross-Attention Base
    run_experiment("cross_attention", fusion_mode="cross", dropout=0.1, lr=1e-3, weight_decay=0.01)
    # 2. Cross-Attention Regularizado
    run_experiment("cross_attention_reg", fusion_mode="cross", dropout=0.3, lr=3e-4, weight_decay=0.05)
    # 3. Consolidar comparativa
    generate_comparative_summary()
