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
from src.training.plots import SERIE_1, SERIE_2, SERIE_3, SERIE_4, SERIE_5, aplicar_estilo_cientifico

SEEDS = [7, 42, 123, 456, 999]


def run_experiment(
    exp_name: str,
    fusion_mode: str,
    dropout: float,
    lr: float,
    weight_decay: float,
    seeds: Sequence[int] = SEEDS,
) -> None:
    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}] (fusion={fusion_mode}, dropout={dropout}, lr={lr}, wd={weight_decay})")
    print("=" * 88)
    run_hybrid_experiment(
        exp_name=exp_name,
        fusion_mode=fusion_mode,
        fusion_heads=1,
        d_model=32,
        d_ff=128,
        num_layers=1,
        n_heads=1,
        pos_encoding="sinusoidal",
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=15,
        no_early_stopping=True,
        seeds=seeds,
    )


def generate_comparative_summary(
    base_agg: Path = Path("results/aggregate/hybrid_baseline"),
    base_fig: Path = Path("results/figures/hybrid_baseline"),
) -> None:
    """Genera la figura de comparación directa entre Late Fusion y Cross-Attention."""
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
        print("⚠️  No hay suficientes experimentos ejecutados para generar la comparativa de Cross-Attention.")
        return

    df_comp = pd.DataFrame(resumenes)
    base_agg.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(base_agg / "cross_attention_vs_late_fusion_comparison.csv", index=False)

    aplicar_estilo_cientifico()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.2))

    # Panel 1: Dinámica de Validación (Val BCE Loss a lo largo de las 15 épocas)
    colors = [SERIE_5, SERIE_2, SERIE_1, SERIE_3]
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
    ax2.bar(x - w / 2, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
            width=w, capsize=4, color=SERIE_1, label="Train PR-AUC (Memorización)")
    ax2.bar(x + w / 2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Generalización)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["config_label"], fontsize=9)
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Brecha de Generalización $\\Delta(\\text{Train} - \\text{Test})$")
    ax2.legend(loc="lower right")

    # Panel 3: Test PR-AUC y Test ROC-AUC
    ax3.bar(x - w / 2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Primaria)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comp["config_label"], fontsize=9)
    ax3.set_ylabel("Test PR-AUC", color=SERIE_3)
    ax3.set_ylim([0.65, 0.76])

    ax3_twin = ax3.twinx()
    ax3_twin.errorbar(x + w / 2, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                      fmt="s--", color=SERIE_1, linewidth=2.2, capsize=4, label="Test ROC-AUC (Secundaria)")
    ax3_twin.set_ylabel("Test ROC-AUC", color=SERIE_1)
    ax3_twin.set_ylim([0.95, 1.0])
    ax3_twin.grid(False)
    ax3.set_title("Rendimiento Final en Test (PR-AUC y ROC-AUC)")

    fig.suptitle("Comparativa Exhaustiva: Late Fusion (Mean Pooling) vs. Cross-Attention (Sin Pooling)", fontsize=14)
    fig.tight_layout()

    out_fig = base_fig / "00_cross_attention_vs_late_fusion_comparison.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🎨 Gráfico comparativo de Cross-Attention guardado en: {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de Cross-Attention vs Late Fusion.")
    parser.add_argument("--summary_only", action="store_true", help="Solo genera el gráfico comparativo consolidado.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if not args.summary_only:
        # 1. Cross-Attention Base
        run_experiment("cross_attention", fusion_mode="cross", dropout=0.1, lr=1e-3, weight_decay=0.01, seeds=args.seeds)
        # 2. Cross-Attention Regularizado
        run_experiment("cross_attention_reg", fusion_mode="cross", dropout=0.3, lr=3e-4, weight_decay=0.05, seeds=args.seeds)

    # 3. Consolidar comparativa
    generate_comparative_summary()


if __name__ == "__main__":
    main()
