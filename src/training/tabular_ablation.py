"""Estudio de ablación multi-semilla para la rama tabular.

Evalúa sistemáticamente el impacto de la dimensión de Entity Embeddings ($d_{\\text{emb}}$)
conectada directamente al MLP final clasificador (sin Transformer y sin MLP intermedio previo).

Genera métricas estrictas y limpias a través de 5 semillas aleatorias sin textos explicativos
ni adornos artificiales:
- `01_tabular_pr_auc_by_embedding_dim.png`: Train, Val y Test PR-AUC con barras de error (media ± desvío).
- `02_tabular_roc_auc_and_loss_by_embedding_dim.png`: Test ROC-AUC y Test BCE Loss con barras de error.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.hybrid_transformer.fusion import BTRModel, FusionConfig
from src.hybrid_transformer.tabular_encoder import TabularEncoder, TabularEncoderConfig
from src.training.dataset import build_dataloaders
from src.training.metrics import compute_extended_metrics
from src.training.trainer import Trainer, TrainerConfig, set_seed

OUTPUT_AGG_DIR = Path("results/aggregate")
OUTPUT_FIG_DIR = Path("results/figures/tabular_ablation")


def aplicar_estilo_cientifico() -> None:
    """Configuración estética limpia, sobria y estándar para papers/reportes."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 11,
        "font.size": 11,
        "figure.dpi": 150,
    })


def train_single_seed(
    variant_name: str,
    embedding_dim: Optional[int],
    seed: int,
    loaders: dict,
    artefactos: dict,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    patience: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Entrena una corrida puntual para una combinación de variante y semilla."""
    set_seed(seed)
    tab_config = TabularEncoderConfig(
        num_numeric=artefactos["num_numeric"],
        num_direct=artefactos["num_direct"],
        embedding_cardinalities=artefactos["embedding_cardinalities"],
        embedding_dim=embedding_dim,
        onehot_cardinalities=artefactos["onehot_cardinalities"],
        use_mlp=False,
        dropout=dropout,
    )
    tabular_encoder = TabularEncoder(tab_config)

    fusion_config = FusionConfig(
        mode="late",
        d_text=0,
        d_tab=tabular_encoder.output_dim,
        hidden_dims=[64],
        dropout=dropout,
        activation="gelu",
    )
    model = BTRModel(text_encoder=None, tabular_encoder=tabular_encoder, fusion_config=fusion_config)

    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            seed=seed,
            verbose=False,
        ),
    )
    trainer.fit(loaders["train"], loaders["val"])

    train_logits, train_targets = trainer.predict(loaders["train"])
    val_logits, val_targets = trainer.predict(loaders["val"])
    test_logits, test_targets = trainer.predict(loaders["test"])

    train_m = compute_extended_metrics(train_targets, train_logits, prefix="train_")
    val_m = compute_extended_metrics(val_targets, val_logits, prefix="val_")
    test_m = compute_extended_metrics(test_targets, test_logits, prefix="test_")

    return {
        "variant": variant_name,
        "seed": seed,
        "embedding_dim_label": f"d={embedding_dim}" if embedding_dim is not None else "auto (d~6)",
        "embedding_total_dim": tab_config.embedding_output_dim,
        "input_dim": tab_config.input_dim,
        "params_total": model.get_num_params(),
        "best_epoch": trainer.best_epoch,
        "train_pr_auc": train_m["train_pr_auc"],
        "train_roc_auc": train_m["train_roc_auc"],
        "train_bce": train_m["train_bce"],
        "val_pr_auc": val_m["val_pr_auc"],
        "val_roc_auc": val_m["val_roc_auc"],
        "val_bce": val_m["val_bce"],
        "test_pr_auc": test_m["test_pr_auc"],
        "test_roc_auc": test_m["test_roc_auc"],
        "test_bce": test_m["test_bce"],
    }


def plot_pr_auc_comparison(df_agg: pd.DataFrame, output_dir: Path = OUTPUT_FIG_DIR) -> Path:
    """Figura 1: Curvas de PR-AUC (Train, Val, Test) con barras de error reales."""
    aplicar_estilo_cientifico()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df_agg))
    labels = df_agg["embedding_dim_label"].tolist()

    # Curva Train
    ax.errorbar(
        x, df_agg["train_pr_auc_mean"], yerr=df_agg["train_pr_auc_std"],
        fmt="s--", color="#1f77b4", linewidth=2.0, capsize=4, capthick=1.2,
        label="Train PR-AUC",
    )

    # Curva Validación
    ax.errorbar(
        x, df_agg["val_pr_auc_mean"], yerr=df_agg["val_pr_auc_std"],
        fmt="o-", color="#ff7f0e", linewidth=2.2, capsize=4, capthick=1.2,
        label="Validation PR-AUC",
    )

    # Curva Test
    ax.errorbar(
        x, df_agg["test_pr_auc_mean"], yerr=df_agg["test_pr_auc_std"],
        fmt="^-", color="#2ca02c", linewidth=2.2, capsize=4, capthick=1.2,
        label="Test PR-AUC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Dimensión de Entity Embeddings ($d_{\\text{emb}}$)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC vs Dimensión de Embedding (Media ± Desvío Estándar, 5 Semillas)")
    ax.legend(loc="lower right")

    salida_path = output_dir / "01_tabular_pr_auc_by_embedding_dim.png"
    fig.savefig(salida_path, bbox_inches="tight")
    plt.close(fig)
    return salida_path


def plot_roc_and_loss_comparison(df_agg: pd.DataFrame, output_dir: Path = OUTPUT_FIG_DIR) -> Path:
    """Figura 2: Test ROC-AUC y Test BCE Loss con barras de error."""
    aplicar_estilo_cientifico()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(df_agg))
    labels = df_agg["embedding_dim_label"].tolist()

    # Panel 1: ROC-AUC
    ax1.errorbar(
        x, df_agg["test_roc_auc_mean"], yerr=df_agg["test_roc_auc_std"],
        fmt="o-", color="#2ca02c", linewidth=2.0, capsize=4, capthick=1.2,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_xlabel("Dimensión de Entity Embeddings ($d_{\\text{emb}}$)")
    ax1.set_ylabel("Test ROC-AUC")
    ax1.set_title("Test ROC-AUC")

    # Panel 2: BCE Loss
    ax2.errorbar(
        x, df_agg["test_bce_mean"], yerr=df_agg["test_bce_std"],
        fmt="s-", color="#d62728", linewidth=2.0, capsize=4, capthick=1.2,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_xlabel("Dimensión de Entity Embeddings ($d_{\\text{emb}}$)")
    ax2.set_ylabel("Test BCE Loss")
    ax2.set_title("Test Binary Cross-Entropy Loss")

    fig.suptitle("Evaluación en Test: ROC-AUC y Función de Pérdida (5 Semillas)", fontsize=14)
    fig.tight_layout()

    salida_path = output_dir / "02_tabular_roc_auc_and_loss_by_embedding_dim.png"
    fig.savefig(salida_path, bbox_inches="tight")
    plt.close(fig)
    return salida_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de ablación multi-semilla para la rama tabular.")
    parser.add_argument("--data_dir", type=str, default="resources/datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 456, 999])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plots_only", action="store_true", help="Solo regenera las figuras desde summary.csv.")
    args = parser.parse_args()

    if args.plots_only and (OUTPUT_AGG_DIR / "tabular_ablation_summary.csv").exists():
        print("🎨 Regenerando figuras desde resultados existentes...")
        df_agg = pd.read_csv(OUTPUT_AGG_DIR / "tabular_ablation_summary.csv")
        f1 = plot_pr_auc_comparison(df_agg)
        f2 = plot_roc_and_loss_comparison(df_agg)
        print(f"   ✓ {f1.name}")
        print(f"   ✓ {f2.name}")
        print("✅ Figuras actualizadas correctamente.")
        return

    print("=" * 88)
    print(f"🔬 ESTUDIO MULTI-SEMILLA TABULAR (5 Seeds: {args.seeds})")
    print("=" * 88)

    loaders, artefactos = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_text=False,
        use_tabular=True,
        seed=42,
    )

    variantes = [
        ("d_emb_2", 2),
        ("d_emb_4", 4),
        ("d_emb_auto", None),
        ("d_emb_8", 8),
        ("d_emb_16", 16),
        ("d_emb_32", 32),
    ]

    registros_totales = []
    for nombre, dim in variantes:
        label = f"d={dim}" if dim is not None else "auto"
        print(f"📦 Variante [{nombre}] ({label})...")
        for s in args.seeds:
            res = train_single_seed(
                variant_name=nombre,
                embedding_dim=dim,
                seed=s,
                loaders=loaders,
                artefactos=artefactos,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                patience=args.patience,
                device=args.device,
            )
            registros_totales.append(res)

    df_all = pd.DataFrame(registros_totales)

    # Limpiar figuras previas
    if OUTPUT_FIG_DIR.exists():
        shutil.rmtree(OUTPUT_FIG_DIR)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AGG_DIR.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(OUTPUT_AGG_DIR / "tabular_ablation_all_seeds.csv", index=False)

    agregados = []
    for (var, label), sub in df_all.groupby(["variant", "embedding_dim_label"], sort=False):
        agregados.append({
            "variant": var,
            "embedding_dim_label": label,
            "embedding_total_dim": int(sub["embedding_total_dim"].iloc[0]),
            "input_dim": int(sub["input_dim"].iloc[0]),
            "params_total": int(sub["params_total"].iloc[0]),
            "train_pr_auc_mean": float(sub["train_pr_auc"].mean()),
            "train_pr_auc_std": float(sub["train_pr_auc"].std()),
            "val_pr_auc_mean": float(sub["val_pr_auc"].mean()),
            "val_pr_auc_std": float(sub["val_pr_auc"].std()),
            "test_pr_auc_mean": float(sub["test_pr_auc"].mean()),
            "test_pr_auc_std": float(sub["test_pr_auc"].std()),
            "test_roc_auc_mean": float(sub["test_roc_auc"].mean()),
            "test_roc_auc_std": float(sub["test_roc_auc"].std()),
            "test_bce_mean": float(sub["test_bce"].mean()),
            "test_bce_std": float(sub["test_bce"].std()),
        })

    df_agg = pd.DataFrame(agregados)
    df_agg.to_csv(OUTPUT_AGG_DIR / "tabular_ablation_summary.csv", index=False)
    (OUTPUT_AGG_DIR / "tabular_ablation_summary.json").write_text(
        json.dumps(agregados, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("📊 RESULTADOS FINALES (Media ± Desvío Estándar en 5 Semillas)")
    print("=" * 88)
    columnas_tabla = [
        "embedding_dim_label", "input_dim", "params_total",
        "val_pr_auc_mean", "val_pr_auc_std", "test_pr_auc_mean", "test_pr_auc_std",
        "test_roc_auc_mean", "test_bce_mean"
    ]
    print(df_agg[columnas_tabla].to_string(index=False))

    print("\n🎨 Guardando figuras...")
    f1 = plot_pr_auc_comparison(df_agg)
    f2 = plot_roc_and_loss_comparison(df_agg)

    print(f"   ✓ {f1.name}")
    print(f"   ✓ {f2.name}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
