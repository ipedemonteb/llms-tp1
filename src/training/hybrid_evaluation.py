"""Evaluación del Modelo Híbrido Simple (Transformer + Tabular Directo).

Arquitectura evaluada:
- Rama de Texto: Transformer Encoder (d_model, d_ff, L capas, H cabezales, pos_encoding, pooling).
- Rama Tabular: Entity Embeddings + Numéricas directas (SIN MLP tabular previo).
- Fusión: Concatenación simple [e_text || e_tab] sin mecanismos complejos.
- Clasificador: ClassifierHead (Linear(d_text + d_tab -> 64) -> GELU -> Dropout(0.1) -> Linear(64 -> 1)).

Todos los hiperparámetros se reciben explícitamente por línea de comandos / parámetros.
Evalúa estabilidad multi-semilla y guarda métricas completas en CSV, JSON y figuras.

Uso:
    uv run python -m src.training.hybrid_evaluation \
        --d_model 96 \
        --d_ff 384 \
        --num_layers 1 \
        --n_heads 1 \
        --pos_encoding sinusoidal \
        --pooling mean \
        --seeds 7 42 123 456 999
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
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder
from src.training.dataset import build_dataloaders
from src.training.metrics import compute_extended_metrics
from src.training.trainer import Trainer, TrainerConfig, set_seed

OUTPUT_AGG_DIR = Path("results/aggregate/hybrid_baseline")
OUTPUT_FIG_DIR = Path("results/figures/hybrid_baseline")


def aplicar_estilo_cientifico() -> None:
    """Configuración estética limpia y sobria para figuras científicas."""
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
        "legend.fontsize": 10.5,
        "font.size": 11,
        "figure.dpi": 150,
    })


def run_single_seed(
    seed: int,
    loaders: dict,
    artefactos: dict,
    d_model: int,
    d_ff: int,
    num_layers: int,
    n_heads: int,
    pos_encoding: str,
    pooling: str,
    embedding_dim: Optional[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: Optional[int],
    restore_best: bool,
    fusion_mode: str,
    fusion_heads: int,
    device: Optional[str],
) -> Dict[str, Any]:
    """Entrena una corrida puntual del modelo híbrido con una semilla específica."""
    set_seed(seed)

    # 1. Rama de Texto (Transformer)
    text_config = TextTransformerConfig(
        vocab_size=artefactos["vocab_size"],
        max_seq_len=128,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        pos_encoding_type=pos_encoding,
        pooling_mode=pooling if fusion_mode != "cross" else "none",
        pad_token_id=artefactos["pad_token_id"],
    )
    text_encoder = TextTransformerEncoder(text_config)

    # 2. Rama Tabular (Sin MLP, directo)
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

    # 3. Fusión (Late Fusion o Cross-Attention) + Clasificador final
    fusion_config = FusionConfig(
        mode=fusion_mode,
        d_text=text_encoder.config.d_model,
        d_tab=tabular_encoder.output_dim,
        n_heads=fusion_heads,
        hidden_dims=[64],
        dropout=dropout,
        activation="gelu",
    )
    model = BTRModel(text_encoder=text_encoder, tabular_encoder=tabular_encoder, fusion_config=fusion_config)
    params_breakdown = model.param_breakdown()

    # 4. Entrenamiento
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            restore_best=restore_best,
            device=device,
            seed=seed,
            verbose=False,
        ),
    )
    trainer.fit(loaders["train"], loaders["val"])

    # 5. Evaluación de métricas completas
    train_logits, train_targets = trainer.predict(loaders["train"])
    val_logits, val_targets = trainer.predict(loaders["val"])
    test_logits, test_targets = trainer.predict(loaders["test"])

    train_m = compute_extended_metrics(train_targets, train_logits, prefix="train_")
    val_m = compute_extended_metrics(val_targets, val_logits, prefix="val_")
    test_m = compute_extended_metrics(test_targets, test_logits, prefix="test_")

    return {
        "seed": seed,
        "d_model": d_model,
        "d_tab": tabular_encoder.output_dim,
        "fused_dim": fusion_config.fused_dim,
        "fusion_mode": fusion_mode,
        "params_text": params_breakdown["texto"],
        "params_tabular": params_breakdown["tabular"],
        "params_cross": params_breakdown.get("cross_attention", 0),
        "params_head": params_breakdown["cabeza"],
        "params_total": params_breakdown["total"],
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
        "test_brier": test_m.get("test_brier_score", float("nan")),
        "test_lift": test_m.get("test_lift", float("nan")),
        "test_best_f1": test_m.get("test_best_f1", float("nan")),
        "history": trainer.history,
    }


def plot_hybrid_results(df_runs: pd.DataFrame, all_histories: Any, output_dir: Path, filename: str = "01_hybrid_simple_evaluation.png", title_suffix: str = "") -> Path:
    """Genera una figura limpia de 3 paneles con las 3 métricas de la consigna (BCE Loss, PR-AUC y ROC-AUC)."""
    aplicar_estilo_cientifico()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.0))

    # Panel 1: Curva de Pérdida Agregada (Media ± Desvío Estándar en 5 semillas por época)
    if all_histories:
        # Asegurar formato lista de listas de diccionarios
        if isinstance(all_histories, dict):
            hist_list = list(all_histories.values())
        elif isinstance(all_histories, list) and len(all_histories) > 0 and isinstance(all_histories[0], dict):
            hist_list = [all_histories]
        else:
            hist_list = all_histories

        filas_epocas = []
        for s_idx, h in enumerate(hist_list):
            for row in h:
                filas_epocas.append({
                    "seed_idx": s_idx,
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                })
        df_ep = pd.DataFrame(filas_epocas)
        agg_ep = df_ep.groupby("epoch").agg({
            "train_loss": ["mean", "std"],
            "val_loss": ["mean", "std"],
        }).reset_index()

        ep = agg_ep["epoch"]
        tr_mean = agg_ep["train_loss"]["mean"]
        tr_std = agg_ep["train_loss"]["std"].fillna(0)
        vl_mean = agg_ep["val_loss"]["mean"]
        vl_std = agg_ep["val_loss"]["std"].fillna(0)

        # Curva Train con banda de error
        ax1.plot(ep, tr_mean, "o--", color="#1f77b4", linewidth=2.0, label="Train Loss (media)")
        ax1.fill_between(ep, tr_mean - tr_std, tr_mean + tr_std, color="#1f77b4", alpha=0.2)

        # Curva Val con banda de error
        ax1.plot(ep, vl_mean, "s-", color="#d62728", linewidth=2.2, label="Val Loss (media)")
        ax1.fill_between(ep, vl_mean - vl_std, vl_mean + vl_std, color="#d62728", alpha=0.2)

        ax1.set_xlabel("Época")
        ax1.set_ylabel("BCE Loss")
        ax1.set_title("Evolución de Pérdida (BCE Loss, $\\mu \\pm \\sigma$, 5 Semillas)")
        ax1.legend(loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Historial no disponible", ha="center", va="center")

    x = np.arange(len(df_runs))
    seeds = [f"Seed {s}" for s in df_runs["seed"]]
    w = 0.25

    # Panel 2: PR-AUC por Semilla (Train vs Val vs Test) - Métrica Primaria
    ax2.bar(x - w, df_runs["train_pr_auc"], width=w, color="#1f77b4", label="Train PR-AUC")
    ax2.bar(x, df_runs["val_pr_auc"], width=w, color="#ff7f0e", label="Val PR-AUC")
    ax2.bar(x + w, df_runs["test_pr_auc"], width=w, color="#2ca02c", label="Test PR-AUC")
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds)
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("PR-AUC por Semilla (Métrica Primaria)")
    ax2.legend(loc="lower right")

    # Panel 3: ROC-AUC por Semilla (Train vs Val vs Test) - Métrica Secundaria
    ax3.bar(x - w, df_runs["train_roc_auc"], width=w, color="#1f77b4", label="Train ROC-AUC")
    ax3.bar(x, df_runs["val_roc_auc"], width=w, color="#ff7f0e", label="Val ROC-AUC")
    ax3.bar(x + w, df_runs["test_roc_auc"], width=w, color="#2ca02c", label="Test ROC-AUC")
    ax3.set_xticks(x)
    ax3.set_xticklabels(seeds)
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title("ROC-AUC por Semilla (Métrica Secundaria)")
    ax3.set_ylim([0.90, 1.0])
    ax3.legend(loc="lower right")

    titulo = f"Evaluación del Modelo Híbrido {title_suffix}".strip()
    fig.suptitle(titulo if titulo else "Evaluación Completa del Modelo Híbrido", fontsize=14)
    fig.tight_layout()

    salida_path = output_dir / filename
    fig.savefig(salida_path, bbox_inches="tight")
    plt.close(fig)
    return salida_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del Modelo Híbrido Simple (Transformer + Tabular Directo).")
    parser.add_argument("--exp_name", type=str, default=None, help="Nombre del experimento para aislar salidas.")
    parser.add_argument("--d_model", type=int, default=96, help="Dimensión latente del Transformer.")
    parser.add_argument("--d_ff", type=int, default=384, help="Dimensión FFN del Transformer.")
    parser.add_argument("--num_layers", type=int, default=1, help="Cantidad de capas del Transformer.")
    parser.add_argument("--n_heads", type=int, default=1, help="Cabezales de atención.")
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal", help="Tipo de codificación posicional.")
    parser.add_argument("--pooling", type=str, default="mean", help="Estrategia de pooling de texto.")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Dimensión de embeddings tabulares (None=auto).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 456, 999],
                        help="Semillas aleatorias para evaluación estadística.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_early_stopping", action="store_true", help="Desactiva early stopping y corre todas las épocas sin restaurar.")
    parser.add_argument("--fusion_mode", type=str, default="late", choices=["late", "cross"], help="Estrategia de fusión: late o cross.")
    parser.add_argument("--fusion_heads", type=int, default=1, help="Cabezales de cross-attention.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plots_only", action="store_true", help="Solo regenera las figuras desde los CSVs.")
    args = parser.parse_args()

    agg_dir = OUTPUT_AGG_DIR if args.exp_name is None else (OUTPUT_AGG_DIR / args.exp_name)
    fig_dir = OUTPUT_FIG_DIR if args.exp_name is None else (OUTPUT_FIG_DIR / args.exp_name)
    fig_filename = "01_hybrid_simple_evaluation.png" if args.exp_name is None else f"01_hybrid_{args.exp_name}_evaluation.png"
    title_suffix = f"({args.exp_name})" if args.exp_name else "(5 Semillas)"

    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds_csv = agg_dir / "hybrid_evaluation_seeds.csv"
    history_json = agg_dir / "hybrid_evaluation_histories.json"
    if not history_json.exists():
        history_json = agg_dir / "hybrid_evaluation_history.json"

    if args.plots_only and seeds_csv.exists():
        print(f"🎨 Regenerando figura en {fig_dir}...")
        df_runs = pd.read_csv(seeds_csv)
        hist = json.loads(history_json.read_text(encoding="utf-8")) if history_json.exists() else None
        fig_path = plot_hybrid_results(df_runs, hist, fig_dir, filename=fig_filename, title_suffix=title_suffix)
        print(f"   ✓ {fig_path.name}")
        print("✅ Figura actualizada correctamente con ROC-AUC, PR-AUC y BCE Loss.")
        return

    patience = None if args.no_early_stopping else args.patience
    restore_best = False if args.no_early_stopping else True

    print("=" * 88)
    print(f"🔬 EVALUACIÓN DEL MODELO HÍBRIDO [{'CROSS-ATTENTION' if args.fusion_mode == 'cross' else 'LATE FUSION'}]")
    print(f"   • Texto: d_model={args.d_model}, d_ff={args.d_ff}, L={args.num_layers}, H={args.n_heads}, {args.pos_encoding}")
    print(f"   • Tabular: embedding_dim={args.embedding_dim or 'auto'}, use_mlp=False (directo)")
    print(f"   • Fusión: mode={args.fusion_mode} (heads={args.fusion_heads}) -> ClassifierHead(64 -> 1)")
    print(f"   • Early Stopping: {'Desactivado (corre todas las épocas)' if args.no_early_stopping else f'Activado (patience={args.patience})'}")
    print(f"   • Semillas: {args.seeds}")
    print("=" * 88)

    loaders, artefactos = build_dataloaders(
        batch_size=args.batch_size,
        use_text=True,
        use_tabular=True,
        seed=42,
    )

    registros = []
    example_history = None

    for s in args.seeds:
        print(f"🚀 Ejecutando Seed {s}...")
        res = run_single_seed(
            seed=s,
            loaders=loaders,
            artefactos=artefactos,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            n_heads=args.n_heads,
            pos_encoding=args.pos_encoding,
            pooling=args.pooling,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=patience,
            restore_best=restore_best,
            fusion_mode=args.fusion_mode,
            fusion_heads=args.fusion_heads,
            device=args.device,
        )
        registros.append(res)
        if example_history is None:
            example_history = res["history"]
        print(f"   ✓ Seed {s:>3} | Época: {res['best_epoch']} | Val PR-AUC: {res['val_pr_auc']:.4f} | Test PR-AUC: {res['test_pr_auc']:.4f} | ROC: {res['test_roc_auc']:.4f}")

    df_runs = pd.DataFrame(registros)
    all_histories = [r["history"] for r in registros]
    df_runs_save = df_runs.drop(columns=["history"])
    df_runs_save.to_csv(agg_dir / "hybrid_evaluation_seeds.csv", index=False)
    (agg_dir / "hybrid_evaluation_histories.json").write_text(json.dumps(all_histories, indent=2), encoding="utf-8")

    summary = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "n_heads": args.n_heads,
        "pos_encoding": args.pos_encoding,
        "pooling": args.pooling,
        "embedding_dim": args.embedding_dim or "auto",
        "params_text": int(df_runs["params_text"].iloc[0]),
        "params_tabular": int(df_runs["params_tabular"].iloc[0]),
        "params_head": int(df_runs["params_head"].iloc[0]),
        "params_total": int(df_runs["params_total"].iloc[0]),
        "fused_dim": int(df_runs["fused_dim"].iloc[0]),
        "train_pr_auc_mean": float(df_runs["train_pr_auc"].mean()),
        "train_pr_auc_std": float(df_runs["train_pr_auc"].std()),
        "val_pr_auc_mean": float(df_runs["val_pr_auc"].mean()),
        "val_pr_auc_std": float(df_runs["val_pr_auc"].std()),
        "test_pr_auc_mean": float(df_runs["test_pr_auc"].mean()),
        "test_pr_auc_std": float(df_runs["test_pr_auc"].std()),
        "test_roc_auc_mean": float(df_runs["test_roc_auc"].mean()),
        "test_roc_auc_std": float(df_runs["test_roc_auc"].std()),
        "test_bce_mean": float(df_runs["test_bce"].mean()),
        "test_bce_std": float(df_runs["test_bce"].std()),
        "test_lift_mean": float(df_runs["test_lift"].mean()),
    }

    df_sum = pd.DataFrame([summary])
    df_sum.to_csv(agg_dir / "hybrid_evaluation_summary.csv", index=False)
    (agg_dir / "hybrid_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("📊 RESUMEN ESTADÍSTICO CONSOLIDADO (5 SEMILLAS)")
    print("=" * 88)
    cols = ["fused_dim", "params_total", "val_pr_auc_mean", "val_pr_auc_std", "test_pr_auc_mean", "test_pr_auc_std", "test_roc_auc_mean", "test_bce_mean", "test_lift_mean"]
    print(df_sum[cols].to_string(index=False))

    fig_path = plot_hybrid_results(df_runs, all_histories, fig_dir, filename=fig_filename, title_suffix=title_suffix)
    print(f"\n🎨 Figura guardada: {fig_path.name}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
