"""Estudio de ablación e hiperparámetros para la rama del Transformer (Texto Puro).

Evalúa sistemáticamente la rama de texto aislada conectada directamente al clasificador final
(sin variables tabulares):
1. Dimensión latente (d_model ∈ [32, 48, 64, 96])
2. Cantidad de capas / bloques (L ∈ [1, 2, 3, 4])
3. Cantidad de cabezales de atención (H ∈ [1, 2, 4, 8])
4. Positional Encoding (sinusoidal vs learned vs none)
5. Estrategia de Pooling (mean vs cls vs max)

Métricas reportadas:
- PR-AUC en Train, Validación y Test (con media ± desvío estándar multi-semilla).
- ROC-AUC y BCE Loss en Test.
- Cantidad de parámetros entrenables del Transformer y de la cabeza.

Uso:
    uv run python -m src.training.transformer_ablation --study d_model
    uv run python -m src.training.transformer_ablation --study all --seeds 7 42 123
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
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder
from src.training.dataset import build_dataloaders
from src.training.metrics import compute_extended_metrics
from src.training.plots import SERIE_1, SERIE_2, SERIE_3, SERIE_5, aplicar_estilo_cientifico
from src.training.trainer import Trainer, TrainerConfig, set_seed

OUTPUT_AGG_DIR = Path("results/aggregate/transformer_ablation")
OUTPUT_FIG_DIR = Path("results/figures/transformer_ablation")


def train_single_text_run(
    d_model: int,
    n_heads: int,
    d_ff: int,
    num_layers: int,
    pos_encoding: str,
    pooling: str,
    seed: int,
    loaders: dict,
    artefactos: dict,
    max_length: int = 128,
    epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    patience: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Entrena una variante puntual de texto puro y devuelve métricas completas."""
    set_seed(seed)

    text_config = TextTransformerConfig(
        vocab_size=artefactos["vocab_size"],
        max_seq_len=max_length,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        pos_encoding_type=pos_encoding,
        pooling_mode=pooling,
        pad_token_id=artefactos["pad_token_id"],
    )
    text_encoder = TextTransformerEncoder(text_config)

    fusion_config = FusionConfig(
        mode="late",
        d_text=d_model,
        d_tab=0,
        hidden_dims=[64],
        dropout=dropout,
        activation="gelu",
    )
    model = BTRModel(text_encoder=text_encoder, tabular_encoder=None, fusion_config=fusion_config)

    params_breakdown = model.param_breakdown()

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
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "pos_encoding": pos_encoding,
        "pooling": pooling,
        "seed": seed,
        "params_text": params_breakdown["texto"],
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
    }


def aggregate_runs(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calcula medias y desvíos estándar agrupados por un hiperparámetro."""
    agregados = []
    for val, sub in df.groupby(group_col, sort=False):
        agregados.append({
            group_col: val,
            "params_total": int(sub["params_total"].iloc[0]),
            "best_epoch_mean": float(sub["best_epoch"].mean()),
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
            "test_lift_mean": float(sub["test_lift"].mean()),
        })
    return pd.DataFrame(agregados)


# =====================================================================================
# GENERACIÓN DE FIGURAS CIENTÍFICAS
# =====================================================================================

def plot_ablation_study(
    df_agg: pd.DataFrame,
    param_col: str,
    param_label: str,
    title: str,
    output_path: Path,
) -> Path:
    """Genera un gráfico limpio de 2 paneles (PR-AUC a la izq, ROC y BCE a la der)."""
    aplicar_estilo_cientifico()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(df_agg))
    labels = [str(v) for v in df_agg[param_col]]

    # Panel 1: PR-AUC (Train, Val, Test)
    ax1.errorbar(
        x, df_agg["train_pr_auc_mean"], yerr=df_agg["train_pr_auc_std"],
        fmt="s--", color="#1f77b4", linewidth=1.8, capsize=4, label="Train PR-AUC",
    )
    ax1.errorbar(
        x, df_agg["val_pr_auc_mean"], yerr=df_agg["val_pr_auc_std"],
        fmt="o-", color="#ff7f0e", linewidth=2.0, capsize=4, label="Validation PR-AUC",
    )
    ax1.errorbar(
        x, df_agg["test_pr_auc_mean"], yerr=df_agg["test_pr_auc_std"],
        fmt="^-", color="#2ca02c", linewidth=2.2, capsize=4, label="Test PR-AUC",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(param_label)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("PR-AUC (Train vs Val vs Test)")
    ax1.legend(loc="best")

    # Panel 2: ROC-AUC y BCE Loss en Test
    ax2.errorbar(
        x, df_agg["test_roc_auc_mean"], yerr=df_agg["test_roc_auc_std"],
        fmt="o-", color="#2ca02c", linewidth=2.0, capsize=4, label="Test ROC-AUC (eje izq)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel(param_label)
    ax2.set_ylabel("Test ROC-AUC", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax2_twin = ax2.twinx()
    ax2_twin.errorbar(
        x, df_agg["test_bce_mean"], yerr=df_agg["test_bce_std"],
        fmt="s--", color="#d62728", linewidth=1.8, capsize=4, label="Test BCE Loss (eje der)",
    )
    ax2_twin.set_ylabel("Test BCE Loss", color="#d62728")
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")
    ax2_twin.grid(False)

    ax2.set_title("Test ROC-AUC y BCE Loss")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =====================================================================================
# EJECUCIÓN DE ESTUDIOS ESPECÍFICOS
# =====================================================================================

def run_d_model_study(loaders: dict, artefactos: dict, seeds: List[int], epochs: int, device: Optional[str]) -> pd.DataFrame:
    """Estudio 1: Dimensión Latente d_model ∈ [32, 48, 64, 96]."""
    print("\n" + "=" * 70)
    print("🔹 ESTUDIO 1: DIMENSIÓN LATENTE (d_model ∈ [32, 48, 64, 96])")
    print("=" * 70)
    d_models = [32, 48, 64, 96]
    registros = []
    for d in d_models:
        d_ff = 4 * d
        print(f"  • d_model={d} (d_ff={d_ff}, L=2, H=4)...")
        for s in seeds:
            res = train_single_text_run(
                d_model=d, n_heads=4, d_ff=d_ff, num_layers=2,
                pos_encoding="sinusoidal", pooling="mean",
                seed=s, loaders=loaders, artefactos=artefactos,
                epochs=epochs, device=device,
            )
            registros.append(res)

    df_raw = pd.DataFrame(registros)
    df_agg = aggregate_runs(df_raw, "d_model")
    return df_agg, df_raw


def run_layers_study(loaders: dict, artefactos: dict, seeds: List[int], epochs: int, device: Optional[str], best_d_model: int = 64) -> pd.DataFrame:
    """Estudio 2: Cantidad de Capas L ∈ [1, 2, 3, 4]."""
    print("\n" + "=" * 70)
    print(f"🔹 ESTUDIO 2: CANTIDAD DE CAPAS (L ∈ [1, 2, 3, 4] con d_model={best_d_model})")
    print("=" * 70)
    layers = [1, 2, 3, 4]
    registros = []
    for l in layers:
        print(f"  • num_layers={l} (d_model={best_d_model}, H=4)...")
        for s in seeds:
            res = train_single_text_run(
                d_model=best_d_model, n_heads=4, d_ff=4 * best_d_model, num_layers=l,
                pos_encoding="sinusoidal", pooling="mean",
                seed=s, loaders=loaders, artefactos=artefactos,
                epochs=epochs, device=device,
            )
            registros.append(res)

    df_raw = pd.DataFrame(registros)
    df_agg = aggregate_runs(df_raw, "num_layers")
    return df_agg, df_raw


def run_heads_study(loaders: dict, artefactos: dict, seeds: List[int], epochs: int, device: Optional[str], best_d_model: int = 64, best_l: int = 2) -> pd.DataFrame:
    """Estudio 3: Cantidad de Cabezales H ∈ [1, 2, 4, 8]."""
    print("\n" + "=" * 70)
    print(f"🔹 ESTUDIO 3: CABEZALES DE ATENCIÓN (H ∈ [1, 2, 4, 8] con d_model={best_d_model}, L={best_l})")
    print("=" * 70)
    heads = [1, 2, 4, 8]
    registros = []
    for h in heads:
        print(f"  • n_heads={h} (d_k={best_d_model // h})...")
        for s in seeds:
            res = train_single_text_run(
                d_model=best_d_model, n_heads=h, d_ff=4 * best_d_model, num_layers=best_l,
                pos_encoding="sinusoidal", pooling="mean",
                seed=s, loaders=loaders, artefactos=artefactos,
                epochs=epochs, device=device,
            )
            registros.append(res)

    df_raw = pd.DataFrame(registros)
    df_agg = aggregate_runs(df_raw, "n_heads")
    return df_agg, df_raw


def run_pos_encoding_study(loaders: dict, artefactos: dict, seeds: List[int], epochs: int, device: Optional[str], best_d_model: int = 64, best_l: int = 2) -> pd.DataFrame:
    """Estudio 4: Positional Encoding (sinusoidal vs learned vs none)."""
    print("\n" + "=" * 70)
    print("🔹 ESTUDIO 4: POSITIONAL ENCODING (sinusoidal vs learned vs none)")
    print("=" * 70)
    pos_types = ["sinusoidal", "learned", "none"]
    registros = []
    for pos in pos_types:
        print(f"  • pos_encoding={pos}...")
        for s in seeds:
            res = train_single_text_run(
                d_model=best_d_model, n_heads=4, d_ff=4 * best_d_model, num_layers=best_l,
                pos_encoding=pos, pooling="mean",
                seed=s, loaders=loaders, artefactos=artefactos,
                epochs=epochs, device=device,
            )
            registros.append(res)

    df_raw = pd.DataFrame(registros)
    df_agg = aggregate_runs(df_raw, "pos_encoding")
    return df_agg, df_raw


def run_pooling_study(loaders: dict, artefactos: dict, seeds: List[int], epochs: int, device: Optional[str], best_d_model: int = 64, best_l: int = 2) -> pd.DataFrame:
    """Estudio 5: Pooling (mean vs cls vs max)."""
    print("\n" + "=" * 70)
    print("🔹 ESTUDIO 5: POOLING (mean vs cls vs max)")
    print("=" * 70)
    pool_types = ["mean", "cls", "max"]
    registros = []
    for pool in pool_types:
        print(f"  • pooling={pool}...")
        for s in seeds:
            res = train_single_text_run(
                d_model=best_d_model, n_heads=4, d_ff=4 * best_d_model, num_layers=best_l,
                pos_encoding="sinusoidal", pooling=pool,
                seed=s, loaders=loaders, artefactos=artefactos,
                epochs=epochs, device=device,
            )
            registros.append(res)

    df_raw = pd.DataFrame(registros)
    df_agg = aggregate_runs(df_raw, "pooling")
    return df_agg, df_raw


# =====================================================================================
# MAIN RUNNER
# =====================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de ablación de la rama Transformer (Texto Puro).")
    parser.add_argument("--study", type=str, default="all",
                        choices=["d_model", "layers", "heads", "pos_encoding", "pooling", "all"],
                        help="Estudio puntual a ejecutar o 'all' para la batería completa.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123],
                        help="Semillas aleatorias para evaluar estabilidad.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plots_only", action="store_true", help="Solo regenera figuras desde CSVs existentes.")
    args = parser.parse_args()

    OUTPUT_AGG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        print("🎨 Regenerando figuras desde CSVs guardados...")
        archivos = {
            "d_model": ("d_model", "Dimensión Latente ($d_{\\text{model}}$)", "01_transformer_d_model_pr_auc.png"),
            "layers": ("num_layers", "Cantidad de Capas ($L$)", "02_transformer_layers_pr_auc.png"),
            "heads": ("n_heads", "Cabezales de Atención ($H$)", "03_transformer_heads_pr_auc.png"),
            "pos_encoding": ("pos_encoding", "Tipo de Positional Encoding", "04_transformer_pos_encoding_pr_auc.png"),
            "pooling": ("pooling", "Estrategia de Pooling", "05_transformer_pooling_pr_auc.png"),
        }
        for nombre, (col, lbl, fig_name) in archivos.items():
            csv_p = OUTPUT_AGG_DIR / f"transformer_ablation_{nombre}.csv"
            if csv_p.exists():
                df = pd.read_csv(csv_p)
                f = plot_ablation_study(df, col, lbl, f"Ablación de Texto: {lbl}", OUTPUT_FIG_DIR / fig_name)
                print(f"   ✓ {f.name}")
        return

    print("=" * 88)
    print(f"🔬 ESTUDIO TRANSFORMER PURO (Texto) — Semillas: {args.seeds}")
    print("=" * 88)

    loaders, artefactos = build_dataloaders(
        batch_size=args.batch_size,
        use_text=True,
        use_tabular=False,
        seed=42,
    )

    print(f"📦 Vocabulario BPE: {artefactos['vocab_size']} tokens | Max Seq: 128")

    estudios_a_correr = [args.study] if args.study != "all" else ["d_model", "layers", "heads", "pos_encoding", "pooling"]

    best_d = 64
    best_l = 2

    for est in estudios_a_correr:
        if est == "d_model":
            df_agg, df_raw = run_d_model_study(loaders, artefactos, args.seeds, args.epochs, args.device)
            df_agg.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_d_model.csv", index=False)
            df_raw.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_d_model_raw.csv", index=False)
            f = plot_ablation_study(df_agg, "d_model", "Dimensión Latente ($d_{\\text{model}}$)",
                                    "Ablación Transformer: Dimensión Latente ($d_{\\text{model}}$)",
                                    OUTPUT_FIG_DIR / "01_transformer_d_model_pr_auc.png")
            print(f"📊 Resumen d_model:\n{df_agg[['d_model', 'params_total', 'val_pr_auc_mean', 'test_pr_auc_mean', 'test_roc_auc_mean']].to_string(index=False)}")
            print(f"   ✓ Figura guardada: {f.name}")
            best_d = int(df_agg.loc[df_agg["val_pr_auc_mean"].idxmax()]["d_model"])

        elif est == "layers":
            df_agg, df_raw = run_layers_study(loaders, artefactos, args.seeds, args.epochs, args.device, best_d_model=best_d)
            df_agg.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_layers.csv", index=False)
            df_raw.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_layers_raw.csv", index=False)
            f = plot_ablation_study(df_agg, "num_layers", "Cantidad de Capas ($L$)",
                                    "Ablación Transformer: Cantidad de Capas ($L$)",
                                    OUTPUT_FIG_DIR / "02_transformer_layers_pr_auc.png")
            print(f"📊 Resumen layers:\n{df_agg[['num_layers', 'params_total', 'val_pr_auc_mean', 'test_pr_auc_mean', 'test_roc_auc_mean']].to_string(index=False)}")
            print(f"   ✓ Figura guardada: {f.name}")
            best_l = int(df_agg.loc[df_agg["val_pr_auc_mean"].idxmax()]["num_layers"])

        elif est == "heads":
            df_agg, df_raw = run_heads_study(loaders, artefactos, args.seeds, args.epochs, args.device, best_d_model=best_d, best_l=best_l)
            df_agg.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_heads.csv", index=False)
            df_raw.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_heads_raw.csv", index=False)
            f = plot_ablation_study(df_agg, "n_heads", "Cabezales de Atención ($H$)",
                                    "Ablación Transformer: Cabezales de Atención ($H$)",
                                    OUTPUT_FIG_DIR / "03_transformer_heads_pr_auc.png")
            print(f"📊 Resumen heads:\n{df_agg[['n_heads', 'params_total', 'val_pr_auc_mean', 'test_pr_auc_mean', 'test_roc_auc_mean']].to_string(index=False)}")
            print(f"   ✓ Figura guardada: {f.name}")

        elif est == "pos_encoding":
            df_agg, df_raw = run_pos_encoding_study(loaders, artefactos, args.seeds, args.epochs, args.device, best_d_model=best_d, best_l=best_l)
            df_agg.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_pos_encoding.csv", index=False)
            df_raw.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_pos_encoding_raw.csv", index=False)
            f = plot_ablation_study(df_agg, "pos_encoding", "Tipo de Positional Encoding",
                                    "Ablación Transformer: Positional Encoding",
                                    OUTPUT_FIG_DIR / "04_transformer_pos_encoding_pr_auc.png")
            print(f"📊 Resumen pos_encoding:\n{df_agg[['pos_encoding', 'params_total', 'val_pr_auc_mean', 'test_pr_auc_mean', 'test_roc_auc_mean']].to_string(index=False)}")
            print(f"   ✓ Figura guardada: {f.name}")

        elif est == "pooling":
            df_agg, df_raw = run_pooling_study(loaders, artefactos, args.seeds, args.epochs, args.device, best_d_model=best_d, best_l=best_l)
            df_agg.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_pooling.csv", index=False)
            df_raw.to_csv(OUTPUT_AGG_DIR / "transformer_ablation_pooling_raw.csv", index=False)
            f = plot_ablation_study(df_agg, "pooling", "Estrategia de Pooling",
                                    "Ablación Transformer: Estrategia de Pooling",
                                    OUTPUT_FIG_DIR / "05_transformer_pooling_pr_auc.png")
            print(f"📊 Resumen pooling:\n{df_agg[['pooling', 'params_total', 'val_pr_auc_mean', 'test_pr_auc_mean', 'test_roc_auc_mean']].to_string(index=False)}")
            print(f"   ✓ Figura guardada: {f.name}")

    print("\n" + "=" * 88)
    print(f"✅ ESTUDIO TRANSFORMER COMPLETADO. Resultados en {OUTPUT_AGG_DIR}/ y {OUTPUT_FIG_DIR}/")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
