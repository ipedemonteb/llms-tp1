"""Script de evaluación del Modelo Híbrido con Rama Tabular Desredundada.

Elimina de la rama tabular todas las variables redundantes que ya están presentes en el texto:
- Removidas de tabla: brand, category, storage_type, unit_of_measure, net_weight_oz, title_tag,
                       allergens, num_ingredients, has_allergens.
- Conservadas en tabla: price, price_span, price_per_oz, volume, nutrition_score,
                         country_of_origin (Embedding), day_of_week (One-Hot).

Configuración solicitada:
- d_model = 16, d_ff = 64, num_layers = 1, n_heads = 1
- weight_decay = 0.05, dropout = 0.25, lr = 0.0003, epochs = 15
- Sin MLP tabular por defecto, pero con flag `--use_tabular_mlp` para activarlo opcionalmente.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.hybrid_transformer.fusion import BTRModel, FusionConfig
from src.hybrid_transformer.tabular_encoder import TabularEncoder, TabularEncoderConfig, TabularPreprocessor
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder
from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.dataset import SupermarketDataset
from src.training.metrics import compute_extended_metrics
from src.training.trainer import Trainer, TrainerConfig, set_seed

# Campos estrictamente no redundantes para la rama tabular
NON_REDUNDANT_NUMERIC_FIELDS = [
    "price",
    "price_span",
    "price_per_oz",
    "volume",
    "nutrition_score",
]

NON_REDUNDANT_LOG1P_FIELDS = [
    "price_per_oz",
    "volume",
]

NON_REDUNDANT_EMBEDDING_FIELDS = [
    "country_of_origin",
]

NON_REDUNDANT_ONEHOT_FIELDS = [
    "day_of_week",
]

OUTPUT_AGG_DIR = Path("results/aggregate/hybrid_baseline")
OUTPUT_FIG_DIR = Path("results/figures/hybrid_baseline")


def build_non_redundant_dataloaders(
    data_dir: Union[str, Path] = "resources/datasets",
    tokenizer_path: Union[str, Path] = "resources/tokenizer/bpe_tokenizer.json",
    max_length: int = 128,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """Construye DataLoaders con texto completo y variables tabulares estrictamente desredundadas."""
    data_dir = Path(data_dir)
    splits = {}
    for nombre in ("train", "val", "test"):
        ruta = data_dir / f"transformer_{nombre}.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró {ruta}.")
        splits[nombre] = pd.read_csv(ruta)

    tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path, max_length=max_length)

    preprocessor = TabularPreprocessor(
        numeric_fields=NON_REDUNDANT_NUMERIC_FIELDS,
        log1p_fields=NON_REDUNDANT_LOG1P_FIELDS,
        direct_fields=[],
        embedding_fields=NON_REDUNDANT_EMBEDDING_FIELDS,
        onehot_fields=NON_REDUNDANT_ONEHOT_FIELDS,
    ).fit(splits["train"])

    datasets = {
        nombre: SupermarketDataset(
            df=df,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            text_fields=["title_clean", "description", "ingredients"],
            max_length=max_length,
        )
        for nombre, df in splits.items()
    }

    generador = torch.Generator().manual_seed(seed)
    loaders = {
        nombre: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(nombre == "train"),
            generator=generador if nombre == "train" else None,
            drop_last=(nombre == "train"),
        )
        for nombre, ds in datasets.items()
    }

    artefactos = {
        "tokenizer": tokenizer,
        "preprocessor": preprocessor,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "num_numeric": len(preprocessor.numeric_fields),
        "num_direct": len(preprocessor.direct_fields),
        "embedding_cardinalities": [
            preprocessor.embedding_cardinalities[c] for c in preprocessor.embedding_fields
        ],
        "onehot_cardinalities": [
            preprocessor.onehot_cardinalities[c] for c in preprocessor.onehot_fields
        ],
    }

    return loaders, artefactos


def run_single_seed(
    seed: int,
    loaders: Dict[str, DataLoader],
    artefactos: Dict[str, Any],
    d_model: int,
    d_ff: int,
    num_layers: int,
    n_heads: int,
    pos_encoding: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: Optional[int],
    restore_best: bool,
    fusion_mode: str,
    fusion_heads: int,
    use_tabular_mlp: bool,
    d_tab: int,
    tab_hidden_dims: Optional[List[int]],
    head_hidden_dims: Optional[List[int]],
    device: Optional[str],
) -> Dict[str, Any]:
    """Entrena una corrida puntual con una semilla aleatoria fija."""
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
        pooling_mode="none" if fusion_mode == "cross" else "mean",
        pad_token_id=artefactos["pad_token_id"],
    )
    text_encoder = TextTransformerEncoder(text_config)

    # 2. Rama Tabular Desredundada (17 dimensiones de entrada)
    tab_config = TabularEncoderConfig(
        num_numeric=artefactos["num_numeric"],
        num_direct=artefactos["num_direct"],
        embedding_cardinalities=artefactos["embedding_cardinalities"],
        embedding_dim=None,  # auto
        onehot_cardinalities=artefactos["onehot_cardinalities"],
        use_mlp=use_tabular_mlp,
        hidden_dims=tab_hidden_dims if tab_hidden_dims is not None else [64],
        d_tab=d_tab,
        dropout=dropout,
        activation="gelu",
    )
    tabular_encoder = TabularEncoder(tab_config)

    # 3. Fusión y Clasificador Final
    fusion_config = FusionConfig(
        mode=fusion_mode,
        d_text=text_encoder.config.d_model,
        d_tab=tabular_encoder.output_dim,
        n_heads=fusion_heads,
        hidden_dims=head_hidden_dims if head_hidden_dims is not None else [64],
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

    # 5. Evaluación de Métricas
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
        "use_tabular_mlp": use_tabular_mlp,
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
        "test_lift": test_m.get("test_lift_top_decile", float("nan")),
        "history": trainer.history,
    }


def plot_evaluation_results(
    df_runs: pd.DataFrame,
    all_histories: Optional[List[List[Dict[str, Any]]]],
    fig_dir: Path,
    filename: str = "01_hybrid_non_redundant_tabular_evaluation.png",
    title_suffix: str = "(Tabular Desredundado)",
) -> Path:
    """Genera la figura de 3 paneles limpia con bandas de error y distribuciones estadísticas."""
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

    # Panel 1: Curvas de Aprendizaje (BCE Loss)
    if all_histories and len(all_histories) > 0:
        filas = []
        for s_idx, h in enumerate(all_histories):
            for row in h:
                filas.append({
                    "seed_idx": s_idx,
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                })
        df_hist = pd.DataFrame(filas)
        agg_hist = df_hist.groupby("epoch").agg({
            "train_loss": ["mean", "std"],
            "val_loss": ["mean", "std"],
        }).reset_index()

        epochs = agg_hist["epoch"]
        tr_mean = agg_hist["train_loss"]["mean"]
        tr_std = agg_hist["train_loss"]["std"].fillna(0)
        vl_mean = agg_hist["val_loss"]["mean"]
        vl_std = agg_hist["val_loss"]["std"].fillna(0)

        ax1.plot(epochs, tr_mean, label="Train Loss", color="#1f77b4", linewidth=2.0)
        ax1.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, color="#1f77b4", alpha=0.18)
        ax1.plot(epochs, vl_mean, label="Val Loss", color="#ff7f0e", linewidth=2.0)
        ax1.fill_between(epochs, vl_mean - vl_std, vl_mean + vl_std, color="#ff7f0e", alpha=0.18)

        ax1.set_xlabel("Época")
        ax1.set_ylabel("BCE Loss")
        ax1.set_title(f"Evolución de Pérdida (5 Semillas)")
        ax1.legend(loc="upper right")

    # Panel 2: PR-AUC (Train, Val, Test)
    x = np.arange(len(df_runs))
    w = 0.26
    seeds = df_runs["seed"].tolist()

    ax2.bar(x - w, df_runs["train_pr_auc"], width=w, label="Train", color="#1f77b4", alpha=0.85)
    ax2.bar(x, df_runs["val_pr_auc"], width=w, label="Val", color="#ff7f0e", alpha=0.85)
    ax2.bar(x + w, df_runs["test_pr_auc"], width=w, label="Test", color="#2ca02c", alpha=0.85)

    tr_m, tr_s = df_runs["train_pr_auc"].mean(), df_runs["train_pr_auc"].std()
    te_m, te_s = df_runs["test_pr_auc"].mean(), df_runs["test_pr_auc"].std()
    ax2.axhline(tr_m, color="#1f77b4", linestyle="--", alpha=0.6, label=f"Train μ={tr_m:.3f}")
    ax2.axhline(te_m, color="#2ca02c", linestyle="--", alpha=0.6, label=f"Test μ={te_m:.3f}")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S-{s}" for s in seeds])
    ax2.set_xlabel("Semilla")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title(f"PR-AUC por Semilla (Test μ={te_m:.4f} ± {te_s:.4f})")
    ax2.legend(loc="lower right", fontsize=8.5)

    # Panel 3: ROC-AUC (Train, Val, Test)
    ax3.bar(x - w, df_runs["train_roc_auc"], width=w, label="Train", color="#1f77b4", alpha=0.85)
    ax3.bar(x, df_runs["val_roc_auc"], width=w, label="Val", color="#ff7f0e", alpha=0.85)
    ax3.bar(x + w, df_runs["test_roc_auc"], width=w, label="Test", color="#2ca02c", alpha=0.85)

    r_tr_m = df_runs["train_roc_auc"].mean()
    r_te_m, r_te_s = df_runs["test_roc_auc"].mean(), df_runs["test_roc_auc"].std()
    ax3.axhline(r_tr_m, color="#1f77b4", linestyle="--", alpha=0.6, label=f"Train μ={r_tr_m:.3f}")
    ax3.axhline(r_te_m, color="#2ca02c", linestyle="--", alpha=0.6, label=f"Test μ={r_te_m:.3f}")

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"S-{s}" for s in seeds])
    ax3.set_xlabel("Semilla")
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title(f"ROC-AUC por Semilla (Test μ={r_te_m:.4f} ± {r_te_s:.4f})")
    ax3.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(f"Evaluación del Modelo Híbrido {title_suffix}", fontsize=13.5, y=0.99)
    fig.tight_layout()

    out_path = fig_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del Modelo Híbrido con Rama Tabular Desredundada.")
    parser.add_argument("--exp_name", type=str, default="non_redundant_tabular", help="Nombre del experimento.")
    parser.add_argument("--d_model", type=int, default=16, help="Dimensión latente del Transformer.")
    parser.add_argument("--d_ff", type=int, default=64, help="Dimensión FFN del Transformer.")
    parser.add_argument("--num_layers", type=int, default=1, help="Capas del Transformer.")
    parser.add_argument("--n_heads", type=int, default=1, help="Cabezales de atención.")
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal")
    parser.add_argument("--fusion_mode", type=str, default="cross", choices=["cross", "late"])
    parser.add_argument("--fusion_heads", type=int, default=1)
    parser.add_argument("--use_tabular_mlp", action="store_true", help="Activa el MLP en la rama tabular.")
    parser.add_argument("--d_tab", type=int, default=16, help="Dimensión de salida e_tab del MLP tabular.")
    parser.add_argument("--tab_hidden_dims", nargs="+", type=int, default=[32], help="Capas ocultas del MLP tabular.")
    parser.add_argument("--head_hidden_dims", nargs="*", type=int, default=[64], help="Capas ocultas del clasificador final.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 456, 999])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    agg_dir = OUTPUT_AGG_DIR / args.exp_name
    fig_dir = OUTPUT_FIG_DIR / args.exp_name
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("🔬 EXPERIMENTO: MODELO HÍBRIDO CON RAMA TABULAR DESREDUNDADA")
    print(f"   • Texto: d_model={args.d_model}, d_ff={args.d_ff}, L={args.num_layers}, H={args.n_heads}")
    print(f"   • Tabular: 17 variables limpias (5 numéricas, 1 embedding país, 1 one-hot día)")
    print(f"   • Tabular MLP: {'Activado (d_tab=' + str(args.d_tab) + ')' if args.use_tabular_mlp else 'Desactivado (passthrough directo)'}")
    print(f"   • Fusión: mode={args.fusion_mode} (heads={args.fusion_heads})")
    print(f"   • Hiperparámetros: lr={args.lr}, weight_decay={args.weight_decay}, dropout={args.dropout}, epochs={args.epochs}")
    print(f"   • Semillas: {args.seeds}")
    print("=" * 88)

    loaders, artefactos = build_non_redundant_dataloaders(
        batch_size=args.batch_size,
        seed=42,
    )

    registros = []
    all_histories = []

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
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=None,
            restore_best=False,
            fusion_mode=args.fusion_mode,
            fusion_heads=args.fusion_heads,
            use_tabular_mlp=args.use_tabular_mlp,
            d_tab=args.d_tab,
            tab_hidden_dims=args.tab_hidden_dims,
            head_hidden_dims=args.head_hidden_dims,
            device=args.device,
        )
        all_histories.append(res["history"])
        registros.append(res)
        print(f"   ✓ Seed {s:>3} | Test PR-AUC: {res['test_pr_auc']:.4f} | ROC: {res['test_roc_auc']:.4f} | BCE: {res['test_bce']:.4f}")

    df_runs = pd.DataFrame(registros)
    df_runs_save = df_runs.drop(columns=["history"])
    df_runs_save.to_csv(agg_dir / "hybrid_evaluation_seeds.csv", index=False)
    (agg_dir / "hybrid_evaluation_histories.json").write_text(json.dumps(all_histories, indent=2), encoding="utf-8")

    summary = {
        "exp_name": args.exp_name,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "n_heads": args.n_heads,
        "use_tabular_mlp": args.use_tabular_mlp,
        "params_text": int(df_runs["params_text"].iloc[0]),
        "params_tabular": int(df_runs["params_tabular"].iloc[0]),
        "params_cross": int(df_runs["params_cross"].iloc[0]),
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
    cols = ["fused_dim", "params_total", "train_pr_auc_mean", "val_pr_auc_mean", "test_pr_auc_mean", "test_pr_auc_std", "test_roc_auc_mean", "test_bce_mean"]
    print(df_sum[cols].to_string(index=False))

    fig_filename = f"01_hybrid_{args.exp_name}_evaluation.png"
    fig_path = plot_evaluation_results(df_runs, all_histories, fig_dir, filename=fig_filename, title_suffix=f"({args.exp_name})")
    print(f"\n🎨 Figura guardada en: {fig_path}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
