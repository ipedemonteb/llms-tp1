"""Script de evaluación del impacto de reducir el tamaño de vocabulario (vocab_size)
en el Hybrid Transformer con Cross-Attention y d_model=16 (sin MLP tabular).

Compara:
- vocab_size = 261  (Byte-level puro / mínimo BPE, 0 fusiones)
- vocab_size = 512  (BPE compacto, ~250 fusiones)
- vocab_size = 1024 (BPE intermedio)
- vocab_size = 1720 (BPE completo original / baseline)

Arquitectura:
- Rama de Texto: Transformer con d_model=16, d_ff=64, num_layers=1, n_heads=1, sinusoidal, sin pooling.
- Rama Tabular: Entity Embeddings + Numéricas directas (SIN MLP tabular previo, use_mlp=False).
- Fusión: Cross-Attention (el vector tabular e_tab actúa como Query sobre los tokens de texto e_text).
- Clasificador: ClassifierHead con Linear(d_text + d_tab -> 64) -> GELU -> Dropout -> Linear(64 -> 1).

Genera:
1. Figuras de 3 paneles aisladas para cada vocab_size en results/figures/hybrid_baseline/crossatt_d16_vocab_<V>/
2. Gráfico comparativo consolidado en results/figures/hybrid_baseline/00_crossatt_d16_vocab_size_comparison.png
3. Tablas agregadas en results/aggregate/hybrid_baseline/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tokenizer.bpe import ByteLevelBPETokenizer
from src.training.experiments.hybrid_evaluation import run_hybrid_experiment
from src.training.plots import SERIE_1, SERIE_2, SERIE_3, SERIE_4, SERIE_5, aplicar_estilo_cientifico

SEEDS = [7, 42, 123, 456, 999]
DEFAULT_VOCAB_SIZES = [261, 512, 1024, 1720]

OUTPUT_AGG_DIR = Path("results/aggregate/hybrid_baseline")
OUTPUT_FIG_DIR = Path("results/figures/hybrid_baseline")
TOKENIZER_DIR = Path("resources/tokenizer")
DATASET_TRAIN_PATH = Path("resources/datasets/transformer_train.csv")


def ensure_tokenizer(vocab_size: int, train_csv_path: Path = DATASET_TRAIN_PATH, tokenizer_dir: Path = TOKENIZER_DIR) -> Path:
    """Garantiza la existencia de un tokenizador BPE entrenado exclusivamente sobre train para el vocab_size dado."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if vocab_size == 1720 or vocab_size == 2048:
        default_tok = tokenizer_dir / "bpe_tokenizer.json"
        if default_tok.exists():
            return default_tok

    target_path = tokenizer_dir / f"bpe_vocab_{vocab_size}.json"
    if target_path.exists():
        return target_path

    print(f"🔤 Entrenando nuevo tokenizador BPE (vocab_size={vocab_size}) sobre {train_csv_path}...")
    df_train = pd.read_csv(train_csv_path)
    texts = df_train["text"].dropna().astype(str).tolist()

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab_size, show_progress=False)
    tok.save(target_path)
    print(f"   ✓ Tokenizador guardado en {target_path} (vocabulario efectivo: {tok.vocab_size} tokens)")
    return target_path


def run_experiment_for_vocab(
    vocab_size: int,
    d_model: int = 16,
    d_ff: int = 64,
    num_layers: int = 1,
    n_heads: int = 1,
    pos_encoding: str = "sinusoidal",
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 15,
    no_early_stopping: bool = True,
    patience: Optional[int] = 5,
    seeds: Sequence[int] = SEEDS,
    device: Optional[str] = None,
) -> Path:
    """Ejecuta la evaluación multi-semilla para un tamaño de vocabulario específico."""
    tok_path = ensure_tokenizer(vocab_size)
    tok = ByteLevelBPETokenizer.from_file(tok_path)
    eff_vocab = tok.vocab_size

    exp_name = f"crossatt_d16_vocab_{eff_vocab}"
    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}] (vocab_size={eff_vocab}, d_model={d_model}, Cross-Attention, sin MLP tab)")
    print("=" * 88)

    run_hybrid_experiment(
        exp_name=exp_name,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        n_heads=n_heads,
        pos_encoding=pos_encoding,
        pooling="none",
        fusion_mode="cross",
        fusion_heads=n_heads,
        tabular_mlp=False,
        use_tabular_mlp=False,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        no_early_stopping=no_early_stopping,
        patience=patience,
        tokenizer_path=tok_path,
        seeds=seeds,
        device=device,
    )
    return tok_path


def generate_vocab_comparison_summary(
    vocab_sizes: List[int],
    base_agg: Path = OUTPUT_AGG_DIR,
    base_fig: Path = OUTPUT_FIG_DIR,
) -> None:
    """Genera la figura consolidada y el CSV comparativo de impacto de vocab_size."""
    resumenes = []
    histories_by_config = {}

    for vs in vocab_sizes:
        exp_name = f"crossatt_d16_vocab_{vs}"
        agg_p = base_agg / exp_name
        summary_p = agg_p / "hybrid_evaluation_summary.csv"
        hist_p = agg_p / "hybrid_evaluation_histories.json"
        if not hist_p.exists():
            hist_p = agg_p / "hybrid_evaluation_history.json"

        if summary_p.exists():
            df = pd.read_csv(summary_p)
            df["vocab_size_eval"] = vs
            df["config_label"] = f"Vocab {vs}\n({int(df['params_text'].iloc[0]):,} txt par)"
            resumenes.append(df.iloc[0])
        if hist_p.exists():
            histories_by_config[vs] = json.loads(hist_p.read_text(encoding="utf-8"))

    if len(resumenes) < 2:
        print("⚠️  No hay suficientes experimentos ejecutados para generar la comparativa consolidada de vocab_size.")
        return

    df_comp = pd.DataFrame(resumenes).sort_values("vocab_size_eval")
    base_agg.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    comp_csv_path = base_agg / "crossatt_d16_vocab_size_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n📊 Tabla comparativa guardada en: {comp_csv_path}")

    aplicar_estilo_cientifico()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.2))

    # Panel 1: Dinámica de Validación (Val BCE Loss a lo largo de las épocas)
    colors = [SERIE_1, SERIE_3, SERIE_4, SERIE_5]
    for idx, (vs, all_hists) in enumerate(histories_by_config.items()):
        color = colors[idx % len(colors)]
        filas = []
        for s_idx, h in enumerate(all_hists):
            for row in h:
                filas.append({"epoch": row["epoch"], "val_loss": row["val_loss"]})
        df_ep = pd.DataFrame(filas)
        agg = df_ep.groupby("epoch")["val_loss"].agg(["mean", "std"]).reset_index()
        ep = agg["epoch"]
        vl_mean = agg["mean"]
        vl_std = agg["std"].fillna(0)

        label_text = f"Vocab {vs} ({int(df_comp.loc[df_comp['vocab_size_eval'] == vs, 'params_total'].iloc[0]):,} par)"
        ax1.plot(ep, vl_mean, "o-", color=color, linewidth=2.0, label=label_text)
        ax1.fill_between(ep, vl_mean - vl_std, vl_mean + vl_std, color=color, alpha=0.12)

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Validation BCE Loss")
    ax1.set_title("Evolución de Pérdida en Val (Media ± $\\sigma$)")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: PR-AUC (Train vs Test) y Brecha de Generalización
    x = np.arange(len(df_comp))
    w = 0.35
    ax2.bar(x - w / 2, df_comp["train_pr_auc_mean"], yerr=df_comp["train_pr_auc_std"],
            width=w, capsize=4, color=SERIE_1, label="Train PR-AUC (Memorización)")
    ax2.bar(x + w / 2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Generalización)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["config_label"], fontsize=9.5)
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Brecha de Generalización $\\Delta(\\text{Train} - \\text{Test})$")
    ax2.legend(loc="lower right")

    # Panel 3: Test PR-AUC y Test ROC-AUC vs Conteo de Parámetros
    ax3.bar(x - w / 2, df_comp["test_pr_auc_mean"], yerr=df_comp["test_pr_auc_std"],
            width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Primaria)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comp["config_label"], fontsize=9.5)
    ax3.set_ylabel("Test PR-AUC", color=SERIE_3)
    ax3.set_ylim([0.65, 0.76])

    ax3_twin = ax3.twinx()
    ax3_twin.errorbar(x + w / 2, df_comp["test_roc_auc_mean"], yerr=df_comp["test_roc_auc_std"],
                      fmt="s--", color=SERIE_1, linewidth=2.2, capsize=4, label="Test ROC-AUC (Secundaria)")
    ax3_twin.set_ylabel("Test ROC-AUC", color=SERIE_1)
    ax3_twin.set_ylim([0.94, 0.98])
    ax3_twin.grid(False)
    ax3.set_title("Rendimiento Final en Test (PR-AUC y ROC-AUC)")

    fig.suptitle(
        "Impacto del Tamaño de Vocabulario (Vocab Size) en Hybrid Transformer con Cross-Attention ($d_{\\text{model}}=16$)",
        fontsize=14,
    )
    fig.tight_layout()

    out_fig = base_fig / "00_crossatt_d16_vocab_size_comparison.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"🎨 Gráfico comparativo consolidado guardado en: {out_fig}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación de reducción de vocab_size en Hybrid Transformer con Cross-Attention y d_model=16."
    )
    parser.add_argument(
        "--vocab_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_VOCAB_SIZES,
        help="Lista de tamaños de vocabulario a evaluar (por defecto: 261, 512, 1024, 1720).",
    )
    parser.add_argument("--d_model", type=int, default=16, help="Dimensión latente del Transformer.")
    parser.add_argument("--d_ff", type=int, default=64, help="Dimensión de la capa intermedia FFN.")
    parser.add_argument("--num_layers", type=int, default=1, help="Número de capas de Transformer.")
    parser.add_argument("--n_heads", type=int, default=1, help="Número de cabezales de atención.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Tasa de dropout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay L2.")
    parser.add_argument("--epochs", type=int, default=15, help="Cantidad de épocas de entrenamiento.")
    parser.add_argument("--no_early_stopping", action="store_true", default=True, help="Corre todas las épocas sin early stopping.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Semillas aleatorias para evaluación estadística.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--summary_only", action="store_true", help="Solo genera el gráfico comparativo consolidado sin reentrenar.")
    args = parser.parse_args()

    # Normalizar vocab sizes a efectivos
    target_vocabs = []
    for vs in args.vocab_sizes:
        tok_p = ensure_tokenizer(vs)
        eff_vs = ByteLevelBPETokenizer.from_file(tok_p).vocab_size
        target_vocabs.append(eff_vs)
    target_vocabs = sorted(list(set(target_vocabs)))

    if not args.summary_only:
        for vs in target_vocabs:
            run_experiment_for_vocab(
                vocab_size=vs,
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_layers=args.num_layers,
                n_heads=args.n_heads,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                no_early_stopping=args.no_early_stopping,
                seeds=args.seeds,
                device=args.device,
            )

    generate_vocab_comparison_summary(target_vocabs)


if __name__ == "__main__":
    main()
