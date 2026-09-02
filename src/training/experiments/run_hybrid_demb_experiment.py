"""Script de evaluación del impacto de la dimensión de Entity Embeddings (d_emb = 2)
y la presencia de MLP Tabular en el Hybrid Transformer con Cross-Attention y d_model=16.

Compara:
- d_model = 16, d_ff = 64, num_layers = 1, n_heads = 1
- Cross-Attention (Query = vector tabular, Key/Value = tokens de texto)
- Vocabulario BPE optimizado (vocab_size = 512)
- Rama Tabular: embedding_dim = 2 con o sin MLP intermedio
- Regularización: dropout = 0.1, weight_decay = 0.01, lr = 0.001
- 5 semillas aleatorias (7, 42, 123, 456, 999), 15 épocas completas

Genera:
1. Figuras de 3 paneles en results/figures/hybrid_baseline/<exp_name>/
2. Gráfico comparativo consolidado en results/figures/hybrid_baseline/00_hybrid_demb_comparison.png
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
OUTPUT_AGG_DIR = Path("results/aggregate/hybrid_baseline")
OUTPUT_FIG_DIR = Path("results/figures/hybrid_baseline")
TOKENIZER_DIR = Path("resources/tokenizer")
DATASET_TRAIN_PATH = Path("resources/datasets/transformer_train.csv")


def ensure_tokenizer(vocab_size: int, train_csv_path: Path = DATASET_TRAIN_PATH, tokenizer_dir: Path = TOKENIZER_DIR) -> Path:
    """Garantiza la existencia de un tokenizador BPE entrenado para el vocab_size dado."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if vocab_size in (1720, 2048):
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


def run_experiment(
    exp_name: str = "crossatt_d16_vocab_512_demb2_mlp",
    embedding_dim: int = 2,
    tabular_mlp: bool = True,
    d_tab: int = 16,
    tab_hidden_dims: Optional[List[int]] = None,
    vocab_size: int = 512,
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
) -> dict:
    """Ejecuta la evaluación multi-semilla para la configuración con d_emb y MLP tabular."""
    tok_path = ensure_tokenizer(vocab_size)
    tok = ByteLevelBPETokenizer.from_file(tok_path)
    eff_vocab = tok.vocab_size

    print("\n" + "=" * 88)
    print(f"🚀 INICIANDO EXPERIMENTO [{exp_name}]")
    print(f"   • Arquitectura: d_model={d_model}, d_ff={d_ff}, L={num_layers}, H={n_heads}, Cross-Attention")
    print(f"   • Tokenizador: vocab_size={eff_vocab} ({tok_path.name})")
    print(f"   • Tabular: embedding_dim={embedding_dim}, MLP={'Activado (d_tab=' + str(d_tab) + ')' if tabular_mlp else 'Desactivado (directo)'}")
    print(f"   • Entrenamiento: lr={lr}, weight_decay={weight_decay}, dropout={dropout}, {epochs} épocas")
    print(f"   • Semillas: {list(seeds)}")
    print("=" * 88)

    resultado = run_hybrid_experiment(
        exp_name=exp_name,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        n_heads=n_heads,
        pos_encoding=pos_encoding,
        pooling="none",
        fusion_mode="cross",
        fusion_heads=n_heads,
        embedding_dim=embedding_dim,
        tabular_mlp=tabular_mlp,
        use_tabular_mlp=tabular_mlp,
        d_tab=d_tab,
        tab_hidden_dims=tab_hidden_dims if tab_hidden_dims is not None else [32],
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
    return resultado


def generate_demb_comparison(
    base_agg: Path = OUTPUT_AGG_DIR,
    base_fig: Path = OUTPUT_FIG_DIR,
) -> None:
    """Genera la figura consolidada y el CSV comparando todas las variantes relevantes."""
    comparaciones = [
        ("Tabular Puro\n(d_emb=2)", Path("results/aggregate/tabular_ablation/tabular_ablation_summary.csv"), "d_emb_2"),
        ("Hybrid Base\n(Vocab 1720, d_emb=auto)", base_agg / "crossatt_d16_vocab_1720/hybrid_evaluation_summary.csv", None),
        ("Hybrid Óptimo\n(Vocab 512, d_emb=auto)", base_agg / "crossatt_d16_vocab_512/hybrid_evaluation_summary.csv", None),
        ("Hybrid d_emb=2\n(Sin MLP)", base_agg / "crossatt_d16_vocab_512_demb2/hybrid_evaluation_summary.csv", None),
        ("Hybrid d_emb=2\n(Con MLP Tabular)", base_agg / "crossatt_d16_vocab_512_demb2_mlp/hybrid_evaluation_summary.csv", None),
    ]

    filas = []
    for label, path, variant in comparaciones:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if variant and "variant" in df.columns:
            df = df[df["variant"] == variant]
        if not df.empty:
            row = df.iloc[0].to_dict()
            row["config_label"] = label
            filas.append(row)

    if len(filas) < 2:
        print("⚠️  No hay suficientes experimentos ejecutados para generar la comparativa consolidada.")
        return

    df_comp = pd.DataFrame(filas)
    comp_csv_path = base_agg / "hybrid_demb_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n📊 Tabla comparativa guardada en: {comp_csv_path}")

    aplicar_estilo_cientifico()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))

    x = np.arange(len(df_comp))
    w = 0.35
    labels = df_comp["config_label"].tolist()

    # Panel 1: Parámetros Totales
    params = df_comp["params_total"].astype(int)
    ax1.bar(x, params, width=0.45, color=SERIE_1, edgecolor="none", alpha=0.9)
    for i, p in enumerate(params):
        ax1.text(i, p + max(params) * 0.02, f"{p:,}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Parámetros Totales")
    ax1.set_title("A. Complejidad Paramétrica")
    ax1.set_ylim(0, max(params) * 1.18)

    # Panel 2: PR-AUC (Train vs Test) y Brecha de Generalización
    train_pr = df_comp["train_pr_auc_mean"]
    train_std = df_comp.get("train_pr_auc_std", pd.Series([0.0] * len(df_comp)))
    test_pr = df_comp["test_pr_auc_mean"]
    test_std = df_comp.get("test_pr_auc_std", pd.Series([0.0] * len(df_comp)))

    ax2.bar(x - w / 2, train_pr, yerr=train_std, width=w, capsize=4, color=SERIE_2, label="Train PR-AUC (Memorización)")
    ax2.bar(x + w / 2, test_pr, yerr=test_std, width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Generalización)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("B. Generalización y Brecha de Sobreajuste")
    ax2.set_ylim([0.60, 0.90])
    ax2.legend(loc="lower right")

    # Panel 3: Test PR-AUC vs Test ROC-AUC
    ax3.bar(x - w / 2, test_pr, yerr=test_std, width=w, capsize=4, color=SERIE_3, label="Test PR-AUC (Primaria)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.set_ylabel("Test PR-AUC", color=SERIE_3)
    ax3.set_ylim([0.68, 0.78])

    ax3_twin = ax3.twinx()
    test_roc = df_comp["test_roc_auc_mean"]
    test_roc_std = df_comp.get("test_roc_auc_std", pd.Series([0.0] * len(df_comp)))
    ax3_twin.errorbar(x + w / 2, test_roc, yerr=test_roc_std, fmt="s--", color=SERIE_5, linewidth=2.2, capsize=4, label="Test ROC-AUC (Secundaria)")
    ax3_twin.set_ylabel("Test ROC-AUC", color=SERIE_5)
    ax3_twin.set_ylim([0.955, 0.978])
    ax3_twin.grid(False)
    ax3.set_title("C. Métricas Finales en Test (PR-AUC y ROC-AUC)")

    fig.suptitle(
        "Impacto de $d_{\\text{emb}}=2$ y MLP Tabular en Hybrid Transformer ($d_{\\text{model}}=16$, Vocab 512)",
        fontsize=13.5,
    )
    fig.tight_layout()

    out_fig = base_fig / "00_hybrid_demb_comparison.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"🎨 Gráfico comparativo consolidado guardado en: {out_fig}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación de d_emb=2 y MLP Tabular en Hybrid Transformer con Cross-Attention y d_model=16."
    )
    parser.add_argument("--exp_name", type=str, default="crossatt_d16_vocab_512_demb2_mlp", help="Nombre del experimento.")
    parser.add_argument("--embedding_dim", type=int, default=2, help="Dimensión para Entity Embeddings.")
    parser.add_argument("--tabular_mlp", action="store_true", default=True, help="Activa el MLP en la rama tabular.")
    parser.add_argument("--no_tabular_mlp", dest="tabular_mlp", action="store_false", help="Desactiva el MLP tabular.")
    parser.add_argument("--d_tab", type=int, default=16, help="Dimensión de salida e_tab del MLP tabular.")
    parser.add_argument("--tab_hidden_dims", nargs="+", type=int, default=[32], help="Dimensiones ocultas del MLP tabular.")
    parser.add_argument("--vocab_size", type=int, default=512, help="Tamaño de vocabulario BPE.")
    parser.add_argument("--d_model", type=int, default=16, help="Dimensión latente del Transformer.")
    parser.add_argument("--d_ff", type=int, default=64, help="Dimensión FFN.")
    parser.add_argument("--num_layers", type=int, default=1, help="Capas de Transformer.")
    parser.add_argument("--n_heads", type=int, default=1, help="Cabezales de atención.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Tasa de dropout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay L2.")
    parser.add_argument("--epochs", type=int, default=15, help="Épocas de entrenamiento.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Semillas aleatorias.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--summary_only", action="store_true", help="Solo genera el gráfico comparativo sin reentrenar.")
    args = parser.parse_args()

    if not args.summary_only:
        run_experiment(
            exp_name=args.exp_name,
            embedding_dim=args.embedding_dim,
            tabular_mlp=args.tabular_mlp,
            d_tab=args.d_tab,
            tab_hidden_dims=args.tab_hidden_dims,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seeds=args.seeds,
            device=args.device,
        )

    generate_demb_comparison()


if __name__ == "__main__":
    main()
