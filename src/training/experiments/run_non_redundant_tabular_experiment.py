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
from pathlib import Path
from typing import Sequence

from src.training.experiments.hybrid_evaluation import run_hybrid_experiment

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
SEEDS = [7, 42, 123, 456, 999]


def run_non_redundant_experiment(
    exp_name: str = "non_redundant_tabular_d16",
    d_model: int = 16,
    d_ff: int = 64,
    num_layers: int = 1,
    n_heads: int = 1,
    pos_encoding: str = "sinusoidal",
    fusion_mode: str = "cross",
    fusion_heads: int = 1,
    use_tabular_mlp: bool = False,
    d_tab: int = 16,
    tab_hidden_dims: Sequence[int] = (32,),
    head_hidden_dims: Sequence[int] = (64,),
    seeds: Sequence[int] = SEEDS,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.0003,
    weight_decay: float = 0.05,
    dropout: float = 0.25,
    device: str | None = None,
    plots_only: bool = False,
) -> dict:
    """Ejecuta el experimento de desredundancia tabular usando el pipeline unificado."""
    print("=" * 88)
    print("🔬 EXPERIMENTO: MODELO HÍBRIDO CON RAMA TABULAR DESREDUNDADA")
    print(f"   • Texto: d_model={d_model}, d_ff={d_ff}, L={num_layers}, H={n_heads}")
    print(f"   • Tabular: 17 variables limpias (5 numéricas, 1 embedding país, 1 one-hot día)")
    print(f"   • Tabular MLP: {'Activado (d_tab=' + str(d_tab) + ')' if use_tabular_mlp else 'Desactivado (passthrough directo)'}")
    print(f"   • Fusión: mode={fusion_mode} (heads={fusion_heads})")
    print(f"   • Hiperparámetros: lr={lr}, weight_decay={weight_decay}, dropout={dropout}, epochs={epochs}")
    print(f"   • Semillas: {list(seeds)}")
    print("=" * 88)

    return run_hybrid_experiment(
        exp_name=exp_name,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        n_heads=n_heads,
        pos_encoding=pos_encoding,
        pooling="mean" if fusion_mode != "cross" else "none",
        fusion_mode=fusion_mode,
        fusion_heads=fusion_heads,
        tabular_mlp=use_tabular_mlp,
        d_tab=d_tab,
        tab_hidden_dims=list(tab_hidden_dims),
        head_hidden_dims=list(head_hidden_dims),
        text_fields=["title_clean", "description", "ingredients"],
        numeric_fields=NON_REDUNDANT_NUMERIC_FIELDS,
        log1p_fields=NON_REDUNDANT_LOG1P_FIELDS,
        embedding_fields=NON_REDUNDANT_EMBEDDING_FIELDS,
        onehot_fields=NON_REDUNDANT_ONEHOT_FIELDS,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        no_early_stopping=True,
        device=device,
        plots_only=plots_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del Modelo Híbrido con Rama Tabular Desredundada.")
    parser.add_argument("--exp_name", type=str, default="non_redundant_tabular_d16", help="Nombre del experimento.")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plots_only", action="store_true", help="Solo regenera las figuras desde CSVs existentes.")
    args = parser.parse_args()

    run_non_redundant_experiment(
        exp_name=args.exp_name,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        pos_encoding=args.pos_encoding,
        fusion_mode=args.fusion_mode,
        fusion_heads=args.fusion_heads,
        use_tabular_mlp=args.use_tabular_mlp,
        d_tab=args.d_tab,
        tab_hidden_dims=args.tab_hidden_dims,
        head_hidden_dims=args.head_hidden_dims,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        device=args.device,
        plots_only=args.plots_only,
    )


if __name__ == "__main__":
    main()
