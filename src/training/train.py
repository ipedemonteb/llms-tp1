"""Punto de entrada CLI para entrenar el modelo de predicción de BTR.

Ensambla los DataLoaders, construye el modelo según la configuración pedida, entrena con early
stopping y evalúa una única vez sobre test con el mejor checkpoint de validación.

Ejemplos:

    # Modelo híbrido completo con late fusion (configuración por defecto)
    uv run python -m src.training.train

    # Baseline de texto puro (sin rama tabular)
    uv run python -m src.training.train --no_tabular --run_name baseline_texto

    # Baseline tabular puro (sin Transformer)
    uv run python -m src.training.train --no_text --run_name baseline_tabular

    # Ablación del módulo de fusión
    uv run python -m src.training.train --fusion cross --run_name cross_attention

    # Ablación de la arquitectura del Transformer
    uv run python -m src.training.train --d_model 32 --num_layers 1 --n_heads 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.hybrid_transformer.fusion import BTRModel, FusionConfig
from src.hybrid_transformer.tabular_encoder import TabularEncoder, TabularEncoderConfig
from src.hybrid_transformer.text_encoder import TextTransformerConfig, TextTransformerEncoder
from src.training.dataset import build_dataloaders
from src.training.metrics import lift_over_baseline
from src.training.trainer import Trainer, TrainerConfig


def build_model(args, artefactos) -> BTRModel:
    """Construye el modelo a partir de los argumentos CLI y los metadatos de los datos."""
    text_encoder = None
    if args.use_text:
        text_encoder = TextTransformerEncoder(TextTransformerConfig(
            vocab_size=artefactos["vocab_size"],
            max_seq_len=args.max_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pos_encoding_type=args.pos_encoding,
            pooling_mode="none" if args.fusion == "cross" else args.pooling,
            pad_token_id=artefactos["pad_token_id"],
        ))

    tabular_encoder = None
    if args.use_tabular:
        tabular_encoder = TabularEncoder(TabularEncoderConfig(
            num_numeric=artefactos["num_numeric"],
            cardinalities=artefactos["cardinalities"],
            categorical_encoding=args.cat_encoding,
            d_tab=args.d_tab,
            dropout=args.dropout,
        ))

    # El modo 'cross' exige ambas ramas; con una sola se degrada a 'late'
    modo = args.fusion if (args.use_text and args.use_tabular) else "late"
    fusion_config = FusionConfig(mode=modo, n_heads=args.n_heads, dropout=args.dropout)

    return BTRModel(text_encoder=text_encoder, tabular_encoder=tabular_encoder, fusion_config=fusion_config)


def main(argv: Optional[list] = None) -> dict:
    parser = argparse.ArgumentParser(description="Entrena el modelo de predicción de BTR.")

    # Datos
    parser.add_argument("--data_dir", type=str, default="resources/datasets")
    parser.add_argument("--tokenizer_path", type=str, default="resources/tokenizer/bpe_tokenizer.json")
    parser.add_argument("--text_fields", type=str, default=None,
                        help="Lista separada por comas de los campos de texto. Por defecto usa DEFAULT_TEXT_FIELDS.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)

    # Ramas activas
    parser.add_argument("--no_text", dest="use_text", action="store_false", help="Desactiva la rama de texto.")
    parser.add_argument("--no_tabular", dest="use_tabular", action="store_false", help="Desactiva la rama tabular.")
    parser.set_defaults(use_text=True, use_tabular=True)

    # Arquitectura del Transformer
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal",
                        choices=["sinusoidal", "learned", "none"])

    # Rama tabular y fusión
    parser.add_argument("--d_tab", type=int, default=32)
    parser.add_argument("--cat_encoding", type=str, default="onehot", choices=["onehot", "embedding"])
    parser.add_argument("--fusion", type=str, default="late", choices=["late", "cross"])

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Peso de la clase positiva en la BCE. Por defecto sin ponderar.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # Salida
    parser.add_argument("--run_name", type=str, default="late_fusion")
    parser.add_argument("--results_dir", type=str, default="results/runs")

    args = parser.parse_args(argv)
    text_fields = [c.strip() for c in args.text_fields.split(",")] if args.text_fields else None

    print("=" * 88)
    print(f"  ENTRENAMIENTO DE BTR — run: {args.run_name}")
    print("=" * 88)

    loaders, artefactos = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        text_fields=text_fields,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_text=args.use_text,
        use_tabular=args.use_tabular,
        seed=args.seed,
    )

    print(f"\n📦 Splits: {artefactos['sizes']}")
    print(f"   BTR por split: " + ", ".join(f"{k}={v:.4f}" for k, v in artefactos["positive_rate"].items()))
    print(f"   Ramas activas: texto={args.use_text} | tabular={args.use_tabular} | fusión={args.fusion}")

    model = build_model(args, artefactos)
    print(f"\n🧠 Parámetros por componente: {model.param_breakdown()}\n")

    trainer = Trainer(model, TrainerConfig(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, pos_weight=args.pos_weight, device=args.device, seed=args.seed,
    ))
    trainer.fit(loaders["train"], loaders["val"])

    print("\n" + "=" * 88)
    print("  EVALUACIÓN FINAL SOBRE TEST (mejor checkpoint de validación)")
    print("=" * 88)
    test_metrics = trainer.evaluate(loaders["test"], prefix="test_")
    lift = lift_over_baseline(test_metrics, prefix="test_")

    print(f"  PR-AUC    : {test_metrics['test_pr_auc']:.4f}   "
          f"(línea base = {test_metrics['test_pr_auc_baseline']:.4f}"
          f"{f', lift = {lift:.2f}x' if lift else ''})")
    print(f"  ROC-AUC   : {test_metrics['test_roc_auc']:.4f}   (línea base = 0.5000)")
    print(f"  BCE       : {test_metrics['test_bce']:.4f}")
    print("=" * 88)

    salida = Path(args.results_dir) / args.run_name
    salida.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(salida / "checkpoint.pt")
    resumen = {
        "run_name": args.run_name,
        "args": vars(args),
        "param_breakdown": model.param_breakdown(),
        "best_epoch": trainer.best_epoch,
        "best_val_pr_auc": trainer.best_val_pr_auc,
        "test_metrics": test_metrics,
        "test_lift": lift,
        "history": trainer.history,
    }
    (salida / "summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    trainer.history_dataframe().to_csv(salida / "history.csv", index=False)
    print(f"\n💾 Resultados guardados en: {salida}/\n")

    return resumen


if __name__ == "__main__":
    main()
