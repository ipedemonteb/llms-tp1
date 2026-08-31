"""Punto de entrada CLI para entrenar el Transformer "pelado".

Implementa la Fase 5 del plan (`src/raw_transformer/PLAN.md`), aplicando la decisión D6:
el loop de entrenamiento NO se reimplementa — se reutiliza `src.training.trainer.Trainer`
(AdamW, `BCEWithLogitsLoss`, early stopping por PR-AUC de validación y restauración del
mejor checkpoint), que consume los batches del `RawSerializedDataset` sin adaptación
porque comparten las claves `input_ids` / `attention_mask` / `labels`.

Decisiones de comparabilidad con `hybrid_transformer`:
- **Mismos defaults de entrenamiento** que `src/training/train.py`: AdamW lr=1e-3,
  weight_decay=0.01, batch_size=64, 20 épocas, patience=5, seed=42, BCE sin ponderar.
- **`--auto_pos_weight`** calcula n_neg/n_pos sobre train (≈ 6,6) como ablation del
  manejo del desbalance; el default queda sin ponderar para que la comparación contra
  las corridas del hybrid sea justa.
- **Resultados en `results/runs_raw/`**, separados de `results/runs/` (cuyo agregador
  está acoplado al parser del hybrid) pero con el mismo formato interno de `summary.json`.

Ejemplos:

    # Defaults comparables con el hybrid (d_model=64, 2 capas, 4 cabezales)
    uv run python -m src.raw_transformer.train

    # Ablation del manejo de desbalance
    uv run python -m src.raw_transformer.train --auto_pos_weight

    # Variante con menos capacidad
    uv run python -m src.raw_transformer.train --d_model 32 --num_layers 1 --n_heads 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.raw_transformer.dataset import build_dataloaders, load_tokenizer
from src.raw_transformer.model import RawTransformerClassifier, RawTransformerConfig
from src.training.metrics import lift_over_baseline
from src.training.trainer import Trainer, TrainerConfig, set_seed


def build_run_name(args: argparse.Namespace, parser_factory) -> str:
    """Genera un nombre autodescriptivo: siempre arquitectura y seed, y los
    hiperparámetros secundarios solo cuando difieren del default (mismo criterio que
    `src.training.config.build_run_name`)."""
    defaults = vars(parser_factory().parse_args([]))
    valores = vars(args)

    partes: List[str] = [valores["data_prefix"], f"d{valores['d_model']}", f"L{valores['num_layers']}",
                         f"H{valores['n_heads']}"]

    opcionales = [("d_ff", "ff"), ("pooling", "pool"), ("pos_encoding", "pos"),
                  ("max_length", "len"), ("dropout", "do"), ("lr", "lr"),
                  ("weight_decay", "wd"), ("batch_size", "bs")]
    for clave, prefijo in opcionales:
        if valores.get(clave) != defaults.get(clave):
            valor = valores[clave]
            partes.append(f"{prefijo}{valor:g}" if isinstance(valor, float) else f"{prefijo}{valor}")

    if valores.get("auto_pos_weight"):
        partes.append("pw")
    partes.append(f"s{valores['seed']}")
    return "_".join(partes)


def build_model(args, vocab_size: int, pad_token_id: int) -> RawTransformerClassifier:
    """Construye el clasificador a partir de los argumentos CLI y los metadatos del tokenizador."""
    return RawTransformerClassifier(RawTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding,
        pooling_mode=args.pooling,
        pad_token_id=pad_token_id,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    ))


def param_breakdown(model: RawTransformerClassifier) -> dict:
    """Desglose de parámetros por componente, con la misma forma que `BTRModel.param_breakdown`."""
    emb = model.encoder.embedding.weight.numel()
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total = model.get_num_params()
    return {
        "embeddings": emb,
        "encoder": total - emb - head,
        "cabeza": head,
        "total": total,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI."""
    parser = argparse.ArgumentParser(description="Entrena el Transformer pelado para predecir BTR.")

    # Datos
    parser.add_argument("--data_dir", type=str, default="resources/datasets")
    parser.add_argument("--data_prefix", type=str, default="raw",
                        help="Prefijo de los CSV serializados ('raw' = preset all; 'raw_po' = product_only).")
    parser.add_argument("--tokenizer_path", type=str, default="resources/tokenizer/bpe_tokenizer_raw.json")
    parser.add_argument("--max_length", type=int, default=256,
                        help="256 cubre el 100%% del corpus serializado (decisión D4).")
    parser.add_argument("--batch_size", type=int, default=64)

    # Arquitectura (mismos defaults que el hybrid para la comparación controlada)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal",
                        choices=["sinusoidal", "learned", "none"])
    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument("--head_dropout", type=float, default=0.2)

    # Entrenamiento (mismos defaults que src/training/train.py)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Peso de la clase positiva en la BCE. Por defecto sin ponderar.")
    parser.add_argument("--auto_pos_weight", action="store_true", default=False,
                        help="Calcula pos_weight = n_neg/n_pos sobre train (ablation del desbalance).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # Salida
    parser.add_argument("--run_name", type=str, default=None,
                        help="Nombre del directorio de resultados. Por defecto se deriva de los hiperparámetros.")
    parser.add_argument("--results_dir", type=str, default="results/runs_raw")

    return parser


def main(argv: Optional[list] = None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.run_name is None:
        args.run_name = build_run_name(args, build_parser)

    set_seed(args.seed)

    print("=" * 88)
    print(f"  ENTRENAMIENTO DEL TRANSFORMER PELADO — run: {args.run_name}")
    print("=" * 88)

    loaders, datasets = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        prefix=args.data_prefix,
    )
    tokenizer = load_tokenizer(args.tokenizer_path, max_length=args.max_length)

    print(f"\n📦 Splits: " + ", ".join(f"{n}={len(ds)}" for n, ds in datasets.items()))
    print(f"   BTR por split: " + ", ".join(f"{n}={ds.positive_rate:.4f}" for n, ds in datasets.items()))
    print(f"   Truncadas: " + ", ".join(f"{n}={ds.n_truncated}" for n, ds in datasets.items()))

    pos_weight = args.pos_weight
    if args.auto_pos_weight:
        pos_weight = float(datasets["train"].pos_weight().item())
        print(f"   pos_weight automático (n_neg/n_pos sobre train): {pos_weight:.3f}")

    model = build_model(args, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id or 0)
    print(f"\n🧠 Parámetros por componente: {param_breakdown(model)}\n")

    trainer = Trainer(model, TrainerConfig(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, pos_weight=pos_weight, device=args.device, seed=args.seed,
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
        "param_breakdown": param_breakdown(model),
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
