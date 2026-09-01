"""Comparación multi-semilla pareada: Transformer "pelado" vs. `hybrid_transformer`.

Implementa el cierre de la Fase 6 del plan (`src/raw_transformer/PLAN.md`). Reutiliza la
maquinaria estadística de `src.training.aggregate` — media y desvío por configuración, y
diferencias **pareadas por semilla** con intervalo de confianza y t-test — sobre dos
fuentes de corridas que comparten splits (verificado fila por fila) y semillas:

- `results/runs_raw/`: las corridas del raw, agrupadas por preset de campos
  (`raw_all` = los 20 campos, `raw_product_only` = sin contexto de búsqueda).
- `results/runs/`: las corridas del hybrid, agrupadas por `--config` como hace `aggregate`.

El apareamiento por semilla cancela la componente de suerte compartida (mismo split, mismo
orden de shuffle), que con tres semillas es la única forma de distinguir una diferencia
real de ruido: el desvío de las diferencias es mucho menor que la diferencia de los desvíos.

Uso:
    uv run python -m src.raw_transformer.compare
    uv run python -m src.raw_transformer.compare --metric val_pr_auc
    uv run python -m src.raw_transformer.compare --baselines raw_product_only late_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from src.training.aggregate import (
    RESULTS_DIR as HYBRID_RESULTS_DIR,
    Corrida,
    agrupar,
    cargar_corridas,
    comparacion_pareada,
    resumen_por_config,
    tabla_corridas,
    tabla_pareada_markdown,
    tabla_resumen_markdown,
    validar,
)

RAW_RESULTS_DIR = Path("results/runs_raw")
OUTPUT_DIR = Path("results/aggregate")

# Nombre de grupo por prefijo de datos, para que la tabla lea como el informe
PRESET_LABELS = {"raw": "raw_all", "raw_po": "raw_product_only"}

# Las tres comparaciones que responden las preguntas del plan
DEFAULT_BASELINES = ["raw_product_only", "late_fusion", "baseline_tabular"]


def cargar_corridas_raw(results_dir: Path = RAW_RESULTS_DIR) -> Dict[str, List[Corrida]]:
    """Lee los `summary.json` del raw y los agrupa por preset de campos.

    La firma de arquitectura es el `run_name` sin el sufijo de semilla, que `train.py` ya
    construye a partir de los hiperparámetros (misma convención que `build_run_name`).
    """
    grupos: Dict[str, List[Corrida]] = {}
    for ruta in sorted(results_dir.glob("*/summary.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        args = datos.get("args", {})

        metricas = {k: v for k, v in datos.get("test_metrics", {}).items() if v is not None}
        if datos.get("best_val_pr_auc") is not None:
            metricas["val_pr_auc"] = float(datos["best_val_pr_auc"])
        if datos.get("test_lift") is not None:
            metricas["test_lift"] = float(datos["test_lift"])

        prefijo = args.get("data_prefix", "raw")
        grupo = PRESET_LABELS.get(prefijo, prefijo)
        nombre = datos.get("run_name", ruta.parent.name)

        grupos.setdefault(grupo, []).append(Corrida(
            nombre=nombre,
            config=grupo,
            seed=int(args.get("seed", -1)),
            firma="_".join(nombre.split("_")[:-1]),
            epochs=args.get("epochs"),
            patience=args.get("patience"),
            params=int(datos.get("param_breakdown", {}).get("total", 0)),
            best_epoch=int(datos.get("best_epoch", -1)),
            metricas=metricas,
        ))
    return grupos


def cargar_grupos(
    raw_dir: Path = RAW_RESULTS_DIR,
    hybrid_dir: Path = HYBRID_RESULTS_DIR,
) -> Dict[str, List[Corrida]]:
    """Une las corridas del raw y del hybrid en un único diccionario config -> corridas."""
    grupos_hybrid, sueltas = agrupar(cargar_corridas(hybrid_dir)) if hybrid_dir.exists() else ({}, [])
    if sueltas:
        print(f"ℹ️  {len(sueltas)} corridas del hybrid sin --config se ignoran: "
              f"{[c.nombre for c in sueltas]}")
    grupos_raw = cargar_corridas_raw(raw_dir) if raw_dir.exists() else {}

    solapadas = set(grupos_hybrid) & set(grupos_raw)
    if solapadas:
        raise ValueError(f"Nombres de grupo repetidos entre raw y hybrid: {sorted(solapadas)}")
    return {**grupos_raw, **grupos_hybrid}


def comparar(
    metrica: str = "test_pr_auc",
    baselines: Sequence[str] = DEFAULT_BASELINES,
    output_dir: Path = OUTPUT_DIR,
    raw_dir: Path = RAW_RESULTS_DIR,
    hybrid_dir: Path = HYBRID_RESULTS_DIR,
) -> str:
    """Genera las tablas descriptiva y pareadas, las imprime y las guarda en markdown y CSV."""
    grupos = cargar_grupos(raw_dir, hybrid_dir)
    if not grupos:
        raise FileNotFoundError(f"No hay corridas en {raw_dir}/ ni en {hybrid_dir}/.")

    fatales, advertencias = validar(grupos)
    for aviso in advertencias:
        print(f"⚠️  {aviso}")
    if fatales:
        raise ValueError("La grilla de corridas no admite un agregado con sentido:\n  - " + "\n  - ".join(fatales))

    output_dir.mkdir(parents=True, exist_ok=True)

    semillas = {nombre: sorted(c.seed for c in corridas) for nombre, corridas in sorted(grupos.items())}
    print("📦 Corridas por configuración:")
    for nombre, lista in semillas.items():
        print(f"   {nombre:22s} semillas={lista}")

    secciones: List[str] = [f"# Raw Transformer vs. Hybrid — métrica `{metrica}`\n"]

    resumen = resumen_por_config(grupos, metrica)
    secciones.append("## Media por configuración\n\n" + tabla_resumen_markdown(resumen, metrica) + "\n")

    for referencia in baselines:
        if referencia not in grupos:
            print(f"⚠️  Se omite la referencia '{referencia}': no hay corridas.")
            continue
        pareado = comparacion_pareada(grupos, referencia, metrica)
        secciones.append(
            f"## Diferencias pareadas por semilla — referencia `{referencia}`\n\n"
            f"Δ = {referencia} − otra configuración (positivo = la referencia es mejor).\n\n"
            + tabla_pareada_markdown(pareado, referencia) + "\n"
        )
        pareado.to_csv(output_dir / f"raw_vs_hybrid_pareado_{referencia}_{metrica}.csv", index=False)

    informe = "\n".join(secciones)
    print("\n" + informe)

    (output_dir / f"raw_vs_hybrid_{metrica}.md").write_text(informe, encoding="utf-8")
    tabla_corridas(grupos).to_csv(output_dir / "raw_vs_hybrid_corridas.csv", index=False)
    resumen.to_csv(output_dir / f"raw_vs_hybrid_resumen_{metrica}.csv", index=False)
    print(f"💾 Tablas guardadas en {output_dir}/")
    return informe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparación multi-semilla pareada del Transformer pelado contra el hybrid."
    )
    parser.add_argument("--metric", type=str, default="test_pr_auc",
                        help="Métrica a comparar (test_pr_auc, test_roc_auc, val_pr_auc, ...).")
    parser.add_argument("--baselines", type=str, nargs="+", default=DEFAULT_BASELINES,
                        help="Configuraciones de referencia para las tablas pareadas.")
    args = parser.parse_args()
    comparar(metrica=args.metric, baselines=args.baselines)


if __name__ == "__main__":
    main()
