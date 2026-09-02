"""Script ejecutor del Exploratory Data Analysis (EDA).

Ejecuta el pipeline completo de análisis y genera las figuras con matplotlib
en el directorio `results/figures/`.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Permitir importaciones locales desde src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_analysis.dataset import load_raw_data, clean_and_enrich_data, get_dataset_summary
from src.data_analysis.plots import (
    plot_funnel_and_target_distribution,
    plot_btr_by_category_and_storage,
    plot_btr_by_brand,
    plot_btr_by_social_proof_tags,
    plot_numerical_distributions_and_btr,
    plot_query_relative_dynamics,
    plot_text_length_distributions,
    plot_temporal_trends,
    plot_correlation_matrix
)


def run_eda(
    csv_path: str = "resources/datasets/supermarket_products.csv",
    results_dir: str = "results",
    figures_subdir: str = "dataset_info",
):
    """Ejecuta el pipeline completo de EDA y guarda todas las figuras en `results/figures/dataset_info/`."""
    results_path = Path(results_dir)
    figures_path = results_path / "figures" / figures_subdir
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA) - TP1 TRANSFORMERS")
    print("=" * 70)

    print(f"\n📂 1. Cargando datos desde {csv_path}...")
    df_raw = load_raw_data(csv_path)
    print(f"   -> Filas: {df_raw.shape[0]:,}, Columnas: {df_raw.shape[1]}")

    print("\n🧹 2. Limpiando y enriqueciendo dataset con features derivadas...")
    df = clean_and_enrich_data(df_raw)
    summary = get_dataset_summary(df)
    print(f"   -> BTR Global: {summary['btr_global']*100:.2f}%")
    print(f"   -> Cart Rate Global: {summary['cart_global']*100:.2f}%")
    print(f"   -> Queries únicas: {summary['num_queries']:,}")

    print("\n📊 3. Generando visualizaciones con matplotlib en results/figures/...")
    p1 = plot_funnel_and_target_distribution(df, figures_path)
    print(f"   ✓ [1/9] Guardado: {p1.name}")
    p2 = plot_btr_by_category_and_storage(df, figures_path)
    print(f"   ✓ [2/9] Guardado: {p2.name}")
    p3 = plot_btr_by_brand(df, figures_path)
    print(f"   ✓ [3/9] Guardado: {p3.name}")
    p4 = plot_btr_by_social_proof_tags(df, figures_path)
    print(f"   ✓ [4/9] Guardado: {p4.name}")
    p5 = plot_numerical_distributions_and_btr(df, figures_path)
    print(f"   ✓ [5/9] Guardado: {p5.name}")
    p6 = plot_query_relative_dynamics(df, figures_path)
    print(f"   ✓ [6/9] Guardado: {p6.name}")
    p7 = plot_text_length_distributions(df, figures_path)
    print(f"   ✓ [7/9] Guardado: {p7.name}")
    p8 = plot_temporal_trends(df, figures_path)
    print(f"   ✓ [8/9] Guardado: {p8.name}")
    p9 = plot_correlation_matrix(df, figures_path)
    print(f"   ✓ [9/9] Guardado: {p9.name}")

    print("\n" + "=" * 70)
    print("✅ EDA COMPLETADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
