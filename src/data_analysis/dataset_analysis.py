"""Análisis descriptivo exhaustivo del dataset de productos de supermercado.

Este script analiza la información contenida en el dataset en sí mismo,
independientemente de la variable objetivo (BTR). Cubre:
  1. Estructura general y tipos de dato
  2. Valores faltantes (conteo, porcentaje, patrón)
  3. Cardinalidad y valores únicos de variables categóricas
  4. Estadísticas descriptivas y distribuciones de variables numéricas
  5. Detección de outliers (IQR)
  6. Análisis de campos de texto libre (title, description, ingredients, allergens)
  7. Rango temporal y cobertura
  8. Duplicados

Genera un reporte en consola y figuras en results/figures/dataset_info/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Permitir importaciones locales desde src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIGURES_DIR = Path("results/figures/dataset_info")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'figure.autolayout': True,
        'figure.titlesize': 14,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Guardado: {path}")
    return path


def _section(title: str):
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# 1. Estructura general
# ---------------------------------------------------------------------------

def report_structure(df: pd.DataFrame):
    _section("1. ESTRUCTURA GENERAL DEL DATASET")
    print(f"\n  Filas:    {df.shape[0]:,}")
    print(f"  Columnas: {df.shape[1]}")
    print(f"\n  {'Columna':<25} {'Tipo':<15} {'No-Null':>8} {'Null':>6} {'%Null':>7}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*6} {'-'*7}")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null = df[col].isna().sum()
        pct = null / len(df) * 100
        print(f"  {col:<25} {str(df[col].dtype):<15} {non_null:>8,} {null:>6,} {pct:>6.1f}%")


# ---------------------------------------------------------------------------
# 2. Valores faltantes
# ---------------------------------------------------------------------------

def report_missing_values(df: pd.DataFrame):
    _section("2. VALORES FALTANTES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\n  ✅ No hay valores faltantes en el dataset.")
        return
    print(f"\n  {'Columna':<25} {'Faltantes':>10} {'% del total':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col:<25} {count:>10,} {pct:>11.2f}%")
    print(f"\n  Filas completas (sin ningún null): {df.dropna().shape[0]:,} / {len(df):,}"
          f" ({df.dropna().shape[0] / len(df) * 100:.1f}%)")


def plot_missing_values(df: pd.DataFrame):
    """Gráfico de barras con el porcentaje de missings por columna."""
    _setup_style()
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#E74C3C' if p > 0 else '#2ECC71' for p in missing_pct]
    ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xlabel('% de valores faltantes')
    ax.set_title('Porcentaje de Valores Faltantes por Columna')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    for i, (col, val) in enumerate(missing_pct.items()):
        if val > 0:
            ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    _save(fig, "01_missing_values.png")


# ---------------------------------------------------------------------------
# 3. Cardinalidad de variables categóricas
# ---------------------------------------------------------------------------

def report_categorical_cardinality(df: pd.DataFrame):
    _section("3. CARDINALIDAD DE VARIABLES CATEGÓRICAS")
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    print(f"\n  Se detectaron {len(cat_cols)} columnas categóricas/texto/bool.\n")
    print(f"  {'Columna':<25} {'Únicos':>8} {'Top valor':<35} {'Freq Top':>10} {'% Top':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*35} {'-'*10} {'-'*7}")
    for col in cat_cols:
        nunique = df[col].nunique()
        top_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
        top_freq = int(df[col].value_counts().iloc[0]) if nunique > 0 else 0
        top_pct = top_freq / len(df) * 100
        top_display = str(top_val)[:33]
        print(f"  {col:<25} {nunique:>8,} {top_display:<35} {top_freq:>10,} {top_pct:>6.1f}%")


def plot_categorical_distributions(df: pd.DataFrame):
    """Gráficos de barras para las principales variables categóricas de baja cardinalidad."""
    _setup_style()
    low_card_cols = ['category', 'storage_type', 'country_of_origin', 'unit_of_measure', 'cart', 'bought']
    low_card_cols = [c for c in low_card_cols if c in df.columns]

    n_cols = 3
    n_rows = (len(low_card_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else (axes if len(low_card_cols) > 1 else [axes])

    palette = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6', '#E74C3C',
               '#1ABC9C', '#F1C40F', '#34495E', '#D35400', '#8E44AD']

    for i, col in enumerate(low_card_cols):
        counts = df[col].value_counts()
        colors = [palette[j % len(palette)] for j in range(len(counts))]
        axes[i].bar(counts.index.astype(str), counts.values, color=colors,
                    edgecolor='black', alpha=0.85)
        axes[i].set_title(f'Distribución de "{col}" ({df[col].nunique()} únicos)')
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x', rotation=45)
        for bar in axes[i].patches:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2.0, yval,
                         f'{yval:,}', ha='center', va='bottom', fontsize=8)

    # Ocultar axes sobrantes
    for j in range(len(low_card_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribución de Variables Categóricas de Baja Cardinalidad', fontsize=14, fontweight='bold')
    _save(fig, "02_categorical_distributions.png")


# ---------------------------------------------------------------------------
# 4. Estadísticas descriptivas de variables numéricas
# ---------------------------------------------------------------------------

def report_numerical_stats(df: pd.DataFrame):
    _section("4. ESTADÍSTICAS DESCRIPTIVAS — VARIABLES NUMÉRICAS")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n  Se detectaron {len(num_cols)} columnas numéricas.\n")
    stats = df[num_cols].describe().T
    stats['missing'] = df[num_cols].isnull().sum()
    stats['skew'] = df[num_cols].skew()
    cols_display = ['count', 'missing', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew']
    print(stats[cols_display].to_string())


def plot_numerical_histograms(df: pd.DataFrame):
    """Histogramas individuales para cada variable numérica del dataset."""
    _setup_style()
    num_cols = ['price', 'net_weight_oz', 'nutrition_score', 'filter_price_min', 'filter_price_max']
    num_cols = [c for c in num_cols if c in df.columns]

    n_cols_grid = 3
    n_rows = (len(num_cols) + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=35, color='#2980B9', edgecolor='black', alpha=0.8)
        axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                        label=f'Media: {data.mean():.2f}')
        axes[i].axvline(data.median(), color='orange', linestyle=':', linewidth=1.5,
                        label=f'Mediana: {data.median():.2f}')
        axes[i].set_title(f'Distribución de "{col}"')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend(fontsize=8)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribuciones de Variables Numéricas (individuales)', fontsize=14, fontweight='bold')
    _save(fig, "03_numerical_histograms.png")


def plot_boxplots(df: pd.DataFrame):
    """Boxplots para detectar outliers en variables numéricas clave."""
    _setup_style()
    num_cols = ['price', 'net_weight_oz', 'nutrition_score', 'filter_price_min', 'filter_price_max']
    num_cols = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(num_cols), figsize=(3.5 * len(num_cols), 6))
    if len(num_cols) == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        bp = axes[i].boxplot(df[col].dropna(), patch_artist=True, orientation='vertical',
                             boxprops=dict(facecolor='#AED6F1', edgecolor='black'),
                             medianprops=dict(color='red', linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=3, alpha=0.5))
        axes[i].set_title(col, fontsize=11)
        axes[i].set_ylabel('Valor')

    fig.suptitle('Boxplots — Detección de Outliers', fontsize=14, fontweight='bold')
    _save(fig, "04_boxplots_outliers.png")


# ---------------------------------------------------------------------------
# 5. Detección de outliers (IQR)
# ---------------------------------------------------------------------------

def report_outliers(df: pd.DataFrame):
    _section("5. DETECCIÓN DE OUTLIERS (método IQR × 1.5)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir booleanas convertidas a int
    num_cols = [c for c in num_cols if df[c].nunique() > 2]

    print(f"\n  {'Columna':<25} {'Q1':>10} {'Q3':>10} {'IQR':>10} {'Outliers':>10} {'%':>7}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
    for col in num_cols:
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((data < lower) | (data > upper)).sum())
        pct = n_outliers / len(data) * 100
        print(f"  {col:<25} {q1:>10.2f} {q3:>10.2f} {iqr:>10.2f} {n_outliers:>10,} {pct:>6.1f}%")


# ---------------------------------------------------------------------------
# 6. Análisis de campos de texto libre
# ---------------------------------------------------------------------------

def report_text_fields(df: pd.DataFrame):
    _section("6. ANÁLISIS DE CAMPOS DE TEXTO LIBRE")
    text_cols = ['title', 'description', 'ingredients', 'allergens', 'brand',
                 'package_size', 'dimensions_in']
    text_cols = [c for c in text_cols if c in df.columns]

    for col in text_cols:
        series = df[col].dropna()
        lengths = series.astype(str).str.len()
        word_counts = series.astype(str).str.split().str.len()
        print(f"\n  📝 Columna: {col}")
        print(f"     Valores no-nulos: {len(series):,} / {len(df):,}")
        print(f"     Valores únicos:   {series.nunique():,}")
        print(f"     Largo (chars):    min={lengths.min()}, media={lengths.mean():.1f}, "
              f"max={lengths.max()}")
        print(f"     Largo (words):    min={word_counts.min()}, media={word_counts.mean():.1f}, "
              f"max={word_counts.max()}")
        # Mostrar los 5 valores más frecuentes si cardinalidad < 50
        if series.nunique() <= 50:
            print(f"     Top 5 valores:")
            for val, count in series.value_counts().head(5).items():
                print(f"       - \"{val}\" → {count:,} ({count/len(df)*100:.1f}%)")


def plot_text_analysis(df: pd.DataFrame):
    """Distribución de longitud de los campos de texto libre principales."""
    _setup_style()
    text_cols = ['title', 'description', 'ingredients']
    text_cols = [c for c in text_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(text_cols), figsize=(6 * len(text_cols), 5))
    if len(text_cols) == 1:
        axes = [axes]

    colors = ['#2980B9', '#D35400', '#27AE60']
    for i, col in enumerate(text_cols):
        word_counts = df[col].fillna('').str.split().str.len()
        axes[i].hist(word_counts, bins=25, color=colors[i], edgecolor='black', alpha=0.8)
        axes[i].axvline(word_counts.mean(), color='red', linestyle='--', linewidth=1.5,
                        label=f'Media: {word_counts.mean():.1f}')
        axes[i].set_title(f'Longitud de "{col}" (en palabras)')
        axes[i].set_xlabel('Cantidad de palabras')
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend()

    fig.suptitle('Distribución de Longitud de Campos de Texto', fontsize=14, fontweight='bold')
    _save(fig, "05_text_lengths.png")


def plot_allergens_ingredients(df: pd.DataFrame):
    """Frecuencia de alérgenos individuales e ingredientes más comunes."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Alérgenos individuales
    allergens_series = df['allergens'].dropna().str.split(',').explode().str.strip()
    allergen_counts = allergens_series.value_counts()
    axes[0].barh(allergen_counts.index, allergen_counts.values, color='#E74C3C',
                 edgecolor='black', alpha=0.85)
    axes[0].set_title('Frecuencia de Alérgenos Individuales')
    axes[0].set_xlabel('Apariciones')
    for bar in axes[0].patches:
        axes[0].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                     f'{int(bar.get_width()):,}', va='center', fontsize=9)

    # Top 15 ingredientes
    ingredients_series = df['ingredients'].dropna().str.split(',').explode().str.strip()
    ing_counts = ingredients_series.value_counts().head(15).sort_values(ascending=True)
    axes[1].barh(ing_counts.index, ing_counts.values, color='#27AE60',
                 edgecolor='black', alpha=0.85)
    axes[1].set_title('Top 15 Ingredientes Más Frecuentes')
    axes[1].set_xlabel('Apariciones')
    for bar in axes[1].patches:
        axes[1].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                     f'{int(bar.get_width()):,}', va='center', fontsize=9)

    fig.suptitle('Análisis de Alérgenos e Ingredientes', fontsize=14, fontweight='bold')
    _save(fig, "06_allergens_ingredients.png")


# ---------------------------------------------------------------------------
# 7. Rango temporal y cobertura
# ---------------------------------------------------------------------------

def report_temporal_coverage(df: pd.DataFrame):
    _section("7. RANGO TEMPORAL Y COBERTURA")
    if 'timestamp' not in df.columns:
        print("\n  ⚠️  No se encontró la columna 'timestamp'.")
        return
    ts = pd.to_datetime(df['timestamp'], utc=True)
    print(f"\n  Fecha mínima: {ts.min()}")
    print(f"  Fecha máxima: {ts.max()}")
    print(f"  Rango total:  {(ts.max() - ts.min()).days} días")
    print(f"\n  Eventos por año:")
    year_counts = ts.dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"    {year}: {count:,} eventos ({count/len(df)*100:.1f}%)")
    print(f"\n  Eventos por mes (top 5):")
    month_counts = ts.dt.to_period('M').value_counts().sort_values(ascending=False).head(5)
    for month, count in month_counts.items():
        print(f"    {month}: {count:,} eventos")


def plot_temporal_coverage(df: pd.DataFrame):
    """Histograma de eventos por mes y por hora del día."""
    _setup_style()
    if 'timestamp' not in df.columns:
        return
    ts = pd.to_datetime(df['timestamp'], utc=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Eventos por mes
    months = ts.dt.to_period('M').astype(str)
    month_counts = months.value_counts().sort_index()
    axes[0].bar(range(len(month_counts)), month_counts.values, color='#3498DB',
                edgecolor='black', alpha=0.85)
    # Mostrar solo cada N etiquetas para que no se solapen
    step = max(1, len(month_counts) // 12)
    tick_positions = range(0, len(month_counts), step)
    axes[0].set_xticks(list(tick_positions))
    axes[0].set_xticklabels([month_counts.index[i] for i in tick_positions], rotation=45, fontsize=8)
    axes[0].set_title('Cantidad de Eventos por Mes')
    axes[0].set_ylabel('Eventos')
    axes[0].set_xlabel('Mes')

    # Eventos por hora del día
    hour_counts = ts.dt.hour.value_counts().sort_index()
    axes[1].bar(hour_counts.index, hour_counts.values, color='#F39C12',
                edgecolor='black', alpha=0.85, width=0.8)
    axes[1].set_title('Distribución de Eventos por Hora del Día (UTC)')
    axes[1].set_xlabel('Hora (0-23)')
    axes[1].set_ylabel('Eventos')
    axes[1].set_xticks(range(0, 24))

    fig.suptitle('Cobertura Temporal del Dataset', fontsize=14, fontweight='bold')
    _save(fig, "07_temporal_coverage.png")


# ---------------------------------------------------------------------------
# 8. Análisis de Configuraciones de Búsqueda (query_id)
# ---------------------------------------------------------------------------

def report_query_structure(df: pd.DataFrame):
    _section("8. ANÁLISIS DE CONFIGURACIONES DE BÚSQUEDA (query_id)")
    if 'query_id' not in df.columns:
        print("\n  ⚠️  No se encontró la columna 'query_id'.")
        return

    n_queries = df['query_id'].nunique()
    prods_per_query = df.groupby('query_id')['price'].count()
    query_configs = df[['filter_category', 'filter_storage_type', 'filter_price_min', 'filter_price_max']].drop_duplicates()

    print(f"\n  Total de configuraciones de búsqueda (query_id): {n_queries:,}")
    print(f"  Configuraciones únicas de filtros:               {len(query_configs):,}")
    print(f"  Eventos de impresión por query_id:")
    print(f"    - Mínimo:  {prods_per_query.min()}")
    print(f"    - Media:   {prods_per_query.mean():.2f}")
    print(f"    - Mediana: {prods_per_query.median():.0f}")
    print(f"    - Máximo:  {prods_per_query.max()}")


def plot_query_structure(df: pd.DataFrame):
    """Visualiza la distribución de eventos registrados por configuración de query."""
    _setup_style()
    if 'query_id' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    prods_per_query = df.groupby('query_id')['price'].count()
    size_counts = prods_per_query.value_counts().sort_index()
    ax.bar(size_counts.index, size_counts.values, color='#16A085', edgecolor='black', alpha=0.85)
    ax.set_title('Distribución de Eventos por Configuración de Query (query_id)')
    ax.set_xlabel('Cantidad de Eventos / Productos por query_id')
    ax.set_ylabel('Cantidad de query_ids')
    for bar in ax.patches:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{int(yval):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    _save(fig, "08_query_structure.png")


# ---------------------------------------------------------------------------
# 9. Duplicados
# ---------------------------------------------------------------------------

def report_duplicates(df: pd.DataFrame):
    _section("9. ANÁLISIS DE DUPLICADOS")
    n_dup_full = df.duplicated().sum()
    print(f"\n  Filas completamente duplicadas: {n_dup_full:,} / {len(df):,}"
          f" ({n_dup_full/len(df)*100:.2f}%)")

    # Duplicados por subconjuntos clave
    key_sets = {
        'title': ['title'],
        'title + price': ['title', 'price'],
        'title + price + query_id': ['title', 'price', 'query_id'],
        'query_id': ['query_id'],
    }
    print(f"\n  Valores únicos por combinación de columnas clave:")
    for label, cols in key_sets.items():
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            n_unique = df[valid_cols].drop_duplicates().shape[0]
            print(f"    {label:<35} → {n_unique:,} combinaciones únicas")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset_analysis(csv_path: str = "resources/supermarket_products.csv"):
    """Ejecuta el análisis descriptivo completo del dataset."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  📋 ANÁLISIS DESCRIPTIVO DEL DATASET — supermarket_products.csv")
    print("=" * 72)

    print(f"\n  Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas\n")

    # --- Reportes en consola ---
    report_structure(df)
    report_missing_values(df)
    report_categorical_cardinality(df)
    report_numerical_stats(df)
    report_outliers(df)
    report_text_fields(df)
    report_temporal_coverage(df)
    report_query_structure(df)
    report_duplicates(df)

    # --- Figuras ---
    _section("10. GENERANDO FIGURAS")
    print(f"\n  Directorio de salida: {FIGURES_DIR.resolve()}\n")
    plot_missing_values(df)
    plot_categorical_distributions(df)
    plot_numerical_histograms(df)
    plot_boxplots(df)
    plot_text_analysis(df)
    plot_allergens_ingredients(df)
    plot_temporal_coverage(df)
    plot_query_structure(df)

    print(f"\n{'=' * 72}")
    print("  ✅ ANÁLISIS DESCRIPTIVO COMPLETADO")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    run_dataset_analysis()

