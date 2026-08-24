"""Módulo de visualización y generación de gráficos para el análisis exploratorio usando exclusivamente matplotlib."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def setup_matplotlib_style():
    """Configura el estilo base para los gráficos con matplotlib."""
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


def plot_funnel_and_target_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la distribución de la variable objetivo (bought) y el embudo de conversión (Funnel)."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "01_funnel_and_target_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Distribución del Target 'bought'
    bought_counts = df['bought'].value_counts().sort_index()
    bought_pcts = df['bought'].value_counts(normalize=True).sort_index() * 100
    bars1 = axes[0].bar(['No Comprado (False)', 'Comprado (True)'], bought_counts, color=['#4A90E2', '#50E3C2'], width=0.5, edgecolor='black', alpha=0.85)
    axes[0].set_title('Distribución de Variable Objetivo (bought - BTR)')
    axes[0].set_ylabel('Cantidad de Impresiones')
    for bar, pct in zip(bars1, bought_pcts):
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 150, f'{yval:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    axes[0].set_ylim(0, 10000)

    # Subplot 2: Funnel de Conversión (Impresión -> Cart -> Bought)
    total_impressions = len(df)
    total_cart = int(df['cart'].sum())
    total_bought = int(df['bought'].sum())
    stages = ['1. Impresiones Totales', '2. Agregado al Carrito (cart)', '3. Compra Final (bought)']
    values = [total_impressions, total_cart, total_bought]
    pcts_stage = [100.0, (total_cart / total_impressions) * 100, (total_bought / total_impressions) * 100]
    conversion_from_prev = [100.0, (total_cart / total_impressions) * 100, (total_bought / total_cart) * 100 if total_cart > 0 else 0.0]

    bars2 = axes[1].barh(stages[::-1], values[::-1], color=['#2ECC71', '#F39C12', '#34495E'], height=0.5, edgecolor='black', alpha=0.85)
    axes[1].set_title('Embudo de Conversión de Búsqueda (Funnel)')
    axes[1].set_xlabel('Eventos')
    for bar, val, pct, conv in zip(bars2, values[::-1], pcts_stage[::-1], conversion_from_prev[::-1]):
        axes[1].text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2.0,
                     f'{val:,} ({pct:.1f}% del total)\n[Conv. paso: {conv:.1f}%]',
                     ha='left', va='center', fontsize=9, fontweight='bold')
    axes[1].set_xlim(0, 13500)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_btr_by_category_and_storage(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica el BTR por categoría y por tipo de almacenamiento."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "02_btr_by_category_and_storage.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Categorías ordenadas por BTR descendente
    cat_stats = df.groupby('category')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=True)
    btr_global = df['bought'].mean() * 100

    y_pos = np.arange(len(cat_stats))
    bars1 = axes[0].barh(y_pos, cat_stats['mean'] * 100, color='#3498DB', edgecolor='black', alpha=0.85, height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(cat_stats.index)
    axes[0].axvline(btr_global, color='red', linestyle='--', linewidth=1.5, label=f'BTR Global ({btr_global:.1f}%)')
    axes[0].set_title('BTR por Categoría de Producto')
    axes[0].set_xlabel('Buy Through Rate (%)')
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc='lower right')
    for bar, count in zip(bars1, cat_stats['count']):
        axes[0].text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2.0, f'{bar.get_width():.1f}% (N={count})', ha='left', va='center', fontsize=8.5)
    axes[0].set_xlim(0, max(cat_stats['mean'] * 100) + 6)

    # Storage type
    storage_stats = df.groupby('storage_type')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    bars2 = axes[1].bar(storage_stats.index, storage_stats['mean'] * 100, color=['#E67E22', '#9B59B6', '#1ABC9C'], width=0.45, edgecolor='black', alpha=0.85)
    axes[1].axhline(btr_global, color='red', linestyle='--', linewidth=1.5, label=f'BTR Global ({btr_global:.1f}%)')
    axes[1].set_title('BTR por Tipo de Almacenamiento')
    axes[1].set_ylabel('Buy Through Rate (%)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc='upper right')
    for bar, count in zip(bars2, storage_stats['count']):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.3, f'{bar.get_height():.1f}%\n(N={count:,})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, max(storage_stats['mean'] * 100) + 4)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_btr_by_brand(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica el BTR por Marca."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "03_btr_by_brand.png"

    brand_stats = df.groupby('brand')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=True)
    btr_global = df['bought'].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(brand_stats))
    colors = ['#E74C3C' if m < (btr_global/100) else '#2ECC71' for m in brand_stats['mean']]
    bars = ax.barh(y_pos, brand_stats['mean'] * 100, color=colors, edgecolor='black', alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(brand_stats.index)
    ax.axvline(btr_global, color='black', linestyle='--', linewidth=1.5, label=f'Promedio Global ({btr_global:.1f}%)')
    ax.set_title('Buy Through Rate (BTR) por Marca')
    ax.set_xlabel('Buy Through Rate (%)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='lower right')

    for bar, count in zip(bars, brand_stats['count']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.0, f'{bar.get_width():.1f}% (N={count})', ha='left', va='center', fontsize=9)
    ax.set_xlim(0, max(brand_stats['mean'] * 100) + 4)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_btr_by_social_proof_tags(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica el impacto determinante de los tags de reputación/social proof extraídos del título."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "04_btr_by_social_proof_tags.png"

    tag_stats = df.groupby('title_tag')['bought'].agg(['count', 'mean']).sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(tag_stats))
    colors = ['#27AE60' if m > 0.4 else ('#F39C12' if m > 0.0 else '#95A5A6') for m in tag_stats['mean']]
    bars = ax.barh(y_pos, tag_stats['mean'] * 100, color=colors, edgecolor='black', alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tag_stats.index)
    ax.set_title('Impacto de Badges / Tags Semánticos del Título en el BTR')
    ax.set_xlabel('Buy Through Rate (%)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, count in zip(bars, tag_stats['count']):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2.0, f'{bar.get_width():.1f}% (N={count})', ha='left', va='center', fontsize=8.5)
    ax.set_xlim(0, 80)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_numerical_distributions_and_btr(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica histogramas y relación de BTR por cuantiles para variables numéricas clave."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "05_numerical_distributions_and_btr.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Price Distribution and BTR
    axes[0, 0].hist(df[df['bought'] == False]['price'], bins=30, alpha=0.6, label='No Comprado', color='#3498DB', density=True)
    axes[0, 0].hist(df[df['bought'] == True]['price'], bins=30, alpha=0.6, label='Comprado', color='#2ECC71', density=True)
    axes[0, 0].set_title('Distribución de Precios (Price)')
    axes[0, 0].set_xlabel('Precio ($ USD)')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].legend()

    # 2. Nutrition Score Distribution and BTR
    axes[0, 1].hist(df[df['bought'] == False]['nutrition_score'], bins=25, alpha=0.6, label='No Comprado', color='#3498DB', density=True)
    axes[0, 1].hist(df[df['bought'] == True]['nutrition_score'], bins=25, alpha=0.6, label='Comprado', color='#2ECC71', density=True)
    axes[0, 1].set_title('Distribución de Nutrition Score')
    axes[0, 1].set_xlabel('Nutrition Score (0 - 100)')
    axes[0, 1].set_ylabel('Densidad')
    axes[0, 1].legend()

    # 3. Net Weight (oz)
    axes[1, 0].hist(df[df['bought'] == False]['net_weight_oz'], bins=30, alpha=0.6, label='No Comprado', color='#3498DB', density=True)
    axes[1, 0].hist(df[df['bought'] == True]['net_weight_oz'], bins=30, alpha=0.6, label='Comprado', color='#2ECC71', density=True)
    axes[1, 0].set_title('Distribución de Peso Neto (net_weight_oz)')
    axes[1, 0].set_xlabel('Peso Neto (oz)')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].legend()

    # 4. Volume (cu in)
    valid_vol = df['volume_cu_in'].dropna()
    axes[1, 1].hist(df[df['bought'] == False]['volume_cu_in'], bins=30, alpha=0.6, label='No Comprado', color='#3498DB', density=True)
    axes[1, 1].hist(df[df['bought'] == True]['volume_cu_in'], bins=30, alpha=0.6, label='Comprado', color='#2ECC71', density=True)
    axes[1, 1].set_title('Distribución de Volumen Físico (Pulgadas Cúbicas)')
    axes[1, 1].set_xlabel('Volumen (in³)')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_query_relative_dynamics(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la dinámica competitiva dentro de cada query (ranking de precio y posición relativa)."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "06_query_relative_dynamics.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BTR por Rango de Precio dentro de la Query (int rank)
    int_ranks = df[df['price_rank_in_query'].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    rank_stats = int_ranks.groupby('price_rank_in_query')['bought'].agg(['count', 'mean'])
    bars1 = axes[0].bar(rank_stats.index.astype(int), rank_stats['mean'] * 100, color='#8E44AD', width=0.5, edgecolor='black', alpha=0.85)
    axes[0].set_title('BTR según Ranking de Precio dentro de la Búsqueda')
    axes[0].set_xlabel('Ranking de Precio en la Query (1 = más barato)')
    axes[0].set_ylabel('Buy Through Rate (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, count in zip(bars1, rank_stats['count']):
        axes[0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.4, f'{bar.get_height():.1f}%\n(N={count})', ha='center', va='bottom', fontsize=8.5)
    axes[0].set_ylim(0, 22)

    # Cantidad de productos por query
    query_sizes = df.groupby('query_id')['price'].count()
    size_counts = query_sizes.value_counts().sort_index()
    bars2 = axes[1].bar(size_counts.index, size_counts.values, color='#16A085', width=0.5, edgecolor='black', alpha=0.85)
    axes[1].set_title('Distribución de Cantidad de Productos por Búsqueda (Query Size)')
    axes[1].set_xlabel('Cantidad de Productos Mostrados')
    axes[1].set_ylabel('Cantidad de Queries')
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, max(size_counts.values) + 50)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_text_length_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la distribución de longitud de palabras y caracteres para Title y Description."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "07_text_length_distributions.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Longitud de Títulos
    title_words = df['title_word_count']
    axes[0].hist(title_words, bins=15, color='#2980B9', edgecolor='black', alpha=0.8)
    axes[0].axvline(title_words.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Media: {title_words.mean():.1f} palabras')
    axes[0].axvline(title_words.max(), color='orange', linestyle=':', linewidth=1.5, label=f'Máx: {title_words.max()} palabras')
    axes[0].set_title('Distribución de Longitud de Palabras: Title')
    axes[0].set_xlabel('Cantidad de Palabras')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()

    # Longitud de Descripciones
    desc_words = df['desc_word_count']
    axes[1].hist(desc_words, bins=15, color='#D35400', edgecolor='black', alpha=0.8)
    axes[1].axvline(desc_words.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Media: {desc_words.mean():.1f} palabras')
    axes[1].axvline(desc_words.max(), color='purple', linestyle=':', linewidth=1.5, label=f'Máx: {desc_words.max()} palabras')
    axes[1].set_title('Distribución de Longitud de Palabras: Description')
    axes[1].set_xlabel('Cantidad de Palabras')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_temporal_trends(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la evolución temporal del BTR por mes y por día de la semana."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "08_temporal_trends.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Evolución mensual
    monthly = df.groupby('year_month')['bought'].agg(['count', 'mean']).sort_index()
    axes[0].plot(monthly.index, monthly['mean'] * 100, marker='o', color='#2980B9', linewidth=2, label='BTR Mensual (%)')
    btr_global = df['bought'].mean() * 100
    axes[0].axhline(btr_global, color='red', linestyle='--', linewidth=1.5, label=f'Promedio Global ({btr_global:.1f}%)')
    axes[0].set_title('Evolución Temporal del BTR por Mes')
    axes[0].set_ylabel('BTR (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()

    # BTR por día de la semana
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('day_name')['bought'].agg(['count', 'mean']).reindex(days_order)
    bars = axes[1].bar(dow_stats.index, dow_stats['mean'] * 100, color='#27AE60', width=0.5, edgecolor='black', alpha=0.85)
    axes[1].axhline(btr_global, color='red', linestyle='--', linewidth=1.5, label=f'Promedio Global ({btr_global:.1f}%)')
    axes[1].set_title('BTR según Día de la Semana')
    axes[1].set_ylabel('BTR (%)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend()
    for bar, count in zip(bars, dow_stats['count']):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.3, f'{bar.get_height():.1f}%\n(N={count:,})', ha='center', va='bottom', fontsize=8.5)
    axes[1].set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la matriz de correlación de variables numéricas y targets con matplotlib."""
    setup_matplotlib_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "09_correlation_matrix.png"

    num_cols = [
        'cart', 'bought', 'price', 'filter_price_min', 'filter_price_max',
        'net_weight_oz', 'nutrition_score', 'length_in', 'width_in', 'height_in',
        'volume_cu_in', 'num_ingredients', 'num_allergens', 'price_rank_in_query'
    ]
    available_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(available_cols))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(available_cols, rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(available_cols, fontsize=9)

    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Matriz de Correlaciones Lineales (Pearson)', pad=30, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file
