"""Detección de títulos que refieren al mismo producto con variaciones menores.

Aplica 3 enfoques incrementales:
  1. Quitar el tag entre paréntesis y buscar duplicados exactos.
  2. Quitar tag + package size y buscar duplicados exactos.
  3. Similitud difusa (fuzzy matching) dentro de cada marca para
     encontrar pares con alta similitud pero no idénticos.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers de normalización
# ---------------------------------------------------------------------------

def strip_tag(title: str) -> str:
    """Remueve el tag entre paréntesis al final del título. Ej: '(Well Reviewed)'."""
    return re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()


def strip_tag_and_size(title: str) -> str:
    """Remueve el tag y el package size. Ej: '- 10 oz (Well Reviewed)'."""
    # Primero quitar el tag
    no_tag = strip_tag(title)
    # Luego quitar el patrón '- <número> <unidad>' al final
    return re.sub(r'\s*-\s*\d+\.?\d*\s*(oz|ct|lb|kg|g|ml|l|pk|pack)\s*$', '', no_tag, flags=re.IGNORECASE).strip()


# ---------------------------------------------------------------------------
# Enfoque 1: Quitar tag y buscar duplicados
# ---------------------------------------------------------------------------

def find_duplicates_without_tag(df: pd.DataFrame):
    print("=" * 80)
    print("  ENFOQUE 1: Quitar tag entre paréntesis y buscar títulos duplicados")
    print("=" * 80)

    df = df.copy()
    df['title_no_tag'] = df['title'].apply(strip_tag)

    n_original_unique = df['title'].nunique()
    n_no_tag_unique = df['title_no_tag'].nunique()
    n_collapsed = n_original_unique - n_no_tag_unique

    print(f"\n  Títulos únicos originales:     {n_original_unique:,}")
    print(f"  Títulos únicos sin tag:        {n_no_tag_unique:,}")
    print(f"  Títulos que colapsan (nuevos duplicados): {n_collapsed:,}")

    # Mostrar grupos donde distintos títulos originales comparten el mismo título sin tag
    dup_groups = df.groupby('title_no_tag').filter(lambda g: g['title'].nunique() > 1)
    if dup_groups.empty:
        print("\n  No se encontraron títulos que colapsen al quitar el tag.")
        return

    groups = dup_groups.groupby('title_no_tag')
    print(f"\n  Grupos de productos con mismo título base pero distinto tag: {groups.ngroups:,}")
    print(f"  Filas involucradas: {len(dup_groups):,}")

    shown = 0
    for base_title, group in groups:
        if shown >= 10:
            print(f"\n  ... y {groups.ngroups - 10} grupos más.")
            break
        variants = group['title'].unique()
        print(f"\n  📦 \"{base_title}\"")
        for v in variants:
            count = (group['title'] == v).sum()
            print(f"     → \"{v}\" ({count} filas)")
        shown += 1


# ---------------------------------------------------------------------------
# Enfoque 2: Quitar tag + package size y buscar duplicados
# ---------------------------------------------------------------------------

def find_duplicates_without_tag_and_size(df: pd.DataFrame):
    print(f"\n\n{'=' * 80}")
    print("  ENFOQUE 2: Quitar tag + package size y buscar títulos duplicados")
    print("=" * 80)

    df = df.copy()
    df['title_no_tag'] = df['title'].apply(strip_tag)
    df['title_base'] = df['title'].apply(strip_tag_and_size)

    n_no_tag_unique = df['title_no_tag'].nunique()
    n_base_unique = df['title_base'].nunique()
    n_collapsed = n_no_tag_unique - n_base_unique

    print(f"\n  Títulos únicos sin tag:              {n_no_tag_unique:,}")
    print(f"  Títulos únicos sin tag ni size:       {n_base_unique:,}")
    print(f"  Títulos adicionales que colapsan:     {n_collapsed:,}")

    # Mostrar grupos donde distintos títulos (sin tag) comparten el mismo base
    dup_groups = df.groupby('title_base').filter(lambda g: g['title_no_tag'].nunique() > 1)
    if dup_groups.empty:
        print("\n  No se encontraron títulos que colapsen al quitar tag + size.")
        return

    groups = dup_groups.groupby('title_base')
    print(f"\n  Grupos de mismo producto con distinto tamaño: {groups.ngroups:,}")
    print(f"  Filas involucradas: {len(dup_groups):,}")

    shown = 0
    for base_title, group in groups:
        if shown >= 10:
            print(f"\n  ... y {groups.ngroups - 10} grupos más.")
            break
        variants = group[['title_no_tag', 'package_size', 'price']].drop_duplicates().sort_values('price')
        print(f"\n  📦 \"{base_title}\"")
        for _, row in variants.iterrows():
            count = (group['title_no_tag'] == row['title_no_tag']).sum()
            print(f"     → \"{row['title_no_tag']}\" | size={row['package_size']} | "
                  f"price=${row['price']:.2f} ({count} filas)")
        shown += 1


# ---------------------------------------------------------------------------
# Enfoque 3: Similitud difusa (fuzzy matching) por marca
# ---------------------------------------------------------------------------

def find_fuzzy_matches(df: pd.DataFrame, threshold: float = 0.85):
    print(f"\n\n{'=' * 80}")
    print(f"  ENFOQUE 3: Similitud difusa (SequenceMatcher, umbral ≥ {threshold})")
    print("=" * 80)

    df = df.copy()
    df['title_no_tag'] = df['title'].apply(strip_tag)

    # Agrupamos por marca para reducir comparaciones
    total_comparisons = 0
    all_matches = []

    brands = df['brand'].unique()
    print(f"\n  Comparando títulos (sin tag) dentro de cada marca ({len(brands)} marcas)...")

    for brand in sorted(brands):
        unique_titles = df[df['brand'] == brand]['title_no_tag'].unique()
        n = len(unique_titles)
        comparisons = n * (n - 1) // 2
        total_comparisons += comparisons

        for i in range(n):
            for j in range(i + 1, n):
                ratio = SequenceMatcher(None, unique_titles[i], unique_titles[j]).ratio()
                if ratio >= threshold and unique_titles[i] != unique_titles[j]:
                    all_matches.append({
                        'brand': brand,
                        'title_a': unique_titles[i],
                        'title_b': unique_titles[j],
                        'similarity': ratio,
                    })

    print(f"  Comparaciones realizadas: {total_comparisons:,}")
    print(f"  Pares con similitud ≥ {threshold}: {len(all_matches):,}")

    if not all_matches:
        print("\n  No se encontraron pares con similitud suficiente.")
        return

    matches_df = pd.DataFrame(all_matches).sort_values('similarity', ascending=False)

    # Agrupar por rangos de similitud
    bins = [(0.95, 1.0), (0.90, 0.95), (0.85, 0.90)]
    for low, high in bins:
        count = ((matches_df['similarity'] >= low) & (matches_df['similarity'] < high)).sum()
        print(f"  Similitud [{low:.0%} - {high:.0%}): {count:,} pares")

    print(f"\n  Top 20 pares más similares:")
    print(f"  {'-'*78}")
    for idx, row in matches_df.head(20).iterrows():
        print(f"\n  [{row['similarity']:.1%}] ({row['brand']})")
        print(f"    A: \"{row['title_a']}\"")
        print(f"    B: \"{row['title_b']}\"")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print(f"\n\n{'=' * 80}")
    print("  RESUMEN FINAL")
    print("=" * 80)

    df = df.copy()
    df['title_no_tag'] = df['title'].apply(strip_tag)
    df['title_base'] = df['title'].apply(strip_tag_and_size)

    print(f"\n  Títulos originales únicos:            {df['title'].nunique():,}")
    print(f"  Títulos únicos (sin tag):             {df['title_no_tag'].nunique():,}")
    print(f"  Títulos únicos (sin tag ni size):     {df['title_base'].nunique():,}")

    # Tags encontrados
    tags = df['title'].str.extract(r'\(([^)]*)\)')[0]
    print(f"\n  Tags encontrados en títulos:")
    for tag, count in tags.value_counts().items():
        print(f"    \"{tag}\": {count:,} ({count/len(df)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("  🔍 DETECCIÓN DE TÍTULOS QUE REFIEREN AL MISMO PRODUCTO")
    print("=" * 80)

    df = pd.read_csv("resources/supermarket_products.csv")
    print(f"\n  Dataset cargado: {len(df):,} filas, {df['title'].nunique():,} títulos únicos\n")

    find_duplicates_without_tag(df)
    find_duplicates_without_tag_and_size(df)
    find_fuzzy_matches(df, threshold=0.85)
    print_summary(df)

    print()


if __name__ == "__main__":
    main()
