"""Análisis de la estructura de oraciones en las descripciones del dataset."""

import re
from collections import Counter

import pandas as pd


def count_sentences(text: str) -> int:
    """Cuenta oraciones separadas por punto."""
    if not isinstance(text, str) or not text.strip():
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)


def main():
    df = pd.read_csv("resources/supermarket_products.csv")

    df['n_sentences'] = df['description'].apply(count_sentences)

    print("=" * 70)
    print("  ANÁLISIS DE ORACIONES EN LAS DESCRIPCIONES")
    print("=" * 70)

    # Distribución de cantidad de oraciones
    counts = df['n_sentences'].value_counts().sort_index()
    print(f"\n  Distribución de cantidad de oraciones por descripción:\n")
    print(f"  {'Oraciones':>10} {'Filas':>8} {'%':>7}")
    print(f"  {'-'*10} {'-'*8} {'-'*7}")
    for n, c in counts.items():
        print(f"  {n:>10} {c:>8,} {c/len(df)*100:>6.1f}%")

    print(f"\n  Total filas: {len(df):,}")
    print(f"  Media: {df['n_sentences'].mean():.2f} oraciones")

    # Mostrar ejemplos para cada cantidad de oraciones
    print(f"\n{'=' * 70}")
    print("  EJEMPLOS POR CANTIDAD DE ORACIONES")
    print("=" * 70)

    for n in sorted(df['n_sentences'].unique()):
        examples = df[df['n_sentences'] == n]['description'].head(3)
        print(f"\n  --- {n} oración(es) ---")
        for desc in examples:
            print(f"    \"{desc}\"")

    # Analizar la estructura: ¿qué dice cada oración?
    most_common = df['n_sentences'].mode().iloc[0]
    subset = df[df['n_sentences'] == most_common]

    print(f"\n{'=' * 70}")
    print(f"  ESTRUCTURA DE LAS DESCRIPCIONES CON {most_common} ORACIONES")
    print("=" * 70)

    for i in range(most_common):
        sentence_i = subset['description'].apply(
            lambda x: [s.strip() for s in x.split('.') if s.strip()][i]
            if len([s.strip() for s in x.split('.') if s.strip()]) > i else ''
        )
        # Primeras palabras más comunes
        first_words = sentence_i.str.split().str[:3].str.join(' ')
        top_starts = first_words.value_counts().head(10)
        print(f"\n  Oración {i+1} — Inicios más frecuentes:")
        for start, c in top_starts.items():
            print(f"    \"{start}...\" → {c:,} veces ({c/len(subset)*100:.1f}%)")

        # Últimas palabras más comunes
        last_words = sentence_i.str.split().str[-3:].str.join(' ')
        top_ends = last_words.value_counts().head(10)
        print(f"\n  Oración {i+1} — Finales más frecuentes:")
        for end, c in top_ends.items():
            print(f"    \"...{end}\" → {c:,} veces ({c/len(subset)*100:.1f}%)")


if __name__ == "__main__":
    main()
