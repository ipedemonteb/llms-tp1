"""Reporte de todas las últimas oraciones distintas en las descripciones del dataset.

Extrae la última oración de cada descripción y reporta todas las variantes
únicas con su frecuencia.
"""

import pandas as pd


def extract_last_sentence(text: str) -> str:
    """Extrae la última oración de una descripción."""
    if not isinstance(text, str) or not text.strip():
        return ''
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences[-1] if sentences else ''


def main():
    df = pd.read_csv("resources/datasets/supermarket_products.csv")

    df['last_sentence'] = df['description'].apply(extract_last_sentence)

    unique_sentences = df['last_sentence'].value_counts()

    print("=" * 80)
    print("  ÚLTIMAS ORACIONES ÚNICAS EN LAS DESCRIPCIONES")
    print("=" * 80)
    print(f"\n  Total de filas: {len(df):,}")
    print(f"  Últimas oraciones distintas: {len(unique_sentences):,}")

    print(f"\n  {'#':>4}  {'Frecuencia':>10}  {'%':>6}  Oración")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*55}")

    for i, (sentence, count) in enumerate(unique_sentences.items(), 1):
        pct = count / len(df) * 100
        print(f"  {i:>4}  {count:>10,}  {pct:>5.1f}%  \"{sentence}\"")

    # Verificar si las de 2 oraciones caen en "Listed under..."
    print(f"\n\n{'=' * 80}")
    print("  DETALLE: ÚLTIMAS ORACIONES QUE EMPIEZAN CON 'Listed under'")
    print("=" * 80)
    listed = df[df['last_sentence'].str.startswith('Listed under')]
    print(f"\n  Filas cuya última oración es 'Listed under...': {len(listed):,}")
    print(f"  (Estas son las descripciones de 2 oraciones, sin frase de reputación)\n")
    for sentence, count in listed['last_sentence'].value_counts().head(15).items():
        print(f"    \"{sentence}\" → {count:,}")

    # Las que NO empiezan con "Listed under" = frases de reputación
    print(f"\n\n{'=' * 80}")
    print("  DETALLE: FRASES DE REPUTACIÓN (últimas oraciones que NO son 'Listed under')")
    print("=" * 80)
    reputation = df[~df['last_sentence'].str.startswith('Listed under')]
    rep_counts = reputation['last_sentence'].value_counts()
    print(f"\n  Filas con frase de reputación: {len(reputation):,}")
    print(f"  Frases de reputación distintas: {len(rep_counts):,}")

    print(f"\n  {'#':>4}  {'Freq':>6}  {'%':>6}  Frase")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*55}")
    for i, (sentence, count) in enumerate(rep_counts.items(), 1):
        pct = count / len(df) * 100
        print(f"  {i:>4}  {count:>6,}  {pct:>5.1f}%  \"{sentence}\"")


if __name__ == "__main__":
    main()
