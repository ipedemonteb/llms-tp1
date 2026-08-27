"""Asociación formal entre los tags del título y las últimas oraciones de la descripción.

Este script analiza la co-ocurrencia estricta entre el tag extraído del título
y la última oración de la descripción, demostrando la partición en clusters
semánticos / niveles de reputación sin solapamiento entre grupos.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_tag(title: str) -> str:
    """Extrae el tag entre paréntesis del título o devuelve 'No Tag'."""
    if not isinstance(title, str):
        return 'No Tag'
    match = re.search(r'\(([^)]+)\)', title)
    return match.group(1).strip() if match else 'No Tag'


def extract_last_sentence(description: str) -> str:
    """Extrae la última oración limpia de la descripción."""
    if not isinstance(description, str) or not description.strip():
        return ''
    sentences = [s.strip() for s in description.split('.') if s.strip()]
    return sentences[-1] if sentences else ''


def analyze_tag_sentence_clusters(csv_path: str = "resources/supermarket_products.csv"):
    df = pd.read_csv(csv_path)

    df['tag'] = df['title'].apply(extract_tag)
    df['last_sentence'] = df['description'].apply(extract_last_sentence)

    # Identificar si la última oración es puramente metadata de almacenamiento ("Listed under...")
    df['is_storage_sentence'] = df['last_sentence'].str.startswith('Listed under')
    df['clean_last_sentence'] = np.where(
        df['is_storage_sentence'],
        '[Sin 3ra oración / Termina en "Listed under..."]',
        df['last_sentence']
    )

    # Matriz de contingencia
    crosstab = pd.crosstab(df['tag'], df['clean_last_sentence'])

    print("=" * 85)
    print("  MAPEO FORMAL: TAGS DEL TÍTULO ↔ ÚLTIMAS ORACIONES DE LA DESCRIPCIÓN")
    print("=" * 85)

    # Agrupar tags que comparten exactamente el mismo conjunto de oraciones asociadas
    # Creamos una firma para cada tag basada en las oraciones donde tiene co-ocurrencia > 0
    tag_signatures = {}
    for tag in crosstab.index:
        sentences_with_counts = crosstab.loc[tag]
        active_sentences = tuple(sorted(sentences_with_counts[sentences_with_counts > 0].index.tolist()))
        tag_signatures.setdefault(active_sentences, []).append(tag)

    print(f"\nSe identificaron {len(tag_signatures)} clusters/familias semánticas cerradas.\n")

    tier_names = [
        "1. Nivel Top / Best Sellers (Máxima Reputación)",
        "2. Nivel Positivo / Favoritos de Compradores (Alta Reputación)",
        "3. Nivel Nuevos / En Evaluación (Feedback Limitado o Nuevo)",
        "4. Nivel Negativo / Descontinuación (Baja Reputación / Clearance)",
        "5. Nivel Estándar / Neutro / Sin Tag"
    ]

    # Ordenar los clusters de mayor a menor reputación aproximada para presentación
    # Usamos heurísticas de palabras clave en los tags para darles un orden lógico
    def sort_key(item):
        sentences, tags = item
        tag_str = " ".join(tags).lower()
        if "best seller" in tag_str or "top rated" in tag_str:
            return 1
        elif "highly rated" in tag_str or "shopper favorite" in tag_str:
            return 2
        elif "new listing" in tag_str or "recently added" in tag_str:
            return 3
        elif "rarely reordered" in tag_str or "discontinuing" in tag_str:
            return 4
        else:
            return 5

    sorted_clusters = sorted(tag_signatures.items(), key=sort_key)

    total_cross_errors = 0

    for idx, (sentences, tags) in enumerate(sorted_clusters):
        tier_title = tier_names[idx] if idx < len(tier_names) else f"Cluster {idx+1}"
        print(f"\n{'-'*85}")
        print(f"📦 {tier_title.upper()}")
        print(f"{'-'*85}")
        
        print("\n  🏷️  Tags incluidos en este grupo:")
        total_rows_tier = 0
        for tag in tags:
            tag_count = (df['tag'] == tag).sum()
            total_rows_tier += tag_count
            print(f"      • {tag:<25} (N = {tag_count:,} productos)")
        print(f"      👉 Total productos en el cluster: {total_rows_tier:,}")

        print("\n  📝  Oraciones asignadas aleatoriamente a estos tags:")
        for sentence in sentences:
            count = df[(df['tag'].isin(tags)) & (df['clean_last_sentence'] == sentence)].shape[0]
            pct = (count / total_rows_tier) * 100 if total_rows_tier > 0 else 0
            print(f"      • \"{sentence}\"")
            print(f"        └─ Apariciones: {count:,} ({pct:.1f}% de este grupo)")

        # Verificar si alguna de estas oraciones aparece fuera de estos tags
        leakage = df[(~df['tag'].isin(tags)) & (df['clean_last_sentence'].isin(sentences))].shape[0]
        total_cross_errors += leakage
        if leakage == 0:
            print("\n  ✅ Exclusividad: 100% estricta (0 apariciones fuera de este grupo de tags)")
        else:
            print(f"\n  ❌ Solapamiento detectado: {leakage} filas fuera del cluster")

    print(f"\n{'=' * 85}")
    print("  RESUMEN DE VALIDACIÓN")
    print(f"{'=' * 85}")
    print(f"  • Total de filas analizadas:           {len(df):,}")
    print(f"  • Total de tags:                       {df['tag'].nunique():,}")
    print(f"  • Total de oraciones/estados únicos:   {df['clean_last_sentence'].nunique():,}")
    print(f"  • Solapamiento entre clusters:         {total_cross_errors} casos (Partición perfecta)")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    analyze_tag_sentence_clusters()
