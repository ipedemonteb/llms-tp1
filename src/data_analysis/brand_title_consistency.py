"""Análisis de marcas: longitud en palabras y verificación de presencia en el título.

Este script responde dos preguntas del EDA sobre la variable `brand`:

1. **¿Cuántas palabras tiene cada marca?** Relevante porque el título entra al
   Transformer como texto tokenizado: una marca de N palabras ocupa N o más tokens
   dentro de la secuencia, siempre en las primeras posiciones.

2. **¿La marca está contenida en el título en todas las filas?** Si se cumple,
   `brand` ya entra implícitamente a la rama de texto del modelo, lo que cambia
   la justificación de cómo tratarla en la rama tabular (ver
   `resources/documentation/feature_planning.md`).

Se verifican tres niveles de inclusión, de más débil a más fuerte:
    - `brand` contenida en `title` (string crudo del catálogo).
    - `brand` contenida en `title_clean` (sin badge entre paréntesis ni sufijo de medida).
    - `title_clean` **empieza** con `brand` (la marca es el prefijo del título).

Uso:
    uv run python -m src.data_analysis.brand_title_consistency
"""

import re
from typing import Any, Dict

import pandas as pd


# Mismos patrones que `src/data_extraction/clean_dataset.py` para mantener coherencia
RE_BADGE = r'\s*\(.*?\)'
RE_SIZE_SUFFIX = r'\s*-\s*[\d\.]+\s*(oz|fl oz|lb|ct|gal)\s*$'


def clean_title(title_series: pd.Series) -> pd.Series:
    """Limpia el título eliminando el badge entre paréntesis y el sufijo de medida comercial."""
    no_tag = title_series.str.replace(RE_BADGE, '', regex=True)
    pure_title = no_tag.str.replace(RE_SIZE_SUFFIX, '', regex=True)
    return pure_title.str.strip()


def count_words(brand: str) -> int:
    """Cuenta las palabras de una marca separando por espacios (ej. 'Cedar House' -> 2)."""
    return len(str(brand).split())


def count_alpha_words(brand: str) -> int:
    """Cuenta solo las palabras alfabéticas, ignorando símbolos sueltos (ej. 'Oak & Grain' -> 2)."""
    return len(re.findall(r'[A-Za-z]+', str(brand)))


def report_brand_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Reporta la longitud en palabras y caracteres de cada marca, con su frecuencia."""
    print("=" * 88)
    print("  1. INVENTARIO DE MARCAS: LONGITUD EN PALABRAS")
    print("=" * 88)

    counts = df['brand'].value_counts()
    stats = pd.DataFrame({'brand': sorted(df['brand'].dropna().unique())})
    stats['palabras'] = stats['brand'].apply(count_words)
    stats['palabras_alfa'] = stats['brand'].apply(count_alpha_words)
    stats['caracteres'] = stats['brand'].str.len()
    stats['filas'] = stats['brand'].map(counts)
    stats['pct_dataset'] = stats['filas'] / len(df) * 100
    stats = stats.sort_values(['palabras', 'brand']).reset_index(drop=True)

    print(f"\n{'marca':<22}{'palabras':>10}{'pal. alfa':>11}{'caracteres':>12}{'filas':>9}{'% dataset':>11}")
    print("-" * 88)
    for _, r in stats.iterrows():
        print(f"{r['brand']:<22}{r['palabras']:>10}{r['palabras_alfa']:>11}"
              f"{r['caracteres']:>12}{r['filas']:>9,}{r['pct_dataset']:>10.2f}%")

    print("\n  Distribución de marcas por cantidad de palabras:")
    for n_words, group in stats.groupby('palabras'):
        etiqueta = "palabra" if n_words == 1 else "palabras"
        nombres = ", ".join(group['brand'].tolist())
        print(f"    {n_words} {etiqueta:<9}: {len(group):>2} marcas  ({nombres})")

    print(f"\n  Marcas únicas:              {len(stats)}")
    print(f"  Palabras por marca (min/max): {stats['palabras'].min()} / {stats['palabras'].max()}")
    print(f"  Promedio de palabras:        {stats['palabras'].mean():.2f}")

    return stats


def verify_brand_in_title(df: pd.DataFrame, stats: pd.DataFrame) -> Dict[str, Any]:
    """Verifica que la marca esté contenida en el título en todas las filas del dataset."""
    print("\n" + "=" * 88)
    print("  2. VERIFICACIÓN: ¿LA MARCA ESTÁ CONTENIDA EN EL TÍTULO?")
    print("=" * 88)

    title_raw = df['title'].fillna('')
    title_cleaned = clean_title(title_raw)
    brand = df['brand'].fillna('')

    # regex=False es necesario: hay marcas con caracteres especiales ('Market Pantry Co.', 'Oak & Grain')
    in_raw = [b != '' and b in t for b, t in zip(brand, title_raw)]
    in_clean = [b != '' and b in t for b, t in zip(brand, title_cleaned)]
    is_prefix = [b != '' and t.startswith(b) for b, t in zip(brand, title_cleaned)]

    total = len(df)
    checks = [
        ("`brand` contenida en `title` (crudo)", in_raw),
        ("`brand` contenida en `title_clean`", in_clean),
        ("`title_clean` EMPIEZA con `brand`", is_prefix),
    ]

    print()
    for label, mask in checks:
        n_ok = sum(mask)
        icono = "✅" if n_ok == total else "❌"
        print(f"  {icono} {label:<45}: {n_ok:>6,}/{total:,} ({n_ok / total * 100:6.2f}%)")

    # Desglose por marca sobre el criterio más fuerte (prefijo)
    print(f"\n  Desglose por marca (criterio: el título empieza con la marca):")
    print(f"\n{'marca':<22}{'filas':>9}{'cumplen':>10}{'fallan':>9}{'%':>9}")
    print("-" * 88)
    df_check = df.assign(_is_prefix=is_prefix)
    fallas_por_marca = {}
    for b in stats['brand']:
        grupo = df_check[df_check['brand'] == b]
        n_ok = int(grupo['_is_prefix'].sum())
        n_fail = len(grupo) - n_ok
        fallas_por_marca[b] = n_fail
        icono = "" if n_fail == 0 else "  ❌"
        print(f"{b:<22}{len(grupo):>9,}{n_ok:>10,}{n_fail:>9,}{n_ok / len(grupo) * 100:>8.2f}%{icono}")

    # Detalle de las filas que fallan, si las hay
    fallidas = df_check[~df_check['_is_prefix']]
    if len(fallidas) > 0:
        print(f"\n  ⚠️  {len(fallidas):,} filas NO cumplen el criterio de prefijo. Primeros 20 casos:")
        print("-" * 88)
        for idx, row in fallidas.head(20).iterrows():
            print(f"    fila {idx}: brand='{row['brand']}'")
            print(f"              title='{row['title']}'")
            print(f"        title_clean='{clean_title(pd.Series([row['title']]))[0]}'")
    else:
        print("\n  ✅ No hay filas que incumplan el criterio de prefijo.")

    return {
        "total_filas": total,
        "n_marcas": len(stats),
        "ok_en_title_raw": sum(in_raw),
        "ok_en_title_clean": sum(in_clean),
        "ok_prefijo": sum(is_prefix),
        "fallas_por_marca": fallas_por_marca,
    }


def print_summary(resultado: Dict[str, Any]) -> None:
    """Imprime el resumen final y la conclusión para el EDA."""
    total = resultado["total_filas"]
    print("\n" + "=" * 88)
    print("  RESUMEN")
    print("=" * 88)
    print(f"  Filas analizadas:                    {total:,}")
    print(f"  Marcas únicas:                       {resultado['n_marcas']}")
    print(f"  Marca presente en `title` crudo:     {resultado['ok_en_title_raw']:,} "
          f"({resultado['ok_en_title_raw'] / total * 100:.2f}%)")
    print(f"  Marca presente en `title_clean`:     {resultado['ok_en_title_clean']:,} "
          f"({resultado['ok_en_title_clean'] / total * 100:.2f}%)")
    print(f"  Marca como prefijo de `title_clean`: {resultado['ok_prefijo']:,} "
          f"({resultado['ok_prefijo'] / total * 100:.2f}%)")

    if resultado["ok_prefijo"] == total:
        print("\n  CONCLUSIÓN: la marca es el prefijo literal del título en el 100% de las filas.")
        print("  Como `title_clean` es el primer campo de la secuencia de texto, `brand` ya entra")
        print("  al Transformer sin necesidad de agregarla como campo adicional. Mantenerla en la")
        print("  rama tabular sigue siendo útil como sesgo inductivo (identidad nítida y barata),")
        print("  no porque falte información en el texto.")
    else:
        n_fail = total - resultado["ok_prefijo"]
        print(f"\n  CONCLUSIÓN: {n_fail:,} filas no tienen la marca como prefijo del título.")
        print("  Revisar esos casos antes de asumir que `brand` entra implícitamente al Transformer.")


def run_brand_title_analysis(
    csv_path: str = "resources/datasets/supermarket_products.csv",
) -> Dict[str, Any]:
    """Ejecuta el análisis completo de longitud de marcas y su presencia en el título."""
    print("=" * 88)
    print("  ANÁLISIS DE MARCAS: PALABRAS POR MARCA Y PRESENCIA EN EL TÍTULO")
    print("=" * 88)
    print(f"\n📂 Cargando datos desde: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   -> Filas: {df.shape[0]:,}, Columnas: {df.shape[1]}")

    faltantes = [c for c in ('brand', 'title') if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas requeridas en {csv_path}: {faltantes}")

    n_brand_null = int(df['brand'].isna().sum())
    n_title_null = int(df['title'].isna().sum())
    if n_brand_null or n_title_null:
        print(f"   ⚠️  Nulos detectados -> brand: {n_brand_null:,} | title: {n_title_null:,}")

    print()
    stats = report_brand_lengths(df)
    resultado = verify_brand_in_title(df, stats)
    print_summary(resultado)
    print("=" * 88 + "\n")

    return resultado


if __name__ == "__main__":
    run_brand_title_analysis()
