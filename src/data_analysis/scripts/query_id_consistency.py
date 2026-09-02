"""Verificación de consistencia de campos compartidos dentro de cada query_id.

Para cada query_id, agrupa todas las filas y verifica si los campos de filtros
(filter_category, filter_price_min, filter_price_max, filter_storage_type)
son idénticos en todas las filas del grupo.

Reporta por cada query_id si hay consistencia o no, y un resumen final.
"""

import pandas as pd


def verify_query_id_consistency(csv_path: str = "resources/datasets/supermarket_products.csv"):
    df = pd.read_csv(csv_path)

    filter_cols = ['filter_category', 'filter_price_min', 'filter_price_max', 'filter_storage_type']
    query_ids = df['query_id'].unique()

    print(f"Total de query_ids únicos: {len(query_ids):,}")
    print(f"Columnas verificadas: {filter_cols}")
    print(f"\n{'='*80}")

    consistent = []
    inconsistent = []

    for qid in sorted(query_ids):
        group = df[df['query_id'] == qid]
        n_rows = len(group)

        # Para cada columna de filtro, verificar si tiene un único valor en el grupo
        col_results = {}
        all_ok = True
        for col in filter_cols:
            unique_vals = group[col].dropna().unique()
            n_unique = len(unique_vals)
            if n_unique <= 1:
                col_results[col] = {"ok": True, "values": unique_vals.tolist()}
            else:
                col_results[col] = {"ok": False, "values": unique_vals.tolist()}
                all_ok = False

        if all_ok:
            consistent.append(qid)
            vals = {col: col_results[col]["values"][0] if col_results[col]["values"] else "NULL" for col in filter_cols}
            print(f"  ✅ {qid} ({n_rows} filas) — Consistente "
                  f"[cat={vals['filter_category']}, price={vals['filter_price_min']}-{vals['filter_price_max']}, "
                  f"storage={vals['filter_storage_type']}]")
        else:
            inconsistent.append(qid)
            print(f"  ❌ {qid} ({n_rows} filas) — INCONSISTENTE:")
            for col in filter_cols:
                r = col_results[col]
                status = "✅" if r["ok"] else "❌"
                print(f"       {status} {col}: {r['values']}")

    # Resumen
    print(f"\n{'='*80}")
    print(f"  RESUMEN")
    print(f"{'='*80}")
    print(f"  Query IDs totales:       {len(query_ids):,}")
    print(f"  Consistentes:            {len(consistent):,} ({len(consistent)/len(query_ids)*100:.1f}%)")
    print(f"  Inconsistentes:          {len(inconsistent):,} ({len(inconsistent)/len(query_ids)*100:.1f}%)")

    if inconsistent:
        print(f"\n  Query IDs inconsistentes: {inconsistent}")


if __name__ == "__main__":
    verify_query_id_consistency()
