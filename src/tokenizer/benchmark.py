"""Suite de pruebas y visualización de diagnóstico para el tokenizador Byte-Level BPE.

Genera 6 figuras diagnósticas usando exclusivamente matplotlib:
1. `01_vocab_size_vs_longitud_y_saturacion.png` — Barrido de vocab_size vs largo de secuencia y saturación de merges.
2. `02_distribucion_longitud_y_truncamiento.png`  — Histograma de largos y análisis de corte para max_length.
3. `03_fertilidad_y_fragmentacion_subpalabras.png` — Fertilidad léxica (tokens/palabra) y distribución de fragmentos.
4. `04_impacto_min_frequency.png`                 — Sensibilidad a la frecuencia mínima de aparición.
5. `05_ley_de_zipf_y_frecuencia_tokens.png`        — Distribución rango-frecuencia log-log (Ley de Zipf).
6. `06_tradeoff_memoria_vs_computo_atencion.png`   — Matriz de embeddings vs operaciones de atención O(T^2).

Uso:
    uv run python -m src.tokenizer.benchmark
    uv run python -m src.tokenizer.benchmark --data_path resources/datasets/transformer_train.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tokenizer.bpe import ByteLevelBPETokenizer

# Paleta editorial consistente con el resto del proyecto
SERIE_1 = "#2a78d6"    # Azul
SERIE_2 = "#eb6834"    # Naranja
SERIE_3 = "#1b9e77"    # Verde
SERIE_4 = "#7570b3"    # Violeta
SERIE_5 = "#e7298a"    # Magenta
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTE = "#8a8985"
GRILLA = "#e6e5e1"
SUPERFICIE = "#fcfcfb"

FIGURES_DIR = Path("results/figures/tokenizer")


def aplicar_estilo() -> None:
    """Configura los estilos base de matplotlib para una presentación nítida y profesional."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    })


def cargar_textos(csv_path: Path) -> List[str]:
    """Carga y arma la secuencia de texto del dataset ('title_clean | description | ingredients')."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecutar primero:\n"
            "  uv run python -m src.data_extraction.build_transformer_dataset"
        )
    df = pd.read_csv(csv_path)
    campos = ["title_clean", "description", "ingredients"]
    for c in campos:
        if c not in df.columns:
            raise KeyError(f"Columna requerida ausente en {csv_path}: {c}")

    textos = (
        df["title_clean"].fillna("")
        + " | "
        + df["description"].fillna("")
        + " | "
        + df["ingredients"].fillna("")
    ).tolist()
    return textos


# --- Figura 1: Barrido de vocab_size vs Longitud y Saturación ----------------------------

def plot_vocab_size_sweep(
    textos_train: List[str],
    textos_eval: List[str],
    vocab_sizes: Sequence[int] = (250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000),
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Analiza cómo la longitud de secuencia disminuye con vocab_size y dónde se satura."""
    aplicar_estilo()

    medias, p50s, p95s, maxs, vocabs_reales = [], [], [], [], []

    for vs in vocab_sizes:
        tok = ByteLevelBPETokenizer().train_from_iterator(
            textos_train, vocab_size=vs, min_frequency=2, show_progress=False
        )
        vocabs_reales.append(tok.vocab_size)
        lens = [len(tok.encode(t, add_special_tokens=True).ids) for t in textos_eval]
        medias.append(np.mean(lens))
        p50s.append(np.median(lens))
        p95s.append(np.percentile(lens, 95))
        maxs.append(np.max(lens))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Compresión de secuencia
    ax1.plot(vocab_sizes, medias, "o-", color=SERIE_1, linewidth=2.2, label="Media de tokens")
    ax1.plot(vocab_sizes, p50s, "s--", color=SERIE_3, linewidth=1.8, label="Mediana (P50)")
    ax1.plot(vocab_sizes, p95s, "^--", color=SERIE_2, linewidth=1.8, label="Percentil 95 (P95)")
    ax1.plot(vocab_sizes, maxs, "d:", color=SERIE_5, linewidth=1.5, label="Máximo")

    sat_vocab = max(vocabs_reales)
    ax1.axvline(sat_vocab, color=TINTA_2, linestyle=":", alpha=0.7, label=f"Saturación de merges ({sat_vocab})")
    ax1.set_xlabel("vocab_size configurado")
    ax1.set_ylabel("Longitud de secuencia (tokens)")
    ax1.set_title("Compresión de Secuencia vs Tamaño de Vocabulario")
    ax1.legend(loc="upper right")

    # Subplot 2: Vocabulario Real vs Solicitado
    ax1_max_v = max(vocab_sizes)
    ax2.plot([0, ax1_max_v], [0, ax1_max_v], "--", color=MUTE, label="Ideal (y = x)")
    ax2.plot(vocab_sizes, vocabs_reales, "o-", color=SERIE_4, linewidth=2.2, label="Vocabulario aprendido")
    ax2.axhline(sat_vocab, color=SERIE_2, linestyle=":", linewidth=1.5, label=f"Límite del corpus ({sat_vocab} tokens)")
    ax2.set_xlabel("vocab_size configurado")
    ax2.set_ylabel("Vocabulario real alcanzado")
    ax2.set_title("Saturación de Merges en el Corpus")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    destino = output_dir / "01_vocab_size_vs_longitud_y_saturacion.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Figura 2: Distribución de Longitud y Análisis de Truncamiento -----------------------

def plot_sequence_length_distribution(
    textos_train: List[str],
    textos_eval: List[str],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Histogramas comparativos de longitud de secuencia y análisis de umbrales max_length."""
    aplicar_estilo()

    vocabs_demo = [500, 1000, 1643]
    colores = [SERIE_2, SERIE_1, SERIE_3]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    max_global = 0
    for vs, col in zip(vocabs_demo, colores):
        tok = ByteLevelBPETokenizer().train_from_iterator(
            textos_train, vocab_size=vs, min_frequency=2, show_progress=False
        )
        lens = [len(tok.encode(t, add_special_tokens=True).ids) for t in textos_eval]
        max_global = max(max_global, max(lens))
        ax.hist(
            lens,
            bins=np.arange(10, 120, 2),
            alpha=0.45,
            color=col,
            label=f"Vocabulario {tok.vocab_size} (media: {np.mean(lens):.1f}, max: {max(lens)})",
            edgecolor=col,
            linewidth=1.2,
        )

    # Umbrales clave de max_length
    ax.axvline(64, color=TINTA, linestyle="--", linewidth=2.0, label="max_length = 64 (0% truncado)")
    ax.axvline(128, color=MUTE, linestyle=":", linewidth=1.5, label="max_length = 128 (desperdicio de padding)")

    ax.set_xlabel("Cantidad de tokens por producto (incluye [CLS] y [SEP])")
    ax.set_ylabel("Cantidad de productos (evaluación)")
    ax.set_title("Distribución de Longitud de Secuencia según Vocabulario")
    ax.legend(loc="upper right")

    fig.tight_layout()
    destino = output_dir / "02_distribucion_longitud_y_truncamiento.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Figura 3: Fertilidad y Fragmentación de Subpalabras --------------------------------

def plot_subword_fertility(
    textos_train: List[str],
    textos_eval: List[str],
    vocab_sizes: Sequence[int] = (300, 500, 800, 1000, 1300, 1643),
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Mide la fertilidad (subpalabras por palabra léxica) y la distribución de fragmentos."""
    aplicar_estilo()

    fertilidades = []
    # Conteo de palabras en fragmentos de 1, 2, 3, >=4 tokens
    distribuciones_frag = {1: [], 2: [], 3: [], "4+": []}

    for vs in vocab_sizes:
        tok = ByteLevelBPETokenizer().train_from_iterator(
            textos_train, vocab_size=vs, min_frequency=2, show_progress=False
        )

        tokens_totales = 0
        palabras_totales = 0
        frags = {1: 0, 2: 0, 3: 0, "4+": 0}

        for texto in textos_eval[:1000]:
            palabras = [p for p in texto.replace("|", " ").split() if p.strip()]
            palabras_totales += len(palabras)
            for palabra in palabras:
                # Tokenizar la palabra individualmente sin [CLS]/[SEP]
                enc = tok.encode(palabra, add_special_tokens=False)
                n_tok = len(enc.ids)
                tokens_totales += n_tok
                if n_tok == 1:
                    frags[1] += 1
                elif n_tok == 2:
                    frags[2] += 1
                elif n_tok == 3:
                    frags[3] += 1
                else:
                    frags["4+"] += 1

        fertilidades.append(tokens_totales / max(1, palabras_totales))
        for k in frags:
            distribuciones_frag[k].append(frags[k] / max(1, palabras_totales) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Fertilidad
    ax1.plot(vocab_sizes, fertilidades, "o-", color=SERIE_1, linewidth=2.2)
    ax1.set_xlabel("vocab_size configurado")
    ax1.set_ylabel("Fertilidad (Tokens BPE / Palabra léxica)")
    ax1.set_title("Fertilidad de Subpalabras vs Tamaño de Vocabulario")
    ax1.axhline(1.0, color=MUTE, linestyle="--", alpha=0.7, label="Ideal palabra entera (1.0)")
    ax1.legend(loc="upper right")

    # Subplot 2: Gráfico de barras apiladas de fragmentación
    x_indices = np.arange(len(vocab_sizes))
    bottom = np.zeros(len(vocab_sizes))
    colores_frag = [SERIE_3, SERIE_1, SERIE_2, SERIE_5]
    labels_frag = ["1 token (palabra entera)", "2 subpalabras", "3 subpalabras", "4+ subpalabras"]

    for k, col, lbl in zip([1, 2, 3, "4+"], colores_frag, labels_frag):
        vals = np.array(distribuciones_frag[k])
        ax2.bar(x_indices, vals, bottom=bottom, label=lbl, color=col, alpha=0.85, width=0.55)
        bottom += vals

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([str(v) for v in vocab_sizes])
    ax2.set_xlabel("vocab_size")
    ax2.set_ylabel("% de palabras en el corpus")
    ax2.set_title("Nivel de Fragmentación de Palabras")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    destino = output_dir / "03_fertilidad_y_fragmentacion_subpalabras.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Figura 4: Impacto de min_frequency --------------------------------------------------

def plot_min_frequency_impact(
    textos_train: List[str],
    textos_eval: List[str],
    min_frequencies: Sequence[int] = (1, 2, 3, 5, 10, 20),
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Evalúa la sensibilidad al umbral de frecuencia mínima para la creación de merges."""
    aplicar_estilo()

    vocabs_obtenidos = []
    longitudes_medias = []

    for mf in min_frequencies:
        tok = ByteLevelBPETokenizer().train_from_iterator(
            textos_train, vocab_size=3000, min_frequency=mf, show_progress=False
        )
        vocabs_obtenidos.append(tok.vocab_size)
        lens = [len(tok.encode(t, add_special_tokens=True).ids) for t in textos_eval]
        longitudes_medias.append(np.mean(lens))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    col1 = SERIE_1
    ax1.set_xlabel("min_frequency (umbral mínimo de apariciones)")
    ax1.set_ylabel("Vocabulario máximo aprendido", color=col1)
    ax1.plot(min_frequencies, vocabs_obtenidos, "o-", color=col1, linewidth=2.2, label="Vocabulario final")
    ax1.tick_params(axis="y", labelcolor=col1)

    ax2 = ax1.twinx()
    col2 = SERIE_2
    ax2.set_ylabel("Longitud media de secuencia (tokens)", color=col2)
    ax2.plot(min_frequencies, longitudes_medias, "s--", color=col2, linewidth=2.0, label="Longitud media")
    ax2.tick_params(axis="y", labelcolor=col2)

    ax1.set_title("Sensibilidad del Vocabulario a min_frequency")
    fig.tight_layout()

    destino = output_dir / "04_impacto_min_frequency.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Figura 5: Ley de Zipf y Rango de Ocurrencias ----------------------------------------

def plot_token_rank_frequency_zipf(
    textos_train: List[str],
    textos_eval: List[str],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Analiza la distribución log-log de rango vs frecuencia de los tokens aprendidos."""
    aplicar_estilo()

    tok = ByteLevelBPETokenizer().train_from_iterator(
        textos_train, vocab_size=1643, min_frequency=2, show_progress=False
    )

    conteo = Counter()
    for t in textos_eval:
        enc = tok.encode(t, add_special_tokens=True)
        conteo.update(enc.ids)

    frecuencias_ordenadas = [f for _, f in conteo.most_common()]
    rangos = np.arange(1, len(frecuencias_ordenadas) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.loglog(rangos, frecuencias_ordenadas, color=SERIE_1, linewidth=2.2, label="Tokens BPE observados")

    # Pendiente teórica de Zipf (1 / rango) normalizada
    if len(frecuencias_ordenadas) > 0:
        f_max = frecuencias_ordenadas[0]
        zipf_teorico = f_max / rangos
        ax.loglog(rangos, zipf_teorico, "--", color=SERIE_2, alpha=0.7, label="Ley de Zipf teórica (1/r)")

    ax.set_xlabel("Rango del token (log)")
    ax.set_ylabel("Frecuencia de aparición (log)")
    ax.set_title("Distribución de Frecuencia de Tokens (Ley de Zipf)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    destino = output_dir / "05_ley_de_zipf_y_frecuencia_tokens.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Figura 6: Trade-off Memoria de Embeddings vs Cómputo de Atención -------------------

def plot_tradeoff_memory_vs_computation(
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Visualiza el costo de parámetros vs la complejidad cuadrática de la atención."""
    aplicar_estilo()

    vocabs = np.linspace(250, 5000, 50)
    d_model = 64
    parametros_emb_k = (vocabs * d_model) / 1000.0  # en miles de parámetros

    longitudes = np.array([32, 48, 64, 96, 128, 160, 256])
    ops_atencion_k = (longitudes ** 2) / 1000.0     # en miles de ops por capa

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Parámetros de la capa de Embedding
    ax1.plot(vocabs, parametros_emb_k, color=SERIE_4, linewidth=2.2)
    ax1.axvline(1600, color=SERIE_1, linestyle="--", label="Vocab recomendado (1.600 -> 102k params)")
    ax1.set_xlabel("vocab_size")
    ax1.set_ylabel("Parámetros de Embedding (miles, d_model=64)")
    ax1.set_title("Tamaño de Matriz de Embeddings (vocab_size × d_model)")
    ax1.legend(loc="upper left")

    # Subplot 2: Complejidad O(T^2) de Atención por Secuencia
    ax2.plot(longitudes, ops_atencion_k, "o-", color=SERIE_2, linewidth=2.2)
    ax2.axvline(64, color=SERIE_3, linestyle="--", label="max_length = 64 (4.1k ops - Recomendado)")
    ax2.axvline(128, color=SERIE_5, linestyle=":", label="max_length = 128 (16.4k ops - 4x cómputo)")
    ax2.set_xlabel("max_length (T)")
    ax2.set_ylabel("Operaciones de Atención por Bloque (miles, T²)")
    ax2.set_title("Costo Computacional de Atención O(T²)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    destino = output_dir / "06_tradeoff_memoria_vs_computo_atencion.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


# --- Función principal / CLI -----------------------------------------------------------

def run_all_benchmarks(
    train_path: Path = Path("resources/datasets/transformer_train.csv"),
    val_path: Path = Path("resources/datasets/transformer_val.csv"),
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Ejecuta todas las pruebas diagnósticas y guarda las figuras generadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    textos_train = cargar_textos(train_path)
    textos_val = cargar_textos(val_path)

    print("=" * 80)
    print(f"📊 GENERANDO FIGURAS DE DIAGNÓSTICO DEL TOKENIZADOR EN: {output_dir}/")
    print(f"   Corpus: {len(textos_train)} textos de train | {len(textos_val)} textos de validación")
    print("=" * 80)

    figuras = []

    print("  [1/6] Analizando barrido de vocab_size vs longitud y saturación...")
    figuras.append(plot_vocab_size_sweep(textos_train, textos_val, output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("  [2/6] Analizando distribución de longitud y corte de max_length...")
    figuras.append(plot_sequence_length_distribution(textos_train, textos_val, output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("  [3/6] Analizando fertilidad y fragmentación de subpalabras...")
    figuras.append(plot_subword_fertility(textos_train, textos_val, output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("  [4/6] Analizando sensibilidad a min_frequency...")
    figuras.append(plot_min_frequency_impact(textos_train, textos_val, output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("  [5/6] Analizando ley de Zipf y frecuencia de tokens...")
    figuras.append(plot_token_rank_frequency_zipf(textos_train, textos_val, output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("  [6/6] Analizando trade-off memoria vs cómputo de atención...")
    figuras.append(plot_tradeoff_memory_vs_computation(output_dir=output_dir))
    print(f"        ✓ {figuras[-1].name}")

    print("=" * 80)
    print(f"🎉 ÉXITO: {len(figuras)} figuras guardadas en {output_dir}/")
    print("=" * 80)
    return figuras


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gráficos de diagnóstico para el tokenizador BPE.")
    parser.add_argument("--train_path", type=str, default="resources/datasets/transformer_train.csv")
    parser.add_argument("--val_path", type=str, default="resources/datasets/transformer_val.csv")
    parser.add_argument("--output_dir", type=str, default="results/figures/tokenizer")
    args = parser.parse_args()

    run_all_benchmarks(
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
