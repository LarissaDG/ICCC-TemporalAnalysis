# -*- coding: utf-8 -*-
"""
04_compute_gif_duration.py
Calcula a duração REAL dos GIFs originais usados no dataset,
compatível com o pipeline do 01_generate_samples.py.
"""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageSequence
import numpy as np


def get_gif_duration(gif_path: Path):
    """
    Calcula:
    - num_frames
    - fps
    - duration_seconds
    usando delays reais do GIF.
    """
    if not gif_path.exists():
        print(f"❌ GIF não encontrado: {gif_path}")
        return None, None, None

    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"❌ Erro ao abrir GIF {gif_path}: {e}")
        return None, None, None

    # Número real de frames no GIF original
    num_frames = getattr(gif, "n_frames", 1)

    # Duração individual de cada frame (ms)
    durations_ms = []
    for i in range(num_frames):
        gif.seek(i)
        durations_ms.append(gif.info.get("duration", 0))  # milissegundos

    total_ms = sum(durations_ms)
    duration_seconds = total_ms / 1000 if total_ms > 0 else 0

    mean_ms = np.mean(durations_ms) if len(durations_ms) else 0
    fps = 1000 / mean_ms if mean_ms > 0 else 0

    return num_frames, fps, duration_seconds


def find_original_gif(gif_name: str, search_root: Path):
    """
    Busca o GIF original no diretório de entrada usado no pipeline.
    Isso é necessário porque o dataset tem apenas frames.
    """
    # Procura recursivamente por arquivos .gif
    for path in search_root.rglob("*.gif"):
        if path.stem == gif_name:
            return path
    return None


def process_durations(csv_path: Path, gif_search_root: Path, output_csv: Path):
    """
    Lê o dataset CSV, encontra para cada gif_name o GIF original
    e calcula sua duração real.
    """

    df = pd.read_csv(csv_path)

    if "gif_name" not in df.columns:
        raise ValueError("❌ O CSV não contém a coluna 'gif_name' — necessário para calcular a duração.")

    unique_gifs = sorted(df["gif_name"].unique().tolist())

    results = []
    for gif_name in unique_gifs:
        gif_path = find_original_gif(gif_name, gif_search_root)

        if gif_path is None:
            print(f"⚠ GIF não encontrado para gif_name={gif_name} dentro de {gif_search_root}")
            results.append({
                "gif_name": gif_name,
                "gif_path": None,
                "num_frames": None,
                "fps": None,
                "duration_seconds": None
            })
            continue

        num_frames, fps, duration_sec = get_gif_duration(gif_path)

        results.append({
            "gif_name": gif_name,
            "gif_path": str(gif_path),
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_sec
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

    print(f"\n✔ Durações calculadas e salvas em:\n{output_csv}")
    return df_out


def main():
    """
    Exemplo de uso:
    - Coloque o caminho do CSV gerado pelo pipeline (original ou ruido)
    - Coloque o caminho da pasta onde os GIFs originais foram baixados
    """
    # Ajuste o CSV de entrada:
    csv_path = Path("/mnt/c/Users/Dell/Downloads/ICCC-TemporalAnalysis/dataset/gifs/gifs_original_dataset.csv")

    # Pasta raiz onde você passou os GIFs para o 01_generate_samples.py
    gif_search_root = Path("/mnt/c/Users/Dell/Downloads/ICCC-TemporalAnalysis/gifs")

    # Pasta de saída
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)

    dataset_name = csv_path.stem.replace(".csv", "")
    output_csv = out_dir / f"{dataset_name}_gifs_durations.csv"

    process_durations(csv_path, gif_search_root, output_csv)


if __name__ == "__main__":
    main()
