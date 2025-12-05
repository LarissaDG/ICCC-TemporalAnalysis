# 02_amostra_videos.py
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path
import pandas as pd

# CONFIG
DOWNLOADS_DIR = Path("downloads")              # onde estão seus videos baixados
AMOSTRA_RAW_DIR = Path("videos_amostrados_raw")  # pasta onde salvar os 100 videos brutos
N = 100
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".gif", ".webm", ".mkv"]


def list_videos(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def sample_videos(files, n):
    filtered = [f for f in files if "tips" not in f.stem.lower()]
    random.shuffle(filtered)
    return filtered[:min(n, len(filtered))]


def main():
    print("=== ETAPA 1: AMOSTRAGEM DE VÍDEOS ===")

    AMOSTRA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list_videos(DOWNLOADS_DIR)
    print(f"Encontrados {len(files)} vídeos no diretório.")

    selected = sample_videos(files, N)
    print(f"Amostrando {len(selected)} vídeos.")

    copied = []
    for src in selected:
        dst = AMOSTRA_RAW_DIR / src.name
        try:
            shutil.copy2(src, dst)
            copied.append({"original": str(src), "copied_to": str(dst)})
        except Exception as e:
            print(f"Erro copiando {src}: {e}")

    # salva tabela de referência
    df = pd.DataFrame(copied)
    df.to_csv(AMOSTRA_RAW_DIR / "amostragem.csv", index=False)
    print(f"✔ Amostragem salva em: {AMOSTRA_RAW_DIR/'amostragem.csv'}")
    print("✔ Copia concluída. Agora suba a pasta 'videos_amostrados_raw' para o SLURM.")

    print("\n=== FIM DA ETAPA 1 ===")


if __name__ == "__main__":
    main()
