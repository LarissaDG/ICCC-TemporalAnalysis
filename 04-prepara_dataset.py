# 04-prepara_dataset.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import random
import shutil
import math
import json
from typing import List
import traceback
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
import imageio
import matplotlib.pyplot as plt

import subprocess
import unicodedata
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from colorama import Fore, Style
from scipy import stats

# =====================================================================
# CONFIGURAÇÃO GLOBAL
# =====================================================================

RAW_DIR = Path("videos_amostrados_raw")
WORK_DIR = Path("videos_amostrados_mp4")
FRAMES_ORIGINAIS_DIR = Path("frames_originais")
FRAMES_RUIDOS_H1_DIR = Path("frames_ruidos_h1")
FRAMES_INTENSIDADES_H2_DIR = Path("frames_intensidades_h2")
ORQUESTRADOR_DIR = Path("orquestrador_experimentos")

NOISE_TYPES = ["blur_sp", "shapes_x", "gaussian"]
VIDEO_EXTS = [".mp4", ".webm", ".mov", ".avi", ".gif", ".mkv"]

random.seed(42)
np.random.seed(42)

imageio.plugins.ffmpeg.ALLOW_EXEC = True
print("✅ imageio configurado para usar ffmpeg para leitura/escrita de vídeo.")

# =====================================================================
# UTIL
# =====================================================================

def list_videos(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_") or "video"

# =====================================================================
# CONVERSÃO MP4 (Usando imageio para robustez)
# =====================================================================

def _convert_single(task):
    src, dst = task
    try:
        if dst.exists():
            return src, dst, True, "", True  # skipped

        reader = imageio.get_reader(str(src), "ffmpeg")
        fps = reader.get_meta_data().get("fps", 30)
        writer = imageio.get_writer(str(dst), fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])

        for frame in reader:
            writer.append_data(frame)
        
        writer.close()
        reader.close()

        return src, dst, True, "", False

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\nERRO ao converter {src.name}: {e}\n")
        return src, dst, False, error_msg, False

def convert_all_to_mp4(input_videos: list[Path], output_dir: Path, n_jobs: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for vid in input_videos:
        base = slugify(vid.stem)
        dst = output_dir / f"{base}.mp4"
        tasks.append((vid, dst))

    print(f"\n🔧 Convertendo {len(tasks)} vídeos para MP4 usando {n_jobs} processos...\n")
    pbar = tqdm(total=len(tasks), desc="Convertendo", unit="vídeo", dynamic_ncols=True)
    converted_paths = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_convert_single, t): t for t in tasks}
        for fut in as_completed(futures):
            src, dst, ok, err, skipped = fut.result()
            if ok and not skipped:
                converted_paths.append(dst)
            pbar.update(1)
            pbar.set_postfix_str(src.name[:35])

    return converted_paths

# =====================================================================
# DURAÇÃO E EXTRAÇÃO DE FRAMES
# =====================================================================

def get_duration(path: Path) -> float:
    try:
        reader = imageio.get_reader(str(path), "ffmpeg")
        meta = reader.get_meta_data()
        reader.close()
        return float(meta.get("duration", 0.0))
    except:
        return 0.0

def extract_sampled_frames(video_path: Path, min_duration: float, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)
    
    num_frames_to_sample = int(min_duration) 
    frame_paths = []

    for i in range(num_frames_to_sample):
        sec = i * 1.0
        idx = int(round(sec * fps))

        try:
            frame = reader.get_data(idx)
            img = Image.fromarray(frame).convert("RGB")
            frame_name = f"{video_path.stem}_frame_{i:04d}.png"
            frame_path = frames_dir / frame_name
            img.save(frame_path)
            frame_paths.append(str(frame_path))
        except IndexError:
            break
        except Exception:
            break

    reader.close()
    return frame_paths, num_frames_to_sample

# =====================================================================
# RUÍDOS
# =====================================================================

def add_blur_sp(img, intensity):
    blur_radius = 1 + intensity * 6
    sp_amount  = intensity * 0.06
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(blurred).astype(np.int16)
    h, w, c = arr.shape
    n = max(1, int(h*w*sp_amount))
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs] = 255
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def add_shapes(img, intensity):
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size)
    d = ImageDraw.Draw(overlay, "RGBA")
    w, h = base.size
    alpha = int(50 + intensity * 205)
    d.line((0,0,w,h), fill=(255,0,0,alpha), width=3)
    d.line((0,h,w,0), fill=(255,0,0,alpha), width=3)
    return Image.alpha_composite(base, overlay).convert("RGB")

def add_gaussian(img, intensity):
    sigma = intensity * 60
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_noise(img, noise_type, intensity):
    if noise_type == "blur_sp": return add_blur_sp(img, intensity)
    if noise_type == "shapes_x": return add_shapes(img, intensity)
    if noise_type == "gaussian": return add_gaussian(img, intensity)
    raise ValueError(noise_type)

# ================================================================
# VISUALIZAÇÃO (simples) — adaptado para novos nomes de colunas
# ================================================================
def visualizar_exemplos_ruido(df: pd.DataFrame, original_col: str, modified_col: str, noise_type_col: str):
    examples = []
    # Usamos o set(NOISE_TYPES) para garantir que pegamos os 3 tipos definidos globalmente
    for noise_type in set(NOISE_TYPES): 
        df_noise = df[df[noise_type_col] == noise_type]
        if not df_noise.empty:
            row = df_noise.iloc[0]
            # Usando .iloc[0] pegamos o primeiro exemplo que aparecer no DF
            orig = Path(row[original_col])
            mod = Path(row[modified_col]) if pd.notna(row[modified_col]) else None
            if orig.exists() and (mod is None or mod.exists()):
                examples.append((noise_type, orig, mod))

    if not examples:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Nenhum exemplo de ruído disponível", ha="center", va="center")
        plt.axis("off")
        return fig

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes] # Garante que axes seja sempre 2D array-like

    for i, (noise, orig, mod) in enumerate(examples):
        ax_orig = axes[i][0]
        ax_mod = axes[i][1]
        ax_orig.imshow(Image.open(orig))
        ax_orig.set_title(f"Original ({noise})")
        ax_orig.axis("off")
        if mod:
            ax_mod.imshow(Image.open(mod))
            ax_mod.set_title(f"Modificado ({noise})")
        else:
            ax_mod.text(0.5, 0.5, "Sem modificado", ha="center")
        ax_mod.axis("off")

    plt.tight_layout()
    return fig


def salvar_visualizacao(df: pd.DataFrame, nome_arquivo: str, original_col: str, modified_col: str, noise_type_col: str) -> Path:
    fig = visualizar_exemplos_ruido(df, original_col, modified_col, noise_type_col)
    try:
        out_path = Path(__file__).resolve().parent / nome_arquivo
    except Exception:
        out_path = Path.cwd() / nome_arquivo
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Visualização salva em: {out_path}")
    return out_path

# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def build_dataset(dataset_name="dataset_final", n_jobs=4):

    print("\n==============================")
    print("📦 MONTANDO DATASET FINAL")
    print("==============================\n")

    for d in [FRAMES_ORIGINAIS_DIR, FRAMES_RUIDOS_H1_DIR, FRAMES_INTENSIDADES_H2_DIR, ORQUESTRADOR_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    raw_videos = list_videos(RAW_DIR)
    print(f"📁 Vídeos encontrados (raw): {len(raw_videos)}")

    converted_paths = convert_all_to_mp4(raw_videos, WORK_DIR, n_jobs=n_jobs)
    if not converted_paths:
        print(f"{Fore.RED}ERRO: 0 vídeos convertidos. Verifique as permissões ou a instalação do ffmpeg no PATH.{Style.RESET_ALL}")
        return

    durations = [get_duration(p) for p in converted_paths]
    min_duration = math.floor(min(durations)) if durations else 1
    print(f"\n✔ Durações calculadas. Duração mínima (T): {min_duration} segundos.\n")
    
    print(f"🖼️ Extraindo frames originais (1/s por {min_duration}s)...")
    all_original_frames_map = {}
    for video_path in tqdm(converted_paths, desc="Extraindo Originais", dynamic_ncols=True):
        frames, num_sampled = extract_sampled_frames(video_path, min_duration, FRAMES_ORIGINAIS_DIR)
        all_original_frames_map[str(video_path.name)] = frames
    
    T = num_sampled
    print(f"✔ Extração de frames originais concluída. T = {T} frames por vídeo.\n")

    # 4) Preparação H1: Ruído fixo por vídeo (aleatório)
    print("💥 Gerando frames ruidosos para H1 (Tipo de ruído fixo por vídeo)...")
    df_h1_orchestrator_rows = []
    
    video_names = list(all_original_frames_map.keys())
    random.shuffle(video_names)
    num_videos = len(video_names)
    chunk_size = num_videos // len(NOISE_TYPES)
    noise_assignment_h1 = {}
    current_index = 0

    for noise_type in NOISE_TYPES:
        for i in range(chunk_size):
            if current_index < num_videos:
                noise_assignment_h1[video_names[current_index]] = noise_type
                current_index += 1
    while current_index < num_videos:
        noise_assignment_h1[video_names[current_index]] = random.choice(NOISE_TYPES)
        current_index += 1

    for video_name, original_frames in tqdm(all_original_frames_map.items(), desc="Aplicando Ruído H1", dynamic_ncols=True):
        noise_type = noise_assignment_h1[video_name]
        for frame_path_str in original_frames:
            frame_path = Path(frame_path_str)
            img = Image.open(frame_path)
            noisy_img = apply_noise(img, noise_type, 1.0) 
            noisy_frame_name = f"H1_{noise_type}_{frame_path.name}"
            noisy_frame_path = FRAMES_RUIDOS_H1_DIR / noisy_frame_name
            noisy_img.save(noisy_frame_path)

            df_h1_orchestrator_rows.append({
                "video_name": video_name,
                "frame_original_path": str(frame_path),
                "frame_ruidoso_path_h1": str(noisy_frame_path),
                "noise_type_h1": noise_type,
                "noise_intensity_h1": 1.0,
                "frame_index": frame_path.stem.split('_')[-1],
                "experiment_type": "H1_H5"
            })
    
    # 5) Preparação H2: Ruído com escala de intensidade temporal por vídeo (aleatório)
    print("📉 Gerando frames ruidosos para H2 (Escala de intensidade temporal)...")
    df_h2_orchestrator_rows = []
    
    if T > 1:
        intensities = np.linspace(0.0, 1.0, T)
    else:
        intensities = [1.0]
    
    video_names_h2 = list(all_original_frames_map.keys())
    random.shuffle(video_names_h2)
    noise_assignment_h2 = {}
    current_index = 0
    for noise_type in NOISE_TYPES:
        for i in range(chunk_size):
            if current_index < num_videos:
                noise_assignment_h2[video_names_h2[current_index]] = noise_type
                current_index += 1
    while current_index < num_videos:
        noise_assignment_h2[video_names_h2[current_index]] = random.choice(NOISE_TYPES)
        current_index += 1

    for video_name, original_frames in tqdm(all_original_frames_map.items(), desc="Aplicando Ruído H2", dynamic_ncols=True):
        noise_type = noise_assignment_h2[video_name]
        for i, frame_path_str in enumerate(original_frames):
            frame_path = Path(frame_path_str)
            img = Image.open(frame_path)
            intensity = intensities[i]
            noisy_img = apply_noise(img, noise_type, intensity)
            noisy_frame_name = f"H2_{noise_type}_int{int(intensity*100):03d}_{frame_path.name}"
            noisy_frame_path = FRAMES_INTENSIDADES_H2_DIR / noisy_frame_name
            noisy_img.save(noisy_frame_path)

            df_h2_orchestrator_rows.append({
                "video_name": video_name,
                "frame_original_path": str(frame_path),
                "frame_intensidade_path_h2": str(noisy_frame_path),
                "noise_type_h2": noise_type,
                "noise_intensity_h2": intensity,
                "frame_index": i,
                "experiment_type": "H2"
            })

    # 6) Gerar CSVs Orquestradores e Visualizações
    print("📊 Gerando arquivos CSVs e visualizações orquestradores...")
    df_h1 = pd.DataFrame(df_h1_orchestrator_rows)
    df_h2 = pd.DataFrame(df_h2_orchestrator_rows)

    df_h1.to_csv(ORQUESTRADOR_DIR / "orquestrador_H1_H5.csv", index=False)
    df_h2.to_csv(ORQUESTRADOR_DIR / "orquestrador_H2.csv", index=False)
    
    # Gerar visualizações H1 e H2
    salvar_visualizacao(df_h1, "visualizacao_H1_H5.png", "frame_original_path", "frame_ruidoso_path_h1", "noise_type_h1")
    salvar_visualizacao(df_h2, "visualizacao_H2.png", "frame_original_path", "frame_intensidade_path_h2", "noise_type_h2")

    print("✨ Pipeline de preparação de dataset concluída.")


# =====================================================================
# PONTO DE ENTRADA (MAIN)
# =====================================================================

if __name__ == "__main__":
    build_dataset(n_jobs=4)
