# 05-processa_dataset.py
# -*- coding: utf-8 -*-
"""
Processa os frames já extraídos:
- Aplica ruídos H1 e H2
- Gera datasets
- Gera visualizações
- Gera os CSVs orquestradores

Este script NÃO converte vídeos e NÃO extrai frames.
Ele assume que:
- videos_amostrados_mp4/ já existe (convertidos)
- frames_originais/ já existe (frames 1 FPS)
"""

from pathlib import Path
import random
import math
import json
import traceback

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style

import sys
import imageio   # <-- ADICIONADO
import math

SHAPES_CACHE = None

# ============================================================
# DIRETÓRIOS
# ============================================================

FRAMES_ORIGINAIS_DIR = Path("frames_originais")
FRAMES_RUIDOS_H1_DIR = Path("frames_ruidos_h1")
FRAMES_INTENSIDADES_H2_DIR = Path("frames_intensidades_h2")
ORQUESTRADOR_DIR = Path("orquestrador_experimentos")

NOISE_TYPES = ["blur_sp", "shapes_x", "gaussian"]

random.seed(42)
np.random.seed(42)

# ============================================================
# RUÍDOS — MESMOS DO SEU CÓDIGO ORIGINAL
# ============================================================

# =====================================================
# COLOR SAMPLING
# =====================================================
def sample_color_from_image(frame, n_samples=5):
    arr = np.array(frame.convert("RGB"))
    h, w, _ = arr.shape
    ys = np.random.randint(0, h, n_samples)
    xs = np.random.randint(0, w, n_samples)
    samples = arr[ys, xs]
    mean_color = samples.mean(axis=0).astype(int)
    return tuple(int(c) for c in mean_color)

# =====================================================
# RUIDOS
# =====================================================
def add_blur_saltpepper(frame, intensity = 100):

    #Regra de três intensidade:
    blur_radius = 100*intensity/100
    sp_amount = 100*intensity/100

    """
    blur_radius: raio do Gaussian Blur varia de 0 a 100
    sp_amount: porcentagem de pixels a receber sal/pimenta (ex: 5 = 5%). Varia de 0 a 100
    """
    # 1) Aplica blur primeiro
    arr = np.array(frame.filter(ImageFilter.GaussianBlur(blur_radius))).astype(np.int16)

    # Garante RGB
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    h, w, c = arr.shape

    # 2) Converte percentual → número de pixels
    total_pixels = h * w
    n = int((sp_amount / 100) * total_pixels)
    n = max(1, n)

    # 3) Pixels brancos ("sal")
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 255

    # 4) Pixels pretos ("pimenta")
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 0

    # 5) Retorna imagem
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), blur_radius, sp_amount


def add_shapes_x(frame, intensity=100):
    global SHAPES_CACHE

    alpha = math.ceil(255 * intensity / 100)

    frame_background = frame.convert("RGBA")
    w, h = frame.size
    width = max(3, int(min(w, h) * 0.02))

    # -------------------------
    # SE JÁ TEM CACHE → REAPLICAR
    # -------------------------
    if SHAPES_CACHE is not None:
        shapes_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shapes_layer, "RGBA")

        for item in SHAPES_CACHE:
            kind = item["kind"]
            c = item["color"][:-1] + (alpha,)  # atualiza alpha dinamicamente

            if kind == "rect":
                draw.rectangle(item["coords"], fill=c)
            elif kind == "ellipse":
                draw.ellipse(item["coords"], fill=c)
            elif kind == "line":
                draw.line(item["coords"], fill=(255, 0, 0, alpha), width=item["width"])

        frame_background.alpha_composite(shapes_layer)
    else:
        # -------------------------
        # PRIMEIRA EXECUÇÃO → CRIAR CACHE
        # -------------------------
        SHAPES_CACHE = []
        shapes_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shapes_layer, "RGBA")

        # --- FORMAS ALEATÓRIAS ---
        num_shapes = random.randint(2, 5)
        for _ in range(num_shapes):
            c = sample_color_from_image(frame_background, n_samples=3)
            c_alpha = c + (alpha,)

            if random.random() < 0.5:  # retângulo
                x0 = random.randint(0, w-1)
                y0 = random.randint(0, h-1)
                x1 = min(w, x0 + random.randint(20, int(w*0.25)))
                y1 = min(h, y0 + random.randint(20, int(h*0.25)))
                coords = [x0, y0, x1, y1]
                draw.rectangle(coords, fill=c_alpha)

                SHAPES_CACHE.append({
                    "kind": "rect",
                    "coords": coords,
                    "color": c_alpha
                })

            else:  # elipse
                cx = random.randint(0, w-1)
                cy = random.randint(0, h-1)
                r = random.randint(10, int(min(w, h)*0.12))
                coords = [cx-r, cy-r, cx+r, cy+r]
                draw.ellipse(coords, fill=c_alpha)

                SHAPES_CACHE.append({
                    "kind": "ellipse",
                    "coords": coords,
                    "color": c_alpha
                })

        # --- LINHAS EM X ---
        coords1 = (0, 0, w, h)
        coords2 = (0, h, w, 0)

        draw.line(coords1, fill=(255, 0, 0, alpha), width=width)
        draw.line(coords2, fill=(255, 0, 0, alpha), width=width)

        SHAPES_CACHE.append({
            "kind": "line",
            "coords": coords1,
            "width": width,
            "color": (255, 0, 0, alpha)
        })
        SHAPES_CACHE.append({
            "kind": "line",
            "coords": coords2,
            "width": width,
            "color": (255, 0, 0, alpha)
        })

        frame_background.alpha_composite(shapes_layer)

    # Final → colar no fundo branco
    background = Image.new("RGB", (w, h), (255, 255, 255))
    background.paste(frame_background, mask=frame_background.split()[3])

    return background, alpha


def add_gaussian_noise(frame, intensity=100):
    #regra de três intensity
    sigma = 1000*intensity/100

    #varia de 0 a 1000
    rgb = frame.convert("RGB")
    arr = np.array(rgb).astype(np.int16)
    base_color = np.array(sample_color_from_image(rgb, n_samples=50))
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = arr + noise + (base_color - base_color.mean())
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), sigma

# ============================================================
# VISUALIZAÇÃO
# ============================================================

def visualizar_exemplos_ruido(df, original_col, modified_col, noise_type_col):
    examples = []
    for noise_type in NOISE_TYPES:
        df_noise = df[df[noise_type_col] == noise_type]
        if not df_noise.empty:
            row = df_noise.iloc[0]
            orig = Path(row[original_col])
            mod = Path(row[modified_col])
            if orig.exists() and mod.exists():
                examples.append((noise_type, orig, mod))

    if not examples:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Nenhum exemplo disponível", ha="center")
        plt.axis("off")
        return fig

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (noise, orig, mod) in enumerate(examples):
        ax1, ax2 = axes[i]
        ax1.imshow(Image.open(orig))
        ax1.set_title(f"Original ({noise})")
        ax1.axis("off")

        ax2.imshow(Image.open(mod))
        ax2.set_title(f"Modificado ({noise})")
        ax2.axis("off")

    plt.tight_layout()
    return fig

def salvar_visualizacao(df, nome_arquivo, orig_col, mod_col, type_col):
    fig = visualizar_exemplos_ruido(df, orig_col, mod_col, type_col)
    out = ORQUESTRADOR_DIR / nome_arquivo
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Visualização salva em: {out}")
    return out

# =====================================================
# GERAR N INTENSIDADES DIFERENTES DE CADA RUÍDO
# =====================================================
def generate_noise_sweep(img, n=5, out_dir="ruido_tuning_sweep"):
    """
    Gera n variações de intensidade para:
      - blur + salt & pepper
      - shapes_x
      - gaussian noise

    Salva todas as imagens em 'ruido_tuning_sweep'.
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    img = img.convert("RGB")

    # -------------------------
    # ORIGINAL
    # -------------------------
    img.save(out / "original.png")

    #Divide o 100% de intensidade pelo numero de frames
    # O passo é calculado para garantir que o primeiro e o último (1 e 100) sejam incluídos
    passo = 99 / (n - 1) 

    vetor_intensity = []
    for i in range(n):
        # Arredondamos para evitar problemas com números de ponto flutuante se necessário
        valor = 1 + round(i * passo) 
        vetor_intensity.append(valor)

    for i in range(n):

        # 1) BLUR + SALT & PEPPER
        im, blur, sp = add_blur_saltpepper(img, vetor_intensity[i])

        fname = f"blur_sp_level_{i}_b_{blur}_sp_{sp}.png"
        im.save(out / fname)
        print(f"✔ salvo: {fname}")

        # 2) SHAPES_X COM ALPHA VARIÁVEL

        im,alpha = add_shapes_x(img, vetor_intensity[i])

        fname = f"shapes_x_level_{i}_a_{alpha}.png"
        im.save(out / fname)
        print(f"✔ salvo: {fname}")
        
        # 3) GAUSSIAN NOISE
        im, sigma = add_gaussian_noise(img, vetor_intensity[i])

        fname = f"gaussian_level_{i}_s_{sigma}.png"
        im.save(out / fname)
        print(f"✔ salvo: {fname}")

    print("\n🎉 Sweep completo! Imagens salvas em:", out.absolute())


def apply_noise(img, noise_type, intensity):
    if noise_type == "blur_sp": return add_blur_sp(img, intensity)
    if noise_type == "shapes_x": return add_shapes(img, intensity)
    if noise_type == "gaussian": return add_gaussian(img, intensity)
    raise ValueError(noise_type)

# ============================================================
# PROCESSAMENTO DO DATASET
# ============================================================

def build_dataset_processado():

    # Cria pastas necessárias
    for d in [FRAMES_RUIDOS_H1_DIR, FRAMES_INTENSIDADES_H2_DIR, ORQUESTRADOR_DIR]:
        d.mkdir(exist_ok=True)

    print("\n==============================")
    print("📦 PROCESSANDO DATASET")
    print("==============================\n")

    # ============================================================
    # 1. Carrega todos os frames originais
    # ============================================================

    print("📁 Carregando frames originais...")
    all_frames = sorted(FRAMES_ORIGINAIS_DIR.glob("*.png"))
    if not all_frames:
        print(Fore.RED + "ERRO: Nenhum frame encontrado." + Style.RESET_ALL)
        return

    # Agrupa por vídeo
    videos = {}
    for f in all_frames:
        vid = "_".join(f.stem.split("_")[:-2])  # nome_base_frame_0001
        videos.setdefault(vid, []).append(f)

    # ============================================================
    # 2. H1 — ruído fixo por vídeo
    # ============================================================

    print("💥 Aplicando H1...")
    df_h1_rows = []

    vid_names = list(videos.keys())
    random.shuffle(vid_names)

    noise_assign = {}
    chunk = len(vid_names) // len(NOISE_TYPES)
    idx = 0

    for nt in NOISE_TYPES:
        for _ in range(chunk):
            if idx < len(vid_names):
                noise_assign[vid_names[idx]] = nt
                idx += 1

    # restantes aleatórios
    while idx < len(vid_names):
        noise_assign[vid_names[idx]] = random.choice(NOISE_TYPES)
        idx += 1

    for vid, frame_list in tqdm(videos.items(), desc="H1", ncols=90):
        noise_type = noise_assign[vid]
        for f in frame_list:
            img = Image.open(f)
            noisy = apply_noise(img, noise_type, 1.0)
            out = FRAMES_RUIDOS_H1_DIR / f"H1_{noise_type}_{f.name}"
            noisy.save(out)

            df_h1_rows.append({
                "video_name": vid,
                "frame_original_path": str(f),
                "frame_ruidoso_path_h1": str(out),
                "noise_type_h1": noise_type,
                "noise_intensity_h1": 1.0,
                "frame_index": f.stem.split("_")[-1],
                "experiment_type": "H1_H5"
            })

    df_h1 = pd.DataFrame(df_h1_rows)
    df_h1.to_csv(ORQUESTRADOR_DIR / "orquestrador_H1_H5.csv", index=False)

    # ============================================================
    # 3. H2 — ruído com intensidade crescente no tempo
    # ============================================================

    print("📉 Aplicando H2...")
    df_h2_rows = []

    vid_names2 = list(videos.keys())
    random.shuffle(vid_names2)
    assign2 = {}
    idx = 0

    for nt in NOISE_TYPES:
        for _ in range(chunk):
            if idx < len(vid_names2):
                assign2[vid_names2[idx]] = nt
                idx += 1
    while idx < len(vid_names2):
        assign2[vid_names2[idx]] = random.choice(NOISE_TYPES)
        idx += 1

    for vid, frame_list in tqdm(videos.items(), desc="H2", ncols=90):
        noise_type = assign2[vid]
        T = len(frame_list)

        if T > 1:
            intensities = np.linspace(0.0, 1.0, T)
        else:
            intensities = [1.0]

        for i, f in enumerate(frame_list):
            img = Image.open(f)
            noisy = apply_noise(img, noise_type, intensities[i])
            out = FRAMES_INTENSIDADES_H2_DIR / f"H2_{noise_type}_int{int(intensities[i]*100):03d}_{f.name}"
            noisy.save(out)

            df_h2_rows.append({
                "video_name": vid,
                "frame_original_path": str(f),
                "frame_intensidade_path_h2": str(out),
                "noise_type_h2": noise_type,
                "noise_intensity_h2": float(intensities[i]),
                "frame_index": i,
                "experiment_type": "H2"
            })

    df_h2 = pd.DataFrame(df_h2_rows)
    df_h2.to_csv(ORQUESTRADOR_DIR / "orquestrador_H2.csv", index=False)

    # ============================================================
    # 4. VISUALIZAÇÕES
    # ============================================================

    salvar_visualizacao(df_h1, "visualizacao_H1_H5.png",
                        "frame_original_path", "frame_ruidoso_path_h1", "noise_type_h1")

    salvar_visualizacao(df_h2, "visualizacao_H2.png",
                        "frame_original_path", "frame_intensidade_path_h2", "noise_type_h2")

    print("\n✨ PROCESSAMENTO FINALIZADO!")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_dataset_processado()

