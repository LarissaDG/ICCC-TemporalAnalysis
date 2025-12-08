# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import imageio
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ============================================================
#   CONFIGURAÇÕES GERAIS
# ============================================================

FRAMES_DIR = Path("frames_originais")
VIDEOS_DIR = Path("videos_amostrados_mp4")

OUT_DIR_H0 = Path("frames_h0")
OUT_DIR_H1 = Path("frames_h1")
OUT_DIR_H2 = Path("frames_h2")

for d in [OUT_DIR_H0, OUT_DIR_H1, OUT_DIR_H2]:
    os.makedirs(d, exist_ok=True)

# ============================================================
#   RUÍDOS
# ============================================================

def add_blur_sp(img, level):
    if level <= 0:
        return img

    img = img.copy().filter(ImageFilter.GaussianBlur(radius=level))

    arr = np.array(img)
    s_vs_p = 0.5
    amount = min(0.05 * level, 1.0)

    num_salt = int(np.ceil(amount * arr.size * s_vs_p))
    num_pepper = int(np.ceil(amount * arr.size * (1 - s_vs_p)))

    coords = [np.random.randint(0, i - 1, num_salt) for i in arr.shape]
    arr[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in arr.shape]
    arr[coords[0], coords[1], :] = 0

    return Image.fromarray(arr)


def add_shapes_x(img, intensity, video_name):
    """
    shapes_x com CACHE por vídeo.
    (Mas o cache deve ser reiniciado fora daqui!)
    """
    global SHAPES_CACHE
    if video_name not in SHAPES_CACHE:
        SHAPES_CACHE[video_name] = []
        np.random.seed(int(video_name) % 999_999)

        base_shapes = []
        for _ in range(3):
            x1, y1 = np.random.randint(0, img.width), np.random.randint(0, img.height)
            x2, y2 = np.random.randint(0, img.width), np.random.randint(0, img.height)
            thick = np.random.randint(1, 5)
            base_shapes.append((x1, y1, x2, y2, thick))

        SHAPES_CACHE[video_name] = base_shapes

    base = SHAPES_CACHE[video_name]
    out = img.copy()
    draw = ImageDraw.Draw(out)

    k = min(max(int(intensity), 0), len(base))
    for (x1, y1, x2, y2, thick) in base[:k]:
        draw.line((x1, y1, x2, y2), fill="white", width=thick)

    return out


def add_gaussian_noise(img, level):
    if level <= 0:
        return img

    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level * 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_noise_sweep_single(img, noise_type, intensity, video_name):
    if noise_type == "blur_sp":
        return add_blur_sp(img, intensity)
    elif noise_type == "shapes_x":
        return add_shapes_x(img, intensity, video_name)
    elif noise_type == "gaussian":
        return add_gaussian_noise(img, intensity)
    return img


# ============================================================
#   LEITURA DE FRAMES
# ============================================================

def load_frames():
    frames = []
    for f in sorted(FRAMES_DIR.glob("*_frame_*.png")):
        stem = f.stem
        video_name, _, idx = stem.split("_")
        frame_index = int(idx.lstrip("0") or "0")

        frames.append({
            "video_name": video_name,
            "frame_path": f,
            "frame_index": frame_index
        })
    return frames


# ============================================================
#   ROUND-ROBIN PARA ATRIBUIR RUÍDOS
# ============================================================

def assign_noise_types(videos):
    noise_types = ["blur_sp", "shapes_x", "gaussian"]
    mapping = {}
    for i, v in enumerate(sorted(videos)):
        mapping[v] = noise_types[i % 3]
    return mapping


# ============================================================
#   METADADOS DO VÍDEO
# ============================================================

def read_video_metadata(video_name):
    path = VIDEOS_DIR / f"{video_name}.mp4"
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()

    duration = meta.get("duration", None)
    nframes = meta.get("nframes", None)

    try:
        reader.close()
    except:
        pass

    return duration, nframes


# ============================================================
#   FUNÇÃO PROCESSAR UM ÚNICO FRAME (para multiprocessamento)
# ============================================================

def process_single_frame(args):
    (
        fr,                   # dict do frame
        noise_type,           # tipo de ruído
        intensity_h2,         # intensidade progressiva
        duration,             # duração do vídeo
        nframes_video         # nº de frames do vídeo
    ) = args

    # cada processo precisa de IMG recarregada
    img = Image.open(fr["frame_path"])
    video_name = fr["video_name"]

    # H0
    out_h0 = OUT_DIR_H0 / fr["frame_path"].name
    img.save(out_h0)

    # H1 (intensidade fixa 6)
    out_h1 = OUT_DIR_H1 / fr["frame_path"].name
    img_h1 = generate_noise_sweep_single(img, noise_type, 6, video_name)
    img_h1.save(out_h1)

    # H2 (crescente)
    out_h2 = OUT_DIR_H2 / fr["frame_path"].name
    img_h2 = generate_noise_sweep_single(img, noise_type, intensity_h2, video_name)
    img_h2.save(out_h2)

    return {
        "video_name": fr["video_name"],
        "frame_index": fr["frame_index"],
        "original_frame": str(fr["frame_path"]),
        "h0_path": str(out_h0),
        "h1_path": str(out_h1),
        "h2_path": str(out_h2),
        "noise_type": noise_type,
        "intensity_h1": 6,
        "intensity_h2": intensity_h2,
        "video_duration": duration,
        "video_nframes": nframes_video
    }


# ============================================================
#   PROCESSAMENTO TOTAL (com multiprocessamento)
# ============================================================

def process_all():
    global SHAPES_CACHE
    SHAPES_CACHE = {}

    frames = load_frames()
    videos = sorted({f["video_name"] for f in frames})
    noise_assignment = assign_noise_types(videos)

    rows = []

    pool = Pool(cpu_count())

    for v in tqdm(videos, desc="Processando vídeos", ncols=100):
        # resetar shapes_x cache para este vídeo
        if v in SHAPES_CACHE:
            SHAPES_CACHE.pop(v)

        duration, nframes_video = read_video_metadata(v)

        video_frames = sorted(
            [f for f in frames if f["video_name"] == v],
            key=lambda x: x["frame_index"]
        )

        N = len(video_frames)
        intensities = np.linspace(0, 10, N)

        args = [
            (
                fr,
                noise_assignment[v],
                float(intensities[i]),
                duration,
                nframes_video
            )
            for i, fr in enumerate(video_frames)
        ]

        results = pool.map(process_single_frame, args)
        rows.extend(results)

    pool.close()
    pool.join()

    df = pd.DataFrame(rows)
    df.to_csv("metadata_final.csv", index=False, encoding="utf-8")

    print("\n✔ Finalizado! CSV salvo como metadata_final.csv")


# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    process_all()
