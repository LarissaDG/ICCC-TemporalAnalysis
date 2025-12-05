# temporal_pipeline_experiments_build.py
# -*- coding: utf-8 -*-

"""
VERSÃO REESCRITA:
- Agora NÃO amostra nem renomeia mais vídeos.
- Usa diretamente a pasta videos_amostrados_raw/ como entrada.
- Só faz: conversão → extração → H1/H2 → CSVs.
- Ideal para rodar no SLURM.
"""

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
# CONFIGURAÇÃO
# =====================================================================

RAW_DIR = Path("videos_amostrados_raw")     # << agora SEU PONTO DE ENTRADA
WORK_DIR = Path("videos_amostrados_mp4")    # onde os mp4 convertidos vão
DATASET_DIR = Path("dataset")               # onde vai ficar o dataset final

NOISE_TYPES = ["blur_sp", "shapes_x", "gaussian"]
VIDEO_EXTS = [".mp4", ".webm", ".mov", ".avi", ".gif", ".mkv"]

SCORE_COLUMNS = [
    "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color", "The sense of order",
    "Details and texture", "The overall", "Mood"
]

random.seed(42)


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
# CONVERSÃO MP4
# =====================================================================

def _convert_single(task):
    src, dst = task
    try:
        if dst.exists():
            return src, dst, True, "", True  # skipped

        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vcodec", "libx264",
            "-acodec", "aac",
            str(dst)
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode != 0:
            return src, dst, False, proc.stderr.decode("utf-8"), False

        return src, dst, True, "", False

    except Exception as e:
        return src, dst, False, str(e), False


def convert_all_to_mp4(input_videos: list[Path], output_dir: Path, n_jobs: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    mapping = []

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

            if ok:
                converted_paths.append(dst)

            mapping.append({
                "original": str(src),
                "converted": str(dst),
                "success": ok,
                "error": err,
                "skipped": skipped
            })

            pbar.update(1)
            pbar.set_postfix_str(src.name[:35])

    pd.DataFrame(mapping).to_csv(output_dir / "conversao.csv", index=False)
    return converted_paths


# =====================================================================
# DURAÇÃO DE VÍDEO
# =====================================================================

def get_duration(path: Path) -> float:
    try:
        reader = imageio.get_reader(str(path))
        meta = {}
        try: meta = reader.get_meta_data()
        except: pass

        if "duration" in meta:
            reader.close()
            return float(meta["duration"])

        fps = meta.get("fps", None)
        nframes = meta.get("nframes", None)

        if fps and nframes:
            reader.close()
            return nframes / fps

        # fallback: contar frames
        cnt = 0
        for _ in reader:
            cnt += 1
        reader.close()
        return cnt / (fps or 25)

    except:
        return 0.0


# =====================================================================
# EXTRAÇÃO DE FRAMES
# =====================================================================

def extract_frame_at_second(path: Path, sec: float):
    try:
        reader = imageio.get_reader(str(path))
        meta = {}
        try: meta = reader.get_meta_data()
        except: pass

        fps = meta.get("fps", None)

        if fps:
            idx = int(round(sec * fps))
            try:
                frame = reader.get_data(idx)
            except:
                frame = reader.get_data(max(0, len(reader)-1))
            reader.close()
            return Image.fromarray(frame).convert("RGB")

        # fallback
        last = None
        for f in reader:
            last = f
        reader.close()

        if last is None:
            return None

        return Image.fromarray(last).convert("RGB")

    except Exception:
        return None


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


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def build_dataset(dataset_name="dataset_final", n_h1_repeats=5, n_jobs=4):

    print("\n==============================")
    print("📦 MONTANDO DATASET FINAL")
    print("==============================\n")

    raw_videos = list_videos(RAW_DIR)
    print(f"📁 Vídeos encontrados (raw): {len(raw_videos)}")

    # ------------------------------------------------------------
    # 1) Converter para MP4
    # ------------------------------------------------------------
    converted = convert_all_to_mp4(raw_videos, WORK_DIR, n_jobs=n_jobs)
    print(f"\n✔ Conversão para MP4 concluída: {len(converted)} vídeos\n")

    # ------------------------------------------------------------
    # 2) Durações
    # ------------------------------------------------------------
    df_meta = []
    for p in converted:
        df_meta.append({
            "path": str(p),
            "name": p.stem,
            "duration": get_duration(p)
        })

    df_meta = pd.DataFrame(df_meta)
    df_meta = df_meta.sort_values("duration").reset_index(drop=True)

    T = max(1, int(df_meta["duration"].min()))
    print(f"⏱ Usando T = {T} segundos (1 FPS)\n")

    # ------------------------------------------------------------
    # Criar pastas dataset/
    # ------------------------------------------------------------
    root = DATASET_DIR / dataset_name
    orig_dir = root / "frames_originais"
    h1_dir   = root / "frames_h1"
    h2_dir   = root / "frames_h2"
    exp_dir  = root / "orquestrador"

    for p in [root, orig_dir, h1_dir, h2_dir, exp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 3) Extrair frames 1 FPS
    # ------------------------------------------------------------
    rows_orig = []
    for idx, row in df_meta.iterrows():
        vid = Path(row["path"])
        name = row["name"]

        for i in range(T):
            img = extract_frame_at_second(vid, i)
            if img is None:
                continue

            fname = f"{name}_frame{i+1}.png"
            out = orig_dir / fname
            img.save(out)

            rec = {
                "filename": fname,
                "video_name": name,
                "frame_number": i+1,
                "oficial_path": str(out),
                "duration": row["duration"]
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan
            rows_orig.append(rec)

    df_orig = pd.DataFrame(rows_orig)
    df_orig.to_csv(root / "frames_originais.csv", index=False)
    print(f"✔ Frames originais: {len(df_orig)}\n")

    # ------------------------------------------------------------
    # ASSIGN NOISE TYPES
    # ------------------------------------------------------------
    vids = df_orig["video_name"].unique().tolist()
    random.shuffle(vids)

    assignment = {}
    group = len(vids)//len(NOISE_TYPES)

    start = 0
    for i, nt in enumerate(NOISE_TYPES):
        if i < len(NOISE_TYPES)-1:
            chunk = vids[start:start+group]
        else:
            chunk = vids[start:]
        for v in chunk:
            assignment[v] = nt
        start += group

    # ------------------------------------------------------------
    # 4) H1 — intensidade 1.0 constante
    # ------------------------------------------------------------
    h1_rows = []
    for v in vids:
        nt = assignment[v]
        dfv = df_orig[df_orig["video_name"] == v]

        sub = h1_dir / nt
        sub.mkdir(exist_ok=True)

        for _, r in dfv.iterrows():
            orig = Image.open(r["oficial_path"]).convert("RGB")
            mod  = apply_noise(orig, nt, 1.0)

            fname = r["filename"].replace(".png", f"_h1_{nt}.png")
            out = sub / fname
            mod.save(out)

            rec = {
                "filename_orig": r["filename"],
                "filename_mod": fname,
                "video_name": v,
                "frame_number": r["frame_number"],
                "noise_type": nt,
                "noise_intensity": 1.0,
                "oficial_path_orig": r["oficial_path"],
                "oficial_path_mod": str(out)
            }

            for c in SCORE_COLUMNS:
                rec[c] = np.nan

            h1_rows.append(rec)

    df_h1 = pd.DataFrame(h1_rows)
    df_h1.to_csv(exp_dir / "h1_mapping.csv", index=False)
    print(f"✔ H1 gerado: {len(df_h1)}\n")

    # ------------------------------------------------------------
    # 5) H2 — rampa 0→1
    # ------------------------------------------------------------
    h2_rows = []
    for v in vids:
        nt = assignment[v]
        dfv = df_orig[df_orig["video_name"] == v].sort_values("frame_number")

        ramp = np.linspace(0,1,len(dfv))

        sub = h2_dir / v
        sub.mkdir(exist_ok=True)

        for idx,(i,r) in enumerate(dfv.iterrows()):
            orig = Image.open(r["oficial_path"]).convert("RGB")
            inten = float(ramp[idx])
            mod  = apply_noise(orig, nt, inten)

            pct = int(round(inten*100))
            fname = r["filename"].replace(".png", f"_h2_{nt}_i{pct}.png")
            out = sub / fname
            mod.save(out)

            rec = {
                "filename_orig": r["filename"],
                "filename_mod": fname,
                "video_name": v,
                "frame_number": r["frame_number"],
                "noise_type": nt,
                "noise_intensity": inten,
                "oficial_path_orig": r["oficial_path"],
                "oficial_path_mod": str(out)
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan

            h2_rows.append(rec)

    df_h2 = pd.DataFrame(h2_rows)
    df_h2.to_csv(exp_dir / "h2_mapping.csv", index=False)
    print(f"✔ H2 gerado: {len(df_h2)}\n")

    print("\n🎉 Dataset final construído com sucesso!")
    return {
        "root": root,
        "orig_csv": root/"frames_originais.csv",
        "h1_csv": exp_dir/"h1_mapping.csv",
        "h2_csv": exp_dir/"h2_mapping.csv"
    }


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    build_dataset(dataset_name="auto_dataset_final", n_jobs=6)
