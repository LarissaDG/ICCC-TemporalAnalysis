# 01_convert_e_amostra_frames.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import unicodedata
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import imageio
from PIL import Image


# =====================================================================
# CONFIG
# =====================================================================

RAW_DIR = Path("videos_amostrados_raw")
WORK_DIR = Path("videos_amostrados_mp4")
FRAMES_DIR = Path("frames_originais")

VIDEO_EXTS = [".mp4", ".webm", ".mov", ".avi", ".gif", ".mkv"]

np.random.seed(42)

imageio.plugins.ffmpeg.ALLOW_EXEC = True
print("🔧 imageio configurado para usar ffmpeg.")


# =====================================================================
# UTIL
# =====================================================================

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_") or "video"


def list_videos(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])


# =====================================================================
# 1) CONVERSÃO PARA MP4
# =====================================================================

def _convert_single(task):
    src, dst = task
    try:
        if dst.exists():
            return src, dst, True, "", True  # skipped

        reader = imageio.get_reader(str(src), "ffmpeg")
        fps = reader.get_meta_data().get("fps", 30)

        writer = imageio.get_writer(
            str(dst),
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"]
        )

        for frame in reader:
            writer.append_data(frame)

        reader.close()
        writer.close()

        return src, dst, True, "", False

    except Exception:
        return src, dst, False, traceback.format_exc(), False


def convert_all_to_mp4(input_videos, output_dir: Path, n_jobs=4):
    output_dir.mkdir(exist_ok=True)
    tasks = []

    for vid in input_videos:
        base = slugify(vid.stem)
        dst = output_dir / f"{base}.mp4"
        tasks.append((vid, dst))

    print(f"\n🎬 Convertendo {len(tasks)} vídeos para MP4...\n")
    pbar = tqdm(total=len(tasks), desc="Convertendo", unit="vídeo", dynamic_ncols=True)
    converted = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_convert_single, t): t for t in tasks}

        for fut in as_completed(futures):
            src, dst, ok, err, skipped = fut.result()
            if ok and not skipped:
                converted.append(dst)
            pbar.update(1)
            pbar.set_postfix_str(src.name[:35])

    return converted


# =====================================================================
# 2) EXTRAÇÃO DE FRAMES (1 frame/seg)
# =====================================================================

def get_duration(video_path: Path) -> float:
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        meta = reader.get_meta_data()
        reader.close()
        return float(meta.get("duration", 0.0))
    except:
        return 0.0


def extract_sampled_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)

    reader = imageio.get_reader(str(video_path), "ffmpeg")
    meta = reader.get_meta_data()

    fps = meta.get("fps", 25)
    duration = meta.get("duration", 0.0)

    total_frames = int(duration)

    frame_paths = []

    for sec in range(total_frames):
        idx = int(round(sec * fps))

        try:
            frame = reader.get_data(idx)
            img = Image.fromarray(frame).convert("RGB")
            name = f"{video_path.stem}_frame_{sec:04d}.png"
            path = out_dir / name
            img.save(path)
            frame_paths.append(path)

        except Exception:
            break

    reader.close()
    return frame_paths


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run(n_jobs=4):

    print("\n==============================")
    print("🎥 CONVERTER E AMOSTRAR FRAMES")
    print("==============================\n")

    for d in [WORK_DIR, FRAMES_DIR]:
        d.mkdir(exist_ok=True)

    # 1) Encontrar vídeos brutos
    raw_videos = list_videos(RAW_DIR)
    print(f"📁 Vídeos encontrados (raw): {len(raw_videos)}")

    if not raw_videos:
        print("⚠️ Nenhum vídeo encontrado em videos_amostrados_raw/")
        return

    # 2) Converter para MP4
    converted_paths = convert_all_to_mp4(raw_videos, WORK_DIR, n_jobs=n_jobs)

    # 3) Extrair frames amostrados por segundo
    print("\n🖼️ Extraindo frames (1/s)...\n")
    for vid in tqdm(converted_paths, desc="Extraindo Frames", dynamic_ncols=True):
        extract_sampled_frames(vid, FRAMES_DIR)

    print("\n✨ Finalizado! Frames salvos em:")
    print(f"   → {FRAMES_DIR.resolve()}")
    print("✨ Vídeos MP4 salvos em:")
    print(f"   → {WORK_DIR.resolve()}\n")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run(n_jobs=4)
