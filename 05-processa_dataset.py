# -*- coding: utf-8 -*-

import os
import csv
import imageio
import numpy as np
from pathlib import Path

# ============================================================
# BASE PATH — AJUSTE AQUI
# ============================================================

# Exemplo:
# PATH_BASE = Path("/mnt/data/projeto/")
PATH_BASE = Path(".").resolve()   # default: diretório atual

print(f"[DEBUG] PATH_BASE = {PATH_BASE}")

# ============================================================
# CONFIGURAÇÕES DE SUBPASTAS (RELATIVAS AO PATH_BASE)
# ============================================================

VIDEOS_DIR = PATH_BASE / "videos"
FRAMES_OUT = PATH_BASE / "videos_amostrados_mp4"
NOISE_OUT = PATH_BASE / "videos_ruido"

FRAMES_OUT.mkdir(exist_ok=True, parents=True)
NOISE_OUT.mkdir(exist_ok=True, parents=True)

CSV_FRAMES = PATH_BASE / "amostragem_frames.csv"
CSV_NOISE = PATH_BASE / "noise_sweep_map.csv"
CSV_VIDEOS = PATH_BASE / "videos_info.csv"

NOISE_LEVELS = [0.1, 0.2, 0.3]

print(f"[DEBUG] VIDEOS_DIR = {VIDEOS_DIR}")
print(f"[DEBUG] FRAMES_OUT = {FRAMES_OUT}")
print(f"[DEBUG] NOISE_OUT = {NOISE_OUT}")


# ============================================================
# FUNÇÃO: extrair 1 frame por segundo
# ============================================================

def extract_frames_uniform(video_path, out_dir):
    print(f"[DEBUG] Lendo vídeo: {video_path}")

    video_id = video_path.stem
    reader = imageio.get_reader(str(video_path))

    meta = reader.get_meta_data()
    fps = meta["fps"]
    n_frames = meta["nframes"]
    duration = n_frames / fps

    print(f"[DEBUG] FPS={fps}, total_frames={n_frames}, duracao={duration:.2f}s")

    frames_info = []

    # frame indices de 0 até n-1
    frame_indices = np.arange(0, int(duration), 1)
    frame_indices = (frame_indices * fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, n_frames - 1)

    print(f"[DEBUG] Frames que serão extraídos: {frame_indices}")

    for frame_idx in frame_indices:
        frame = reader.get_data(frame_idx)

        out_name = f"{video_id}_frame_{frame_idx:04d}.png"
        out_path = out_dir / out_name

        imageio.imwrite(out_path, frame)

        print(f"[DEBUG] Frame salvo: {out_path}")

        frames_info.append({
            "video_id": video_id,
            "frame_index": int(frame_idx),
            "frame_path": str(out_path.relative_to(PATH_BASE))
        })

    reader.close()

    return frames_info, fps, n_frames, duration


# ============================================================
# FUNÇÃO: gerar 1 imagem ruidosa por intensidade
# ============================================================

def generate_noise_sweep(img_array, intensity):
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_img = img_array.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def process_all_videos():

    print("[DEBUG] Procurando vídeos em:", VIDEOS_DIR)
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))

    print(f"[DEBUG] Vídeos encontrados ({len(videos)}):")
    for v in videos:
        print("   -", v)

    if not videos:
        print("ERRO: Nenhum vídeo encontrado.")
        return

    all_frames = []
    all_noise = []
    all_video_info = []

    for i, video_path in enumerate(videos):
        print(f"\n==============================")
        print(f"[DEBUG] ({i+1}/{len(videos)}) Processando: {video_path.name}")
        print("==============================")

        frames_info, fps, n_frames, duration = extract_frames_uniform(video_path, FRAMES_OUT)

        all_frames.extend(frames_info)
        all_video_info.append({
            "video_id": video_path.stem,
            "total_frames": n_frames,
            "fps": fps,
            "duration_seconds": duration
        })

        # ======================================================
        # GERAR RUIDO PARA CADA FRAME
        # ======================================================

        for f in frames_info:
            img_path = PATH_BASE / f["frame_path"]
            img = imageio.imread(img_path)

            print(f"[DEBUG] Gerando ruídos para frame: {img_path}")

            for intensity in NOISE_LEVELS:
                noisy = generate_noise_sweep(img, intensity)

                out_name = (
                    f"{f['video_id']}_frame_{f['frame_index']:04d}_noise_{intensity:.1f}.png"
                )

                out_path = NOISE_OUT / out_name
                imageio.imwrite(out_path, noisy)

                print(f"[DEBUG] → Ruído {intensity:.1f} salvo: {out_path}")

                all_noise.append({
                    "video_id": f["video_id"],
                    "frame_index": f["frame_index"],
                    "original_frame_path": f["frame_path"],
                    "noisy_frame_path": str(out_path.relative_to(PATH_BASE)),
                    "noise_intensity": intensity
                })

    # ======================================================
    # SALVAR CSVs
    # ======================================================

    print("\n[DEBUG] Salvando CSVs...")

    with open(CSV_FRAMES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frame_index", "frame_path"])
        w.writeheader()
        w.writerows(all_frames)

    with open(CSV_NOISE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "frame_index",
            "original_frame_path", "noisy_frame_path",
            "noise_intensity"
        ])
        w.writeheader()
        w.writerows(all_noise)

    with open(CSV_VIDEOS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "total_frames", "fps", "duration_seconds"
        ])
        w.writeheader()
        w.writerows(all_video_info)

    print("\n✔ PROCESSO FINALIZADO!")
    print("Arquivos gerados:")
    print(" -", CSV_FRAMES)
    print(" -", CSV_NOISE)
    print(" -", CSV_VIDEOS)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_all_videos()
