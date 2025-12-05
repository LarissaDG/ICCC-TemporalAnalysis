# temporal_pipeline_experiments.py
# -*- coding: utf-8 -*-
"""
Pipeline preparado para os experimentos H1, H2 e H5.
Siga as mensagens impressas para checar paths e dependências (ffmpeg/imageio).
"""
import os
from pathlib import Path
import random
import shutil
import math
import json
from typing import List, Dict
import itertools
import traceback

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
import time
# stats
from scipy import stats

# ----------------------
# Configurações (ajuste conforme necessário)
# ----------------------
random.seed(42)
DOWNLOADS_DIR = Path("downloads")           # onde estão os vídeos
DATASET_DIR = Path("dataset")               # onde os datasets serão salvos
RESULTS_DIR = Path("resultados")            # plots e métricas
AMOSTRA_DIR = Path("videos_amostrados")     # onde os vídeos amostrados serão copiados
SAMPLE_N_VIDEOS = 100
INTENSITIES = [0.0, 0.25, 0.5, 0.75, 1.0]   # opções (H2 pode usar rampa contínua)
NOISE_TYPES = ["blur_sp", "shapes_x", "gaussian"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".gif", ".webm", ".mkv"]

SCORE_COLUMNS = [
    "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color", "The sense of order",
    "Details and texture", "The overall", "Mood"
]

# ----------------------
# UTIL
# ----------------------
def list_videos(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} não existe.")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)

def sample_videos(files: List[Path], n: int, output_dir: Path) -> List[Path]:
    """
    Filtra vídeos (remove nomes contendo 'tips'), amostra aleatoriamente n vídeos
    e salva as amostras em outra pasta (output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove 'tips' (case-insensitive)
    filtered = [f for f in files if "tips" not in f.stem.lower()]

    random.shuffle(filtered)
    selected = filtered[:min(n, len(filtered))]

    copied_paths = []
    for src in selected:
        dst = output_dir / src.name
        try:
            shutil.copy2(src, dst)
            copied_paths.append(dst)
        except Exception as e:
            print(f"⚠️ Erro ao copiar {src} -> {dst}: {e}")

    return copied_paths

def slugify(text: str) -> str:
    """Remove emojis, acentos e caracteres problemáticos para criar nomes seguros."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_") or "video"

def _convert_single_video(task):
    """
    Função que será rodada dentro de cada processo.
    task = (src_path, dst_path)
    """
    src, dst = task
    try:
        if not dst.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(src),
                "-vcodec", "libx264",
                "-acodec", "aac",
                str(dst)
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if proc.returncode != 0:
                return (src, dst, False, proc.stderr.decode("utf-8"))
        return (src, dst, True, "")
    except Exception as e:
        return (src, dst, False, str(e))

def convert_to_mp4(videos: list[Path], output_dir: Path) -> list[Path]:
    """
    Converte vídeos para MP4 com barra de progresso aprimorada (ETA, velocidade etc.).
    Pula vídeos que já estejam convertidos (mp4 já existe).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    converted_paths = []

    pbar = tqdm(
        videos,
        desc=Fore.CYAN + "Convertendo vídeos" + Style.RESET_ALL,
        unit="vídeo",
        dynamic_ncols=True,
        colour="green"
    )

    for vid in pbar:
        orig_name = vid.name
        pbar.set_postfix_str(f"{orig_name[:40]}")

        base = slugify(vid.stem)
        mp4_path = output_dir / f"{base}.mp4"

        start = time.time()
        skipped = False
        success = False
        err = ""

        if mp4_path.exists():
            # já convertido -> pular
            skipped = True
            success = True
        else:
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(vid),
                    "-vcodec", "libx264",
                    "-acodec", "aac",
                    str(mp4_path)
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    success = False
                    err = proc.stderr.decode("utf-8", errors="ignore")
                else:
                    success = True
            except Exception as e:
                success = False
                err = str(e)
        elapsed = time.time() - start

        rows.append({
            "original_path": str(vid),
            "original_name": orig_name,
            "converted_name": mp4_path.name,
            "converted_path": str(mp4_path),
            "success": bool(success),
            "error": err,
            "skipped": bool(skipped),
            "elapsed_s": float(round(elapsed, 3))
        })

        if success:
            converted_paths.append(mp4_path)

    pbar.close()

    # Salvar tabela de mapeamento
    df_map = pd.DataFrame(rows)
    df_map.to_csv(output_dir / "nome_conversao.csv", index=False)

    print(f"✔ Arquivo de mapeamento salvo em: {output_dir/'nome_conversao.csv'}")
    return converted_paths


def convert_to_mp4_parallel(videos: list[Path], output_dir: Path, n_jobs: int = 6) -> list[Path]:
    """
    Converte vídeos para mp4 usando múltiplos processos com barra tqdm PRO (ETA, velocidade, etc.)
    Pula vídeos cujo mp4 destino já exista (não submete ao pool).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = []
    mapping_rows = []
    converted_paths = []

    # preparar tasks: se dst existe, não submete; registra como skipped
    for vid in videos:
        base = slugify(vid.stem)
        dst = output_dir / f"{base}.mp4"
        if dst.exists():
            # já convertido: não submeter, apenas registrar
            mapping_rows.append({
                "original_path": str(vid),
                "original_name": vid.name,
                "converted_name": dst.name,
                "converted_path": str(dst),
                "success": True,
                "error": "",
                "skipped": True,
                "elapsed_s": 0.0
            })
            converted_paths.append(dst)
        else:
            tasks_to_run.append((vid, dst))

    total_tasks = len(tasks_to_run)
    print(f"\n🔧 Convertendo {total_tasks} vídeos (de {len(videos)}) para mp4 usando {n_jobs} processos...\n")

    if total_tasks == 0:
        # nada a fazer, salvar mapa e retornar
        df_map = pd.DataFrame(mapping_rows)
        df_map.to_csv(output_dir / "nome_conversao.csv", index=False)
        print(f"✔ Arquivo de mapeamento salvo em: {output_dir/'nome_conversao.csv'}")
        return converted_paths

    pbar = tqdm(
        total=total_tasks,
        desc=Fore.CYAN + "Convertendo vídeos" + Style.RESET_ALL,
        unit="vídeo",
        dynamic_ncols=True,
        colour="green"
    )

    # roda no pool
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_convert_single_video, t): t for t in tasks_to_run}
        for fut in as_completed(futures):
            src, dst, ok, err = fut.result()
            pbar.set_postfix_str(f"{src.name[:40]}")

            # Note: _convert_single_video já checa se dst.exists() dentro do processo;
            # mas aqui submit só incluiu tasks cujo dst não existia no momento do envio.
            elapsed = 0.0
            if ok and dst.exists():
                # sucesso
                converted_paths.append(dst)

            mapping_rows.append({
                "original_path": str(src),
                "original_name": src.name,
                "converted_name": dst.name,
                "converted_path": str(dst),
                "success": bool(ok),
                "error": err,
                "skipped": False,
                "elapsed_s": float(elapsed)
            })

            if not ok:
                print(f"\n❌ Falha ao converter {src.name}: {err}")

            pbar.update(1)

    pbar.close()

    # combinar registros de pré-skipped + processados
    df_map = pd.DataFrame(mapping_rows)
    # se já existiam linhas anteriores em mapping_rows (pré-skipped), elas já estão lá
    df_map.to_csv(output_dir / "nome_conversao.csv", index=False)

    print(f"\n✔ Arquivo de mapeamento salvo em: {output_dir/'nome_conversao.csv'}\n")
    return converted_paths

# ----------------------
# DURAÇÃO (segundos)
# ----------------------
def get_video_duration(path: Path) -> float:
    path = Path(path)
    suf = path.suffix.lower()
    try:
        if suf == ".gif":
            im = Image.open(path)
            durations = []
            i = 0
            while True:
                durations.append(im.info.get("duration", 40))
                i += 1
                try:
                    im.seek(i)
                except EOFError:
                    break
            return sum(durations) / 1000.0
        else:
            reader = imageio.get_reader(str(path))
            meta = {}
            try:
                meta = reader.get_meta_data()
            except Exception:
                pass
            # try duration
            if "duration" in meta and isinstance(meta["duration"], (int, float)):
                duration = float(meta["duration"])
                reader.close()
                return duration
            fps = meta.get("fps", None)
            nframes = meta.get("nframes", None)
            if fps and nframes:
                reader.close()
                return float(nframes) / float(fps)
            # else count frames
            cnt = 0
            for _ in reader:
                cnt += 1
            reader.close()
            if fps:
                return cnt / float(fps)
            return cnt / 25.0
    except Exception as e:
        print(f"[Video duration] Erro em {path}: {e}")
        return 0.0

# ----------------------
# EXTRACAO DE FRAME EM TIMESTAMP (segundos)
# ----------------------
def extract_frame_at_second(path: Path, second: float):
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".gif":
            with Image.open(path) as im:
                durations = []
                frames = []
                i = 0
                try:
                    while True:
                        durations.append(im.info.get("duration", 40))
                        frames.append(im.copy().convert("RGB"))
                        i += 1
                        im.seek(i)
                except EOFError:
                    pass
                cumulative = np.cumsum([0] + durations[:-1]) / 1000.0
                idx = np.searchsorted(cumulative, second)
                idx = min(idx, len(frames)-1)
                return frames[idx]
        else:
            reader = imageio.get_reader(str(path))
            meta = {}
            try:
                meta = reader.get_meta_data()
            except Exception:
                pass
            fps = meta.get("fps", None)
            if fps:
                frame_index = int(round(second * fps))
                try:
                    frame = reader.get_data(frame_index)
                except Exception:
                    try:
                        frame = reader.get_data(len(reader)-1)
                    except Exception:
                        reader.close()
                        return None
                reader.close()
                return Image.fromarray(frame).convert("RGB")
            else:
                last = None
                for f in reader:
                    last = f
                reader.close()
                if last is None:
                    return None
                return Image.fromarray(last).convert("RGB")
    except Exception as e:
        print(f"[extract_frame] erro {path} @ {second}s: {e}")
        return None

# ----------------------
# RUÍDOS (intensity wrappers)
# ----------------------
def add_blur_saltpepper_intensity(frame: Image.Image, intensity: float):
    blur_radius = 1 + intensity * 6.0
    sp_amount = 0.0 + intensity * 0.06
    im_blur = frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(im_blur).astype(np.int16)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    h, w, c = arr.shape
    n = max(1, int(h * w * sp_amount))
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 255
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def add_shapes_x_intensity(frame: Image.Image, intensity: float):
    base = frame.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = base.size
    alpha = int(50 + intensity * 205)
    line_w = max(1, int(min(w,h) * (0.01 + intensity*0.03)))
    draw.line((0,0,w,h), fill=(255,0,0,alpha), width=line_w)
    draw.line((0,h,w,0), fill=(255,0,0,alpha), width=line_w)
    for _ in range(1 + int(3*intensity)):
        if random.random() < 0.5:
            x0 = random.randint(0, w-1)
            y0 = random.randint(0, h-1)
            x1 = min(w-1, x0 + random.randint(10, max(10,int(w*0.2*intensity+20))))
            y1 = min(h-1, y0 + random.randint(10, max(10,int(h*0.2*intensity+20))))
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), alpha)
            draw.rectangle([x0,y0,x1,y1], fill=color)
        else:
            cx = random.randint(0, w-1)
            cy = random.randint(0, h-1)
            r = max(5, int(min(w,h) * (0.02 + 0.1*intensity)))
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), alpha)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    res = Image.alpha_composite(base, overlay).convert("RGB")
    return res

def add_gaussian_noise_intensity(frame: Image.Image, intensity: float):
    sigma = intensity * 60.0
    arr = np.array(frame.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_noise_by_type_and_intensity(img: Image.Image, noise_type: str, intensity: float) -> Image.Image:
    if noise_type == "blur_sp":
        return add_blur_saltpepper_intensity(img, intensity)
    elif noise_type == "shapes_x":
        return add_shapes_x_intensity(img, intensity)
    elif noise_type == "gaussian":
        return add_gaussian_noise_intensity(img, intensity)
    else:
        raise ValueError(noise_type)

# ----------------------
# FUNCAO VISUALIZACAO EXEMPLOS
# ----------------------
def visualizar_exemplos_ruido(dataset_root: Path, out_name: str = "visualizar_exemplos_ruido.png"):
    """
    Cria uma imagem com exemplos (por tipo de ruído) comparando original x ruidoso (H1 intensity=1).
    """
    dataset_root = Path(dataset_root)
    orig_csv = dataset_root / f"{dataset_root.name}_original_frames.csv"
    h1_csv = dataset_root / "orquestrador_experimentos" / "h1_ruidos_mapping.csv"
    if not orig_csv.exists() or not h1_csv.exists():
        print("visualizar_exemplos_ruido: CSVs faltando, não há exemplos.")
        return None
    df_orig = pd.read_csv(orig_csv)
    df_h1 = pd.read_csv(h1_csv)

    examples = []
    for noise in NOISE_TYPES:
        df_n = df_h1[df_h1["noise_type"] == noise]
        if df_n.empty:
            continue
        row = df_n.iloc[0]
        orig_path = Path(row["oficial_path_orig"])
        mod_path = Path(row["oficial_path_mod"])
        if orig_path.exists() and mod_path.exists():
            examples.append((noise, orig_path, mod_path))

    if not examples:
        print("Nenhum exemplo disponível.")
        return None

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (noise, orig, mod) in enumerate(examples):
        ax_o = axes[i][0] if n>1 else axes[0]
        ax_m = axes[i][1] if n>1 else axes[1]
        ax_o.imshow(Image.open(orig))
        ax_o.set_title(f"Original ({noise})")
        ax_o.axis("off")
        ax_m.imshow(Image.open(mod))
        ax_m.set_title(f"Ruidoso ({noise})")
        ax_m.axis("off")

    plt.tight_layout()
    out_path = dataset_root / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Visualização salva em: {out_path}")
    return out_path

# ----------------------
# PIPELINE PRINCIPAL DE PREPARACAO (gera originais, H1 ruidos, H2 intensidades e CSVs)
# ----------------------
def build_dataset_for_experiments(dataset_name: str = "auto_dataset",
                                  downloads_dir: Path = DOWNLOADS_DIR,
                                  n_videos: int = SAMPLE_N_VIDEOS,
                                  intensities_for_h2: List[float] = None,
                                  n_h1_repeats: int = 5):
    if intensities_for_h2 is None:
        intensities_for_h2 = None  # we will create ramp based on T frames

    downloads_dir = Path(downloads_dir)
    files = list_videos(downloads_dir)
    sampled = sample_videos(files, n_videos, AMOSTRA_DIR)
    print(f"Arquivos encontrados: {len(files)} — amostrando {len(sampled)} videos.")

    # Pasta final para trabalhar só com mp4
    print("Convertendo para mp4")
    sampled_mp4 = convert_to_mp4_parallel(sampled, AMOSTRA_DIR)
    print("Conversão concluida")

    # compute durations
    records = []
    for p in sampled:
        dur = get_video_duration(p)
        records.append({"path": str(p), "name": p.stem, "suffix": p.suffix.lower(), "duration_s": float(dur)})
    df_meta = pd.DataFrame(records).sort_values("duration_s").reset_index(drop=True)

    if df_meta.empty:
        raise RuntimeError("Nenhum vídeo para processar.")

    # choose minimal duration (floor to int >=1)
    min_dur = max(1, int(math.floor(df_meta["duration_s"].min())))
    T = min_dur
    print(f"Menor duração (segundos) = {df_meta['duration_s'].min():.3f} → usando T = {T} segundos (1 frame/segundo).")

    # create dataset folders
    dataset_root = Path(DATASET_DIR) / dataset_name
    orig_dir = dataset_root / "frames_originais"
    h1_dir = dataset_root / "frames_ruidos"            # H1: all frames corrupted (intensity=1) per video
    h2_dir = dataset_root / "frames_intensidades"      # H2: ramp intensities along time
    experiments_dir = dataset_root / "orquestrador_experimentos"
    for p in [dataset_root, orig_dir, h1_dir, h2_dir, experiments_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # extract frames 1 fps for T seconds per video and save sequence
    rows_original = []
    for idx, meta in df_meta.iterrows():
        p = Path(meta["path"])
        name = meta["name"]
        dur = float(meta["duration_s"])
        seconds = list(range(T))
        for i, s in enumerate(seconds, start=1):
            img = extract_frame_at_second(p, float(s))
            if img is None:
                continue
            fname = f"{name}_frame{i}.png"
            out = orig_dir / fname
            try:
                img.save(out)
            except Exception as e:
                print(f"Erro salvando {out}: {e}")
                continue
            rec = {
                "filename": fname,
                "video_name": name,
                "video_index": int(idx+1),
                "frame_number": i,
                "oficial_path": str(out),
                "source_video": str(p),
                "duration_seconds": float(dur)
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan
            rows_original.append(rec)

    df_original = pd.DataFrame(rows_original)
    meta_csv = dataset_root / f"{dataset_name}_metadata_videos.csv"
    df_meta.to_csv(meta_csv, index=False)
    orig_csv = dataset_root / f"{dataset_name}_original_frames.csv"
    df_original.to_csv(orig_csv, index=False)
    print(f"✔ Frames originais salvos: {len(df_original)} → {orig_csv}")

    # ----------------------
    # assign noise type per video (balanced-ish but randomized)
    # ----------------------
    video_names = df_original["video_name"].unique().tolist()
    random.shuffle(video_names)
    n_v = len(video_names)
    group_size = n_v // len(NOISE_TYPES)
    assignment = {}
    start = 0
    for i, nt in enumerate(NOISE_TYPES):
        if i < len(NOISE_TYPES)-1:
            group = video_names[start:start+group_size]
        else:
            group = video_names[start:]
        for v in group:
            assignment[v] = nt
        start += group_size

    # if some videos remain unassigned (edge cases), assign randomly
    for v in video_names:
        if v not in assignment:
            assignment[v] = random.choice(NOISE_TYPES)

    # ----------------------
    # H1: create frames_ruidos — same noise type applied to ALL frames of a video (intensity=1.0)
    # ----------------------
    h1_records = []
    for v in video_names:
        nt = assignment[v]
        # create subdir per noise type for clarity
        sub_noise_dir = h1_dir / nt
        sub_noise_dir.mkdir(parents=True, exist_ok=True)
        df_v = df_original[df_original["video_name"] == v]
        for _, row in df_v.iterrows():
            orig_path = Path(row["oficial_path"])
            if not orig_path.exists():
                continue
            try:
                img = Image.open(orig_path).convert("RGB")
            except Exception:
                continue
            # intensity = 1.0 (max) to simulate persistent error
            img_mod = apply_noise_by_type_and_intensity(img, nt, 1.0)
            out_fname = f"{row['filename'].replace('.png','')}_h1_noise_{nt}_i100.png"
            out_path = sub_noise_dir / out_fname
            try:
                img_mod.save(out_path)
            except Exception as e:
                print(f"Erro salvar mod H1 {out_path}: {e}")
                continue
            rec = {
                "filename_orig": row["filename"],
                "filename_mod": out_fname,
                "video_name": v,
                "frame_number": int(row["frame_number"]),
                "noise_type": nt,
                "noise_intensity": 1.0,
                "oficial_path_orig": str(orig_path),
                "oficial_path_mod": str(out_path)
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan
            h1_records.append(rec)
    df_h1 = pd.DataFrame(h1_records)
    h1_csv = experiments_dir / "h1_ruidos_mapping.csv"
    df_h1.to_csv(h1_csv, index=False)
    print(f"✔ H1 ruídos gerados: {len(df_h1)} → {h1_csv}")

    # ----------------------
    # H1 repeats orchestration: for cross-validation-style experiments
    # For each video create n_h1_repeats rows with frames chosen as 'noisy' (e.g., 60% of frames)
    # ----------------------
    h1_repeats_rows = []
    pct_noisy = 0.6
    for rep in range(1, n_h1_repeats+1):
        for v in video_names:
            df_v = df_original[df_original["video_name"]==v]
            frame_nums = df_v["frame_number"].tolist()
            k = max(1, math.ceil(len(frame_nums) * pct_noisy))
            chosen = random.sample(frame_nums, k) if len(frame_nums) >= k else frame_nums
            h1_repeats_rows.append({
                "repeat": rep,
                "video_name": v,
                "chosen_noisy_frames": json.dumps(sorted(chosen))
            })
    df_h1_repeats = pd.DataFrame(h1_repeats_rows)
    df_h1_repeats.to_csv(experiments_dir / "h1_repeats.csv", index=False)
    print(f"✔ H1 repeats salvos: {len(df_h1_repeats)} rows → {experiments_dir/'h1_repeats.csv'}")

    # ----------------------
    # H2: create frames_intensidades — for each video apply intensity ramp across the temporal frames
    # For each video we create modified frames with intensity corresponding to that frame's position
    # ----------------------
    h2_records = []
    for v in video_names:
        nt = assignment[v]
        df_v = df_original[df_original["video_name"] == v].sort_values("frame_number")
        frames_count = len(df_v)
        if frames_count == 0:
            continue
        # intensity ramp: 0..1 across frames (inclusive)
        if frames_count == 1:
            ramp = [0.0]
        else:
            ramp = list(np.linspace(0.0, 1.0, frames_count))
        # create per-video folder inside h2_dir
        sub_v_dir = h2_dir / v
        sub_v_dir.mkdir(parents=True, exist_ok=True)
        for idx, (_, row) in enumerate(df_v.reset_index().iterrows()):
            orig_path = Path(row["oficial_path"])
            if not orig_path.exists():
                continue
            try:
                img = Image.open(orig_path).convert("RGB")
            except Exception:
                continue
            intensity = float(ramp[idx])
            # apply noise with this intensity
            img_mod = apply_noise_by_type_and_intensity(img, nt, intensity)
            pct = int(round(intensity * 100))
            out_fname = f"{row['filename'].replace('.png','')}_h2_noise_{nt}_i{pct}.png"
            out_path = sub_v_dir / out_fname
            try:
                img_mod.save(out_path)
            except Exception as e:
                print(f"Erro salvar mod H2 {out_path}: {e}")
                continue
            rec = {
                "filename_orig": row["filename"],
                "filename_mod": out_fname,
                "video_name": v,
                "frame_number": int(row["frame_number"]),
                "noise_type": nt,
                "noise_intensity": float(intensity),
                "oficial_path_orig": str(orig_path),
                "oficial_path_mod": str(out_path)
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan
            h2_records.append(rec)
    df_h2 = pd.DataFrame(h2_records)
    h2_csv = experiments_dir / "h2_intensity_mapping.csv"
    df_h2.to_csv(h2_csv, index=False)
    print(f"✔ H2 ruídos (intensidades) gerados: {len(df_h2)} → {h2_csv}")

    # ----------------------
    # Save overall orchestration index (summary)
    # ----------------------
    orchestrator_summary = {
        "dataset_root": str(dataset_root),
        "n_videos_sampled": len(video_names),
        "T_seconds": T,
        "n_frames_total": len(df_original),
        "h1_rows": len(df_h1),
        "h2_rows": len(df_h2),
    }
    with open(experiments_dir / "orchestrator_summary.json", "w", encoding="utf-8") as f:
        json.dump(orchestrator_summary, f, indent=2)

    print("✔ Orquestrador gerado em:", experiments_dir)
    # visualize examples
    try:
        visualizar_exemplos_ruido(dataset_root)
    except Exception:
        print("⚠ visualizar_exemplos_ruido falhou:", traceback.format_exc())

    return {
        "dataset_root": dataset_root,
        "meta_csv": meta_csv,
        "orig_csv": orig_csv,
        "h1_csv": h1_csv,
        "h2_csv": h2_csv,
        "experiments_dir": experiments_dir,
        "df_meta": df_meta,
        "df_original": df_original,
        "df_h1": df_h1,
        "df_h2": df_h2,
        "T_seconds": T
    }

# ----------------------
# ENTRYPOINT
# ----------------------
def main_build_dataset():
    print("=== BUILD DATASET FOR EXPERIMENTS ===")
    res = build_dataset_for_experiments(dataset_name="auto_dataset", downloads_dir=DOWNLOADS_DIR, n_videos=SAMPLE_N_VIDEOS, n_h1_repeats=5)
    print("Resultado:", res["dataset_root"])
    print("Próximo passo: rode o avaliador APDDv2 sobre:")
    print(" -", res["orig_csv"])
    print(" -", res["h1_csv"], "(H1 ruídos)")
    print(" -", res["h2_csv"], "(H2 intensidades)")
    print("\nDepois rode compute_all_metrics_from_scores(...) para análise (script adicional).")

if __name__ == "__main__":
    main_build_dataset()
