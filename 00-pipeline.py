# temporal_pipeline_rewrite.py
# -*- coding: utf-8 -*-
"""
Pipeline para análise temporal de vídeos / GIFs:
- lê vídeos de ./downloads
- amostra até 100 vídeos
- calcula duração, extrai 1 frame/segundo por T segundos (T = duração do menor vídeo)
- gera ruídos balanceados (3 tipos) com intensidades (0,25,50,75,100%)
- salva CSVs e imagens organizadas
- fornece funções de análise/estatística para testar H1..H5
"""
import os
from pathlib import Path
import random
import shutil
import math
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
import imageio
import matplotlib.pyplot as plt

# stats
from scipy import stats

# ----------------------
# Configurações
# ----------------------
DOWNLOADS_DIR = Path("downloads")           # onde estão os vídeos
DATASET_DIR = Path("dataset")               # onde os datasets serão salvos
RESULTS_DIR = Path("resultados")            # plots e métricas
AMOSTRA_DIR = Path("amostra")            # plots e métricas
SAMPLE_N_VIDEOS = 100
INTENSITIES = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0% .. 100%
NOISE_TYPES = ["blur_sp", "shapes_x", "gaussian"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".gif", ".webm", ".mkv"]

# Score columns placeholder (same as você usa)
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

def sample_videos(
    files: List[Path], 
    n: int, 
    output_dir: Path
) -> List[Path]:
    """
    Filtra vídeos (remove nomes contendo 'tips'), amostra aleatoriamente n vídeos
    e salva as amostras em outra pasta.

    Retorna a lista dos paths das amostras já copiadas para output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # 1. remover vídeos contendo 'tips'
    # ---------------------------------------
    filtered = [f for f in files if "tips" not in f.stem.lower()]

    # ---------------------------------------
    # 2. embaralhar
    # ---------------------------------------
    random.shuffle(filtered)

    # ---------------------------------------
    # 3. selecionar n
    # ---------------------------------------
    selected = filtered[:min(n, len(filtered))]

    # ---------------------------------------
    # 4. copiar para a pasta de destino
    # ---------------------------------------
    copied_paths = []
    for src in selected:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied_paths.append(dst)

    return copied_paths
# ----------------------
# DURAÇÃO (segundos)
# ----------------------
def get_video_duration(path: Path) -> float:
    """
    Retorna a duração de um vídeo curto (mp4, mov, avi...) ou gif.
    - Para GIF: soma das durações dos frames via PIL.
    - Para outros vídeos: usa metadata do imageio (FFmpeg).
    """
    path = Path(path)
    suf = path.suffix.lower()

    # -----------------------------
    # Caso GIF
    # -----------------------------
    if suf == ".gif":
        try:
            im = Image.open(path)
            durations = []
            i = 0
            while True:
                durations.append(im.info.get("duration", 40))  # default 40 ms
                i += 1
                try:
                    im.seek(i)
                except EOFError:
                    break
            total_ms = sum(durations)
            return total_ms / 1000.0
        except Exception as e:
            print(f"[GIF duration] Erro em {path}: {e}")
            return 0.0

    # -----------------------------
    # Caso vídeo real (mp4, mov…)
    # -----------------------------
    try:
        reader = imageio.get_reader(str(path))
        meta = reader.get_meta_data()

        # Primeira tentativa — FFmpeg fornece diretamente "duration"
        if "duration" in meta and isinstance(meta["duration"], (int, float)):
            duration = float(meta["duration"])
            reader.close()
            return duration

        # Segunda tentativa — usar fps e nframes
        fps = meta.get("fps", None)
        nframes = meta.get("nframes", None)

        if fps and nframes:
            reader.close()
            return float(nframes) / float(fps)

        # Terceira opção — contar frames manualmente
        cnt = 0
        for _ in reader:
            cnt += 1
        reader.close()

        if fps:
            return cnt / float(fps)

        # fallback final — assume-se 25 fps
        return cnt / 25.0

    except Exception as e:
        print(f"[Video duration] Erro em {path}: {e}")
        return 0.0

# ----------------------
# EXTRAIR FRAME EM TIMESTAMP (segundos)
# ----------------------
def extract_frame_at_second(path: Path, second: float) -> Image.Image:
    """
    retorna PIL.Image do frame mais próximo ao tempo (segundos)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".gif":
            with Image.open(path) as im:
                # use cumulative durations to find appropriate frame
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
            meta = reader.get_meta_data()
            fps = meta.get("fps", None)
            if fps:
                frame_index = int(round(second * fps))
                try:
                    frame = reader.get_data(frame_index)
                except IndexError:
                    # fallback to last
                    frame = reader.get_data(len(reader)-1)
                reader.close()
                return Image.fromarray(frame).convert("RGB")
            else:
                # iterate to approximate
                tot = 0
                last = None
                for f in reader:
                    last = f
                reader.close()
                if last is None:
                    raise RuntimeError("No frames")
                return Image.fromarray(last).convert("RGB")
    except Exception as e:
        print(f"[extract_frame] erro {path} @ {second}s: {e}")
        return None

# ----------------------
# RUÍDOS com intensidade
# ----------------------
def add_blur_saltpepper_intensity(frame: Image.Image, intensity: float):
    """
    intensity in [0,1]
    blur radius scaled up to 6, sp_amount scaled up to 0.05
    """
    blur_radius = 1 + intensity * 6.0
    sp_amount = 0.0 + intensity * 0.06
    im_blur = frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(im_blur).astype(np.int16)
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
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
    # draws lines + shapes with alpha scaled by intensity
    base = frame.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = base.size
    alpha = int(50 + intensity * 205)  # from 50..255
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
    # intensity scales sigma from 0..60
    sigma = intensity * 60.0
    arr = np.array(frame.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

# wrapper
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
# PIPELINE PRINCIPAL
# ----------------------
def build_dataset_from_downloads(dataset_name: str = "dataset", downloads_dir: Path = DOWNLOADS_DIR,
                                 n_videos: int = SAMPLE_N_VIDEOS, intensities: List[float] = INTENSITIES):
    downloads_dir = Path(downloads_dir)
    files = list_videos(downloads_dir)
    sampled = sample_videos(files, n_videos, AMOSTRA_DIR)
    print(f"Arquivos encontrados: {len(files)} — amostrando {len(sampled)} videos.")

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
    orig_dir = dataset_root / "original"
    ruido_dir = dataset_root / "ruido"
    orig_dir.mkdir(parents=True, exist_ok=True)
    ruido_dir.mkdir(parents=True, exist_ok=True)

    # We'll save original frames and ruido images. Also build dataframes.
    rows_original = []
    # extract frames 1 fps for T seconds per video
    for _, row in df_meta.iterrows():
        p = Path(row["path"])
        name = row["name"]
        dur = float(row["duration_s"])
        # sample seconds from 0 to T-1 (uniform)
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
                "frame_number": i,
                "oficial_path": str(out),
                "source_video": str(p),
                "duration_seconds": float(dur)
            }
            for c in SCORE_COLUMNS:
                rec[c] = np.nan
            rows_original.append(rec)

    df_original = pd.DataFrame(rows_original)
    # Save metadata CSV
    meta_csv = dataset_root / f"{dataset_name}_metadata_videos.csv"
    df_meta.to_csv(meta_csv, index=False)
    orig_csv = dataset_root / f"{dataset_name}_original_frames.csv"
    df_original.to_csv(orig_csv, index=False)
    print(f"✔ Frames originais salvos: {len(df_original)} → {orig_csv}")

    # ----------------------
    # generate balanced noise assignments: pick 1/3 of frames for each noise type
    # We'll pick frames across the df_original: shuffle indices, split into 3 groups
    all_filenames = df_original["filename"].tolist()
    random.shuffle(all_filenames)
    n = len(all_filenames)
    # for reproducibility, you can set random.seed(...)
    group_size = n // 3
    groups = {
        "blur_sp": all_filenames[:group_size],
        "shapes_x": all_filenames[group_size:2*group_size],
        "gaussian": all_filenames[2*group_size:3*group_size]
    }
    # remaining files (if any) will be kept unmodified
    remaining = all_filenames[3*group_size:]

    # For each frame in groups, generate images for each intensity
    ruido_records = []
    for noise_type, fnames in groups.items():
        # create subfolder per noise type
        sub_noise_dir = ruido_dir / noise_type
        sub_noise_dir.mkdir(parents=True, exist_ok=True)
        for fname in fnames:
            orig_path = orig_dir / fname
            if not orig_path.exists():
                continue
            try:
                img = Image.open(orig_path).convert("RGB")
            except Exception:
                continue
            for intensity in intensities:
                # intensity 0.0 means copy original (we still save) - but we can skip saving if 0 and want to save space
                if intensity == 0.0:
                    # still save a copy under ruido folder to keep consistent naming
                    out_fname = f"{fname.replace('.png','')}_noise_{noise_type}_i{int(intensity*100)}.png"
                    out_path = sub_noise_dir / out_fname
                    if not out_path.exists():
                        shutil.copy(orig_path, out_path)
                else:
                    img_mod = apply_noise_by_type_and_intensity(img, noise_type, intensity)
                    out_fname = f"{fname.replace('.png','')}_noise_{noise_type}_i{int(intensity*100)}.png"
                    out_path = sub_noise_dir / out_fname
                    try:
                        img_mod.save(out_path)
                    except Exception as e:
                        print(f"Erro salvar mod {out_path}: {e}")
                        continue
                rec = {
                    "filename_orig": fname,
                    "filename_mod": out_fname,
                    "video_name": df_original[df_original["filename"]==fname]["video_name"].values[0],
                    "frame_number": df_original[df_original["filename"]==fname]["frame_number"].values[0],
                    "oficial_path": str(out_path),  # points to modified image (in ruido)
                    "filename_path_antes_ruido": str(orig_path),
                    "noise_type": noise_type,
                    "noise_intensity": float(intensity)
                }
                for c in SCORE_COLUMNS:
                    rec[c] = np.nan
                ruido_records.append(rec)

    # save ruido df
    df_ruido = pd.DataFrame(ruido_records)
    ruido_csv = dataset_root / f"{dataset_name}_ruido_frames.csv"
    df_ruido.to_csv(ruido_csv, index=False)
    print(f"✔ Ruído gerado: {len(df_ruido)} imagens → {ruido_csv}")

    return {
        "dataset_root": dataset_root,
        "metadata_csv": meta_csv,
        "original_frames_csv": orig_csv,
        "ruido_frames_csv": ruido_csv,
        "df_meta": df_meta,
        "df_original": df_original,
        "df_ruido": df_ruido,
        "T_seconds": T
    }

# ----------------------
# ANALISES / MÉTRICAS (após você rodar APDDv2 e preencher scores nas CSVs)
# ----------------------
def load_scored_dfs(dataset_root: Path, dataset_name: str):
    root = Path(dataset_root) / dataset_name
    orig_csv = root / f"{dataset_name}_original_frames_scored.csv"
    ruido_csv = root / f"{dataset_name}_ruido_frames_scored.csv"
    if not orig_csv.exists() or not ruido_csv.exists():
        raise FileNotFoundError("CSV avaliados não encontrados. Rode o avaliador APDDv2 e gere *_scored.csv")
    df_orig = pd.read_csv(orig_csv)
    df_ruido = pd.read_csv(ruido_csv)
    return df_orig, df_ruido

def compute_df_diff(df_orig: pd.DataFrame, df_ruido: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna df_diff com diferenças (orig - ruido) por imagem modificada.
    Assume que df_ruido contém colunas filename_mod and filename_orig or filename.
    We'll align by filename_orig.
    """
    # ensure we have filename_orig in df_ruido
    if "filename_orig" not in df_ruido.columns:
        if "filename" in df_ruido.columns:
            df_ruido = df_ruido.rename(columns={"filename": "filename_mod"})
            # try to infer filename_orig by splitting name before '_noise_'
            df_ruido["filename_orig"] = df_ruido["filename_mod"].apply(lambda x: x.split("_noise_")[0] + ".png")
        else:
            raise ValueError("df_ruido missing filename columns.")
    # merge
    df_merge = pd.merge(df_ruido, df_orig, left_on="filename_orig", right_on="filename", suffixes=("_ruido","_orig"))
    # compute diff columns
    for c in SCORE_COLUMNS:
        col_o = c + "_orig" if c+"_orig" in df_merge.columns else c + "_x"
        col_r = c + "_ruido" if c+"_ruido" in df_merge.columns else c + "_y"
        if col_o in df_merge.columns and col_r in df_merge.columns:
            df_merge[c+"_diff"] = df_merge[col_o] - df_merge[col_r]
        else:
            df_merge[c+"_diff"] = np.nan
    # total_change
    diff_cols = [c+"_diff" for c in SCORE_COLUMNS]
    df_merge["total_change"] = df_merge[diff_cols].abs().sum(axis=1)
    return df_merge

# Stability: ICC(1,1) implementation (one-way random)
def intraclass_corr_icc1(data: np.ndarray) -> float:
    """
    data: observations as rows = targets (e.g., frames), cols = raters/measurements (e.g., different sampling runs)
    returns ICC(1,1)
    """
    # Based on Shrout & Fleiss one-way random
    k = data.shape[1]
    n = data.shape[0]
    mean_per_target = np.nanmean(data, axis=1)
    mean_per_rater = np.nanmean(data, axis=0)
    grand_mean = np.nanmean(data)
    # sum squares
    SS_between = k * np.nansum((mean_per_target - grand_mean)**2)
    SS_within = np.nansum((data - mean_per_target[:, None])**2)
    MS_between = SS_between / (n-1)
    MS_within = SS_within / (n*(k-1))
    icc = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    return float(icc)

def test_h1_sampling_consistency(df_orig: pd.DataFrame, n_repeats: int = 5, samples_per_repeat: int = None):
    """
    Re-sample frames per video multiple times and compute ICC & Pearson between aggregated scores.
    This expects df_orig to contain score columns for each frame.
    samples_per_repeat: number of frames to sample per video (if None uses all)
    """
    # For each repeat, for each video compute mean score_total per video using a random subset of frames
    vids = df_orig["video_name"].unique()
    repeats_means = []
    for r in range(n_repeats):
        per_vid_means = []
        for v in vids:
            rows = df_orig[df_orig["video_name"]==v]
            if rows.empty:
                per_vid_means.append(np.nan)
                continue
            if samples_per_repeat and samples_per_repeat < len(rows):
                chosen = rows.sample(samples_per_repeat, replace=False)
            else:
                chosen = rows
            # compute mean across SCORE_COLUMNS (or total)
            chosen_scores = chosen[SCORE_COLUMNS]
            per_vid_means.append(chosen_scores.mean().mean())  # global mean
        repeats_means.append(per_vid_means)
    arr = np.array(repeats_means).T  # shape (n_items, n_repeats)
    icc_val = intraclass_corr_icc1(arr)
    # pairwise Pearson mean
    pearsons = []
    for i in range(arr.shape[1]):
        for j in range(i+1, arr.shape[1]):
            a = arr[:,i]; b = arr[:,j]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() > 1:
                r,_ = stats.pearsonr(a[mask], b[mask])
                pearsons.append(r)
    pearson_mean = float(np.nanmean(pearsons)) if pearsons else np.nan
    return {"icc": icc_val, "pearson_mean_between_repeats": pearson_mean}

# summary metrics: MTD, NIR, PSV, ARS
def compute_summary_metrics_from_diff(df_diff: pd.DataFrame) -> Dict:
    mtd = df_diff["total_change"].abs().mean()
    nir = (df_diff["total_change"] > 0).mean()
    psv = df_diff["total_change"].abs().max()
    # ARS: ratio of mean total scores
    # compute mean orig and ruido totals if columns exist
    orig_total_col = None
    ruido_total_col = None
    for c in SCORE_COLUMNS:
        if c in df_diff.columns and c+"_ruido" in df_diff.columns:
            pass
    # try to compute sum of orig vs ruido cols if present
    # fallback nan
    try:
        # infer columns
        orig_cols = [c + "_orig" for c in SCORE_COLUMNS if c + "_orig" in df_diff.columns]
        ruido_cols = [c + "_ruido" for c in SCORE_COLUMNS if c + "_ruido" in df_diff.columns]
        if orig_cols and ruido_cols:
            mean_orig = df_diff[orig_cols].mean(axis=1).mean()
            mean_ruido = df_diff[ruido_cols].mean(axis=1).mean()
            ars = mean_ruido / (mean_orig + 1e-12)
        else:
            ars = np.nan
    except Exception:
        ars = np.nan
    return {"MTD": float(mtd), "NIR": float(nir), "PSV": float(psv), "ARS": float(ars)}

# H4: analysis by noise type & intensity: group means + ANOVA + Cohen's d
def analyze_by_noise_and_intensity(df_diff: pd.DataFrame, output_dir: Path):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    # group by noise_type and intensity compute mean absolute diff per score
    df = df_diff.copy()
    # ensure noise_type and noise_intensity exist
    if "noise_type" not in df.columns or "noise_intensity" not in df.columns:
        print("df_diff missing noise_type/noise_intensity columns")
        return None
    result_rows = []
    for (nt, ni), g in df.groupby(["noise_type", "noise_intensity"]):
        row = {"noise_type": nt, "noise_intensity": ni, "n": len(g)}
        for sc in SCORE_COLUMNS:
            col = sc + "_diff" if sc + "_diff" in g.columns else None
            if col:
                row[sc + "_mean_abs_diff"] = g[col].abs().mean()
            else:
                row[sc + "_mean_abs_diff"] = np.nan
        result_rows.append(row)
    df_summary = pd.DataFrame(result_rows)
    df_summary.to_csv(out / "by_noise_intensity_summary.csv", index=False)

    # Example bar plot: mean total_change by noise_type (averaged across intensities)
    by_noise = df.groupby("noise_type")["total_change"].mean().reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(by_noise["noise_type"], by_noise["total_change"].abs())
    plt.title("Mean total_change by noise_type")
    plt.ylabel("mean |total_change|")
    plt.tight_layout()
    plt.savefig(out / "mean_total_change_by_noise.png", dpi=250)
    plt.close()

    # ANOVA example: total_change ~ noise_type
    groups = [g["total_change"].dropna().values for _, g in df.groupby("noise_type")]
    if all(len(x) > 1 for x in groups):
        try:
            f, p = stats.f_oneway(*groups)
            with open(out / "anova_noise_type.txt", "w") as ftxt:
                ftxt.write(f"ANOVA F={f}, p={p}\n")
        except Exception as e:
            with open(out / "anova_noise_type.txt", "w") as ftxt:
                ftxt.write("ANOVA error: " + str(e))

    # effect size (Cohen's d) pairwise between noise types (on total_change)
    types = df["noise_type"].unique().tolist()
    cohen_rows = []
    for i in range(len(types)):
        for j in range(i+1, len(types)):
            a = df[df["noise_type"]==types[i]]["total_change"].dropna().values
            b = df[df["noise_type"]==types[j]]["total_change"].dropna().values
            if len(a) > 1 and len(b) > 1:
                d = cohen_d_independent(a,b)
                cohen_rows.append({"g1": types[i], "g2": types[j], "cohen_d": float(d)})
    pd.DataFrame(cohen_rows).to_csv(out / "cohen_d_pairs_total_change.csv", index=False)
    print(f"✔ Análises por ruído salvas em: {out}")
    return df_summary

def cohen_d_independent(x, y):
    nx = len(x); ny = len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd

# H5: pattern mining simplified (signs +/0/-)
def pattern_mining_sign_tuples(df_diff: pd.DataFrame, topk: int = 20):
    """
    Transforma diferenças por score em sinais +/0/- e conta padrões (tuplas).
    Retorna DataFrame com top patterns and counts.
    """
    diffs = df_diff[[c + "_diff" for c in SCORE_COLUMNS if c + "_diff" in df_diff.columns]].copy()
    if diffs.empty:
        print("Nenhuma coluna de diff encontrada para pattern mining.")
        return pd.DataFrame()
    def sign_row(r):
        s = []
        for v in r:
            if pd.isna(v):
                s.append("0")
            elif v > 0.01:
                s.append("+")
            elif v < -0.01:
                s.append("-")
            else:
                s.append("0")
        return tuple(s)
    patterns = diffs.apply(lambda row: sign_row(row.values), axis=1)
    counts = patterns.value_counts().reset_index()
    counts.columns = ["pattern_tuple", "count"]
    # expand pattern into columns
    expanded = counts.copy()
    expanded[SCORE_COLUMNS[:len(expanded.iloc[0]['pattern_tuple'])]] = expanded["pattern_tuple"].apply(lambda t: pd.Series(t))
    return expanded.head(topk)

# ----------------------
# ENTRYPOINT
# ----------------------
def main_build_and_analyze():
    random.seed(42)
    res = build_dataset_from_downloads(dataset_name="auto_dataset", downloads_dir=DOWNLOADS_DIR, n_videos=SAMPLE_N_VIDEOS)
    dataset_root = res["dataset_root"]
    # At this point: you should run your APDDv2 scoring on:
    #   dataset_root / "<dataset>_original_frames.csv"  -> produces *_original_frames_scored.csv
    #   dataset_root / "<dataset>_ruido_frames.csv"     -> produces *_ruido_frames_scored.csv
    print("\nPróximo passo: rode o avaliador APDDv2 com os arquivos CSV gerados.")
    print("Depois, execute compute_all_metrics_from_scores(dataset_root, dataset_name) para analisar.\n")

def compute_all_metrics_from_scores(dataset_root: Path, dataset_name: str):
    """
    Carrega *_original_frames_scored.csv e *_ruido_frames_scored.csv e executa análises:
    - df_diff
    - summary metrics
    - H1 test (ICC)
    - H4 noise analysis
    - H5 pattern mining
    Salva resultados em resultados/<dataset_name>/
    """
    root = Path(dataset_root) / dataset_name
    out = Path(RESULTS_DIR) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(root / f"{dataset_name}_original_frames_scored.csv")
    df_ruido = pd.read_csv(root / f"{dataset_name}_ruido_frames_scored.csv")
    # compute diff
    df_diff = compute_df_diff(df_orig, df_ruido)
    df_diff.to_csv(out / "df_diff.csv", index=False)

    summary = compute_summary_metrics_from_diff(df_diff)
    pd.Series(summary).to_csv(out / "summary_metrics.csv")

    # H1 test
    try:
        h1 = test_h1_sampling_consistency(df_orig, n_repeats=5)
        pd.Series(h1).to_csv(out / "h1_icc_pearson.csv")
    except Exception as e:
        print("Erro H1:", e)

    # H4 analyze by noise
    try:
        analyze_by_noise_and_intensity(df_diff, out)
    except Exception as e:
        print("Erro H4:", e)

    # H5 pattern mining
    try:
        patterns = pattern_mining_sign_tuples(df_diff, topk=30)
        patterns.to_csv(out / "patterns_top.csv", index=False)
    except Exception as e:
        print("Erro H5:", e)

    print("✔ Todas análises concluídas. Resultados em:", out)
    return {
        "df_diff": df_diff,
        "summary": summary,
        "h1": h1 if 'h1' in locals() else None,
        "patterns": patterns if 'patterns' in locals() else None
    }

# ----------------------
# RUN
# ----------------------
if __name__ == "__main__":
    # build dataset from downloads
    main_build_and_analyze()
