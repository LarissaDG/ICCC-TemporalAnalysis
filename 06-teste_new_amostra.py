# 01_generate_samples_uniform_second.py
# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw, ImageSequence
import matplotlib.pyplot as plt

# ================================================================
# GLOBALS
# ================================================================
SCORE_COLUMNS = [
    "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color", "The sense of order",
    "Details and texture", "The overall", "Mood"
]

# ================================================================
# UTILITÁRIAS
# ================================================================
def carregar_paths_simples(txt_path: str = "paths_baixados.txt"):
    txt = Path(txt_path)
    if not txt.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {txt}")
    linhas = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]

    def extrair_path(l):
        return Path(l.split(":", 1)[1].strip()) if ":" in l else Path(l)

    if len(linhas) < 2:
        raise ValueError("O arquivo deve conter ao menos duas linhas com paths.")
    return extrair_path(linhas[0]), extrair_path(linhas[1])


def ultimo_diretorio(path_obj: Path) -> str:
    p = Path(path_obj)
    return p.parts[-1] if p.parts else ""


# ================================================================
# 1) FUNÇÃO AUXILIAR - CALCULA TEMPO TOTAL DO GIF
# ================================================================
def calcular_duracao_gif(gif_path: Path) -> float:
    """
    Retorna a duração do GIF em segundos.
    """
    try:
        with Image.open(gif_path) as im:
            frames = list(ImageSequence.Iterator(im))
            durations = [f.info.get('duration', 40) for f in frames]  # default 40ms se não existir
            total_ms = sum(durations)
            return total_ms / 1000.0
    except Exception as e:
        print(f"⚠ Erro ao calcular duração do GIF {gif_path}: {e}")
        return np.nan


# ================================================================
# 2) EXTRAÇÃO DE FRAMES COM AMOSTRAGEM UNIFORME 1 FRAME/SEGUNDO
# ================================================================
def process_gifs_to_frames(df_name: str, folder_path: Path, output_original_folder: Path) -> pd.DataFrame:
    folder_path = Path(folder_path)
    output_original_folder = Path(output_original_folder)
    if not folder_path.exists():
        raise ValueError(f"Pasta de GIFs não encontrada: {folder_path}")

    output_original_folder.mkdir(parents=True, exist_ok=True)
    gif_files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".gif"]
    print("GIFs encontrados em", folder_path, ":", len(gif_files))

    rows = []
    for gif_file in gif_files:
        try:
            with Image.open(gif_file) as gif:
                frames = list(ImageSequence.Iterator(gif))
                # calcular duração total
                durations = [f.info.get('duration', 40) for f in frames]
                total_ms = sum(durations)
                total_seconds = total_ms / 1000.0
                # amostragem uniforme: 1 frame por segundo
                cumulative_ms = np.cumsum([0]+durations[:-1])
                indices = []
                t = 0
                while t < total_seconds:
                    # encontrar frame mais próximo do tempo t
                    idx = np.searchsorted(cumulative_ms/1000.0, t)
                    idx = min(idx, len(frames)-1)
                    indices.append(idx)
                    t += 1.0
        except Exception as e:
            print(f"⚠ Erro ao abrir GIF {gif_file}: {e}")
            continue

        base_name = gif_file.stem
        for i, frame_idx in enumerate(sorted(set(indices)), start=1):
            frame = frames[frame_idx].convert("RGB")
            fname = f"{base_name}_frame{i}.png"
            out_path = output_original_folder / fname
            try:
                frame.save(out_path)
            except Exception as e:
                print(f"⚠ Erro ao salvar {out_path}: {e}")
                continue

            row = {
                "filename": fname,
                "filename_path_antes_ruido": str(out_path),
                "gif_name": base_name,
                "frame_number": i,
                "gif_duration_seconds": total_seconds,
                "oficial_path": str(out_path)
            }
            for c in SCORE_COLUMNS:
                row[c] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    dataset_root = output_original_folder.parent
    dataset_name = ultimo_diretorio(dataset_root)
    csv_path = dataset_root / f"{dataset_name}_original_dataset.csv"
    cols_order = ["filename", "gif_name", "frame_number", "oficial_path", "filename_path_antes_ruido", "gif_duration_seconds"] + SCORE_COLUMNS
    df = df.reindex(columns=[c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order])
    df.to_csv(csv_path, index=False)
    print(f"✔ CSV salvo em: {csv_path}")
    return df

# ================================================================
# 2) FUNÇÕES DE RUÍDO (operam sobre PIL.Image)
# ================================================================
def add_blur_saltpepper(frame, blur_radius=3, sp_amount=0.02):
    arr = np.array(frame.filter(ImageFilter.GaussianBlur(blur_radius))).astype(np.int16)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, c = arr.shape
    n = max(1, int(h * w * sp_amount))
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 255
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs, :] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_shapes_x(frame):
    frame = frame.convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")
    w, h = frame.size
    width = max(3, int(min(w, h) * 0.02))
    draw.line((0, 0, w, h), fill=(255, 0, 0, 200), width=width)
    draw.line((0, h, w, 0), fill=(255, 0, 0, 200), width=width)
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.5:
            x0 = random.randint(0, frame.width - 1)
            y0 = random.randint(0, frame.height - 1)
            x1 = min(frame.width, x0 + random.randint(20, int(frame.width * 0.25)))
            y1 = min(frame.height, y0 + random.randint(20, int(frame.height * 0.25)))
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255), 150)
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            cx = random.randint(0, frame.width - 1)
            cy = random.randint(0, frame.height - 1)
            r = random.randint(10, int(min(frame.size) * 0.12))
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255), 150)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    return frame.convert("RGB")


def add_gaussian_noise(frame, sigma=25):
    arr = np.array(frame.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ================================================================
# 3) PROCESSAR RUÍDOS (ruido/ contém APENAS imagens modificadas)
# ================================================================
def processar_ruidos(df_name, df_original, frames_original_folder, ruido_folder, percentage=1.0):
    """
    - df_original: DataFrame gerado por process_gifs_to_frames (contém todas as linhas)
    - frames_original_folder: pasta original onde os frames estão (ex: dataset/gifs/original)
    - ruido_folder: pasta do dataset onde guardaremos frames ruidosos (ex: dataset/gifs/ruido)
    - estratégia: selecionar 1 frame por GIF (ou conforme percentage), gerar imagem ruidosa e salvar em ruido_folder
    - o ruido DF é o mesmo df_original atualizado com colunas do ruído e oficial_path apontando para a imagem ruidosa quando existir
    """
    frames_original_folder = Path(frames_original_folder)
    ruido_folder = Path(ruido_folder)
    ruido_folder.mkdir(parents=True, exist_ok=True)

    # Inicializar colunas de texto como None (object) para evitar warnings
    df = df_original.copy()
    df["path_modified"] = None
    df["path_noise_folder"] = None
    df["noise_type"] = None
    df["oficial_path"] = None  # será preenchida com path_modified se houver, senão com filename_path_antes_ruido

    # Garantir que filename_path_antes_ruido existe e aponta para original
    if "filename_path_antes_ruido" not in df.columns:
        # construir a partir de original folder
        df["filename_path_antes_ruido"] = df["filename"].apply(lambda x: str(frames_original_folder / x))

    # extrair gif_num para seleção
    df['gif_num'] = df['filename'].str.extract(r'(\d+)_frame\d+')[0].astype(int)

    gif_ids = sorted(df['gif_num'].unique().tolist())
    random.shuffle(gif_ids)
    num_to_modify = max(1, int(percentage * len(gif_ids))) if len(gif_ids) > 0 else 0

    # dividir em 3 grupos (pode ser menor se num_to_modify < 3)
    group_size = max(1, num_to_modify // 3) if num_to_modify > 0 else 0
    group1 = gif_ids[:group_size]
    group2 = gif_ids[group_size:2*group_size]
    group3 = gif_ids[2*group_size:3*group_size]

    def pick_one_frame_for_group(group):
        picks = []
        for gid in group:
            candidates = df[df['gif_num'] == gid]['filename'].tolist()
            if candidates:
                picks.append(random.choice(candidates))
        return picks

    picks1 = pick_one_frame_for_group(group1)
    picks2 = pick_one_frame_for_group(group2)
    picks3 = pick_one_frame_for_group(group3)

    def apply_and_save(picks, noise_type):
        for fname in picks:
            orig_path = frames_original_folder / fname
            if not orig_path.exists():
                # pula se original inexistente
                continue
            try:
                with Image.open(orig_path) as im:
                    im_rgb = im.convert("RGB")
                    if noise_type == "blur_sp":
                        im_mod = add_blur_saltpepper(im_rgb)
                    elif noise_type == "shapes_x":
                        im_mod = add_shapes_x(im_rgb)
                    elif noise_type == "gaussian":
                        im_mod = add_gaussian_noise(im_rgb)
                    else:
                        continue

                    dst_path = ruido_folder / fname
                    im_mod.save(dst_path)
                    # atualizar df: path_modified, path_noise_folder, noise_type, oficial_path -> apontar para mod
                    mask = df["filename"] == fname
                    df.loc[mask, "path_modified"] = str(dst_path)
                    df.loc[mask, "path_noise_folder"] = str(ruido_folder)
                    df.loc[mask, "noise_type"] = noise_type
                    df.loc[mask, "oficial_path"] = str(dst_path)
            except Exception as e:
                print(f"⚠ Erro ao aplicar ruído em {orig_path}: {e}")
                continue

    # Aplicar ruídos seletivamente
    apply_and_save(picks1, "blur_sp")
    apply_and_save(picks2, "shapes_x")
    apply_and_save(picks3, "gaussian")

    # Para linhas que não foram modificadas, oficial_path deve apontar para a original
    df.loc[df["oficial_path"].isna(), "oficial_path"] = df.loc[df["oficial_path"].isna(), "filename_path_antes_ruido"]

    # Salvar CSV do ruido NO NÍVEL DO DATASET
    dataset_root = ruido_folder.parent
    dataset_name = ultimo_diretorio(dataset_root)
    csv_path = dataset_root / f"{dataset_name}_ruido_dataset.csv"

    # Garantir que todas as colunas originais permaneçam; salvar DF completo
    df.to_csv(csv_path, index=False)
    print(f"✔ CSV salvo em: {csv_path}")
    print(f"✔ {sum(df['path_modified'].notna())} frames modificados e salvos em: {ruido_folder}")

    return df


# ================================================================
# VISUALIZAÇÃO (simples) — mantém como antes
# ================================================================
def visualizar_exemplos_ruido(df):
    examples = []
    for noise_type in ["blur_sp", "shapes_x", "gaussian"]:
        df_noise = df[df['noise_type'] == noise_type]
        if not df_noise.empty:
            row = df_noise.iloc[0]
            orig = Path(row['filename_path_antes_ruido'])
            mod = Path(row['path_modified']) if pd.notna(row['path_modified']) and row['path_modified'] else None
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
        axes = [axes]

    for i, (noise, orig, mod) in enumerate(examples):
        ax_orig = axes[i][0] if n > 1 else axes[0]
        ax_mod = axes[i][1] if n > 1 else axes[1]
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


def salvar_visualizacao(df: pd.DataFrame, nome_arquivo: str = "visualizar_exemplos_ruido.png") -> Path:
    fig = visualizar_exemplos_ruido(df)
    try:
        out_path = Path(__file__).resolve().parent / nome_arquivo
    except Exception:
        out_path = Path.cwd() / nome_arquivo
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Visualização salva em: {out_path}")
    return out_path


# ================================================================
# EXECUTAR PIPELINE (ajustado para nova estrutura)
# ================================================================
def executar_pipeline(folder_path, destino_base, df_name="dataset_crawler"):
    """
    folder_path: pasta com GIFs (input)
    destino_base: pasta do dataset (ex: dataset/gifs)
    """
    folder_path = Path(folder_path)
    destino_base = Path(destino_base)
    destino_original = destino_base / "original"
    destino_ruido = destino_base / "ruido"

    destino_original.mkdir(parents=True, exist_ok=True)
    destino_ruido.mkdir(parents=True, exist_ok=True)

    print(f"\n=== PROCESSANDO DATASET: {destino_base.name} ===")
    print("  original:", destino_original)
    print("  ruido   :", destino_ruido)

    df_original = process_gifs_to_frames(df_name, folder_path, destino_original)
    df_ruido = processar_ruidos(df_name, df_original.copy(), destino_original, destino_ruido)

    salvar_visualizacao(df_ruido)

    return df_original, df_ruido


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("\n=== CARREGANDO PATHS ===")
    try:
        p1, p2 = carregar_paths_simples()
    except Exception as e:
        print("❌ Erro ao carregar paths:", e)
        raise

    print("Path 1 carregado:", p1)
    print("Path 2 carregado:", p2)

    base_dir = Path("dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    print("Base dir:", base_dir)

    destino_1 = base_dir / ultimo_diretorio(p1)
    destino_2 = base_dir / ultimo_diretorio(p2)
    destino_1.mkdir(parents=True, exist_ok=True)
    destino_2.mkdir(parents=True, exist_ok=True)

    df1_original, df1_ruido = executar_pipeline(p1, destino_1, df_name=str(destino_1.name))
    df2_original, df2_ruido = executar_pipeline(p2, destino_2, df_name=str(destino_2.name))

    print("\n✔ PIPELINES COMPLETOS!")
