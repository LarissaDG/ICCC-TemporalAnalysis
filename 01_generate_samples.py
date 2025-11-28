# 01_generate_samples_fixed.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw, ImageSequence
import matplotlib.pyplot as plt

# ================================================================
# Config / Globals
# ================================================================
SCORE_COLUMNS = [
    "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color", "The sense of order",
    "Details and texture", "The overall", "Mood"
]


# ================================================================
# Funções de paths
# ================================================================
def carregar_paths_simples(txt_path: str = "paths_baixados.txt"):
    """
    Lê as duas linhas do arquivo e retorna path_1 e path_2 diretamente.
    Formato esperado no TXT:
        GIFs: caminho1
        Story: caminho2
    Retorna Path, Path
    """
    txt = Path(txt_path)
    if not txt.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {txt}")

    linhas = [linha.strip() for linha in txt.read_text(encoding="utf-8").splitlines() if linha.strip()]

    if len(linhas) < 2:
        raise ValueError("O arquivo deve conter pelo menos duas linhas com paths.")

    # Extrair somente o que vem depois do ':', ignorando label se houver
    def extrair_path(l):
        if ":" in l:
            p = l.split(":", 1)[1].strip()
        else:
            p = l.strip()
        return Path(p)

    return extrair_path(linhas[0]), extrair_path(linhas[1])


def ultimo_diretorio(path_obj: Path) -> str:
    """
    Retorna o último segmento do path (nome da pasta/arquivo)
    """
    path = Path(path_obj)
    if not path.parts:
        return ""
    return str(path.parts[-1])


# ================================================================
# Processamento de GIFs → Frames
# ================================================================
def process_gifs_to_frames(df_name: str, folder_path: Path, output_path: Path, max_frames: int = 24) -> pd.DataFrame:
    """
    Extrai frames de todos os GIFs em folder_path (até max_frames cada),
    salva em output_path/frames e retorna um DataFrame com metadados.
    """
    folder_path = Path(folder_path)
    output_path = Path(output_path)
    if not folder_path.exists():
        raise ValueError(f"Pasta não encontrada: {folder_path}")

    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # listar apenas arquivos .gif
    gif_files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".gif"]

    print("GIFs encontrados em", folder_path, ":", len(gif_files))

    data = []
    for gif_file in gif_files:
        try:
            with Image.open(gif_file) as gif:
                # Itera com ImageSequence (mais robusto)
                frames = []
                for i, frame in enumerate(ImageSequence.Iterator(gif)):
                    if i >= max_frames:
                        break
                    frames.append(frame.convert("RGB"))
        except Exception as e:
            print(f"⚠ Erro ao abrir {gif_file}: {e}")
            continue

        base_name = gif_file.stem
        for i, frame in enumerate(frames, start=1):
            png_name = f"{base_name}_frame{i}.png"
            png_path = frames_dir / png_name
            # salvar frame
            try:
                frame.save(png_path)
            except Exception as e:
                print(f"⚠ Erro ao salvar frame {png_path}: {e}")
                continue

            row = {
                "filename": png_name,
                "filename_path_antes_ruido": str(png_path),
            }
            for col in SCORE_COLUMNS:
                row[col] = np.nan
            data.append(row)

    df = pd.DataFrame(data)
    if not df.empty:
        # tenta extrair um número do nome do GIF se existir; se falhar, coloca -1
        extracted = df["filename"].str.extract(r'(\d+)_frame\d+')
        if extracted is not None and extracted.shape[1] > 0:
            df["gif_num"] = pd.to_numeric(extracted[0], errors="coerce").fillna(-1).astype(int)
        else:
            df["gif_num"] = -1
    else:
        df["gif_num"] = pd.Series(dtype=int)

    csv_name = output_path / f"gif_frames_scores_original_{df_name}.csv"
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_name, index=False)
    print(f"CSV salvo em: {csv_name}")

    return df


# ================================================================
# Funções de ruído (opera sobre PIL.Image)
# ================================================================
def add_blur_saltpepper(frame: Image.Image, blur_radius: int = 3, sp_amount: float = 0.02) -> Image.Image:
    im_blur = frame.copy().filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(im_blur).astype(np.int16)
    if arr.ndim < 3:
        arr = np.stack([arr]*3, axis=-1)
    h, w, c = arr.shape
    num_pixels = max(1, int(h * w * sp_amount))

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    arr[ys, xs, :] = 255

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    arr[ys, xs, :] = 0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_shapes_x(frame: Image.Image) -> Image.Image:
    frame = frame.copy().convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")

    width = max(3, int(min(frame.size) * 0.02))
    draw.line((0, 0, frame.width, frame.height), fill=(255, 0, 0, 200), width=width)
    draw.line((0, frame.height, frame.width, 0), fill=(255, 0, 0, 200), width=width)

    for _ in range(random.randint(1, 3)):
        if random.random() < 0.5:
            x0 = random.randint(0, frame.width - 1)
            y0 = random.randint(0, frame.height - 1)
            x1 = min(frame.width, x0 + random.randint(20, max(20, int(frame.width * 0.25))))
            y1 = min(frame.height, y0 + random.randint(20, max(20, int(frame.height * 0.25))))
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255), 150)
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            cx = random.randint(0, frame.width - 1)
            cy = random.randint(0, frame.height - 1)
            r = random.randint(10, max(10, int(min(frame.size) * 0.12)))
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255), 150)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

    return frame.convert("RGB")


def add_gaussian_noise(frame: Image.Image, sigma: float = 25) -> Image.Image:
    arr = np.array(frame.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ================================================================
# Pipeline de ruído
# ================================================================
def processar_ruidos(df_name, df, frames_folder, ruido_folder, percentage=1.0):

    # Criar pasta ruido/frames
    frames_ruido = Path(ruido_folder) / "frames"
    frames_ruido.mkdir(parents=True, exist_ok=True)

    # Criar colunas do dataframe
    df["path_modified"] = np.nan
    df["path_noise_folder"] = np.nan
    df["noise_type"] = np.nan

    # Listar frames originais
    original_files = [
        f for f in os.listdir(frames_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Copiar originais para ruido/frames
    for fname in original_files:
        src = os.path.join(frames_folder, fname)
        dst = frames_ruido / fname
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Identificar gif_num
    df['gif_num'] = df['filename'].str.extract(r'(\d+)_frame\d+')[0].astype(int)

    # ------------------------------
    # Função interna: aplicar ruído
    # ------------------------------
    def apply_noise(file_list, noise_type):
        for img_path in file_list:
            img_name = os.path.basename(img_path)
            im = Image.open(img_path)

            if noise_type == "blur_sp":
                im_mod = add_blur_saltpepper(im)
            elif noise_type == "shapes_x":
                im_mod = add_shapes_x(im)
            elif noise_type == "gaussian":
                im_mod = add_gaussian_noise(im)
            else:
                raise ValueError("Tipo de ruído desconhecido!")

            im_mod.save(img_path)

            mask = df["filename"] == img_name
            df.loc[mask, "path_modified"] = img_path
            df.loc[mask, "path_noise_folder"] = frames_ruido
            df.loc[mask, "noise_type"] = noise_type

    # Selecionar GIFs que terão ruído
    gif_ids = df['gif_num'].unique()
    num_gifs_modificados = int(percentage * len(gif_ids))
    random.shuffle(gif_ids)

    group_size = max(1, num_gifs_modificados // 3)

    group1 = gif_ids[:group_size]
    group2 = gif_ids[group_size:2*group_size]
    group3 = gif_ids[2*group_size:3*group_size]

    def select_one_frame(gif_group):
        frames_sel = []
        for gif in gif_group:
            frames = df[df['gif_num'] == gif]['filename'].tolist()
            if frames:
                chosen = random.choice(frames)
                frames_sel.append(str(frames_ruido / chosen))
        return frames_sel

    frames_group1 = select_one_frame(group1)
    frames_group2 = select_one_frame(group2)
    frames_group3 = select_one_frame(group3)

    # Aplicar ruído
    apply_noise(frames_group1, "blur_sp")
    apply_noise(frames_group2, "shapes_x")
    apply_noise(frames_group3, "gaussian")

    # ------------------------------
    # Salvar CSV NO NÍVEL CERTO
    # (mesmo nível das pastas 'original' e 'ruido')
    # ------------------------------
    dataset_root = Path(ruido_folder).parent
    csv_path = dataset_root / f"gif_frames_scores_ruido_{df_name}.csv"

    df.to_csv(csv_path, index=False)

    print(f"✔ CSV salvo em: {csv_path}")
    print(f"✔ {num_gifs_modificados} GIFs modificados.")

    return df
# ================================================================
# Visualização (RETORNA figura Matplotlib)
# ================================================================
def visualizar_exemplos_ruido(df: pd.DataFrame) -> plt.Figure:
    """
    Retorna uma figura com pares (original vs modificado) para os 3 tipos de ruído,
    contendo até 3 linhas (uma por tipo de ruído).
    """
    examples = []
    for noise_type in ["blur_sp", "shapes_x", "gaussian"]:
        df_noise = df[df['noise_type'] == noise_type]
        if not df_noise.empty:
            row = df_noise.iloc[0]
            orig = Path(row['filename_path_antes_ruido'])
            mod = Path(row['path_modified'])
            if orig.exists() and mod.exists():
                examples.append((noise_type, orig, mod))

    if not examples:
        # cria figura vazia com aviso
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Nenhum exemplo de ruído disponível", ha="center", va="center")
        plt.axis("off")
        return fig

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]  # normaliza para iteração

    for i, (noise, orig, mod) in enumerate(examples):
        ax_orig = axes[i][0] if n > 1 else axes[0]
        ax_mod = axes[i][1] if n > 1 else axes[1]
        ax_orig.imshow(Image.open(orig))
        ax_orig.set_title(f"Original ({noise})")
        ax_orig.axis("off")
        ax_mod.imshow(Image.open(mod))
        ax_mod.set_title(f"Modificado ({noise})")
        ax_mod.axis("off")

    plt.tight_layout()
    return fig


def salvar_visualizacao(df: pd.DataFrame, nome_arquivo: str = "visualizar_exemplos_ruido.png") -> Path:
    """
    Salva a figura retornada por visualizar_exemplos_ruido() ao lado do script.
    """
    fig = visualizar_exemplos_ruido(df)
    try:
        out_path = Path(__file__).resolve().parent / nome_arquivo
    except Exception:
        # fallback (ex.: ambiente interativo onde __file__ não existe)
        out_path = Path.cwd() / nome_arquivo

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Visualização salva em: {out_path}")
    return out_path


# ================================================================
# Executor do pipeline para uma pasta
# ================================================================
def executar_pipeline(folder_path: Path, destino_base: Path, df_name: str = "dataset_crawler"):
    print(f"\n=== PROCESSANDO: {folder_path} ===")
    destino_original = destino_base / "original"
    destino_ruido = destino_base / "ruido"
    destino_original.mkdir(parents=True, exist_ok=True)
    destino_ruido.mkdir(parents=True, exist_ok=True)

    print("✔ Pastas criadas:")
    print("  -", destino_original)
    print("  -", destino_ruido)

    # 1) GIFs → frames
    df_original = process_gifs_to_frames(df_name=df_name, folder_path=folder_path, output_path=destino_original, max_frames=24)

    # 2) aplicar ruídos nas cópias
    frames_folder = destino_original / "frames"
    df_ruido = processar_ruidos(df_name=df_name, df=df_original, frames_folder=frames_folder, ruido_folder=destino_ruido, percentage=1.0)

    # 3) salvar visualização ao lado do script
    salvar_visualizacao(df_ruido)

    return df_original, df_ruido


# ================================================================
# MAIN — Arquivo único pronto para execução
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

    # Base onde os dados serão salvos (crie dataset ou altere para uma pasta local)
    base_dir = Path("dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    print("Base dir:", base_dir)

    def criar_destino_local(path_obj: Path) -> Path:
        destino = base_dir / ultimo_diretorio(path_obj)
        destino.mkdir(parents=True, exist_ok=True)
        print("✔ Criado destino:", destino)
        return destino

    destino_1 = criar_destino_local(p1)
    destino_2 = criar_destino_local(p2)

    print("\n=== EXECUTANDO PIPELINE EM p1 ===")
    df1_original, df1_ruido = executar_pipeline(p1, destino_1)

    print("\n=== EXECUTANDO PIPELINE EM p2 ===")
    df2_original, df2_ruido = executar_pipeline(p2, destino_2)

    print("\n✔ Tudo finalizado com sucesso! 🎉")
