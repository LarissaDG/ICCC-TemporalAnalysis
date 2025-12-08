"""
ruido_tuning_lab.py — Laboratório de ruídos local
Gera imagens e SEMPRE salva em PNG, independente de GUI.

Executar:
    python3 ruido_tuning_lab.py imagem.png
"""

import sys
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import imageio   # <-- ADICIONADO
import math

SHAPES_CACHE = None

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

# =====================================================
# LABORATÓRIO
# =====================================================
def run_lab(img_path):
    out_dir = Path("ruido_preview")
    out_dir.mkdir(exist_ok=True)

    img = Image.open(img_path).convert("RGB")

    r1, _, _ = add_blur_saltpepper(img)
    r2, _ = add_shapes_x(img)
    r3, _ = add_gaussian_noise(img)

    r1.save(out_dir / "blur_saltpepper.png")
    r2.save(out_dir / "shapes_x.png")
    r3.save(out_dir / "gaussian.png")
    img.save(out_dir / "original.png")

    # Painel
    titles = ["Original", "Blur+S&P", "Shapes/X", "Gaussian"]
    images = [img, r1, r2, r3]

    plt.figure(figsize=(18, 6))
    for i, (title, im) in enumerate(zip(titles, images)):
        plt.subplot(1, 5, i+1)
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    panel_path = out_dir / "ruido_preview_panel.png"
    plt.savefig(panel_path, dpi=150)
    plt.close()

    print("✔ Painel salvo em:", panel_path)

    print("Intensidades")
    generate_noise_sweep(img, n=8, out_dir="ruido_tuning_sweep")
    print("🎉 Laboratório concluído!\n")


# =====================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 ruido_tuning_lab.py imagem.png")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print("Arquivo não encontrado:", img_path)
        sys.exit(1)

    run_lab(img_path)
