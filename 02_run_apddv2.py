import os
import sys
from pathlib import Path
sys.path.append('/home_cerberus/disk3/larissa.gomide/APDDv2/')

import torch
import numpy as np
import warnings
import models.clip as clip
warnings.filterwarnings("ignore")
from models.aesclip import AesCLIP_reg
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import pandas as pd


# =======================================================
#   CLI
# =======================================================
def init():
    parser = argparse.ArgumentParser(description="PyTorch Aesthetic Scoring")

    parser.add_argument(
        "--single",
        action="store_true",
        help="Se ativo, processa apenas a primeira linha do CSV"
    )

    args = parser.parse_args()
    return args


opt = init()


# =======================================================
#   Funções de apoio
# =======================================================
def get_score(opt, y_pred):
    score_np = y_pred.data.cpu().numpy()
    return y_pred, score_np


def load_model(weight_path, device):
    try:
        base_weight = "/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/0.AesCLIP_weight--e11-train2.4314-test4.0253_best.pth"
        model = AesCLIP_reg(clip_name='ViT-B/16', weight=base_weight)
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        print(f"Modelo carregado com sucesso: {weight_path}")
        return model
    except Exception as e:
        print(f"Falha ao carregar modelo de {weight_path}: {e}")
        return None


def evaluate_image(image_path, models_dict, preprocess, opt, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao abrir imagem {image_path}: {e}")
        return {col: np.nan for col in models_dict.keys()}

    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Erro ao preprocessar imagem {image_path}: {e}")
        return {col: np.nan for col in models_dict.keys()}

    scores = {}
    for col, model in models_dict.items():
        if model is not None:
            try:
                pred = model(image_input)
                _, pred_val = get_score(opt, pred)

                if isinstance(pred_val, np.ndarray) and pred_val.size == 1:
                    pred_val = pred_val.item()

                if col == "Total aesthetic score":
                    pred_val = pred_val * 10

                scores[col] = pred_val
            except Exception as e:
                print(f"Erro ao prever {col} para imagem {image_path}: {e}")
                scores[col] = np.nan
        else:
            scores[col] = np.nan

    return scores


# =======================================================
#   PROCESSAMENTO DO CSV
# =======================================================
def process_csv(base_dir, input_csv, output_csv, models_dict, preprocess, opt, device, cols_to_compare):

    print(f"\n📄 Lendo CSV: {input_csv}")

    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except Exception:
        df = pd.read_csv(input_csv, encoding="latin1")

    if opt.single:
        print("⚠️ Rodando SOMENTE a primeira linha (--single)")
        df = df.iloc[[0]].copy()

    for idx, row in df.iterrows():
        image_path = os.path.join("dataset", row.get("oficial_path"))

        if not image_path or not os.path.exists(image_path):
            print(f"⚠️ Imagem NÃO encontrada na linha {idx}: {image_path}")
            for col in cols_to_compare:
                df.loc[idx, col] = np.nan
            continue

        scores = evaluate_image(image_path, models_dict, preprocess, opt, device)
        for col in cols_to_compare:
            df.loc[idx, col] = scores.get(col, np.nan)

        print(f"✔ Linha {idx} processada")

    df.to_csv(output_csv, index=False)
    print(f"💾 Arquivo salvo em: {output_csv}")


# =======================================================
#   MAIN
# =======================================================
def main():

    # ---------------------------------------------------
    #  GPU
    # ---------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, preprocess = clip.load('ViT-B/16', device)

    # ---------------------------------------------------
    #  Carregar modelos
    # ---------------------------------------------------
    score_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth", device)
    theme_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/2.Theme and logic_reg_weight--e5-train0.3792-test0.5953_best.pth", device)
    creativity_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/3.Creativity_reg_weight--e5-train0.4212-test0.7122_best.pth", device)
    layout_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/4.Layout and composition_reg_weight--e6-train0.2783-test0.6342_best.pth", device)
    space_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/5.Space and perspective_reg_weight--e7-train0.2168-test0.5998_best.pth", device)
    sense_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/Model_6.pth", device)
    light_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/7.Light and shadow_reg_weight--e7-train0.1937-test0.6518_best.pth", device)
    color_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/8.Color_reg_weight--e5-train0.2905-test0.5871_best.pth", device)
    details_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/9.Details and texture_reg_weight--e4-train0.4385-test0.7034_best.pth", device)
    overall_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/10.The overall_reg_weight--e3-train0.5131-test0.6343_best.pth", device)
    mood_model = load_model("/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/11.Mood_reg_weight--e7-train0.3108-test0.7097_best.pth", device)

    cols_to_compare = [
        "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
        "Space and perspective", "The sense of order", "Light and shadow", "Color",
        "Details and texture", "The overall", "Mood"
    ]

    models_dict = {
        "Total aesthetic score": score_model,
        "Theme and logic": theme_model,
        "Creativity": creativity_model,
        "Layout and composition": layout_model,
        "Space and perspective": space_model,
        "The sense of order": sense_model,
        "Light and shadow": light_model,
        "Color": color_model,
        "Details and texture": details_model,
        "The overall": overall_model,
        "Mood": mood_model
    }

    # ---------------------------------------------------
    #  Diretório base com as duas pastas
    # ---------------------------------------------------
    base_dir = Path("/sonic_home/larissa.gomide/dataset")

    folders = ["gif_storyboarding", "gifs"]

    for folder in folders:
        folder_path = base_dir / "dataset" / folder

        print(f"\n📁 Processando pasta: {folder_path}")

        # ============================================================
        #   FILTRO: só pegar *_dataset.csv e ignorar tudo com _scored
        # ============================================================
        csv_files = []
        for f in folder_path.glob("*_dataset*.csv"):
            name = f.name

            # Ignorar CSVs com mais de um _scored
            if name.count("_scored") > 1:
                continue

            # Ignorar arquivos que JÁ são scored
            if name.endswith("_scored.csv"):
                continue

            # Aceitar apenas o original: *_dataset.csv
            if name.endswith("_dataset.csv"):
                csv_files.append(f)

        csv_files = sorted(csv_files)

        if not csv_files:
            print(f"⚠️ Nenhum CSV ORIGINAL encontrado em: {folder_path}")
            continue

        for csv_in in csv_files:
            csv_out = folder_path / (csv_in.stem + "_scored.csv")

            print(f"\n➡️ Entrada: {csv_in}")
            print(f"⬅️ Saída : {csv_out}")

            process_csv(
                base_dir,
                str(csv_in),
                str(csv_out),
                models_dict,
                preprocess,
                opt,
                device,
                cols_to_compare
            )

    print("\n✔ TODOS OS CSVs foram processados com sucesso!")


if __name__ == "__main__":
    main()