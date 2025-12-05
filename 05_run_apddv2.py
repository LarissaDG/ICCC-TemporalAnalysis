# score_temporal_dataset.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Caminho do APDDv2 (ajuste se necessário)
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
    parser = argparse.ArgumentParser(description="APDDv2 Scoring de Frames")

    parser.add_argument(
        "--single",
        action="store_true",
        help="Se ativo, processa apenas a primeira linha do CSV"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root do dataset temporal (ex: dataset/auto_dataset)"
    )

    return parser.parse_args()


opt = init()


# =======================================================
#   Funções APDDv2
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
        print(f"✔ Modelo OK: {weight_path}")
        return model
    except Exception as e:
        print(f"❌ Falha ao carregar modelo {weight_path}: {e}")
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
        print(f"Erro no preprocess {image_path}: {e}")
        return {col: np.nan for col in models_dict.keys()}

    scores = {}
    for col, model in models_dict.items():
        if model is not None:
            try:
                pred = model(image_input)
                _, pred_val = get_score(opt, pred)

                if isinstance(pred_val, np.ndarray) and pred_val.size == 1:
                    pred_val = float(pred_val.item())

                # "Total aesthetic score" é reescalado ×10
                if col == "Total aesthetic score":
                    pred_val *= 10

                scores[col] = pred_val
            except Exception as e:
                print(f"Erro ao prever {col} para {image_path}: {e}")
                scores[col] = np.nan
        else:
            scores[col] = np.nan

    return scores


# =======================================================
#   PROCESSAR CSV ÚNICO
# =======================================================
def process_csv(input_csv, output_csv, dataset_root, models_dict, preprocess, opt, device, cols):

    print(f"\n📄 Lendo: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except Exception:
        df = pd.read_csv(input_csv, encoding="latin1")

    if opt.single:
        df = df.iloc[[0]].copy()

    for idx, row in df.iterrows():

        # Coluna padrão usada pelo seu pipeline
        img_path = row.get("oficial_path")

        if not isinstance(img_path, str):
            print(f"⚠️ Linha {idx} sem path")
            for c in cols:
                df.loc[idx, c] = np.nan
            continue

        full_path = Path(dataset_root) / img_path

        if not full_path.exists():
            print(f"❌ Imagem não encontrada: {full_path}")
            for c in cols:
                df.loc[idx, c] = np.nan
            continue

        scores = evaluate_image(full_path, models_dict, preprocess, opt, device)

        for c in cols:
            df.loc[idx, c] = scores.get(c, np.nan)

        print(f"✔ Frame {idx} scored")

    df.to_csv(output_csv, index=False)
    print(f"💾 Salvo: {output_csv}")


# =======================================================
#   MAIN (PROCESSA OS 3 CSVs DO PIPELINE TEMPORAL)
# =======================================================
def main():

    # 1) GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, preprocess = clip.load('ViT-B/16', device)

    # 2) Carregar modelos APDDv2
    model_paths = {
        "Total aesthetic score": "1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth",
        "Theme and logic": "2.Theme and logic_reg_weight--e5-train0.3792-test0.5953_best.pth",
        "Creativity": "3.Creativity_reg_weight--e5-train0.4212-test0.7122_best.pth",
        "Layout and composition": "4.Layout and composition_reg_weight--e6-train0.2783-test0.6342_best.pth",
        "Space and perspective": "5.Space and perspective_reg_weight--e7-train0.2168-test0.5998_best.pth",
        "The sense of order": "Model_6.pth",
        "Light and shadow": "7.Light and shadow_reg_weight--e7-train0.1937-test0.6518_best.pth",
        "Color": "8.Color_reg_weight--e5-train0.2905-test0.5871_best.pth",
        "Details and texture": "9.Details and texture_reg_weight--e4-train0.4385-test0.7034_best.pth",
        "The overall": "10.The overall_reg_weight--e3-train0.5131-test0.6343_best.pth",
        "Mood": "11.Mood_reg_weight--e7-train0.3108-test0.7097_best.pth"
    }

    base_w = "/home_cerberus/disk3/larissa.gomide/APDDv2/modle_weights/"
    models_dict = {k: load_model(base_w + v, device) for k, v in model_paths.items()}

    cols = list(models_dict.keys())

    # 3) CSVs do dataset temporal
    root = Path(opt.dataset_root)

    csvs = {
        "orig": root / "auto_dataset_original_frames_scored.csv",
        "h1": root / "orquestrador_experimentos" / "h1_ruidos_mapping_scored.csv",
        "h2": root / "orquestrador_experimentos" / "h2_intensity_mapping_scored.csv"
    }

    inputs = {
        "orig": root / "auto_dataset_original_frames.csv",
        "h1": root / "orquestrador_experimentos" / "h1_ruidos_mapping.csv",
        "h2": root / "orquestrador_experimentos" / "h2_intensity_mapping.csv"
    }

    # 4) Processar os 3 CSVs
    for key in ["orig", "h1", "h2"]:
        print("\n===================================================")
        print(f"PROCESSANDO → {key.upper()}")
        print("===================================================")

        process_csv(
            input_csv=inputs[key],
            output_csv=csvs[key],
            dataset_root=root,
            models_dict=models_dict,
            preprocess=preprocess,
            opt=opt,
            device=device,
            cols=cols
        )

    print("\n🎉 FINALIZADO — Todos os frames foram avaliados APDDv2!\n")


if __name__ == "__main__":
    main()
