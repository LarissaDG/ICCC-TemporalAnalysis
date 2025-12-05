# -*- coding: utf-8 -*-
"""
03_visualizations.py
Visualizações e métricas do APDDv2
Compatível com o pipeline novo (com oficial_path, sem pastas frames redundantes)
"""

# ============================================================
# IMPORTS
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import wilcoxon


# ============================================================
# CONFIG
# ============================================================
SCORE_COLUMNS = [
    "Total aesthetic score", "Theme and logic", "Creativity", "Layout and composition",
    "Space and perspective", "Light and shadow", "Color", "Details and texture",
    "The overall", "Mood", "The sense of order"
]


# ============================================================
# Helpers
# ============================================================
def ensure_gif_columns(df: pd.DataFrame):
    df = df.copy()

    if "gif_name" not in df.columns:
        df["gif_name"] = df["filename"].apply(lambda x: x.split("_frame")[0])

    if "frame_number" not in df.columns:
        df["frame_number"] = df["filename"].apply(
            lambda x: int(x.split("_frame")[1].split(".")[0])
        )

    return df


# ============================================================
# Preparação
# ============================================================
def prepare_score_dfs(df_original_raw, df_ruido_raw):

    cols = ["filename", "oficial_path"] + SCORE_COLUMNS

    df_original = df_original_raw[cols].sort_values("filename").reset_index(drop=True)
    df_ruido = df_ruido_raw[cols + ["noise_type"]].sort_values("filename").reset_index(drop=True)

    if not df_original["filename"].equals(df_ruido["filename"]):
        print("⚠️ Filenames não alinham — fazendo merge seguro")
        common = set(df_original["filename"]) & set(df_ruido["filename"])
        df_original = df_original[df_original["filename"].isin(common)]
        df_ruido = df_ruido[df_ruido["filename"].isin(common)]

    df_original = ensure_gif_columns(df_original)
    df_ruido = ensure_gif_columns(df_ruido)

    return df_original, df_ruido


# ============================================================
# Diff frame-a-frame
# ============================================================
def compute_diff(df_name, df_original, df_ruido, output_path):
    df_diff = df_ruido.copy()

    for col in SCORE_COLUMNS:
        df_diff[col] = df_original[col] - df_ruido[col]

    df_diff["total_change"] = df_diff[SCORE_COLUMNS].abs().sum(axis=1)

    csv_path = output_path / f"df_diff_{df_name}.csv"
    df_diff.to_csv(csv_path, index=False)
    print(f"✔ df_diff salvo em: {csv_path}")

    return df_diff


"""def statistical_analysis(df_original, df_ruido, df_diff, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results = []

    for col in SCORE_COLUMNS:
        orig = df_original[col].values
        ruido = df_ruido[col].values
        delta = orig - ruido
        
        # Wilcoxon
        stat, p = wilcoxon(orig, ruido)

        # Effect size (Cohen’s d)
        pooled_std = np.sqrt((orig.std()**2 + ruido.std()**2) / 2)
        d = (orig.mean() - ruido.mean()) / pooled_std

        results.append({
            "score": col,
            "mean_original": orig.mean(),
            "mean_ruido": ruido.mean(),
            "mean_delta": delta.mean(),
            "std_original": orig.std(),
            "std_ruido": ruido.std(),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "cohen_d": d,
        })

    df_stats = pd.DataFrame(results)
    df_stats.to_csv(output_dir / "estatisticas_impacto_ruido.csv", index=False)
    print("✔ Estatísticas salvas:", output_dir / "estatisticas_impacto_ruido.csv")
    return df_stats
"""
# ============================================================
# STD
# ============================================================
def compute_std_diff(df_name, df_original, df_ruido, output_path):
    std_o = df_original.groupby("gif_name")[SCORE_COLUMNS].std()
    std_r = df_ruido.groupby("gif_name")[SCORE_COLUMNS].std()

    std_o.columns = [c + "_std_original" for c in std_o.columns]
    std_r.columns = [c + "_std_ruido" for c in std_r.columns]

    df_std = pd.concat([std_o, std_r], axis=1).reset_index()

    for col in SCORE_COLUMNS:
        df_std[f"{col}_std_diff"] = (
            df_std[f"{col}_std_original"] - df_std[f"{col}_std_ruido"]
        )

    csv_path = output_path / f"df_std_{df_name}.csv"
    df_std.to_csv(csv_path, index=False)
    print(f"✔ df_std salvo em: {csv_path}")

    return df_std


# ============================================================
# Plot frame scores
# ============================================================
def plot_frame_scores(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    SCORE_COLUMNS_PLOT = [c for c in SCORE_COLUMNS if c != "Total aesthetic score"]

    for gif in df["gif_name"].unique():
        subset = df[df["gif_name"] == gif].sort_values("frame_number")

        noisy_frames = subset[subset["noise_type"].notna()]["frame_number"].tolist() \
                        if "noise_type" in subset.columns else []

        plt.figure(figsize=(14, 6))

        # -----------------------------
        # 1) PLOT DAS CURVAS
        # -----------------------------
        for col in SCORE_COLUMNS_PLOT:
            plt.plot(
                subset["frame_number"],
                subset[col],
                label=col,
                marker="o"
            )

        # -----------------------------
        # 2) DESTACAR FRAMES MODIFICADOS COM RUIDO
        # -----------------------------
        for nf in noisy_frames:
            # Linha vertical
            plt.axvline(
                nf,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="Frame ruidoso" if nf == noisy_frames[0] else None
            )

            # Sombra leve para reforçar visualmente
            plt.axvspan(
                nf - 0.4, nf + 0.4,
                color="red",
                alpha=0.15
            )

        # -----------------------------
        # 3) AJUSTE DE ESCALA GLOBAL
        # -----------------------------
        plt.ylim(0, 10)      # <<< TUDO vai de 0 a 10
        plt.xlim(subset["frame_number"].min() - 0.5,
                 subset["frame_number"].max() + 0.5)

        plt.title(f"Evolução dos Scores — {gif}")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out = output_dir / f"{gif}_scores.png"
        plt.savefig(out, dpi=300)
        plt.close()

    print(f"✔ Plots salvos em: {output_dir}")



# ============================================================
# Heatmap
# ============================================================
def plot_heatmap_changed_frames(df_diff, output_path):

    df = df_diff.copy()
    df["changed"] = (df["total_change"] > 0).astype(int)

    if "gif_num" not in df.columns:
        df["gif_num"] = pd.factorize(df["gif_name"])[0] + 1

    pivot = df.pivot_table(
        index="frame_number",
        columns="gif_num",
        values="changed",
        fill_value=0
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    hover_text = []
    for frame in pivot.index:
        row_text = []
        for gifnum in pivot.columns:
            subset = df[(df["frame_number"] == frame) & (df["gif_num"] == gifnum)]
            if subset.empty:
                row_text.append("Sem dados")
                continue

            row = subset.iloc[0]

            text = f"<b>GIF:</b> {row['gif_name']}<br>"
            text += f"<b>Frame:</b> {frame}<br>"
            text += f"<b>Total Change:</b> {row['total_change']:.4f}<br><br>"

            for col in SCORE_COLUMNS:
                text += f"{col}: {abs(row[col]):.4f}<br>"

            row_text.append(text)
        hover_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            text=hover_text,
            hoverinfo="text",
            colorscale=[[0, "white"], [1, "red"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="Heatmap — Frames Alterados",
        xaxis_title="GIF",
        yaxis_title="Frame",
        height=900,
        width=1600,
    )

    fig.write_html(str(output_path))
    print(f"✔ Heatmap salvo em: {output_path}")


# ============================================================
# Diferença média por tipo de ruído
# ============================================================
def plot_mean_difference_by_noise(df_diff, output_path):

    df_changed = df_diff[df_diff["total_change"] > 0].copy()

    df_long = df_changed.melt(
        id_vars=["noise_type"],
        value_vars=SCORE_COLUMNS,
        var_name="score",
        value_name="diff",
    )

    df_plot = df_long.groupby(["noise_type", "score"]).mean().reset_index()

    # -------------------------------
    #  GRÁFICO DE BARRAS AGRUPADAS
    # -------------------------------
    plt.figure(figsize=(16, 7))

    scores = SCORE_COLUMNS
    noise_types = df_plot["noise_type"].unique()

    x = np.arange(len(scores))  # posições no eixo X
    width = 0.8 / len(noise_types)  # divide o espaço entre os grupos

    for i, noise in enumerate(noise_types):
        sub = df_plot[df_plot["noise_type"] == noise]["diff"].values
        plt.bar(x + i * width, sub, width=width, label=noise)

    plt.xticks(x + width * (len(noise_types) - 1) / 2, scores, rotation=45, ha="right")
    plt.ylabel("Diferença média")
    plt.title("Diferença Média dos Scores por Tipo de Ruído (Barras)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✔ Gráfico salvo: {output_path}")



# ============================================================
# Rankings + PNG
# ============================================================
def compute_rankings(df_original, df_ruido, output_path):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df_original["score_total"] = df_original[SCORE_COLUMNS].sum(axis=1)

    best = df_original.nlargest(5, "score_total")
    worst = df_original.nsmallest(5, "score_total")

    best.to_csv(output_path / "top5_best_images.csv", index=False)
    worst.to_csv(output_path / "top5_worst_images.csv", index=False)

    # ----- gerar PNG -----
    def gerar_png(df_subset, saida_png, titulo):
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = ImageFont.load_default()

        w_img = 256
        h_img = 256
        margem = 20

        canvas = Image.new("RGB", (w_img + 2 * margem, len(df_subset) * (h_img + 80)), "white")
        draw = ImageDraw.Draw(canvas)

        y = 0
        for _, row in df_subset.iterrows():
            try:
                im = Image.open(row["oficial_path"]).convert("RGB")
                im = im.resize((w_img, h_img))
            except:
                im = Image.new("RGB", (w_img, h_img), "gray")
                ImageDraw.Draw(im).text((10, 10), "Erro", fill="red")

            canvas.paste(im, (margem, y))
            draw.text(
                (margem, y + h_img + 10),
                f"{row['filename']} — Score: {row['score_total']:.2f}",
                fill="black",
                font=font,
            )
            y += h_img + 80

        canvas.save(saida_png)
        print(f"✔ PNG salvo em: {saida_png}")

    gerar_png(best, output_path / "top5_best_images.png", "Top 5 melhores")
    gerar_png(worst, output_path / "top5_worst_images.png", "Top 5 piores")

    print("✔ Rankings prontos!")


# ============================================================
# PROCESSAMENTO POR DATASET
# ============================================================
def process_dataset(original_csv, ruido_csv, dataset_name, output_dir):

    print("\n===================================")
    print(f"📌 Processando dataset: {dataset_name}")
    print("===================================\n")

    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df_original_raw = pd.read_csv(original_csv)
    df_ruido_raw = pd.read_csv(ruido_csv)

    df_original, df_ruido = prepare_score_dfs(df_original_raw, df_ruido_raw)

    df_diff = compute_diff(dataset_name, df_original, df_ruido, output_dir)
    #df_stats = statistical_analysis(df_original, df_ruido, df_diff, output_dir)
    compute_std_diff(dataset_name, df_original, df_ruido, output_dir)


    plot_frame_scores(df_original, output_dir / "original_plots")
    plot_frame_scores(df_ruido, output_dir / "ruido_plots")
    plot_heatmap_changed_frames(df_diff, output_dir / "heatmap.html")
    plot_mean_difference_by_noise(df_diff, output_dir / "diff_media.png")

    compute_rankings(df_original, df_ruido, output_dir)

    print(f"🎉 Finalizado: {dataset_name}\n")


# ============================================================
# MAIN
# ============================================================
def main():

    datasets = [
        {
            "name": "gifs",
            "original": Path("/mnt/c/Users/Dell/Downloads/dataset/dataset/gifs/gifs_original_dataset_scored.csv"),
            "ruido": Path("/mnt/c/Users/Dell/Downloads/dataset/dataset/gifs/gifs_ruido_dataset_scored.csv"),
        },
        {
            "name": "gif_storyboarding",
            "original": Path("/mnt/c/Users/Dell/Downloads/dataset/dataset/gif_storyboarding/gif_storyboarding_original_dataset_scored.csv"),
            "ruido": Path("/mnt/c/Users/Dell/Downloads/dataset/dataset/gif_storyboarding/gif_storyboarding_ruido_dataset_scored.csv"),
        },
    ]

    for d in datasets:
        process_dataset(
            original_csv=d["original"],
            ruido_csv=d["ruido"],
            dataset_name=d["name"],
            output_dir="resultados"
        )

    print("\n🎉 Todas as visualizações concluídas!\n")


# ============================================================
# EXECUTA MAIN
# ============================================================
if __name__ == "__main__":
    main()
