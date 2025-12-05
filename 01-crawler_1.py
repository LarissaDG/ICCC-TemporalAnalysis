import subprocess
import csv
import json
import os

CHANNEL_SHORTS_URL = "https://www.youtube.com/@ArtsyLolaCo/shorts"
OUTPUT_DIR = "downloads"
CSV_FILE = "shorts_ids.csv"


def get_shorts_video_ids(channel_url):
    """
    Usa yt-dlp para listar todos os vídeos de um canal/shorts e extrair os videoIds.
    """
    print("↳ Coletando IDs dos shorts com yt-dlp...")

    # Comando yt-dlp para listar vídeos em JSON
    result = subprocess.run(
        ["yt-dlp", "-j", "--flat-playlist", channel_url],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Erro ao executar yt-dlp:", result.stderr)
        return []

    video_ids = []
    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
            vid = data.get("id")
            if vid:
                video_ids.append(vid)
        except json.JSONDecodeError:
            continue

    # Remove duplicados mantendo a ordem
    unique_ids = list(dict.fromkeys(video_ids))
    print(f"✔ {len(unique_ids)} IDs encontrados.")
    return unique_ids


def save_ids_to_csv(ids, filename=CSV_FILE):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    print(f"↳ Salvando IDs em {filename} ...")
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id"])
        for vid in ids:
            writer.writerow([vid])
    print("✔ IDs salvos com sucesso.")


def download_videos(ids, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    print("\n📥 Iniciando download dos vídeos com yt-dlp...\n")

    for i, vid in enumerate(ids, start=1):
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"({i}/{len(ids)}) Baixando: {url}")

        subprocess.run([
            "yt-dlp",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            url
        ])

    print("\n✔ Todos os downloads finalizados.\n")


def main():
    video_ids = get_shorts_video_ids(CHANNEL_SHORTS_URL)

    if not video_ids:
        print("Nenhum vídeo encontrado. Encerrando.")
        return

    save_ids_to_csv(video_ids)
    download_videos(video_ids)


if __name__ == "__main__":
    main()
