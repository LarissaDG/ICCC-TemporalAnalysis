# renomear_videos_numerados.py
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

PASTA = Path("videos_amostrados_raw")

def main():
    if not PASTA.exists():
        print("❌ Pasta não encontrada:", PASTA)
        return

    arquivos = sorted([f for f in PASTA.iterdir() if f.is_file()])

    if not arquivos:
        print("❌ Nenhum arquivo encontrado em", PASTA)
        return

    print(f"📁 Encontrados {len(arquivos)} arquivos. Renomeando...")

    registros = []
    contador = 1

    for arquivo in arquivos:
        ext = arquivo.suffix.lower()
        novo_nome = f"{contador}{ext}"
        novo_caminho = PASTA / novo_nome

        # Evitar sobrescrever sem querer
        if novo_caminho.exists():
            print(f"⚠️ Pulando {arquivo.name}: {novo_nome} já existe.")
            continue

        arquivo.rename(novo_caminho)

        registros.append({
            "original_name": arquivo.name,
            "new_name": novo_nome,
            "original_path": str(arquivo),
            "new_path": str(novo_caminho)
        })

        contador += 1

    # salvar CSV
    df = pd.DataFrame(registros)
    df.to_csv(PASTA / "renomeacao.csv", index=False)

    print(f"✔ Renomeação concluída. CSV salvo em: {PASTA/'renomeacao.csv'}")


if __name__ == "__main__":
    main()
