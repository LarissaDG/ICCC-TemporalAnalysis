# -*- coding: utf-8 -*-
import os
import gdown
import zipfile
from pathlib import Path


def baixar_por_id(google_id: str) -> Path:
    print("\n=== BAIXANDO ARQUIVO ===")

    antes = set(os.listdir("."))

    cmd = f'gdown "{google_id}"'
    print("Executando:", cmd)
    os.system(cmd)

    depois = set(os.listdir("."))
    novos = depois - antes

    if not novos:
        print("⚠ Nenhum arquivo detectado!")
        return None

    nome = list(novos)[0]
    caminho = Path(nome)
    print(f"✔ Download concluído: {caminho}")
    return caminho



def unzip_simples(zip_path: Path, destino: Path = None) -> Path:
    """
    Extrai o ZIP mantendo a estrutura original e retorna o path
    da PRIMEIRA pasta criada dentro do ZIP.

    Se o ZIP não tiver pasta raiz, retorna o destino.
    """
    print("\n=== EXTRAINDO ZIP ===")

    if destino is None:
        destino = Path(zip_path.stem)

    destino.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(destino)
            nomes = z.namelist()
    except Exception as e:
        print(f"❌ Erro ao extrair {zip_path}: {e}")
        return None

    print(f"✔ ZIP extraído: {zip_path} → {destino}")

    # Descobrir a pasta raiz do zip
    pastas = sorted({
        (destino / nome.split("/")[0])
        for nome in nomes
        if "/" in nome  # tem subpasta
    })

    # Se existe uma pasta raiz, retorna ela
    if pastas:
        raiz = pastas[0]
        print(f"✔ Pasta raiz detectada: {raiz}")
        return raiz

    # Caso não tenha subpastas no zip, retorna o destino
    print(f"⚠ Nenhuma pasta interna detectada. Retornando: {destino}")
    return destino



if __name__ == "__main__":
    print("=== PIPELINE: DOWNLOAD + UNZIP ===")

    id_gifs = "1-c1hi-V2jg-HS27fzblObbnm1NgiSpcB"
    id_story = "1hlDB_8bHjKZaVs8xZLqFqwIFZFLqxig8"

    zip1 = baixar_por_id(id_gifs)
    zip2 = baixar_por_id(id_story)

    paths_salvos = []

    # 2) Extrair e pegar APENAS a pasta raiz
    if zip1:
        pasta_gifs = unzip_simples(zip1)
        print("\nPasta raiz do ZIP 1:", pasta_gifs)
        paths_salvos.append(f"GIFs: {pasta_gifs}")

    if zip2:
        pasta_story = unzip_simples(zip2)
        print("\nPasta raiz do ZIP 2:", pasta_story)
        paths_salvos.append(f"Story: {pasta_story}")

    # === SALVAR EM TXT ===
    arquivo_txt = "paths_baixados.txt"
    with open(arquivo_txt, "w", encoding="utf-8") as f:
        for linha in paths_salvos:
            f.write(str(linha) + "\n")

    print(f"\n✔ Paths salvos em: {arquivo_txt}")
    print("=== FINALIZADO ===")

