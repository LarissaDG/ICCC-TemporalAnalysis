def build_dataset_processado():

    print("\n==============================")
    print("📦 PROCESSANDO DATASET")
    print("==============================\n")

    # Garantir diretórios
    for d in [FRAMES_RUIDOS_H1_DIR, FRAMES_INTENSIDADES_H2_DIR, ORQUESTRADOR_DIR]:
        d.mkdir(exist_ok=True)

    # ============================================================
    # 1. Carregar informações dos vídeos convertidos (para duração)
    # ============================================================

    print("⏱️ Lendo durações dos vídeos convertidos...")
    video_durations = {}
    converted_mp4 = sorted(Path("videos_amostrados_mp4").glob("*.mp4"))

    for vid in converted_mp4:
        try:
            meta = imageio.get_reader(str(vid), "ffmpeg").get_meta_data()
            dur = float(meta.get("duration", 0.0))
        except:
            dur = 0.0
        video_durations[vid.stem] = dur


    # ============================================================
    # 2. Carregar frames originais e montar base ORIGINAL
    # ============================================================

    print("📁 Carregando frames originais...")
    all_frames = sorted(FRAMES_ORIGINAIS_DIR.glob("*.png"))
    if not all_frames:
        print(Fore.RED + "ERRO: Nenhum frame encontrado em frames_originais/." + Style.RESET_ALL)
        return

    videos = {}     # vid → [(frame_idx, frame_path)]
    original_rows = []

    for f in all_frames:
        parts = f.stem.split("_")
        vid = "_".join(parts[:-2])
        frame_idx = int(parts[-1])

        videos.setdefault(vid, []).append((frame_idx, f))

    # ordenar frames dentro de cada vídeo
    for vid in videos:
        videos[vid] = sorted(videos[vid], key=lambda x: x[0])

    # Construir CSV original
    for vid, flist in videos.items():
        tempo_total = video_durations.get(vid, len(flist))

        for frame_idx, f in flist:
            original_rows.append({
                "video_name": vid,
                "frame_index": frame_idx,
                "frame_original_path": str(f),
                "tempo_total_video": tempo_total
            })

    df_original = pd.DataFrame(original_rows)
    df_original.to_csv("original.csv", index=False)



    # ============================================================
    # 3. H1 — Ruído intermitente (3 frames por vídeo)
    # ============================================================

    print("💥 Aplicando H1 intermitente...")

    df_h1_rows = []

    vid_names = list(videos.keys())
    random.shuffle(vid_names)

    # Balanceamento de tipos de ruído
    noise_assign = {}
    chunk = len(vid_names) // len(NOISE_TYPES)
    idx = 0

    for nt in NOISE_TYPES:
        for _ in range(chunk):
            if idx < len(vid_names):
                noise_assign[vid_names[idx]] = nt
                idx += 1

    while idx < len(vid_names):
        noise_assign[vid_names[idx]] = random.choice(NOISE_TYPES)
        idx += 1


    for vid, flist in tqdm(videos.items(), desc="H1", ncols=90):
        noise_type = noise_assign[vid]

        # sortear 3 frames para sofrer ruído
        frame_indices = [idx for idx, p in flist]
        noisy_positions = set(random.sample(frame_indices, min(3, len(flist))))

        tempo_total = video_durations.get(vid, len(flist))

        for frame_idx, f in flist:
            img = Image.open(f)

            if frame_idx in noisy_positions:
                # ruído total intensidade = 100
                if noise_type == "blur_sp":
                    noisy_img, _, _ = add_blur_saltpepper(img, 100)
                elif noise_type == "shapes_x":
                    noisy_img, _ = add_shapes_x(img, 100)
                elif noise_type == "gaussian":
                    noisy_img, _ = add_gaussian_noise(img, 100)

                out = FRAMES_RUIDOS_H1_DIR / f"H1_{noise_type}_{f.name}"
                noisy_img.save(out)
                noisy_path = str(out)
                flag = 1
            else:
                noisy_path = str(f)  # permanece igual ao original
                flag = 0

            df_h1_rows.append({
                "video_name": vid,
                "frame_index": frame_idx,
                "frame_original_path": str(f),
                "frame_h1_path": noisy_path,
                "ruido_aplicado": flag,
                "noise_type_h1": noise_type,
                "tempo_total_video": tempo_total,
                "experiment_type": "H1_intermitente"
            })

    df_h1 = pd.DataFrame(df_h1_rows)
    df_h1.to_csv("h1_intermitente.csv", index=False)
    df_h1.to_csv(ORQUESTRADOR_DIR / "orquestrador_H1.csv", index=False)



    # ============================================================
    # 4. H2 — Gradual usando generate_noise_sweep
    # ============================================================

    print("📉 Aplicando H2 gradual (generate_noise_sweep)...")

    df_h2_rows = []

    vid_names2 = list(videos.keys())
    random.shuffle(vid_names2)

    assign2 = {}
    idx = 0
    for nt in NOISE_TYPES:
        for _ in range(chunk):
            if idx < len(vid_names2):
                assign2[vid_names2[idx]] = nt
                idx += 1
    while idx < len(vid_names2):
        assign2[vid_names2[idx]] = random.choice(NOISE_TYPES)
        idx += 1


    for vid, flist in tqdm(videos.items(), desc="H2", ncols=90):
        noise_type = assign2[vid]
        tempo_total = video_durations.get(vid, len(flist))
        T = len(flist)

        # precisamos de intensidades crescentes → usamos sweep
        # mas sweep salva imagens, nós só aproveitamos a lógica de intensidades
        intensities = np.linspace(1, 100, T).astype(int)

        for intensity, (frame_idx, f) in zip(intensities, flist):
            img = Image.open(f)

            # Aplicando cada ruído manualmente, mas usando intensidade do sweep
            if noise_type == "blur_sp":
                noisy_img, _, _ = add_blur_saltpepper(img, intensity)
            elif noise_type == "shapes_x":
                noisy_img, _ = add_shapes_x(img, intensity)
            elif noise_type == "gaussian":
                noisy_img, _ = add_gaussian_noise(img, intensity)

            out = FRAMES_INTENSIDADES_H2_DIR / f"H2_{noise_type}_int{intensity:03d}_{f.name}"
            noisy_img.save(out)

            df_h2_rows.append({
                "video_name": vid,
                "frame_index": frame_idx,
                "frame_original_path": str(f),
                "frame_h2_path": str(out),
                "noise_type_h2": noise_type,
                "noise_intensity_h2": intensity,
                "tempo_total_video": tempo_total,
                "experiment_type": "H2_gradual"
            })

    df_h2 = pd.DataFrame(df_h2_rows)
    df_h2.to_csv("h2_gradual.csv", index=False)
    df_h2.to_csv(ORQUESTRADOR_DIR / "orquestrador_H2.csv", index=False)


    print("\n✨ PROCESSAMENTO FINALIZADO COM SUCESSO!")
