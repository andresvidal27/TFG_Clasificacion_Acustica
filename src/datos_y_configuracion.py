"""
datos_y_configuracion.py
------------------------
Este archivo centraliza la configuración del proyecto (frecuencias, rutas)
y todas las funciones necesarias para preparar los datos:
1. Generación del índice unificado (dataset_index.csv)
2. Extracción de características (Espectrogramas Log-Mel)
3. Extracción de embeddings (CNN14 PANNs)
"""

import os
import sys
import urllib.request
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import torch

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONSTANTES DE CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ESC50_DIR = BASE_DIR / "data_esc50"
DATA_URBAN_DIR = BASE_DIR / "data_urban"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Parámetros de Audio para CNN
SR = 22050
DURATION = 5.0
N_SAMPLES = 110250
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 11025

SEED = 42

# Mapa de Clases
CLASS_MAP = {
    0: "glass_breaking", 1: "gun_shot", 2: "dog_bark", 3: "siren",
    4: "car_horn", 5: "crying_baby", 6: "thunderstorm", 7: "fireworks",
    8: "clock_alarm", 9: "door_knock", 10: "background", 11: "screaming",
}
NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}

# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO DE AUDIO
# ==============================================================================
def preprocess_audio(filepath: str) -> np.ndarray:
    """Carga y normaliza un audio a exactamente N_SAMPLES muestras."""
    signal, _ = librosa.load(filepath, sr=SR, mono=True)
    if len(signal) > N_SAMPLES:
        signal = signal[:N_SAMPLES]
    elif len(signal) < N_SAMPLES:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)))
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal.astype(np.float32)

def compute_logmel(signal: np.ndarray) -> np.ndarray:
    """Calcula el espectrograma Log-Mel usando parámetros de configuración."""
    S = librosa.feature.melspectrogram(
        y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    return np.log(S + 1e-6).astype(np.float32)

def add_awgn(signal: np.ndarray, snr_db: float = None) -> np.ndarray:
    """Añade ruido blanco gaussiano (AWGN) a la señal para un SNR dado."""
    if snr_db is None:
        return signal
    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return (signal + noise).astype(np.float32)

# ==============================================================================
# 3. CONSTRUCCIÓN DEL DATASET Y PRECOMPUTACIÓN
# ==============================================================================
def generar_indice_dataset():
    """Genera el dataset_index.csv combinando ESC-50 y UrbanSound8K."""
    output_path = BASE_DIR / "dataset_index.csv"
    if output_path.exists():
        print(f"[Info] {output_path.name} ya existe. Saltando generación.")
        return pd.read_csv(output_path)

    print("Generando dataset_index.csv...")
    esc50 = pd.read_csv(DATA_ESC50_DIR / "esc50.csv")
    urban = pd.read_csv(DATA_URBAN_DIR / "UrbanSound8K.csv")

    rows = []
    # 1. Añadir sonidos puros de ESC-50
    esc_only = ["glass_breaking", "crying_baby", "thunderstorm", "fireworks", "clock_alarm"]
    for cat in esc_only:
        for _, r in esc50[esc50["category"] == cat].iterrows():
            rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID[cat], "label_name": cat, "source": "esc50"})
    
    for _, r in esc50[esc50["category"] == "door_wood_knock"].iterrows():
        rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID["door_knock"], "label_name": "door_knock", "source": "esc50"})

    # 2. Clases combinadas (Dog bark, siren, car horn, gun shot)
    for cat_esc, cat_urb, label in [("dog", "dog_bark", "dog_bark"), ("siren", "siren", "siren"), ("car_horn", "car_horn", "car_horn")]:
        esc_subset = esc50[esc50["category"] == cat_esc]
        for _, r in esc_subset.iterrows():
            rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID[label], "label_name": label, "source": "esc50"})
        urb_subset = urban[urban["class"] == cat_urb].copy()
        if len(urb_subset) > (300 - len(esc_subset)): urb_subset = urb_subset.sample(n=(300 - len(esc_subset)), random_state=SEED)
        for _, r in urb_subset.iterrows():
            rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID[label], "label_name": label, "source": "urban"})

    # Gun shot (Urban)
    urb_gun = urban[urban["class"] == "gun_shot"].sample(n=300, random_state=SEED) if len(urban[urban["class"] == "gun_shot"]) > 300 else urban[urban["class"] == "gun_shot"]
    for _, r in urb_gun.iterrows():
        rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID["gun_shot"], "label_name": "gun_shot", "source": "urban"})

    # 3. Background
    bg_urban = urban[urban["class"].isin(["air_conditioner", "children_playing", "drilling", "engine_idling", "jackhammer", "street_music"])]
    danger_cats = {"glass_breaking", "crying_baby", "thunderstorm", "fireworks", "clock_alarm", "door_wood_knock", "dog", "siren", "car_horn"}
    bg_esc = esc50[~esc50["category"].isin(danger_cats)]
    bg_rows = []
    for _, r in bg_esc.iterrows(): bg_rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID["background"], "label_name": "background", "source": "esc50"})
    for _, r in bg_urban.iterrows(): bg_rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID["background"], "label_name": "background", "source": "urban"})
    bg_df = pd.DataFrame(bg_rows)
    if len(bg_df) > 700: bg_df = bg_df.sample(n=700, random_state=SEED)
    rows.extend(bg_df.to_dict("records"))

    # 4. Gritos
    gritos_dir = DATA_ESC50_DIR / "gritos"
    if gritos_dir.exists():
        gritos = [{"filepath": str(f), "label_id": NAME_TO_ID["screaming"], "label_name": "screaming", "source": "esc50"} for f in list(gritos_dir.glob("*.wav"))[:300]]
        rows.extend(gritos)

    df = pd.DataFrame(rows)
    # Split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"[OK] dataset_index.csv creado con {len(df)} audios.")
    return df

def precomputar_features_cnn(df: pd.DataFrame):
    """Calcula y guarda los espectrogramas Log-Mel para la CNN."""
    out_csv = BASE_DIR / "dataset_index_features.csv"
    if out_csv.exists():
        print(f"[Info] {out_csv.name} ya existe. Saltando features CNN.")
        return pd.read_csv(out_csv)
        
    print("Precomputando Log-Mels para CNN...")
    FEATURES_DIR.mkdir(exist_ok=True)
    paths = []
    for idx, row in df.iterrows():
        try:
            logmel = compute_logmel(preprocess_audio(row["filepath"]))
            path = FEATURES_DIR / f"{row['label_id']}_{idx}.npy"
            np.save(path, logmel)
            paths.append(str(path))
        except Exception:
            paths.append("")
    df["feature_path"] = paths
    df.to_csv(out_csv, index=False)
    print("[OK] Features generadas.")
    return df

def precomputar_embeddings_transfer(df: pd.DataFrame):
    """Descarga el modelo PANNs y extrae los embeddings (2048 dims)."""
    out_csv = BASE_DIR / "dataset_index_emb.csv"
    if out_csv.exists():
        print(f"[Info] {out_csv.name} ya existe. Saltando embeddings Transfer.")
        return pd.read_csv(out_csv)

    print("Precomputando Embeddings PANNs (Transfer Learning)...")
    FEATURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "Cnn14_mAP=0.431.pth"
    if not ckpt_path.exists():
        print("Descargando checkpoint CNN14 (AudioSet)...")
        urllib.request.urlretrieve("https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth", str(ckpt_path))
    
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv", str(csv_path))

    from panns_inference import AudioTagging
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    at = AudioTagging(checkpoint_path=str(ckpt_path), device=device)
    
    emb_paths = []
    for idx, row in df.iterrows():
        try:
            audio, _ = librosa.load(row['filepath'], sr=32000, mono=True)
            _, embedding = at.inference(audio[None, :])
            path = FEATURES_DIR / f"emb_{row['label_id']}_{idx}.npy"
            np.save(path, embedding[0])
            emb_paths.append(str(path))
        except Exception:
            emb_paths.append("")
            
    df["embedding_path"] = emb_paths
    df.to_csv(out_csv, index=False)
    print("[OK] Embeddings generados.")
    return df

if __name__ == "__main__":
    df_base = generar_indice_dataset()
    precomputar_features_cnn(df_base.copy())
    precomputar_embeddings_transfer(df_base.copy())
    print("\n[ÉXITO] Datos y características preparados completamente.")
