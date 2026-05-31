"""
preparar_datos_v2.py
--------------------
Este script aplica Data Augmentation (ruido y cambio de tono) a las muestras de
entrenamiento y luego extrae los embeddings PANNs. Genera features_v2/ y
dataset_index_emb_v2.csv.
"""

import os
import sys
import urllib.request
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_V2_DIR = BASE_DIR / "features_v2"
MODELS_DIR = BASE_DIR / "models"
SEED = 42

def extract_panns_embeddings_with_augmentation():
    input_csv = BASE_DIR / "dataset_index.csv"
    output_csv = BASE_DIR / "dataset_index_emb_v2.csv"
    
    if not input_csv.exists():
        print(f"Error: No se encuentra {input_csv}. Por favor, corre datos_y_configuracion.py primero.")
        return
        
    print(f"Cargando {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Preparar modelo PANNs
    FEATURES_V2_DIR.mkdir(exist_ok=True)
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
    print(f"Cargando modelo PANNs en {device}...")
    at = AudioTagging(checkpoint_path=str(ckpt_path), device=device)
    
    new_rows = []
    
    print(f"Procesando {len(df)} audios y aplicando Data Augmentation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filepath = row['filepath']
        label_id = row['label_id']
        label_name = row['label_name']
        split = row['split']
        source = row['source']
        
        try:
            # Cargar audio original
            audio_orig, sr = librosa.load(filepath, sr=32000, mono=True)
            
            # 1. Original: extraer y guardar embedding
            _, emb_orig = at.inference(audio_orig[None, :])
            path_orig = FEATURES_V2_DIR / f"emb_{label_id}_{idx}_orig.npy"
            np.save(path_orig, emb_orig[0])
            
            new_rows.append({
                "filepath": filepath, "label_id": label_id, "label_name": label_name,
                "source": source, "split": split, "embedding_path": str(path_orig),
                "augmentation": "none"
            })
            
            # 2. Solo aplicar Aumentación si es del conjunto de entrenamiento
            if split == 'train':
                # Augmentation A: Inyección de Ruido (SNR 10dB)
                signal_power = np.mean(audio_orig**2)
                if signal_power > 0:
                    snr_linear = 10 ** (10 / 10) # 10dB SNR
                    noise_power = signal_power / snr_linear
                    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_orig))
                    audio_noise = (audio_orig + noise).astype(np.float32)
                    
                    _, emb_noise = at.inference(audio_noise[None, :])
                    path_noise = FEATURES_V2_DIR / f"emb_{label_id}_{idx}_noise.npy"
                    np.save(path_noise, emb_noise[0])
                    
                    new_rows.append({
                        "filepath": filepath, "label_id": label_id, "label_name": label_name,
                        "source": source, "split": split, "embedding_path": str(path_noise),
                        "augmentation": "noise_10db"
                    })
                
                # Augmentation B: Pitch Shift (+2 semitonos)
                audio_pitch = librosa.effects.pitch_shift(y=audio_orig, sr=sr, n_steps=2)
                _, emb_pitch = at.inference(audio_pitch[None, :])
                path_pitch = FEATURES_V2_DIR / f"emb_{label_id}_{idx}_pitch_up.npy"
                np.save(path_pitch, emb_pitch[0])
                
                new_rows.append({
                    "filepath": filepath, "label_id": label_id, "label_name": label_name,
                    "source": source, "split": split, "embedding_path": str(path_pitch),
                    "augmentation": "pitch_+2"
                })

                # Augmentation C: Pitch Shift (-2 semitonos)
                audio_pitch_down = librosa.effects.pitch_shift(y=audio_orig, sr=sr, n_steps=-2)
                _, emb_pitch_down = at.inference(audio_pitch_down[None, :])
                path_pitch_down = FEATURES_V2_DIR / f"emb_{label_id}_{idx}_pitch_down.npy"
                np.save(path_pitch_down, emb_pitch_down[0])
                
                new_rows.append({
                    "filepath": filepath, "label_id": label_id, "label_name": label_name,
                    "source": source, "split": split, "embedding_path": str(path_pitch_down),
                    "augmentation": "pitch_-2"
                })

        except Exception as e:
            print(f"Error procesando {filepath}: {e}")
            pass
            
    # Guardar nuevo dataset index
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    print(f"\n[ÉXITO] Creado nuevo dataset con {len(new_df)} embeddings en {output_csv}.")
    
    # Contar la distribución en entrenamiento
    train_count = len(new_df[new_df['split'] == 'train'])
    print(f"Muestras originales + aumentadas en Entrenamiento: {train_count}")

if __name__ == "__main__":
    extract_panns_embeddings_with_augmentation()
