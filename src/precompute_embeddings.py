"""
precompute_embeddings.py
------------------------
Calcula los embeddings de 2048 dimensiones utilizando el modelo PANNs CNN14
pre-entrenado en AudioSet. Guarda los vectores en features/emb_{label_id}_{indice}.npy
y genera un nuevo indice dataset_index_emb.csv.
"""

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
# pyrefly: ignore [missing-import]
import torch


# Asegurar que src/ este en el path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR

# Rutas
INDEX_CSV = BASE_DIR / "dataset_index.csv"
OUTPUT_CSV = BASE_DIR / "dataset_index_emb.csv"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "Cnn14_mAP=0.431.pth"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"

def download_checkpoint_if_needed():
    MODELS_DIR.mkdir(exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        print(f"Descargando checkpoint CNN14 en {CHECKPOINT_PATH}...")
        urllib.request.urlretrieve(CHECKPOINT_URL, str(CHECKPOINT_PATH))
        print("Descarga completada.")

def download_panns_metadata():
    """Descarga los metadatos de AudioSet para panns_inference (evita usar wget en Windows)"""
    panns_data_dir = Path.home() / "panns_data"
    panns_data_dir.mkdir(exist_ok=True)
    csv_path = panns_data_dir / "class_labels_indices.csv"
    
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        print(f"Descargando metadatos de AudioSet en {csv_path}...")
        urllib.request.urlretrieve(url, str(csv_path))

def main():
    download_checkpoint_if_needed()
    download_panns_metadata()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Cargando modelo PANNs CNN14 en {device}...")
    
    try:
        # pyrefly: ignore [missing-import]
        from panns_inference import AudioTagging

    except ImportError:
        print("[ERROR] No se pudo importar panns_inference. Ejecuta: pip install panns-inference")
        sys.exit(1)
        
    # Inicializar el modelo especificando nuestra ruta de checkpoint local
    at = AudioTagging(checkpoint_path=str(CHECKPOINT_PATH), device=device)
    
    df = pd.read_csv(INDEX_CSV)
    print(f"Clips a procesar: {len(df)}")
    FEATURES_DIR.mkdir(exist_ok=True)
    
    emb_paths = []
    errors = []
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"  {100*(idx+1)//total}% ({idx+1}/{total})", flush=True)
        out_name = f"emb_{row['label_id']}_{idx}.npy"
        out_path = FEATURES_DIR / out_name
        
        try:
            # PANNs espera audio a 32000Hz
            audio, _ = librosa.load(row['filepath'], sr=32000, mono=True)
            # Expandir dimensiones a (batch_size, time)
            audio = audio[None, :] 
            
            # at.inference devuelve (clipwise_output, embedding)
            # embedding tiene forma (1, 2048)
            _, embedding = at.inference(audio)
            
            # Guardar el vector 1D de 2048 elementos
            np.save(out_path, embedding[0])
            emb_paths.append(str(out_path))
        except Exception as e:
            print(f"\n[ERROR] idx={idx} | {row['filepath']}\n  -> {e}")
            emb_paths.append("")
            errors.append(idx)
            
    df["embedding_path"] = emb_paths
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV de embeddings guardado en {OUTPUT_CSV}")
    
    if errors:
        print(f"\n[ADVERTENCIA] {len(errors)} clip(s) fallaron durante el procesado.")
    else:
        print("Procesado completado sin errores [OK]. Forma del embedding: (2048,)")

if __name__ == "__main__":
    main()
