"""
datos_y_configuracion.py
------------------------
Este archivo centraliza la configuración global del proyecto (frecuencias, rutas)
y todas las funciones núcleo necesarias para preparar el audio antes del entrenamiento.

Tareas principales que realiza:
1. Generación del índice unificado: Une dos bases de datos enormes (ESC-50 y UrbanSound8K)
   en un solo archivo 'dataset_index.csv', mapeando las clases inglesas a nuestro CLASS_MAP.
2. Extracción de características (Espectrogramas Log-Mel): Convierte audios brutos en "imágenes"
   de frecuencias (espectrogramas) para que las lea el modelo CNN personalizado.
3. Extracción de embeddings (Transfer Learning - PANNs): Pasa los audios por un modelo preentrenado 
   de alta capacidad y extrae sus capas ocultas, realizando Data Augmentation en el proceso.
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
# 1. CONSTANTES Y CONFIGURACIÓN GLOBAL
# ==============================================================================
# Rutas usando pathlib. BASE_DIR es la carpeta raíz del proyecto.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ESC50_DIR = BASE_DIR / "data_esc50"
DATA_URBAN_DIR = BASE_DIR / "data_urban"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Parámetros estrictos de Audio (especialmente para la CNN Base)
SR = 22050               # Sample Rate: 22.05 kHz (calidad radio)
DURATION = 5.0           # Todos los audios deben durar exactamente 5 segundos
N_SAMPLES = 110250       # 22050 * 5 = 110250 muestras. Si falta audio, se rellena con 0; si sobra, se recorta.
N_FFT = 2048             # Tamaño de ventana para la Transformada Rápida de Fourier
HOP_LENGTH = 512         # Cuánto avanza la ventana de Fourier en el tiempo
N_MELS = 128             # Número de bandas de frecuencia Mel en el espectrograma (eje Y)
FMIN = 20                # Frecuencia mínima audible
FMAX = 11025             # Frecuencia máxima (Nyquist = SR / 2)

SEED = 42                # Semilla para que los splits de datos sean reproducibles siempre de la misma forma

# Mapa central de Clases (Id -> Nombre). Cualquier audio que no encaje en los primeros 7 será "fondo".
CLASS_MAP = {
    0: "rotura_cristal", 1: "disparo", 2: "ladrido_perro", 3: "sirena",
    4: "bebe_llorando", 5: "llamar_puerta", 6: "grito", 7: "fondo"
}
NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()} # Diccionario inverso ("grito" -> 6)

# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO MATEMÁTICO DE AUDIO
# ==============================================================================
def preprocess_audio(filepath: str) -> np.ndarray:
    """Carga y normaliza un audio crudo a exactamente N_SAMPLES muestras (5s)."""
    # sr=SR remuestrea cualquier audio (de 44kHz, 48kHz...) a nuestros 22.05kHz obligatorios. mono=True aplana estéreo.
    signal, _ = librosa.load(filepath, sr=SR, mono=True)
    
    # Recorte (truncado) o Relleno (padding) para cumplir los 5 segundos sí o sí
    if len(signal) > N_SAMPLES:
        signal = signal[:N_SAMPLES]
    elif len(signal) < N_SAMPLES:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)))
        
    # Normalización de volumen al rango [-1, 1] dividiendo por el pico máximo
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal.astype(np.float32)

def compute_logmel(signal: np.ndarray) -> np.ndarray:
    """Calcula el espectrograma Mel y le aplica logaritmo para simular el oído humano."""
    # S contiene la "energía" de las frecuencias. Forma: [N_MELS, Tiempos]
    S = librosa.feature.melspectrogram(
        y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    # Se pasa a decibelios logarítmicos (+ 1e-6 para evitar log(0)).
    return np.log(S + 1e-6).astype(np.float32)

def add_awgn(signal: np.ndarray, snr_db: float = None) -> np.ndarray:
    """Añade ruido blanco gaussiano (AWGN) a la señal para un SNR (Signal-to-Noise Ratio) dado."""
    if snr_db is None:
        return signal
    signal_power = np.mean(signal**2)
    if signal_power == 0: # Si es silencio absoluto
        return signal
    # Conversión de decibelios a lineal
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return (signal + noise).astype(np.float32)

# ==============================================================================
# 3. CONSTRUCCIÓN Y ESTRUCTURACIÓN DE LOS DATASETS
# ==============================================================================
def generar_indice_dataset():
    """Genera el dataset_index.csv unificando las clases de las diferentes DBs."""
    output_path = BASE_DIR / "dataset_index.csv"
    if output_path.exists():
        print(f"[Info] {output_path.name} ya existe. Saltando generación.")
        return pd.read_csv(output_path)

    print("Generando dataset_index.csv...")
    esc50 = pd.read_csv(DATA_ESC50_DIR / "esc50.csv")
    urban = pd.read_csv(DATA_URBAN_DIR / "UrbanSound8K.csv")

    rows = []
    
    # 1. ESC-50 provee varias categorías puras
    esc_only = {"glass_breaking": "rotura_cristal", "crying_baby": "bebe_llorando"}
    for cat, esp_cat in esc_only.items():
        for _, r in esc50[esc50["category"] == cat].iterrows():
            rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID[esp_cat], "label_name": esp_cat, "source": "esc50"})
    
    # Puertas (Knock)
    for _, r in esc50[esc50["category"] == "door_wood_knock"].iterrows():
        rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID["llamar_puerta"], "label_name": "llamar_puerta", "source": "esc50"})

    # 2. Clases compartidas entre ESC-50 y UrbanSound (Perros y Sirenas)
    # Se limitan las muestras de UrbanSound para que no opaquen al resto del dataset (máx 300 sumando ambos).
    for cat_esc, cat_urb, label in [("dog", "dog_bark", "ladrido_perro"), ("siren", "siren", "sirena")]:
        esc_subset = esc50[esc50["category"] == cat_esc]
        for _, r in esc_subset.iterrows():
            rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID[label], "label_name": label, "source": "esc50"})
        urb_subset = urban[urban["class"] == cat_urb].copy()
        if len(urb_subset) > (300 - len(esc_subset)): 
            urb_subset = urb_subset.sample(n=(300 - len(esc_subset)), random_state=SEED)
        for _, r in urb_subset.iterrows():
            rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID[label], "label_name": label, "source": "urban"})

    # 3. Disparos (Gun shot, exclusivo UrbanSound)
    urb_gun = urban[urban["class"] == "gun_shot"].sample(n=300, random_state=SEED) if len(urban[urban["class"] == "gun_shot"]) > 300 else urban[urban["class"] == "gun_shot"]
    for _, r in urb_gun.iterrows():
        rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID["disparo"], "label_name": "disparo", "source": "urban"})

    # 4. Clase BACKGROUND (fondo). Son sonidos "basura" que ignoramos (viento, motor, silencio, etc.)
    # Esto le enseña al modelo a NO alarmarse constantemente.
    esc50_used_cats = ["glass_breaking", "crying_baby", "door_wood_knock", "dog", "siren"]
    esc50_bg = esc50[~esc50["category"].isin(esc50_used_cats)].sample(n=150, random_state=SEED)
    for _, r in esc50_bg.iterrows():
        rows.append({"filepath": str(DATA_ESC50_DIR / "audio" / r["filename"]), "label_id": NAME_TO_ID["fondo"], "label_name": "fondo", "source": "esc50"})

    urban_used_cats = ["dog_bark", "siren", "gun_shot"]
    urban_bg = urban[~urban["class"].isin(urban_used_cats)].sample(n=150, random_state=SEED)
    for _, r in urban_bg.iterrows():
        rows.append({"filepath": str(DATA_URBAN_DIR / f"fold{r['fold']}" / r["slice_file_name"]), "label_id": NAME_TO_ID["fondo"], "label_name": "fondo", "source": "urban"})

    # 5. Gritos (Grabados aparte)
    gritos_dir = DATA_ESC50_DIR / "gritos"
    if gritos_dir.exists():
        gritos = [{"filepath": str(f), "label_id": NAME_TO_ID["grito"], "label_name": "grito", "source": "esc50"} for f in list(gritos_dir.glob("*.wav"))[:300]]
        rows.extend(gritos)

    df = pd.DataFrame(rows)
    
    # ==============================================================================
    # DIVISIONES (SPLITS) - IMPORTANTE
    # ==============================================================================
    # 70% Entrenar, 15% Validar, 15% Test.
    # stratify asegura que la proporción de clases sea idéntica en los tres grupos.
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
    """
    Lee todos los audios, les calcula el espectrograma Log-Mel y los guarda como arrays de Numpy (.npy).
    Esto acelera muchísimo el entrenamiento de la CNN básica porque evita procesar audio en tiempo real.
    """
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
    """
    Toma los audios originales, se los pasa al gigantesco modelo de Google (PANNs CNN14) y guarda los
    'embeddings' (un vector matemático de 2048 dimensiones que resume el significado del audio).
    Aplica también DATA AUGMENTATION intensivo en el grupo de 'train'.
    """
    out_csv = BASE_DIR / "dataset_index_emb.csv"
    if out_csv.exists():
        print(f"[Info] {out_csv.name} ya existe. Saltando embeddings Transfer.")
        return pd.read_csv(out_csv)

    print("Precomputando Embeddings PANNs (con Data Augmentation)...")
    FEATURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 1. Descarga automática de los pesos del modelo PANNs (si no existen localmente)
    ckpt_path = MODELS_DIR / "Cnn14_mAP=0.431.pth"
    if not ckpt_path.exists():
        print("Descargando checkpoint CNN14 (AudioSet)...")
        urllib.request.urlretrieve("https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth", str(ckpt_path))
    
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv", str(csv_path))

    from panns_inference import AudioTagging
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instanciamos PANNs. Nótese que esto carga la red en la memoria (RAM o VRAM de GPU)
    at = AudioTagging(checkpoint_path=str(ckpt_path), device=device)
    
    new_rows = []
    print(f"Procesando audios y aplicando Data Augmentation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filepath = row['filepath']
        label_id = row['label_id']
        label_name = row['label_name']
        split = row['split']
        source = row['source']
        
        try:
            # PANNs exige 32000Hz. Remuestreamos aquí on-the-fly.
            audio_orig, sr = librosa.load(filepath, sr=32000, mono=True)
            
            # --- 1. Audio Original (Extraemos y guardamos su embedding 2048d) ---
            _, emb_orig = at.inference(audio_orig[None, :])
            path_orig = FEATURES_DIR / f"emb_{label_id}_{idx}_orig.npy"
            np.save(path_orig, emb_orig[0])
            
            new_rows.append({
                "filepath": filepath, "label_id": label_id, "label_name": label_name,
                "source": source, "split": split, "embedding_path": str(path_orig),
                "augmentation": "none"
            })
            
            # --- 2. Data Augmentation (OBLIGATORIAMENTE SÓLO AL CONJUNTO TRAIN) ---
            # Si tocamos Val o Test, haríamos trampa falseando los resultados.
            if split == 'train':
                # a) Ruido (+10 dB SNR)
                signal_power = np.mean(audio_orig**2)
                if signal_power > 0:
                    snr_linear = 10 ** (10 / 10) # 10dB
                    noise = np.random.normal(0, np.sqrt(signal_power / snr_linear), len(audio_orig))
                    audio_noise = (audio_orig + noise).astype(np.float32)
                    _, emb_noise = at.inference(audio_noise[None, :])
                    path_noise = FEATURES_DIR / f"emb_{label_id}_{idx}_noise.npy"
                    np.save(path_noise, emb_noise[0])
                    new_rows.append({"filepath": filepath, "label_id": label_id, "label_name": label_name, "source": source, "split": split, "embedding_path": str(path_noise), "augmentation": "noise_10db"})
                
                # b) Pitch Shift Up (Aumentar el tono en 2 semi-tonos)
                audio_pitch = librosa.effects.pitch_shift(y=audio_orig, sr=sr, n_steps=2)
                _, emb_pitch = at.inference(audio_pitch[None, :])
                path_pitch = FEATURES_DIR / f"emb_{label_id}_{idx}_pitch_up.npy"
                np.save(path_pitch, emb_pitch[0])
                new_rows.append({"filepath": filepath, "label_id": label_id, "label_name": label_name, "source": source, "split": split, "embedding_path": str(path_pitch), "augmentation": "pitch_+2"})

                # c) Pitch Shift Down (Bajar el tono en 2 semi-tonos)
                audio_pitch_down = librosa.effects.pitch_shift(y=audio_orig, sr=sr, n_steps=-2)
                _, emb_pitch_down = at.inference(audio_pitch_down[None, :])
                path_pitch_down = FEATURES_DIR / f"emb_{label_id}_{idx}_pitch_down.npy"
                np.save(path_pitch_down, emb_pitch_down[0])
                new_rows.append({"filepath": filepath, "label_id": label_id, "label_name": label_name, "source": source, "split": split, "embedding_path": str(path_pitch_down), "augmentation": "pitch_-2"})

        except Exception:
            pass
            
    # Guardamos este nuevo super-dataset donde 1 solo audio de train se ha multiplicado por 4.
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(out_csv, index=False)
    print(f"[OK] Creado dataset_index_emb con {len(new_df)} embeddings en {out_csv}.")
    return new_df

if __name__ == "__main__":
    # Si se ejecuta este archivo suelto, lanza toda la pipeline de preparación de datos
    df_base = generar_indice_dataset()
    precomputar_features_cnn(df_base.copy())
    precomputar_embeddings_transfer(df_base.copy())
    print("\n[ÉXITO] Datos y características preparados completamente.")
