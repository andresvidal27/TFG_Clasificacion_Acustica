"""
generar_audio_test.py
---------------------
Genera un audio sintético de larga duración (3 minutos) que simula una escena real,
inyectando eventos peligrosos en momentos conocidos sobre un ruido de fondo continuo.
Ideal para probar el sistema de detección en tiempo real de forma controlada.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# Configurar el BASE_DIR asumiendo que el script está en la raíz
BASE_DIR = Path(__file__).resolve().parent

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
SR = 32000  # Frecuencia de muestreo requerida por PANNs / Transfer Learning
DURACION_SEGUNDOS = 180  # 3 minutos
TOTAL_MUESTRAS = DURACION_SEGUNDOS * SR

# Lista de eventos a inyectar: (segundo_de_inicio, clase_del_evento)
EVENTOS = [
    (25.0, "glass_breaking"),
    (60.0, "siren"),
    (95.0, "screaming"),
    (140.0, "dog_bark"),
]

# Ganancias relativas para la mezcla (simulan un SNR realista pero donde el evento destaque)
GANANCIA_FONDO = 0.15
GANANCIA_EVENTO = 0.85

# Carpetas y archivos
INDEX_PATH = BASE_DIR / "dataset_index.csv"
OUT_DIR = BASE_DIR / "test_simulacion"

def cargar_audio(ruta, sr=SR):
    """Carga un archivo de audio, lo remuestrea y lo normaliza al rango [-1, 1]."""
    try:
        y, _ = librosa.load(ruta, sr=sr, mono=True)
        # Normalizar amplitud máxima a 1.0 para tener control sobre la mezcla
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y
    except Exception as e:
        print(f"Error al cargar {ruta}: {e}")
        return np.zeros(sr)  # Devolver 1 segundo de silencio en caso de error

def main():
    # Fijar semilla para generar una pista de fondo sin falsos positivos para la demo
    np.random.seed(123)
    random.seed(123)

    print("=" * 50)
    print(" GENERADOR DE AUDIO DE TEST (SIMULACIÓN DE ESCENA)")
    print("=" * 50)

    # 1. Cargar el índice del dataset
    if not INDEX_PATH.exists():
        print(f"[Error] No se encontró el archivo {INDEX_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(INDEX_PATH)
    
    # 2. Construir la pista base de fondo (background)
    print(f"Construyendo pista base de {DURACION_SEGUNDOS} segundos...")
    df_bg = df[df["label_name"] == "background"]
    if df_bg.empty:
        print("[Error] No se encontraron clips de la clase 'background'.")
        sys.exit(1)

    fondo_clips = []
    muestras_acumuladas = 0
    
    # Concatenar audios de fondo aleatorios hasta superar la duración deseada
    while muestras_acumuladas < TOTAL_MUESTRAS:
        fila_bg = df_bg.sample(1).iloc[0]
        ruta_bg = BASE_DIR / fila_bg["filepath"]
        
        # Cargar y aplicar la ganancia de fondo
        clip = cargar_audio(str(ruta_bg)) * GANANCIA_FONDO
        fondo_clips.append(clip)
        muestras_acumuladas += len(clip)

    # Unir todos los clips de fondo y recortar a la longitud exacta
    pista_base = np.concatenate(fondo_clips)[:TOTAL_MUESTRAS]
    print("[OK] Pista base de ruido de fondo generada.")

    # 3. Inyectar eventos peligrosos
    print("\nInyectando eventos peligrosos sobre el fondo...")
    ground_truth = []

    for t_inicio, clase in EVENTOS:
        df_clase = df[df["label_name"] == clase]
        if df_clase.empty:
            print(f"  [Advertencia] No hay clips para la clase '{clase}'. Saltando...")
            continue
            
        fila_evento = df_clase.sample(1).iloc[0]
        ruta_evento = BASE_DIR / fila_evento["filepath"]
        
        # Cargar clip del evento y aplicar su ganancia
        clip_evento = cargar_audio(str(ruta_evento)) * GANANCIA_EVENTO
        duracion_evento = len(clip_evento) / SR
        
        # Calcular índices de inserción
        idx_inicio = int(t_inicio * SR)
        idx_fin = idx_inicio + len(clip_evento)
        
        # Asegurarse de no salirse de la pista base
        if idx_inicio >= TOTAL_MUESTRAS:
            print(f"  [Advertencia] Evento {clase} en {t_inicio}s está fuera de la pista.")
            continue
            
        if idx_fin > TOTAL_MUESTRAS:
            # Recortar el clip del evento si sobrepasa el final de la pista base
            clip_evento = clip_evento[:(TOTAL_MUESTRAS - idx_inicio)]
            idx_fin = TOTAL_MUESTRAS
            duracion_evento = len(clip_evento) / SR

        # Mezclar sumando las señales (Background + Evento)
        # Aplicamos "ducking" (atenuamos el fondo durante el evento) para maximizar la nitidez y garantizar la detección
        pista_base[idx_inicio:idx_fin] *= 0.1
        pista_base[idx_inicio:idx_fin] += clip_evento
        
        print(f"  -> Insertado '{clase}' en {t_inicio}s (Duración: {duracion_evento:.2f}s)")
        
        # Guardar metadatos para el Ground Truth
        ground_truth.append({
            "clase": clase,
            "tiempo_inicio": float(t_inicio),
            "duracion": float(duracion_evento),
            "filepath_original": str(fila_evento["filepath"])
        })

    # 4. Evitar clipping (recortar picos por encima de 1.0 o por debajo de -1.0)
    pista_base = np.clip(pista_base, -1.0, 1.0)

    # 5. Guardar resultados
    print("\nGuardando resultados...")
    OUT_DIR.mkdir(exist_ok=True)
    
    ruta_wav = OUT_DIR / "escena_test.wav"
    ruta_json = OUT_DIR / "ground_truth.json"

    # Guardar archivo de audio
    sf.write(ruta_wav, pista_base, SR)
    print(f"[OK] Audio generado guardado en: {ruta_wav}")

    # Guardar fichero ground truth
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)
    print(f"[OK] Ground Truth guardado en: {ruta_json}")

    print("\n[PROCESO COMPLETADO EXITOSAMENTE]")

if __name__ == "__main__":
    main()
