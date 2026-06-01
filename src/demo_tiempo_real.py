"""
demo_tiempo_real.py
-------------------
Sistema de detección de eventos acústicos en tiempo real usando el micrófono.
Captura ventanas de audio solapadas y utiliza el modelo Transfer Learning
para predecir y detectar situaciones de peligro.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
import queue

warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import librosa
import matplotlib.pyplot as plt

# Importar configuración y modelo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR, CLASS_MAP, compute_logmel
from entrenar_transfer import TransferHead

# Configuración de Micrófono
SAMPLE_RATE = 32000
VENTANA_SEGUNDOS = 5.0
AVANCE_SEGUNDOS = 2.5 # Solapamiento del 50%
TAMANO_VENTANA = int(VENTANA_SEGUNDOS * SAMPLE_RATE)
TAMANO_AVANCE = int(AVANCE_SEGUNDOS * SAMPLE_RATE)

def main():
    print("="*50)
    print(" SISTEMA DE VIGILANCIA ACÚSTICA EN TIEMPO REAL ")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar umbral óptimo
    umbral_path = BASE_DIR / "models" / "threshold.json"
    if umbral_path.exists():
        try:
            with open(umbral_path) as f:
                data = json.load(f)
                theta = data.get("theta", data.get("selected_threshold", data.get("transfer", {}).get("optimal_threshold", 0.5)))
        except Exception:
            theta = 0.5
    else:
        theta = 0.5
    print(f"[Info] Usando umbral de detección: {theta:.3f}")

    # 2. Cargar modelos
    print("[Info] Cargando redes neuronales...")
    from panns_inference import AudioTagging
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    
    num_clases = len(CLASS_MAP)
    model = TransferHead(num_classes=num_clases).to(device).eval()
    model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", weights_only=True, map_location=device))
    
    alerts_dir = BASE_DIR / "alerts"
    alerts_dir.mkdir(exist_ok=True)
    
    audio_queue = queue.Queue()
    buffer = np.zeros(TAMANO_VENTANA, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0])

    print("\n[LISTO] Escuchando micrófono... (Pulsa Ctrl+C para salir)\n")
    
    ultimo_tiempo_alerta = {}
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=TAMANO_AVANCE):
            while True:
                nuevo_audio = audio_queue.get()
                # Deslizar el buffer
                buffer = np.roll(buffer, -TAMANO_AVANCE)
                buffer[-TAMANO_AVANCE:] = nuevo_audio

                # Inferencia
                with torch.no_grad():
                    # Extraer embedding (panns espera (batch, time))
                    _, emb = panns.inference(buffer[None, :])
                    emb_tensor = torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)
                    # Pasar por la cabeza de Transfer Learning
                    logits = model(emb_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                # Analizar predicción
                id_pred = np.argmax(probs)
                confianza = probs[id_pred]
                clase_pred = CLASS_MAP[id_pred]

                # Solo disparamos si la clase está explícitamente en la lista de alertas
                clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]
                if clase_pred in clases_alerta and confianza >= theta:
                    tiempo_actual = time.time()
                    if tiempo_actual - ultimo_tiempo_alerta.get(clase_pred, 0) > 3.0:
                        ultimo_tiempo_alerta[clase_pred] = tiempo_actual
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        print(f"[{timestamp}] ⚠️ ALERTA DETECTADA: {clase_pred.upper()} (Confianza: {confianza:.2f})")
                        
                        # Guardar clip de audio
                        import soundfile as sf
                        sf.write(alerts_dir / f"alerta_{file_ts}_{clase_pred}.wav", buffer, SAMPLE_RATE)
                        
                        # Guardar espectrograma
                        import librosa.display
                        sig_22k = librosa.resample(buffer, orig_sr=SAMPLE_RATE, target_sr=22050)
                        logmel = compute_logmel(sig_22k)
                        plt.figure(figsize=(6, 4))
                        librosa.display.specshow(logmel, sr=22050, hop_length=512, x_axis='time', cmap='magma')
                        plt.title(f"Alerta: {clase_pred} ({confianza:.2f})")
                        plt.tight_layout()
                        plt.savefig(alerts_dir / f"alerta_{file_ts}_{clase_pred}.png")
                        plt.close()
                else:
                    # No es alerta o no supera el umbral: no hacemos nada
                    pass
                    
    except KeyboardInterrupt:
        print("\n[FIN] Deteniendo captura en tiempo real.")

if __name__ == "__main__":
    main()
