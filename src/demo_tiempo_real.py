"""
demo_tiempo_real.py
-------------------
Sistema de detección de eventos acústicos en tiempo real usando el micrófono.
Captura ventanas de audio solapadas de manera continua y utiliza el modelo 
Transfer Learning preentrenado (PANNs + TransferHead) para predecir si está ocurriendo una situación de peligro.
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

# Importar configuración y modelo (se añade la ruta al path dinámicamente)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR, CLASS_MAP, compute_logmel
from entrenar_transfer import TransferHead

# ==============================================================================
# CONFIGURACIÓN DEL MICRÓFONO Y LA VENTANA DE AUDIO
# ==============================================================================
# El modelo fue entrenado con audios a 32000Hz (32kHz) de 5 segundos de duración.
SAMPLE_RATE = 32000
VENTANA_SEGUNDOS = 5.0
AVANCE_SEGUNDOS = 2.5 # Esto define un "solapamiento" (overlap) del 50%. En vez de esperar 5s para el siguiente análisis, avanza de 2.5s en 2.5s.

# Conversión de segundos a número de muestras (frames) multiplicando por el Sample Rate
TAMANO_VENTANA = int(VENTANA_SEGUNDOS * SAMPLE_RATE)
TAMANO_AVANCE = int(AVANCE_SEGUNDOS * SAMPLE_RATE)

def main():
    print("="*50)
    print(" SISTEMA DE VIGILANCIA ACÚSTICA EN TIEMPO REAL ")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. CARGAR UMBRAL ÓPTIMO (THETA)
    # Busca en el json de los umbrales precalculados cuál es el óptimo para evitar falsas alarmas.
    umbral_path = BASE_DIR / "models" / "threshold.json"
    if umbral_path.exists():
        try:
            with open(umbral_path) as f:
                data = json.load(f)
                # Lee iterativamente la estructura del json tratando de encontrar el umbral. Por defecto usa 0.5.
                theta = data.get("theta", data.get("selected_threshold", data.get("transfer", {}).get("optimal_threshold", 0.5)))
        except Exception:
            theta = 0.5
    else:
        theta = 0.5
    print(f"[Info] Usando umbral de detección: {theta:.3f}")

    # 2. CARGAR MODELOS NEURONALES
    print("[Info] Cargando redes neuronales...")
    from panns_inference import AudioTagging
    # PANNs (Cnn14) es el extractor de características robustas base
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    
    num_clases = len(CLASS_MAP)
    # TransferHead es la cabeza de clasificación (el MLP) que entrenamos nosotros para nuestras 8 clases específicas
    model = TransferHead(num_classes=num_clases).to(device).eval()
    model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", weights_only=True, map_location=device))
    
    # Directorio para guardar automáticamente pruebas/audios cuando hay una alerta
    alerts_dir = BASE_DIR / "alerts"
    alerts_dir.mkdir(exist_ok=True)
    
    # Cola hilo-segura para recibir audio desde el callback del micrófono
    audio_queue = queue.Queue()
    # Buffer que almacena los últimos 5 segundos en todo momento
    buffer = np.zeros(TAMANO_VENTANA, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        """
        Esta función es llamada asíncronamente por sounddevice cada vez que el micrófono
        llena un bloque del tamaño de 'TAMANO_AVANCE' (2.5s).
        Mete la grabación (canal mono -> [:, 0]) a la cola de procesamiento.
        """
        if status: print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0])

    print("\n[LISTO] Escuchando micrófono... (Pulsa Ctrl+C para salir)\n")
    
    ultimo_tiempo_alerta = {}
    
    try:
        # Se abre el stream de entrada del micrófono de manera ininterrumpida
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=TAMANO_AVANCE):
            while True:
                # 1. ESPERAR AUDIO NUEVO: Se bloquea aquí hasta que pasen 2.5s y el micrófono devuelva datos
                nuevo_audio = audio_queue.get()
                
                # 2. ACTUALIZAR BUFFER (Efecto de Desplazamiento/Solapamiento)
                # Desplazamos la memoria vieja hacia la izquierda, tiramos la parte más antigua y metemos la nueva al final
                buffer = np.roll(buffer, -TAMANO_AVANCE)
                buffer[-TAMANO_AVANCE:] = nuevo_audio

                # 3. INFERENCIA NEURONAL (PREDICCIÓN)
                with torch.no_grad(): # Desactivamos el cálculo de gradientes para ir mucho más rápido en Inferencia
                    # Extraer embedding pre-entrenado (PANNs)
                    # panns espera un tensor de forma (batch_size, longitud_muestras), por eso buffer[None, :]
                    _, emb = panns.inference(buffer[None, :])
                    emb_tensor = torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)
                    
                    # Pasar por nuestra cabeza de clasificación Transfer Learning
                    logits = model(emb_tensor)
                    # Aplicamos softmax para convertir los logits (salidas crudas) en probabilidades sumando 1
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                # 4. ANÁLISIS DEL RESULTADO
                id_pred = np.argmax(probs)      # Índice de la probabilidad más alta
                confianza = probs[id_pred]      # Nivel de certeza (0.0 a 1.0)
                clase_pred = CLASS_MAP[id_pred] # Traducción a string (ej: "grito")

                # Solo nos importa alertar si la clase es de "peligro" o interés. El 'fondo' se ignora.
                clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]
                
                # El guard principal: Debe ser una clase de alerta y la confianza debe superar nuestro THETA umbral
                if clase_pred in clases_alerta and confianza >= theta:
                    tiempo_actual = time.time()
                    
                    # Anti-Spam: Solo disparamos la alerta si han pasado más de 3 segundos desde la última de este mismo tipo
                    if tiempo_actual - ultimo_tiempo_alerta.get(clase_pred, 0) > 3.0:
                        ultimo_tiempo_alerta[clase_pred] = tiempo_actual
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        print(f"[{timestamp}] ⚠️ ALERTA DETECTADA: {clase_pred.upper()} (Confianza: {confianza:.2f})")
                        
                        # 5. REGISTRAR EVIDENCIA
                        # Si salta una alerta, guardamos automáticamente el buffer crudo en un archivo .wav
                        import soundfile as sf
                        sf.write(alerts_dir / f"alerta_{file_ts}_{clase_pred}.wav", buffer, SAMPLE_RATE)
                        
                        # Y de manera opcional, se genera un espectrograma mel (.png) de la alerta como prueba visual
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
                    # No es alerta o no supera el umbral: se mantiene el ciclo de escucha en silencio
                    pass
                    
    except KeyboardInterrupt:
        # Se captura el Control+C del usuario de manera limpia para no lanzar error
        print("\n[FIN] Deteniendo captura en tiempo real.")

if __name__ == "__main__":
    main()
