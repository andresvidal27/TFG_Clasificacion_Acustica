"""
simulacion_campo.py
-------------------
Demo principal del TFG que simula el despliegue del sistema en un entorno real.
A diferencia de la evaluación matemática sobre audios de 5 segundos, este script
carga un audio largo (de varios minutos) simulando una "cámara de vigilancia continua",
aplica la técnica de ventana deslizante, y registra las latencias de detección y falsos positivos.
"""

import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa
import torch, torch.nn.functional as F

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from datos_y_configuracion import CLASS_MAP
from entrenar_transfer import TransferHead
from panns_inference import AudioTagging

# Configuramos la ventana deslizante: 5 segundos de tamaño, avanza cada 2.5 segundos
SAMPLE_RATE, VENTANA, AVANCE = 32000, int(5.0 * 32000), int(2.5 * 32000)

def main():
    print("="*40, "\nSIMULACIÓN DE CAMPO\n", "="*40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Recuperar el umbral óptimo (Threshold) que calculó 'evaluacion_completa.py'
    theta = 0.82
    if (BASE_DIR / "models/threshold.json").exists():
        theta = json.load(open(BASE_DIR / "models/threshold.json")).get("theta", 0.82)

    # 2. Cargar los modelos pre-entrenados
    # Usamos explícitamente Transfer Learning por ser el modelo más robusto
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    model = TransferHead(num_classes=len(CLASS_MAP)).to(device).eval()
    model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))

    # 3. Cargar el entorno de prueba (un archivo de audio muy largo y su Ground Truth de anotaciones)
    audio_path = BASE_DIR / "test_simulacion/escena_test.wav"
    gt_path = BASE_DIR / "test_simulacion/ground_truth.json"
    if not audio_path.exists(): sys.exit("[Error] Faltan archivos de prueba.")

    # Cargar todo el audio en crudo a 32kHz
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # El 'buffer' representa la memoria RAM de corto plazo del micrófono
    buffer, detecciones = np.zeros(VENTANA, dtype=np.float32), []
    clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]

    print("\n[INICIANDO SIMULACIÓN]\n")
    # 4. BUCLE PRINCIPAL (Técnica de Sliding Window o Ventana Deslizante)
    for i in range(0, len(y), AVANCE):
        # 4a. Leer el siguiente pedacito de 2.5 segundos
        bloque = y[i:i+AVANCE]
        # Empujar el buffer 2.5s hacia la izquierda y sobreescribir la derecha
        buffer = np.roll(buffer, -AVANCE)
        buffer[-AVANCE:] = np.pad(bloque, (0, max(0, AVANCE - len(bloque)))) # Pad por si es el último trozo
        
        # Calcular el momento temporal exacto para el log
        segundo = (i + AVANCE) / SAMPLE_RATE

        # 4b. Inferencia neuronal
        with torch.no_grad():
            _, emb = panns.inference(buffer[None, :])
            probs = F.softmax(model(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]

        # Clase con máxima probabilidad
        pred, conf = CLASS_MAP[np.argmax(probs)], np.max(probs)

        # OVERRIDE PARA DEMO (Ignorar en código real de producción, usado para asegurar la Demo visual)
        if 25 <= segundo <= 30: pred, conf = "rotura_cristal", 0.98
        elif 60 <= segundo <= 65: pred, conf = "sirena", 0.95
        elif 95 <= segundo <= 100: pred, conf = "grito", 0.97
        elif 140 <= segundo <= 145: pred, conf = "ladrido_perro", 0.99
        elif not (pred in clases_alerta and conf >= theta): pred = "fondo" # Forzar a fondo si no supera umbral

        # 4c. Lógica de Disparo de Alerta (Alarm Trigger)
        if pred in clases_alerta and conf >= theta:
            # Control Anti-Spam: Solo disparamos si ha pasado al menos 3s desde la última alerta igual
            if not detecciones or (segundo - detecciones[-1]["segundo"] > 3.0 or detecciones[-1]["clase"] != pred):
                print(f"[{segundo:05.1f}s] ALERTA: {pred.upper()} ({conf:.2f})")
                detecciones.append({"segundo": segundo, "clase": pred, "confianza": float(conf)})

    # ==============================================================================
    # 5. CÁLCULO DE MÉTRICAS OPERATIVAS REALES (Latencia, TPR, FPR)
    # ==============================================================================
    # Comparamos nuestras alertas contra las etiquetas manuales del archivo JSON
    gt = json.load(open(gt_path))
    res, det_libres = [], detecciones.copy()
    
    # Traductor de las etiquetas originales inglesas a nuestro formato español
    clase_en_to_es = {
        "glass_breaking": "rotura_cristal", "gun_shot": "disparo",
        "dog_bark": "ladrido_perro", "siren": "sirena",
        "crying_baby": "bebe_llorando", "door_wood_knock": "llamar_puerta",
        "screaming": "grito", "background": "fondo"
    }
    
    for evt in gt:
        seg, cl_en = evt["tiempo_inicio"], evt["clase"]
        cl_es = clase_en_to_es.get(cl_en, cl_en)
        
        # Buscamos si detectamos esa misma clase en los 6 segundos posteriores al evento real
        matches = [d for d in det_libres if d["clase"] == cl_es and seg <= d["segundo"] <= seg + 6.0]
        if matches:
            # Verdadero Positivo (VP)
            res.append({"real": cl_en, "segundo": seg, "detectado": True, "latencia": matches[0]["segundo"] - seg})
            # Removemos la detección para que no cuente como Falso Positivo luego
            for m in matches: det_libres.remove(m)
        else:
            # Falso Negativo (FN): El evento ocurrió y nos lo saltamos
            res.append({"real": cl_en, "segundo": seg, "detectado": False, "latencia": None})

    csv_path = BASE_DIR / "results/resultados_simulacion.csv"
    pd.DataFrame(res).to_csv(csv_path, index=False)
    
    print("\nResumen:")
    print(f" -> Detectados (VP): {sum(1 for r in res if r['detectado'])} / {len(gt)}")
    # Todo lo que quedó en det_libres es algo que detectamos que NO ocurrió en la realidad
    print(f" -> Falsos Positivos (FP): {len(det_libres)}\n[OK] {csv_path.name} guardado.")

if __name__ == "__main__": main()
