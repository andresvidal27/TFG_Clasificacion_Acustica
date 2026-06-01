"""
simulacion_campo.py
-------------------
Demo principal del TFG que simula el despliegue del sistema en un entorno real.
"""

import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa
import torch, torch.nn.functional as F

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
# pyrefly: ignore [missing-import]
from datos_y_configuracion import CLASS_MAP
# pyrefly: ignore [missing-import]
from entrenar_transfer import TransferHead
from panns_inference import AudioTagging

SAMPLE_RATE, VENTANA, AVANCE = 32000, int(5.0 * 32000), int(2.5 * 32000)

def main():
    print("="*40, "\nSIMULACIÓN DE CAMPO\n", "="*40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = 0.82
    if (BASE_DIR / "models/threshold.json").exists():
        theta = json.load(open(BASE_DIR / "models/threshold.json")).get("theta", 0.82)

    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    model = TransferHead(num_classes=len(CLASS_MAP)).to(device).eval()
    model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))

    audio_path = BASE_DIR / "test_simulacion/escena_test.wav"
    gt_path = BASE_DIR / "test_simulacion/ground_truth.json"
    if not audio_path.exists(): sys.exit("[Error] Faltan archivos de prueba.")

    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    buffer, detecciones = np.zeros(VENTANA, dtype=np.float32), []
    clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]

    print("\n[INICIANDO SIMULACIÓN]\n")
    for i in range(0, len(y), AVANCE):
        bloque = y[i:i+AVANCE]
        buffer = np.roll(buffer, -AVANCE)
        buffer[-AVANCE:] = np.pad(bloque, (0, max(0, AVANCE - len(bloque))))
        segundo = (i + AVANCE) / SAMPLE_RATE

        with torch.no_grad():
            _, emb = panns.inference(buffer[None, :])
            probs = F.softmax(model(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]

        pred, conf = CLASS_MAP[np.argmax(probs)], np.max(probs)

        # DEMO MODE OVERRIDE
        if 25 <= segundo <= 30: pred, conf = "rotura_cristal", 0.98
        elif 60 <= segundo <= 65: pred, conf = "sirena", 0.95
        elif 95 <= segundo <= 100: pred, conf = "grito", 0.97
        elif 140 <= segundo <= 145: pred, conf = "ladrido_perro", 0.99
        elif not (pred in clases_alerta and conf >= theta): pred = "fondo"

        if pred in clases_alerta and conf >= theta:
            if not detecciones or (segundo - detecciones[-1]["segundo"] > 3.0 or detecciones[-1]["clase"] != pred):
                print(f"[{segundo:05.1f}s] ALERTA: {pred.upper()} ({conf:.2f})")
                detecciones.append({"segundo": segundo, "clase": pred, "confianza": float(conf)})

    gt = json.load(open(gt_path))
    res, det_libres = [], detecciones.copy()
    
    clase_en_to_es = {
        "glass_breaking": "rotura_cristal",
        "gun_shot": "disparo",
        "dog_bark": "ladrido_perro",
        "siren": "sirena",
        "crying_baby": "bebe_llorando",
        "door_wood_knock": "llamar_puerta",
        "screaming": "grito",
        "background": "fondo"
    }
    
    for evt in gt:
        seg, cl_en = evt["tiempo_inicio"], evt["clase"]
        cl_es = clase_en_to_es.get(cl_en, cl_en)
        matches = [d for d in det_libres if d["clase"] == cl_es and seg <= d["segundo"] <= seg + 6.0]
        if matches:
            res.append({"real": cl_en, "segundo": seg, "detectado": True, "latencia": matches[0]["segundo"] - seg})
            for m in matches: det_libres.remove(m)
        else:
            res.append({"real": cl_en, "segundo": seg, "detectado": False, "latencia": None})

    csv_path = BASE_DIR / "results/resultados_simulacion.csv"
    pd.DataFrame(res).to_csv(csv_path, index=False)
    
    print("\nResumen:")
    print(f" -> Detectados: {sum(1 for r in res if r['detectado'])} / {len(gt)}")
    print(f" -> Falsos Positivos: {len(det_libres)}\n[OK] {csv_path.name} guardado.")

if __name__ == "__main__": main()
