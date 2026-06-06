"""
dashboard_simple.py
-------------------
Dashboard simplificado de detección binaria de alarma con Streamlit.
Solo responde UNA pregunta: ¿Hay alarma o no? Y si la hay, ¿en qué segundo?

Ejecutar con:  streamlit run dashboard_simple.py
"""

import sys
import json
from pathlib import Path
import warnings

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src.datos_y_configuracion import SR as CONFIG_SR, CLASS_MAP
from src.entrenar_transfer import TransferHead
from panns_inference import AudioTagging

# Configuración de Streamlit
st.set_page_config(page_title="Detector de Alarmas", layout="centered", page_icon="🚨")

@st.cache_resource
def load_models():
    """Carga los modelos una sola vez al iniciar."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    
    model = TransferHead(num_classes=len(CLASS_MAP)).to(device).eval()
    model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
    
    # Umbrales
    theta = 0.85
    thresholds_por_clase = {}
    min_rms_general = 0.025
    min_rms_por_clase = {}
    if (BASE_DIR / "models/threshold.json").exists():
        try:
            with open(BASE_DIR / "models/threshold.json") as f:
                data = json.load(f)
                theta = data.get("theta", 0.85)
                thresholds_por_clase = data.get("thresholds_por_clase", {})
                min_rms_general = data.get("min_rms_general", 0.025)
                min_rms_por_clase = data.get("min_rms_por_clase", {})
        except Exception:
            pass
            
    return panns, model, device, theta, thresholds_por_clase, min_rms_general, min_rms_por_clase

def detectar_alarmas(y, sr):
    """
    Analiza un audio y devuelve una lista de segundos donde se detectó alarma.
    Retorna: lista de tuplas (segundo, confianza).
    """
    panns, model, device, theta, thresholds_por_clase, min_rms_general, min_rms_por_clase = load_models()
    
    ventana, avance = int(5.0 * sr), int(2.5 * sr)
    clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]
    
    if len(y) < ventana: y = np.pad(y, (0, ventana - len(y)))
        
    alertas = []
    
    for i in range(0, max(1, len(y) - ventana + 1), avance):
        bloque = y[i:i+ventana]
        segundo = min((i + ventana) / sr, len(y)/sr)
        
        with torch.no_grad():
            bloque_32k = librosa.resample(bloque, orig_sr=sr, target_sr=32000)
            _, emb = panns.inference(bloque_32k[None, :])
            probs = F.softmax(model(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            clase_pred = CLASS_MAP[np.argmax(probs)]
            
            if clase_pred in clases_alerta:
                max_prob = np.max(probs)
                segunda_prob = np.sort(probs)[-2]
                t_clase = thresholds_por_clase.get(clase_pred, theta)
                
                if (max_prob >= t_clase) or (max_prob >= (t_clase - 0.15) and (max_prob - segunda_prob) >= 0.30):
                    rms = np.sqrt(np.mean(bloque**2))
                    rms_req = min_rms_por_clase.get(clase_pred, min_rms_general)
                    if rms >= rms_req:
                        # Control anti-spam
                        if not any((segundo - s) <= 3.0 for s, _ in alertas):
                            alertas.append((segundo, max_prob))
                
    return alertas

# ==============================================================================
# UI
# ==============================================================================
st.title("🚨 Detector de Alarmas")
st.caption("Sube o graba un audio y el sistema te dirá si detecta alguna alarma y en qué segundo.")

audio_data = st.audio_input("🎤 Grabar") or st.file_uploader("📁 Subir audio", type=["wav", "mp3", "ogg"])

if audio_data and st.button("Analizar", type="primary", use_container_width=True):
    with st.spinner("Analizando..."):
        audio_data.seek(0)
        try: 
            y, sr = sf.read(audio_data)
        except Exception: 
            audio_data.seek(0)
            y, sr = librosa.load(audio_data, sr=CONFIG_SR, mono=True)
            
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        if sr != CONFIG_SR: y = librosa.resample(y, orig_sr=sr, target_sr=CONFIG_SR)
        
        alertas = detectar_alarmas(y, CONFIG_SR)
        st.session_state.update({"audio_y": y, "audio_sr": CONFIG_SR, "alertas": alertas})

# Resultados
if "alertas" in st.session_state:
    y, sr = st.session_state["audio_y"], st.session_state["audio_sr"]
    alertas = st.session_state["alertas"]
    
    st.divider()
    st.audio(y, sample_rate=sr)
    
    if alertas:
        st.error(f"🚨 **ALARMA DETECTADA** — {len(alertas)} alarma(s)")
        for seg, conf in alertas:
            st.warning(f"⚠️ Alarma en el segundo **{seg:.1f}s** (confianza: {conf:.0%})")
    else:
        st.success("✅ **Sin alarma** — No se detectó ninguna alarma en el audio.")
