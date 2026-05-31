"""
dashboard.py
------------
Dashboard simplificado de análisis interactivo con Streamlit.
"""

import sys
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src.datos_y_configuracion import SR as CONFIG_SR, compute_logmel, add_awgn, CLASS_MAP
from src.entrenar_transfer import TransferHead
from src.entrenar_cnn import AudioCNN
from panns_inference import AudioTagging

st.set_page_config(page_title="Dashboard Acústico", layout="wide")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=device)
    model_transfer = TransferHead(num_classes=len(CLASS_MAP)).to(device).eval()
    model_transfer.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
    
    model_cnn = AudioCNN(num_classes=len(CLASS_MAP)).to(device).eval()
    try:
        model_cnn.load_state_dict(torch.load(BASE_DIR / "models/cnn_base_best.pt", map_location=device, weights_only=True))
    except Exception:
        model_cnn = None

    theta = 0.5
    if (BASE_DIR / "models/threshold.json").exists():
        with open(BASE_DIR / "models/threshold.json") as f:
            theta = json.load(f).get("theta", 0.82)
            
    return panns, model_transfer, model_cnn, device, theta

def analyze_audio_buffer(y, sr, snr=None):
    panns, model_transfer, model_cnn, device, theta = load_models()
    if snr and snr != "Limpio": y = add_awgn(y, int(snr))
        
    ventana, avance = int(5.0 * sr), int(2.5 * sr)
    clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]
    if len(y) < ventana: y = np.pad(y, (0, ventana - len(y)))
        
    res_tf, res_cnn, alertas = [], [], []
    for i in range(0, max(1, len(y) - ventana + 1), avance):
        bloque = y[i:i+ventana]
        segundo = min((i + ventana) / sr, len(y)/sr)
        
        # Transfer
        with torch.no_grad():
            _, emb = panns.inference(bloque[None, :])
            probs_tf = F.softmax(model_transfer(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            clase_pred = CLASS_MAP[np.argmax(probs_tf)]
            res_tf.append({"segundo": segundo, "probs": probs_tf})
            
            if clase_pred in clases_alerta and np.max(probs_tf) >= theta:
                if not any(c == clase_pred and (segundo - s) <= 3.0 for s, c, _ in alertas):
                    alertas.append((segundo, clase_pred, np.max(probs_tf)))
                
        # CNN
        if model_cnn:
            sig_22k = librosa.resample(bloque, orig_sr=sr, target_sr=22050)
            max_val = np.max(np.abs(sig_22k))
            if max_val > 0: sig_22k /= max_val
            logmel = torch.from_numpy(compute_logmel(sig_22k)).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                res_cnn.append({"segundo": segundo, "probs": F.softmax(model_cnn(logmel), dim=1).cpu().numpy()[0]})
                
    return res_tf, res_cnn, alertas, y

def plot_probs(res, title):
    if not res: return None
    df = pd.DataFrame([{"Segundo": r["segundo"], "Clase": CLASS_MAP[i], "Probabilidad": p} for r in res for i, p in enumerate(r["probs"])])
    fig = px.line(df, x="Segundo", y="Probabilidad", color="Clase", title=title, markers=True)
    fig.update_layout(yaxis_range=[0, 1.05], height=400)
    return fig

st.title("🎙️ Analizador de Audio en Vivo")

tabs = st.tabs(["Prueba", "Comparativa", "Espectrogramas", "CNN", "Transfer", "Robustez"])

with tabs[0]:
    audio_data = st.audio_input("Grabar") or st.file_uploader("Subir Audio", type=["wav", "mp3", "ogg"])
    if audio_data and st.button("Analizar"):
        with st.spinner("Analizando..."):
            audio_data.seek(0)
            try: y, sr = sf.read(audio_data)
            except Exception: 
                audio_data.seek(0)
                y, sr = librosa.load(audio_data, sr=CONFIG_SR, mono=True)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            if sr != CONFIG_SR: y = librosa.resample(y, orig_sr=sr, target_sr=CONFIG_SR)
            
            st.session_state.update({"audio_y": y, "audio_sr": CONFIG_SR})
            res_tf, res_cnn, alertas, _ = analyze_audio_buffer(y, CONFIG_SR)
            st.session_state.update({"res_tf": res_tf, "res_cnn": res_cnn, "alertas": alertas})
            
            if alertas:
                for seg, cl, conf in alertas: st.warning(f"⚠️ {cl.upper()} a los {seg:.1f}s ({conf:.2f})")
            else: st.success("✅ Todo tranquilo.")

if "audio_y" in st.session_state:
    y, sr = st.session_state["audio_y"], st.session_state["audio_sr"]
    
    with tabs[1]:
        avg_tf, avg_cnn = np.mean([r["probs"] for r in st.session_state["res_tf"]], axis=0), np.mean([r["probs"] for r in st.session_state["res_cnn"]], axis=0)
        df = pd.DataFrame({"Transfer": avg_tf, "CNN": avg_cnn}, index=[CLASS_MAP[i] for i in range(len(CLASS_MAP))])
        df_filt = df[df.max(axis=1) > 0.05].reset_index().rename(columns={"index": "Clase"}).melt(id_vars="Clase", var_name="Modelo", value_name="Prob")
        st.plotly_chart(px.bar(df_filt, x="Clase", y="Prob", color="Modelo", barmode="group", title="Comparativa Media"), width="stretch")
        
    with tabs[2]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Onda")
        librosa.display.specshow(compute_logmel(librosa.resample(y, orig_sr=sr, target_sr=22050)), sr=22050, x_axis='time', ax=ax2)
        ax2.set_title("Espectrograma")
        st.pyplot(fig)
        
    with tabs[3]: st.plotly_chart(plot_probs(st.session_state["res_cnn"], "CNN"), width="stretch")
    with tabs[4]: st.plotly_chart(plot_probs(st.session_state["res_tf"], "Transfer"), width="stretch")
    with tabs[5]:
        snr = st.select_slider("SNR", ["Limpio", "20", "10", "0"])
        if st.button("Aplicar Ruido"):
            res, _, alertas, y_n = analyze_audio_buffer(y, sr, snr)
            if alertas:
                for seg, cl, conf in alertas: st.warning(f"⚠️ {cl.upper()} a los {seg:.1f}s ({conf:.2f})")
            st.plotly_chart(plot_probs(res, f"Transfer con Ruido {snr}dB"), width="stretch")
