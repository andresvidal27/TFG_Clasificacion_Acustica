"""
dashboard.py
------------
Dashboard simplificado de análisis interactivo con Streamlit.
Esta es la interfaz gráfica (UI) base ORIGINAL (sin el sistema de reentrenamiento por feedback).
Permite grabar audio, subir archivos locales, y ver cómo los modelos (CNN y Transfer Learning)
analizan las ondas segundo a segundo.
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

# Configuración base de la web de Streamlit
st.set_page_config(page_title="Dashboard Acústico", layout="wide")

@st.cache_resource
def load_models():
    """
    Función cacheada para cargar los modelos en memoria de la GPU (o CPU) una sola vez
    al iniciar el dashboard. Evita latencias de lectura de disco.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Extractor base PANNs
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    
    # 2. Cabeza de Transfer Learning
    model_transfer = TransferHead(num_classes=len(CLASS_MAP)).to(device).eval()
    model_transfer.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
    
    # 3. Modelo CNN (por si queremos comparar)
    model_cnn = AudioCNN(num_classes=len(CLASS_MAP)).to(device).eval()
    try:
        model_cnn.load_state_dict(torch.load(BASE_DIR / "models/cnn_base_best.pt", map_location=device, weights_only=True))
    except Exception:
        model_cnn = None

    # 4. Umbral (Theta) para las alertas
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
            
    return panns, model_transfer, model_cnn, device, theta, thresholds_por_clase, min_rms_general, min_rms_por_clase

def analyze_audio_buffer(y, sr, snr=None):
    """
    Toma una señal cruda de audio de cualquier tamaño, la divide en ventanas de 5s 
    solapadas, y extrae las predicciones temporales completas.
    """
    panns, model_transfer, model_cnn, device, theta, thresholds_por_clase, min_rms_general, min_rms_por_clase = load_models()
    
    # Si el usuario selecciona añadir ruido artificial desde la interfaz
    if snr and snr != "Limpio": y = add_awgn(y, int(snr))
        
    ventana, avance = int(5.0 * sr), int(2.5 * sr)
    clases_alerta = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito"]
    
    # Padding de seguridad por si el audio es muy corto
    if len(y) < ventana: y = np.pad(y, (0, ventana - len(y)))
        
    res_tf, res_cnn, alertas = [], [], []
    
    # Bucle de Sliding Window
    for i in range(0, max(1, len(y) - ventana + 1), avance):
        bloque = y[i:i+ventana]
        segundo = min((i + ventana) / sr, len(y)/sr)
        
        # --- Transfer Learning Inferencia ---
        with torch.no_grad():
            bloque_32k = librosa.resample(bloque, orig_sr=sr, target_sr=32000)
            _, emb = panns.inference(bloque_32k[None, :])
            probs_tf = F.softmax(model_transfer(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            clase_pred = CLASS_MAP[np.argmax(probs_tf)]
            res_tf.append({"segundo": segundo, "probs": probs_tf})
            
            # Lógica de Alarmas
            if clase_pred in clases_alerta:
                max_prob = np.max(probs_tf)
                segunda_prob = np.sort(probs_tf)[-2]
                
                # Regla Estricta por clase para ajustar sensibilidad
                t_clase = thresholds_por_clase.get(clase_pred, theta)
                # a) O bien supera el t_clase (ej 0.93 para perro/sirena, 0.82 para el resto)
                # b) O bien supera (t_clase - 0.15) pero la diferencia respecto a la segunda clase es enorme (muy seguro)
                if (max_prob >= t_clase) or (max_prob >= (t_clase - 0.15) and (max_prob - segunda_prob) >= 0.30):
                    # Filtros de volumen general y por clase para requerir más potencia acústica
                    rms = np.sqrt(np.mean(bloque**2))
                    rms_req = min_rms_por_clase.get(clase_pred, min_rms_general)
                    if rms < rms_req:
                        continue
                        
                    # Control anti-spam
                    if not any(c == clase_pred and (segundo - s) <= 3.0 for s, c, _ in alertas):
                        alertas.append((segundo, clase_pred, max_prob))
                
        # --- CNN Base Inferencia ---
        if model_cnn:
            sig_22k = librosa.resample(bloque, orig_sr=sr, target_sr=22050)
            max_val = np.max(np.abs(sig_22k))
            if max_val > 0: sig_22k /= max_val
            # Extraer espectrograma en vivo
            logmel = torch.from_numpy(compute_logmel(sig_22k)).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                res_cnn.append({"segundo": segundo, "probs": F.softmax(model_cnn(logmel), dim=1).cpu().numpy()[0]})
                
    return res_tf, res_cnn, alertas, y

def plot_probs(res, title):
    """Genera gráficas interactivas con Plotly para ver la línea temporal de probabilidades."""
    if not res: return None
    # Convertimos los resultados en un DataFrame largo (tidy format) ideal para gráficos de serie temporal
    df = pd.DataFrame([{"Segundo": r["segundo"], "Clase": CLASS_MAP[i], "Probabilidad": p} 
                       for r in res for i, p in enumerate(r["probs"])])
    fig = px.line(df, x="Segundo", y="Probabilidad", color="Clase", title=title, markers=True)
    fig.update_layout(yaxis_range=[0, 1.05], height=400)
    return fig

# ==============================================================================
# UI MAIN (PÁGINA STREAMLIT)
# ==============================================================================
st.title("🎙️ Analizador de Audio en Vivo")

# Pestañas de navegación
tabs = st.tabs(["Prueba", "Comparativa", "Espectrogramas", "CNN", "Transfer", "Robustez"])

with tabs[0]:
    # Streamlit maneja automáticamente inputs de micrófono y archivos
    audio_data = st.audio_input("Grabar") or st.file_uploader("Subir Audio", type=["wav", "mp3", "ogg"])
    
    if audio_data and st.button("Analizar"):
        with st.spinner("Analizando..."):
            audio_data.seek(0)
            try: 
                y, sr = sf.read(audio_data)
            except Exception: 
                audio_data.seek(0)
                y, sr = librosa.load(audio_data, sr=CONFIG_SR, mono=True)
                
            if len(y.shape) > 1: y = np.mean(y, axis=1) # Estéreo a mono
            if sr != CONFIG_SR: y = librosa.resample(y, orig_sr=sr, target_sr=CONFIG_SR)
            
            # Guardamos resultados en la sesión (memoria caché local) para poder cambiar de pestaña sin re-analizar
            st.session_state.update({"audio_y": y, "audio_sr": CONFIG_SR})
            res_tf, res_cnn, alertas, _ = analyze_audio_buffer(y, CONFIG_SR)
            st.session_state.update({"res_tf": res_tf, "res_cnn": res_cnn, "alertas": alertas})
            
            if alertas:
                for seg, cl, conf in alertas: st.warning(f"⚠️ {cl.upper()} a los {seg:.1f}s ({conf:.2f})")
            else: 
                st.success("✅ Todo tranquilo.")

# Si hay algo cargado en la memoria de la sesión, mostramos las pestañas secundarias
if "audio_y" in st.session_state:
    y, sr = st.session_state["audio_y"], st.session_state["audio_sr"]
    
    with tabs[1]: # Pestaña Comparativa (Histogramas de Probabilidades exactas)
        st.audio(y, sample_rate=sr)
        
        alertas = st.session_state.get("alertas", [])
        opciones = ["Media Global", "Selección Manual"] + [f"{cl.upper()} a los {seg:.1f}s" for seg, cl, _ in alertas]
        opcion_sel = st.selectbox("Seleccionar Instante", opciones)
        
        seg = None
        if opcion_sel == "Media Global":
            # Promedio de toda la pista de audio
            probs_tf = np.mean([r["probs"] for r in st.session_state["res_tf"]], axis=0) if st.session_state["res_tf"] else np.zeros(len(CLASS_MAP))
            probs_cnn = np.mean([r["probs"] for r in st.session_state["res_cnn"]], axis=0) if st.session_state["res_cnn"] else np.zeros(len(CLASS_MAP))
            title = "Comparativa Media Global"
        elif opcion_sel == "Selección Manual":
            duracion = float(len(y) / sr)
            seg = st.slider("Seleccionar segundo", 0.0, duracion, 0.0, step=0.5)
            title = f"Probabilidades en {seg:.1f}s"
        else:
            # Buscamos la alerta correspondiente
            idx = opciones.index(opcion_sel) - 2
            seg = alertas[idx][0]
            title = f"Probabilidades en {opcion_sel}"

        if seg is not None:
            probs_tf = min(st.session_state["res_tf"], key=lambda r: abs(r["segundo"] - seg))["probs"] if st.session_state["res_tf"] else np.zeros(len(CLASS_MAP))
            probs_cnn = min(st.session_state["res_cnn"], key=lambda r: abs(r["segundo"] - seg))["probs"] if st.session_state["res_cnn"] else np.zeros(len(CLASS_MAP))

        # Crear barras comparando qué dice la CNN contra qué dice Transfer Learning
        df = pd.DataFrame({"Transfer": probs_tf, "CNN": probs_cnn}, index=[CLASS_MAP[i] for i in range(len(CLASS_MAP))])
        df_filt = df.reset_index().rename(columns={"index": "Clase"}).melt(id_vars="Clase", var_name="Modelo", value_name="Prob")
        st.plotly_chart(px.bar(df_filt, x="Clase", y="Prob", color="Modelo", barmode="group", title=title), width="stretch")
        
    with tabs[2]: # Espectrograma Visual 2D
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Onda")
        librosa.display.specshow(compute_logmel(librosa.resample(y, orig_sr=sr, target_sr=22050)), sr=22050, x_axis='time', ax=ax2)
        ax2.set_title("Espectrograma")
        st.pyplot(fig)
        
    with tabs[3]: # Timeline de la CNN
        fig_cnn = plot_probs(st.session_state.get("res_cnn", []), "CNN")
        if fig_cnn:
            st.plotly_chart(fig_cnn, use_container_width=True)
        else:
            st.info("No hay datos generados para la CNN.")
    with tabs[4]: # Timeline de Transfer Learning
        fig_tf = plot_probs(st.session_state.get("res_tf", []), "Transfer")
        if fig_tf:
            st.plotly_chart(fig_tf, use_container_width=True)
        else:
            st.info("No hay datos generados para Transfer Learning.")
    with tabs[5]: # Simulación de Ruido en Vivo
        snr = st.select_slider("SNR", ["Limpio", "20", "10", "0"])
        if st.button("Aplicar Ruido"):
            res, _, alertas, y_n = analyze_audio_buffer(y, sr, snr)
            if alertas:
                for seg, cl, conf in alertas: st.warning(f"⚠️ {cl.upper()} a los {seg:.1f}s ({conf:.2f})")
            
            fig_noise = plot_probs(res, f"Transfer con Ruido {snr}dB")
            if fig_noise:
                st.plotly_chart(fig_noise, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar la gráfica.")
