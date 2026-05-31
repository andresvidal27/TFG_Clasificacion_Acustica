"""
dashboard.py
------------
Dashboard de análisis interactivo con Streamlit.
Permite visualizar de forma interactiva y dinámica el análisis en tiempo real 
de los audios capturados o subidos, comparando los modelos entrenados.
"""

import sys
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Para poder importar las funciones del proyecto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src.datos_y_configuracion import SR as CONFIG_SR, compute_logmel, add_awgn, CLASS_MAP
from src.entrenar_transfer import TransferHead
from src.entrenar_cnn import AudioCNN
from panns_inference import AudioTagging

st.set_page_config(page_title="Dashboard Vigilancia Acústica en Vivo", layout="wide")

# ── Funciones de carga de datos ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modelo Transfer Learning (PANNs + Head)
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=device)
    num_clases = len(CLASS_MAP)
    model_transfer = TransferHead(num_classes=num_clases).to(device).eval()
    model_transfer.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
    
    # Modelo CNN Base
    model_cnn = AudioCNN(num_classes=num_clases).to(device).eval()
    try:
        model_cnn.load_state_dict(torch.load(BASE_DIR / "models/audio_cnn_noise_best.pt", map_location=device, weights_only=True))
    except Exception:
        try:
            model_cnn.load_state_dict(torch.load(BASE_DIR / "models/audio_cnn_best.pt", map_location=device, weights_only=True))
        except Exception:
            model_cnn = None # Si no hay CNN base, no fallar

    umbral_path = BASE_DIR / "models" / "threshold.json"
    theta = 0.5
    if umbral_path.exists():
        with open(umbral_path) as f:
            data = json.load(f)
            theta = data.get("theta", data.get("selected_threshold", 0.82))
            
    return panns, model_transfer, model_cnn, device, theta

def analyze_audio_buffer(y, sr, aplicar_ruido_snr=None):
    """
    Analiza el audio con ventana deslizante usando los modelos cargados.
    Si aplicar_ruido_snr tiene un valor (ej. 10, 0), inyecta ruido antes del análisis.
    Devuelve las probabilidades en el tiempo y las alertas.
    """
    panns, model_transfer, model_cnn, device, theta = load_models()
    
    if aplicar_ruido_snr is not None and aplicar_ruido_snr != "Limpio":
        y = add_awgn(y, int(aplicar_ruido_snr))
        
    ventana = int(5.0 * sr)
    avance = int(2.5 * sr)
    clases_alerta = ["glass_breaking", "gun_shot", "dog_bark", "siren", "crying_baby", "door_knock", "screaming"]
    
    if len(y) < ventana:
        y = np.pad(y, (0, ventana - len(y)))
        
    resultados_tf = []
    resultados_cnn = []
    alertas_detectadas = []
    
    for i in range(0, max(1, len(y) - ventana + 1), avance):
        bloque = y[i:i+ventana]
        segundo = min((i + ventana) / sr, len(y)/sr)
        
        # 1. Inferencia Transfer Learning
        with torch.no_grad():
            _, emb = panns.inference(bloque[None, :])
            emb_tensor = torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)
            logits_tf = model_transfer(emb_tensor)
            probs_tf = F.softmax(logits_tf, dim=1).cpu().numpy()[0]
            
            id_pred_tf = np.argmax(probs_tf)
            confianza_tf = probs_tf[id_pred_tf]
            clase_pred_tf = CLASS_MAP[id_pred_tf]
            
            resultados_tf.append({"segundo": segundo, "probs": probs_tf})
            
            if clase_pred_tf in clases_alerta and confianza_tf >= theta:
                alertas_detectadas.append((segundo, clase_pred_tf, float(confianza_tf)))
                
                alerts_dir = BASE_DIR / "alerts"
                if alerts_dir.exists():
                    ruido_str = "" if (aplicar_ruido_snr is None or aplicar_ruido_snr == "Limpio") else f"_SNR{aplicar_ruido_snr}dB"
                    
                    archivo_wav = alerts_dir / f"alerta_{segundo:05.1f}s_{clase_pred_tf}{ruido_str}.wav"
                    sf.write(archivo_wav, bloque, sr)
                    
                    archivo_img = alerts_dir / f"alerta_{segundo:05.1f}s_{clase_pred_tf}{ruido_str}.png"
                    sig_22k = librosa.resample(bloque, orig_sr=sr, target_sr=22050)
                    logmel = compute_logmel(sig_22k)
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(logmel, sr=22050, hop_length=512, x_axis='time', cmap='magma')
                    plt.title(f"ALERTA: {clase_pred_tf.upper()} ({confianza_tf:.2f})")
                    plt.tight_layout()
                    plt.savefig(archivo_img)
                    plt.close()
                
        # 2. Inferencia CNN Base
        if model_cnn is not None:
            # La CNN espera el logmel spectrogram de 22050 Hz en bloque (1, 1, 128, 216)
            sig_22k = librosa.resample(bloque, orig_sr=sr, target_sr=22050)
            
            # Normalizar igual que en el entrenamiento
            max_val = np.max(np.abs(sig_22k))
            if max_val > 0:
                sig_22k = sig_22k / max_val
                
            logmel = compute_logmel(sig_22k)
            # Normalizar logmel de manera similar a preprocessing si es necesario
            logmel_tensor = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_cnn = model_cnn(logmel_tensor)
                probs_cnn = F.softmax(logits_cnn, dim=1).cpu().numpy()[0]
                resultados_cnn.append({"segundo": segundo, "probs": probs_cnn})
                
    return resultados_tf, resultados_cnn, alertas_detectadas, y

def procesar_y_guardar_estado(audio_bytes):
    """Lee el audio subido, extrae la forma de onda y la guarda en la sesión."""
    import shutil
    alerts_dir = BASE_DIR / "alerts"
    if alerts_dir.exists():
        shutil.rmtree(alerts_dir)
    alerts_dir.mkdir(parents=True, exist_ok=True)
    
    audio_bytes.seek(0)
    try:
        y, sr = sf.read(audio_bytes)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != CONFIG_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=CONFIG_SR)
    except Exception:
        audio_bytes.seek(0)
        y, sr = librosa.load(audio_bytes, sr=CONFIG_SR, mono=True)
        
    if len(y) == 0:
        raise ValueError("No se detectaron muestras de audio.")
        
    st.session_state["audio_y"] = y
    st.session_state["audio_sr"] = CONFIG_SR
    
    # Realizar análisis base (limpio)
    res_tf, res_cnn, alertas, _ = analyze_audio_buffer(y, CONFIG_SR, "Limpio")
    st.session_state["res_tf"] = res_tf
    st.session_state["res_cnn"] = res_cnn
    st.session_state["alertas"] = alertas


# ── Funciones Gráficas ───────────────────────────────────────────────────────

def plot_probs_over_time(resultados, title, key_prefix):
    """Genera un gráfico de líneas con las probabilidades a lo largo del tiempo."""
    if not resultados:
        return None
        
    df_data = []
    for res in resultados:
        seg = res["segundo"]
        for i, p in enumerate(res["probs"]):
            df_data.append({"Segundo": seg, "Clase": CLASS_MAP[i], "Probabilidad": p})
            
    df = pd.DataFrame(df_data)
    
    clases_alerta = ["glass_breaking", "gun_shot", "dog_bark", "siren", "crying_baby", "door_knock", "screaming", "background"]
    
    clases_sel = st.multiselect(
        "Filtrar clases en la gráfica:", 
        options=clases_alerta, 
        default=clases_alerta,
        key=f"ms_{key_prefix}"
    )
    
    if not clases_sel:
        st.warning("Selecciona al menos una clase para visualizar.")
        return None
        
    df = df[df["Clase"].isin(clases_sel)]
    
    fig = px.line(df, x="Segundo", y="Probabilidad", color="Clase", title=title, markers=True)
    fig.update_layout(yaxis_range=[0, 1.05], height=500)
    return fig

# ── Interfaz Principal ───────────────────────────────────────────────────────

st.title("🎙️ Dashboard Analizador de Audio en Vivo")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Prueba en Vivo",
    "Comparativa de Modelos", 
    "Espectrogramas", 
    "Resultados CNN", 
    "Resultados Transfer", 
    "Análisis de Robustez"
])

# ── PESTAÑA 1: Prueba en Vivo ───────────────────────────────────────────
with tab1:
    st.header("🔴 Prueba en Vivo: Micrófono y Archivos")
    st.write("Sube un archivo de audio o graba directamente con tu micrófono para detectar si hay situaciones de alerta.")
    
    opcion = st.radio("Selecciona la fuente de audio:", ["🎤 Grabar con Micrófono", "📁 Subir Archivo"], horizontal=True)
    
    audio_data = None
    
    if opcion == "🎤 Grabar con Micrófono":
        audio_data = st.audio_input("Haz clic en el micrófono para empezar a grabar")
    else:
        audio_data = st.file_uploader("Sube tu archivo de audio (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
        
    if audio_data is not None:
        st.audio(audio_data)
        if st.button("Analizar Audio", type="primary"):
            with st.spinner("Procesando y ejecutando redes neuronales sobre el audio..."):
                try:
                    procesar_y_guardar_estado(audio_data)
                    alertas = st.session_state.get("alertas", [])
                    if alertas:
                        st.error(f"¡PELIGRO! Se han detectado {len(alertas)} alertas en este audio.")
                        for seg, clase, conf in alertas:
                            st.warning(f"⚠️ **{clase.upper()}** detectado cerca del segundo {seg:.1f} (Confianza: {conf:.2f})")
                    else:
                        st.success("✅ Todo tranquilo. No se detectaron alertas de peligro en el audio.")
                except ValueError as ve:
                    if str(ve) == "No se detectaron muestras de audio.":
                        st.info("No se detectaron muestras de audio y ya.")
                    else:
                        st.error(f"Ocurrió un error al procesar el audio: {ve}")
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar el audio: {e}")

# Comprobamos si hay audio analizado para renderizar las demás pestañas
audio_cargado = "audio_y" in st.session_state

# ── PESTAÑA 2: Comparativa de Modelos ────────────────────────────────────────
with tab2:
    st.header("Comparativa de Modelos para el Audio Actual")
    if not audio_cargado:
        st.info("Sube o graba un audio en la pestaña 'Prueba en Vivo' y pulsa en 'Analizar Audio'.")
    else:
        res_tf = st.session_state["res_tf"]
        res_cnn = st.session_state["res_cnn"]
        
        if not res_cnn:
            st.warning("No se ha cargado el modelo CNN base.")
        else:
            # Calculamos la probabilidad media de cada clase a lo largo de todo el audio
            avg_probs_tf = np.mean([r["probs"] for r in res_tf], axis=0)
            avg_probs_cnn = np.mean([r["probs"] for r in res_cnn], axis=0)
            
            df_comp = pd.DataFrame({
                "Clase": [CLASS_MAP[i] for i in range(len(CLASS_MAP))],
                "Prob_Transfer": avg_probs_tf,
                "Prob_CNN": avg_probs_cnn
            })
            
            # Filtramos solo las clases que superan un mínimo en al menos uno de los modelos para no saturar
            df_comp = df_comp[(df_comp["Prob_Transfer"] > 0.05) | (df_comp["Prob_CNN"] > 0.05)]
            
            if df_comp.empty:
                st.write("Ninguna clase superó el umbral de 5% de probabilidad.")
            else:
                fig = go.Figure(data=[
                    go.Bar(name='Transfer Learning (PANNs)', x=df_comp['Clase'], y=df_comp['Prob_Transfer'], marker_color='#3b82f6'),
                    go.Bar(name='CNN (Entrenada desde cero)', x=df_comp['Clase'], y=df_comp['Prob_CNN'], marker_color='#94a3b8')
                ])
                fig.update_layout(barmode='group', title="Probabilidad Media Asignada a las Clases Destacadas", yaxis_title="Probabilidad")
                st.plotly_chart(fig, use_container_width=True)

# ── PESTAÑA 3: Explorador de Espectrogramas ──────────────────────────────────
with tab3:
    st.header("Forma de Onda y Espectrograma del Audio Actual")
    if not audio_cargado:
        st.info("Sube o graba un audio en la pestaña 'Prueba en Vivo' y pulsa en 'Analizar Audio'.")
    else:
        y = st.session_state["audio_y"]
        sr = st.session_state["audio_sr"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Forma de Onda (Waveform)")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#3b82f6")
            ax_wave.set_title("Amplitud de la señal")
            st.pyplot(fig_wave)
            
        with col2:
            st.subheader("Espectrograma Log-Mel")
            # Para visualizar mejor, pasamos a 22kHz como usa la CNN base
            sig_22k = librosa.resample(y, orig_sr=sr, target_sr=22050)
            logmel = compute_logmel(sig_22k)
            
            fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(logmel, sr=22050, hop_length=512, x_axis='time', cmap='magma', ax=ax_mel)
            ax_mel.set_title("Energía por Banda de Frecuencia")
            fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
            st.pyplot(fig_mel)

# ── PESTAÑA 4: Modelo CNN ────────────────────────────────────────────────────
with tab4:
    st.header("Análisis a lo Largo del Tiempo: CNN (Desde Cero)")
    if not audio_cargado:
        st.info("Sube o graba un audio en la pestaña 'Prueba en Vivo' y pulsa en 'Analizar Audio'.")
    else:
        res_cnn = st.session_state["res_cnn"]
        if not res_cnn:
            st.warning("No se encuentra el modelo CNN base.")
        else:
            fig = plot_probs_over_time(res_cnn, "Probabilidades de la CNN a lo Largo del Audio", "cnn")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# ── PESTAÑA 5: Modelo Transfer Learning ──────────────────────────────────────
with tab5:
    st.header("Análisis a lo Largo del Tiempo: Transfer Learning")
    if not audio_cargado:
        st.info("Sube o graba un audio en la pestaña 'Prueba en Vivo' y pulsa en 'Analizar Audio'.")
    else:
        res_tf = st.session_state["res_tf"]
        fig = plot_probs_over_time(res_tf, "Probabilidades del Transfer Learning a lo Largo del Audio", "tf")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ── PESTAÑA 6: Robustez ──────────────────────────────────────────────────────
with tab6:
    st.header("Prueba de Robustez ante Ruido Ambiental (AWGN)")
    if not audio_cargado:
        st.info("Sube o graba un audio en la pestaña 'Prueba en Vivo' y pulsa en 'Analizar Audio'.")
    else:
        st.write("Selecciona un nivel de ruido para añadir al audio actual y observa cómo cambian las detecciones del modelo Transfer Learning.")
        
        snr_sel = st.select_slider("Nivel de Ruido a Inyectar (SNR):", options=["Limpio", "20", "10", "0"])
        
        if st.button("Aplicar Ruido y Re-analizar"):
            with st.spinner(f"Inyectando ruido (SNR={snr_sel} dB) y re-evaluando..."):
                y_base = st.session_state["audio_y"]
                sr_base = st.session_state["audio_sr"]
                
                res_tf_noisy, _, alertas_noisy, y_noisy = analyze_audio_buffer(y_base, sr_base, snr_sel)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Audio con Ruido Añadido")
                    # Crear audio temporal en memoria para reproducirlo
                    import io
                    buf = io.BytesIO()
                    sf.write(buf, y_noisy, sr_base, format='wav')
                    buf.seek(0)
                    st.audio(buf)
                    
                    if alertas_noisy:
                        st.error(f"A pesar del ruido, se detectaron {len(alertas_noisy)} alertas.")
                        for seg, clase, conf in alertas_noisy:
                            st.warning(f"⚠️ **{clase.upper()}** (Confianza: {conf:.2f})")
                    else:
                        st.success("No se detectaron alertas.")
                        
                with col2:
                    fig = plot_probs_over_time(res_tf_noisy, f"Probabilidades Transfer Learning con SNR {snr_sel} dB", f"noise_{snr_sel}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

