"""
dashboard.py
------------
Dashboard simplificado de análisis interactivo con Streamlit.
"""

import sys
import json
import time
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
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
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
            bloque_32k = librosa.resample(bloque, orig_sr=sr, target_sr=32000)
            _, emb = panns.inference(bloque_32k[None, :])
            probs_tf = F.softmax(model_transfer(torch.from_numpy(emb[0]).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            clase_pred = CLASS_MAP[np.argmax(probs_tf)]
            res_tf.append({"segundo": segundo, "probs": probs_tf})
            
            if clase_pred in clases_alerta:
                max_prob = np.max(probs_tf)
                segunda_prob = np.sort(probs_tf)[-2]
                
                # Regla: Supera el umbral (theta) O (es mayor a 0.70 y le saca al menos 0.30 de diferencia a la segunda clase)
                if (max_prob >= theta) or (max_prob >= 0.70 and (max_prob - segunda_prob) >= 0.30):
                    if not any(c == clase_pred and (segundo - s) <= 3.0 for s, c, _ in alertas):
                        alertas.append((segundo, clase_pred, max_prob))
                
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

def guardar_feedback(y, sr, seg, true_class, is_center=False):
    panns, _, _, _, _ = load_models()
    
    feedback_dir = BASE_DIR / "data_feedback"
    audio_dir = feedback_dir / "audio"
    emb_dir = feedback_dir / "embeddings"
    audio_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    ventana = int(5.0 * sr)
    if is_center:
        center_sample = int(seg * sr)
        start_sample = max(0, center_sample - ventana // 2)
        end_sample = start_sample + ventana
        if end_sample > len(y):
            end_sample = len(y)
            start_sample = max(0, end_sample - ventana)
    else:
        end_sample = int(seg * sr)
        start_sample = max(0, end_sample - ventana)
    
    bloque = y[start_sample:end_sample]
    if len(bloque) < ventana:
        bloque = np.pad(bloque, (ventana - len(bloque), 0))
    
    timestamp = int(time.time() * 1000)
    filepath = audio_dir / f"fb_{timestamp}.wav"
    sf.write(str(filepath), bloque, sr)
    
    import torch
    bloque_32k = librosa.resample(bloque, orig_sr=sr, target_sr=32000)
    if len(bloque_32k) > 160000:
        bloque_32k = bloque_32k[:160000]
    elif len(bloque_32k) < 160000:
        bloque_32k = np.pad(bloque_32k, (0, 160000 - len(bloque_32k)))
        
    with torch.no_grad():
        _, emb = panns.inference(bloque_32k[None, :])
    
    emb_path = emb_dir / f"fb_{timestamp}.npy"
    np.save(emb_path, emb[0])
    
    from src.datos_y_configuracion import NAME_TO_ID
    label_id = NAME_TO_ID.get(true_class, NAME_TO_ID["fondo"])
    
    csv_path = feedback_dir / "feedback_index.csv"
    df_new = pd.DataFrame([{
        "filepath": str(filepath), 
        "label_id": label_id, 
        "label_name": true_class,
        "embedding_path": str(emb_path)
    }])
    df_new.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

st.title("🎙️ Analizador de Audio en Vivo")

tabs = st.tabs(["Prueba", "Comparativa", "Espectrogramas", "CNN", "Transfer", "Robustez"])

with tabs[0]:
    with st.expander("⚙️ Gestión de Reentrenamiento", expanded=False):
        csv_path = BASE_DIR / "data_feedback" / "feedback_index.csv"
        n_muestras = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
        st.write(f"Muestras de feedback acumuladas: **{n_muestras}**")
        if "reentreno_msg" in st.session_state:
            st.info(st.session_state.pop("reentreno_msg"))
            
        if st.button("🔄 Reentrenar modelo ahora", key="btn_reentrenar_master"):
            with st.spinner("Reentrenando modelo con los nuevos datos..."):
                import subprocess
                result = subprocess.run([sys.executable, "src/entrenar_feedback.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.cache_resource.clear()
                    hist_path = BASE_DIR / "data_feedback" / "historial_reentrenamientos.csv"
                    if hist_path.exists():
                        df_hist = pd.read_csv(hist_path)
                        last = df_hist.iloc[-1]
                        if last["guardado"]:
                            st.session_state["reentreno_msg"] = f"✅ ¡Modelo actualizado y cargado en memoria! F1 pasó de {last['f1_antes']:.4f} a {last['f1_despues']:.4f}."
                        else:
                            st.session_state["reentreno_msg"] = f"⚠️ Reentrenamiento descartado por el Guard (F1 cayó de {last['f1_antes']:.4f} a {last['f1_despues']:.4f}). El modelo anterior sigue activo."
                    else:
                        st.session_state["reentreno_msg"] = "✅ ¡Modelo actualizado con éxito!"
                    st.rerun()
                else:
                    st.error("Hubo un error en el entrenamiento automático.")
                    st.code(result.stderr)
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
            st.session_state.update({"res_tf": res_tf, "res_cnn": res_cnn, "alertas": alertas, "analizado": True})
            for k in list(st.session_state.keys()):
                if k.startswith("fb_") or k == "fb_guardado" or k.startswith("sel_"): del st.session_state[k]
            
    if st.session_state.get("analizado", False):
        alertas = st.session_state.get("alertas", [])
        if st.session_state.get("fb_guardado", False):
            csv_path = BASE_DIR / "data_feedback" / "feedback_index.csv"
            n_muestras = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
            st.success(f"✅ Feedback guardado. Muestras pendientes: {n_muestras}")
        else:
            if alertas:
                st.write("### Resultados detectados")
                correcciones = []
                for idx, (seg, cl, conf) in enumerate(alertas):
                    st.warning(f"⚠️ {cl.upper()} a los {seg:.1f}s (Confianza: {conf*100:.0f}%)")
                    opciones = list(CLASS_MAP.values())
                    idx_cl = opciones.index(cl) if cl in opciones else 0
                    corr_cl = st.selectbox(f"Clase real de la alerta {idx+1}:", opciones, index=idx_cl, key=f"sel_{idx}")
                    correcciones.append((seg, corr_cl))
                
                if st.button("💾 Guardar Feedback", key="btn_guardar_alertas"):
                    for seg, corr_cl in correcciones:
                        guardar_feedback(st.session_state["audio_y"], st.session_state["audio_sr"], seg, corr_cl, is_center=False)
                    st.session_state["fb_guardado"] = True
                    st.rerun()
            else:
                st.success("✅ Todo tranquilo (Ninguna alerta detectada).")
                st.write("¿Hubo algún sonido que el modelo no detectó?")
                
                opciones_miss = ["Ninguno (Todo correcto)"] + [c for c in CLASS_MAP.values() if c != "fondo"]
                corr_cl_miss = st.selectbox("Selecciona la clase real si hubo un error:", opciones_miss, key="sel_noalert")
                
                duracion = float(len(st.session_state["audio_y"]) / st.session_state["audio_sr"])
                if corr_cl_miss != "Ninguno (Todo correcto)":
                    seg_miss = st.number_input("Segundo exacto donde ocurre el sonido:", min_value=0.0, max_value=duracion, value=duracion/2, step=0.5, key="seg_noalert")
                else:
                    seg_miss = min(5.0, duracion)
                
                if st.button("💾 Guardar Feedback", key="btn_guardar_noalert"):
                    clase_a_guardar = "fondo" if corr_cl_miss == "Ninguno (Todo correcto)" else corr_cl_miss
                    is_center_miss = (corr_cl_miss != "Ninguno (Todo correcto)")
                    guardar_feedback(st.session_state["audio_y"], st.session_state["audio_sr"], seg_miss, clase_a_guardar, is_center=is_center_miss)
                    st.session_state["fb_guardado"] = True
                    st.rerun()

if "audio_y" in st.session_state:
    y, sr = st.session_state["audio_y"], st.session_state["audio_sr"]
    
    with tabs[1]:
        st.audio(y, sample_rate=sr)
        
        alertas = st.session_state.get("alertas", [])
        opciones = ["Media Global", "Selección Manual"] + [f"{cl.upper()} a los {seg:.1f}s" for seg, cl, _ in alertas]
        opcion_sel = st.selectbox("Seleccionar Instante", opciones)
        
        seg = None
        if opcion_sel == "Media Global":
            probs_tf = np.mean([r["probs"] for r in st.session_state["res_tf"]], axis=0) if st.session_state["res_tf"] else np.zeros(len(CLASS_MAP))
            probs_cnn = np.mean([r["probs"] for r in st.session_state["res_cnn"]], axis=0) if st.session_state["res_cnn"] else np.zeros(len(CLASS_MAP))
            title = "Comparativa Media Global"
        elif opcion_sel == "Selección Manual":
            duracion = float(len(y) / sr)
            seg = st.slider("Seleccionar segundo", 0.0, duracion, 0.0, step=0.5)
            title = f"Probabilidades en {seg:.1f}s"
        else:
            idx = opciones.index(opcion_sel) - 2
            seg = alertas[idx][0]
            title = f"Probabilidades en {opcion_sel}"

        if seg is not None:
            probs_tf = min(st.session_state["res_tf"], key=lambda r: abs(r["segundo"] - seg))["probs"] if st.session_state["res_tf"] else np.zeros(len(CLASS_MAP))
            probs_cnn = min(st.session_state["res_cnn"], key=lambda r: abs(r["segundo"] - seg))["probs"] if st.session_state["res_cnn"] else np.zeros(len(CLASS_MAP))

        df = pd.DataFrame({"Transfer": probs_tf, "CNN": probs_cnn}, index=[CLASS_MAP[i] for i in range(len(CLASS_MAP))])
        df_filt = df.reset_index().rename(columns={"index": "Clase"}).melt(id_vars="Clase", var_name="Modelo", value_name="Prob")
        st.plotly_chart(px.bar(df_filt, x="Clase", y="Prob", color="Modelo", barmode="group", title=title), width="stretch")
        
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
