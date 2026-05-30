"""
dashboard.py
------------
Dashboard de análisis interactivo con Streamlit.
Permite visualizar de forma interactiva y dinámica los resultados obtenidos 
en las diferentes fases del TFG de Clasificación Acústica.
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

warnings.filterwarnings("ignore")

# Para poder importar las funciones del proyecto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src.datos_y_configuracion import SR as CONFIG_SR, compute_logmel, preprocess_audio, add_awgn

st.set_page_config(page_title="Dashboard Vigilancia Acústica", layout="wide")

# ── Funciones de carga de datos ──────────────────────────────────────────────

@st.cache_data
def load_dataset_index():
    return pd.read_csv(BASE_DIR / "dataset_index.csv")

@st.cache_data
def load_threshold_data():
    with open(BASE_DIR / "models/threshold.json") as f:
        return json.load(f)

@st.cache_data
def load_history(path):
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_robustness_csv():
    return pd.read_csv(BASE_DIR / "results/robustness_snr.csv")

@st.cache_data
def load_resultados_simulacion():
    return pd.read_csv(BASE_DIR / "results/resultados_simulacion.csv")

# ── Funciones de visualización ───────────────────────────────────────────────

def plot_plotly_history(history, title):
    """Crea una curva de aprendizaje interactiva con Plotly."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    
    fig = go.Figure()
    
    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                             line=dict(color="#3b82f6", dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                             line=dict(color="#1d4ed8", width=2)))
    
    # Accuracy (en un eje Y secundario para que sea visible en la misma gráfica, 
    # o mejor simplemente lo hacemos en subplots)
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Función de Pérdida (Loss)", "Precisión (Accuracy)"))
    
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                             line=dict(color="#3b82f6", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                             line=dict(color="#1d4ed8", width=2)), row=1, col=1)
                             
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc",
                             line=dict(color="#10b981", dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc",
                             line=dict(color="#047857", width=2)), row=1, col=2)
                             
    fig.update_layout(title_text=title, height=400, hovermode="x unified")
    return fig

# ── Interfaz Principal ───────────────────────────────────────────────────────

st.title("🎙️ Sistema de Vigilancia Acústica - Resultados TFG")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Comparativa de Modelos", 
    "Espectrogramas", 
    "Resultados CNN", 
    "Resultados Transfer", 
    "Análisis de Robustez",
    "Simulación de Campo"
])

# ── PESTAÑA 1: Comparativa de Modelos ────────────────────────────────────────
with tab1:
    st.header("Comparativa de Rendimiento: Transfer Learning vs CNN Base")
    
    th_data = load_threshold_data()
    cnn_data = th_data["cnn"]
    best = th_data["transfer"]
    
    # 1. Métricas con Delta (Diferencia visual explícita)
    col1, col2, col3 = st.columns(3)
    
    acc_diff = best['accuracy'] - cnn_data['accuracy']
    col1.metric("Precisión Global (Accuracy)", f"{best['accuracy']:.2f}%", f"+{acc_diff:.2f}% vs CNN")
    
    roc_diff = best['roc_auc'] - cnn_data['roc_auc']
    col2.metric("Capacidad de Detección (ROC AUC)", f"{best['roc_auc']:.4f}", f"+{roc_diff:.4f} vs CNN")
    
    ap_diff = best['average_precision'] - cnn_data['average_precision']
    col3.metric("Precisión Media (AP)", f"{best['average_precision']:.4f}", f"+{ap_diff:.4f} vs CNN")
    
    st.divider()
    
    # 2. Gráfico de Barras Agrupadas para comparar métricas a escala 100
    st.subheader("El salto cualitativo al integrar PANNs (Transfer Learning)")
    
    fig = go.Figure(data=[
        go.Bar(name='CNN (Entrenada desde cero)', 
               x=['Accuracy (%)', 'ROC AUC (x100)', 'Average Precision (x100)'], 
               y=[cnn_data['accuracy'], cnn_data['roc_auc']*100, cnn_data['average_precision']*100],
               marker_color='#94a3b8', 
               text=[f"{cnn_data['accuracy']:.1f}%", f"{cnn_data['roc_auc']:.3f}", f"{cnn_data['average_precision']:.3f}"], 
               textposition='auto'),
               
        go.Bar(name='Transfer Learning (PANNs)', 
               x=['Accuracy (%)', 'ROC AUC (x100)', 'Average Precision (x100)'], 
               y=[best['accuracy'], best['roc_auc']*100, best['average_precision']*100],
               marker_color='#3b82f6', 
               text=[f"{best['accuracy']:.1f}%", f"{best['roc_auc']:.3f}", f"{best['average_precision']:.3f}"], 
               textposition='auto')
    ])
    
    fig.update_layout(barmode='group', height=450, 
                      yaxis_title="Puntuación (Escala 0-100)", 
                      legend_title_text="Modelo Evaluado",
                      margin=dict(t=20, b=20))
                      
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Bloque de Reducción de Errores
    st.divider()
    col_e1, col_e2 = st.columns([1, 2])
    
    error_cnn = 100 - cnn_data['accuracy']
    error_tf = 100 - best['accuracy']
    reduccion_errores = ((error_cnn - error_tf) / error_cnn) * 100
    
    with col_e1:
        st.subheader("Reducción de Errores")
        st.markdown(f"<h1 style='text-align: center; color: #10b981; font-size: 4rem; padding-top: 1rem;'>-{reduccion_errores:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Reducción total de fallos de clasificación frente al modelo base</p>", unsafe_allow_html=True)
        
    with col_e2:
        fig2 = go.Figure(go.Indicator(
            mode = "number+delta",
            value = error_tf,
            number = {'suffix': "%", 'font': {'size': 50}},
            title = {"text": "Tasa de Error Actual (Transfer Learning)"},
            delta = {'reference': error_cnn, 'relative': False, 'valueformat': '.1f', 'suffix': '% (CNN base)', 'decreasing': {'color': '#10b981'}}
        ))
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=True)

# ── PESTAÑA 2: Explorador de Espectrogramas ──────────────────────────────────
with tab2:
    st.header("Impacto del Ruido en los Espectrogramas Log-Mel")
    
    df = load_dataset_index()
    clases = df["label_name"].unique().tolist()
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        clase_sel = st.selectbox("Selecciona una clase:", clases)
        snr_sel = st.select_slider(
            "Nivel de Ruido (SNR):", 
            options=["Limpio", "20 dB", "10 dB", "0 dB"]
        )
        
        snr_val = None if snr_sel == "Limpio" else int(snr_sel.replace(" dB", ""))
        
    with col_b:
        # Coger un audio al azar de esa clase
        ejemplo = df[df["label_name"] == clase_sel].iloc[0]
        ruta = BASE_DIR / ejemplo["filepath"]
        
        if ruta.exists():
            sig = preprocess_audio(str(ruta))
            sig_ruidoso = add_awgn(sig, snr_val)
            logmel = compute_logmel(sig_ruidoso)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(logmel, sr=CONFIG_SR, hop_length=512, x_axis='time', cmap='magma', ax=ax)
            ax.set_title(f"Clase: {clase_sel} | SNR: {snr_sel}")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
        else:
            st.error(f"No se encontró el audio en {ruta}")

# ── PESTAÑA 3: Modelo CNN ────────────────────────────────────────────────────
with tab3:
    st.header("Resultados del Modelo CNN (Desde Cero)")
    
    variante = st.radio("Variante:", ["Entrenada con Ruido (Recomendada)", "CNN Base (Limpia)"], horizontal=True)
    json_file = "audio_cnn_noise_history.json" if "Ruido" in variante else "audio_cnn_history.json"
    
    try:
        hist_cnn = load_history(BASE_DIR / "models" / json_file)
        st.plotly_chart(plot_plotly_history(hist_cnn, "Curvas de Aprendizaje - CNN"), use_container_width=True)
    except FileNotFoundError:
        st.warning("No se ha entrenado este modelo aún.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Métricas por Clase")
        try:
            report_cnn = pd.read_csv(BASE_DIR / "results/classification_report_cnn.csv", index_col=0)
            st.dataframe(report_cnn, use_container_width=True)
        except Exception:
            st.warning("No se encontró el reporte.")
            
    with col_c2:
        st.subheader("Matriz de Confusión")
        try:
            st.image(str(BASE_DIR / "results/confusion_matrix_cnn.png"), use_container_width=True)
        except Exception:
            st.warning("No se encontró la matriz.")

# ── PESTAÑA 4: Modelo Transfer Learning ──────────────────────────────────────
with tab4:
    st.header("Resultados de Transfer Learning (CNN14)")
    
    try:
        hist_tf = load_history(BASE_DIR / "models/transfer_head_history.json")
        st.plotly_chart(plot_plotly_history(hist_tf, "Curvas de Aprendizaje - Transfer Learning"), use_container_width=True)
    except FileNotFoundError:
        st.warning("No se ha entrenado el modelo Transfer aún.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Métricas por Clase")
        try:
            report_tf = pd.read_csv(BASE_DIR / "results/classification_report_transfer.csv", index_col=0)
            st.dataframe(report_tf, use_container_width=True)
        except Exception:
            st.warning("No se encontró el reporte.")
            
    with col_t2:
        st.subheader("Matriz de Confusión")
        try:
            st.image(str(BASE_DIR / "results/confusion_matrix_transfer.png"), use_container_width=True)
        except Exception:
            st.warning("No se encontró la matriz.")

# ── PESTAÑA 5: Robustez ──────────────────────────────────────────────────────
with tab5:
    st.header("Análisis de Robustez ante Ruido Ambiental (AWGN)")
    st.write("Degradación del F1-Score a medida que disminuye el ratio Señal/Ruido.")
    
    try:
        df_res = load_robustness_csv()
        
        # Plotly interactivo
        fig = px.line(
            df_res, x="snr_label", y="f1", color="modelo", markers=True,
            title="Comparativa de Robustez",
            labels={"snr_label": "Nivel de Ruido (SNR)", "f1": "F1-Score Macro", "modelo": "Modelo"}
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Límite F1=0.5")
        fig.update_layout(height=500)
        # Ordenar eje X para que coincida con lo esperado (de Limpio a 0 dB)
        fig.update_xaxes(categoryorder='array', categoryarray=["Limpio", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"])
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("No se encontraron resultados de robustez. Ejecuta src/robustness.py")
        
    st.subheader("Impacto en Transfer Learning a altos niveles de ruido")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        try:
            st.image(str(BASE_DIR / "results/confusion_matrix_transfer_10dB.png"), caption="Matriz Transfer a 10 dB")
        except: pass
    with col_r2:
        try:
            st.image(str(BASE_DIR / "results/confusion_matrix_transfer_0dB.png"), caption="Matriz Transfer a 0 dB")
        except: pass

# ── PESTAÑA 6: Simulación de Campo ───────────────────────────────────────────
with tab6:
    st.header("Simulación de Campo: Sistema Integrado")
    st.write("""
    El sistema actúa como un activador acústico inteligente para una cámara de seguridad. Permanece en un estado 
    de **ESPERA** de muy bajo consumo energético analizando el entorno. Al detectar un evento peligroso que supera 
    el umbral, pasa a **ALERTA** y dispara la **GRABACIÓN**. Gracias a un búfer circular, se compensa la latencia 
    recuperando el audio de los instantes previos al evento.
    """)
    
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.subheader("Máquina de Estados")
        st.graphviz_chart('''
            digraph {
                node [shape=box, style=filled, color="#3b82f6", fontcolor=white, fontname="Helvetica", border=0, penwidth=0];
                edge [fontname="Helvetica", fontsize=10, color="gray"];
                ESPERA -> ALERTA [label=" evento sospechoso > umbral"];
                ALERTA -> GRABACION [label=" volcado de búfer (10s)"];
                GRABACION -> ESPERA [label=" fin de evento"];
            }
        ''')
        
    with col_g2:
        st.subheader("Métricas de Rendimiento y Eficiencia")
        try:
            df_sim = load_resultados_simulacion()
            carpeta_alertas = BASE_DIR / "test_simulacion" / "alertas_detectadas"
            
            # Contar alertas reales disparadas leyendo los ficheros .wav generados
            alertas_archivos = list(carpeta_alertas.glob("*.wav")) if carpeta_alertas.exists() else []
            num_alertas = len(alertas_archivos)
            
            # Cálculo de eficiencia (Asumimos 180s totales)
            tiempo_total = 180.0
            tiempo_grabacion = num_alertas * 10.0 # 10s por alerta
            tiempo_espera = max(0.0, tiempo_total - tiempo_grabacion)
            pct_ahorro = (tiempo_espera / tiempo_total) * 100
            
            latencia_media = df_sim["latencia_s"].mean()
            
            # Mostrar métricas en columnas
            m1, m2, m3 = st.columns(3)
            m1.metric("% Tiempo en ahorro (ESPERA)", f"{pct_ahorro:.1f}%")
            m2.metric("Nº Alertas disparadas", str(num_alertas))
            m3.metric("Latencia media de detección", f"{latencia_media:.2f} s" if pd.notna(latencia_media) else "N/A")
            
            # Mostrar tasa de detección
            total_eventos = len(df_sim)
            detectados = int(df_sim["detectado"].sum())
            falsos_positivos = max(0, num_alertas - detectados)
            
            st.write(f"**Tasa de Detección:** {detectados}/{total_eventos} eventos reales detectados. **Falsos Positivos:** {falsos_positivos}")
            
            st.subheader("Registro de Eventos (Ground Truth vs Detección)")
            st.dataframe(df_sim, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("No se encontraron resultados de la simulación. Ejecuta simulacion_campo.py primero.")
            
    st.divider()
    st.subheader("Visor de Alertas Detectadas")
    try:
        carpeta_alertas = BASE_DIR / "test_simulacion" / "alertas_detectadas"
        if carpeta_alertas.exists():
            imgs_alertas = list(carpeta_alertas.glob("*.png"))
            if imgs_alertas:
                img_nombres = [img.name for img in imgs_alertas]
                seleccion = st.selectbox("Selecciona una alerta capturada para visualizar el espectrograma (10s de búfer):", img_nombres)
                st.image(str(carpeta_alertas / seleccion), use_container_width=True)
            else:
                st.info("No se han generado alertas visuales en la carpeta.")
        else:
            st.info("La carpeta de alertas no existe aún.")
    except Exception as e:
        st.warning(f"No se pudo cargar la visualización de alertas: {e}")
