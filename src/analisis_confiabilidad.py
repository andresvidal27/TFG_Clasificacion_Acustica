"""
analisis_confiabilidad.py
-------------------------
Este script analiza y visualiza qué tan confiables son los modelos entrenados.
Se encarga de leer los reportes de clasificación de las redes CNN y Transfer Learning,
y genera gráficos para comparar su desempeño métrica por métrica (Precisión, Recall, F1-Score).
También imprime un análisis diagnóstico de las clases más problemáticas por consola.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuramos las rutas principales del proyecto
# __file__ apunta a este mismo archivo. .parent.parent sube dos niveles (hasta la raíz del proyecto)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def load_reports():
    """
    Carga los archivos CSV generados previamente por evaluacion_completa.py.
    Dichos CSV contienen las métricas de evaluación para cada clase.
    """
    # Se leen los reportes. index_col=0 asume que los nombres de las clases son el índice.
    df_cnn = pd.read_csv(RESULTS_DIR / "classification_report_cnn.csv", index_col=0)
    df_tf = pd.read_csv(RESULTS_DIR / "classification_report_transfer.csv", index_col=0)
    
    # Filtramos explícitamente solo las clases reales para omitir filas resumen como 'accuracy' o 'macro avg'
    clases = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito", "fondo"]
    df_cnn = df_cnn.loc[clases]
    df_tf = df_tf.loc[clases]
    return df_cnn, df_tf

def plot_metrics(df_cnn, df_tf):
    """
    Genera un gráfico de barras comparativo (CNN vs Transfer) para tres métricas:
    1. Precision: Si detecta la clase, ¿qué tan probable es que realmente sea esa clase? (evita falsos positivos)
    2. Recall: De todas las veces que sonó esa clase, ¿cuántas fue capaz de detectar? (evita falsos negativos)
    3. F1-Score: Es la media armónica de las dos anteriores, dando un balance general de acierto.
    """
    metrics = ["precision", "recall", "f1-score"]
    
    # Creamos un lienzo de matplotlib con 3 subgráficos (uno por métrica)
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    x = np.arange(len(df_cnn.index)) # Posiciones en el eje X para las barras
    width = 0.35 # Ancho de las barras
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Dibujamos las barras de la CNN ligeramente a la izquierda
        rects1 = ax.bar(x - width/2, df_cnn[metric], width, label='CNN', color='#1f77b4')
        # Dibujamos las barras del modelo de Transfer ligeramente a la derecha
        rects2 = ax.bar(x + width/2, df_tf[metric], width, label='Transfer', color='#ff7f0e')
        
        # Configuraciones de etiquetas y estilo
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Comparativa de {metric.capitalize().replace("F1-score", "F1-Score")} por Clase')
        ax.set_xticks(x)
        ax.set_xticklabels(df_cnn.index, rotation=45) # Rotamos las etiquetas 45 grados para que quepan bien
        ax.legend()
        ax.set_ylim(0, 1.1) # Las métricas van de 0 a 1. Ponemos 1.1 para dar margen arriba
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Este bucle añade el valor numérico exacto encima de cada barra para facilitar la lectura
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # Desplazamiento de 3 puntos hacia arriba
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    plt.tight_layout() # Ajusta automáticamente los márgenes para que no se superpongan textos
    output_path = RESULTS_DIR / "analisis_confiabilidad_clases.png"
    plt.savefig(output_path, dpi=300) # Guardamos la gráfica en alta calidad (300 dpi)
    print(f"Gráfica guardada en {output_path}")

def analizar_problemas(df_cnn, df_tf):
    """
    Identifica e imprime por consola cuáles son las clases que más le cuestan
    a cada modelo (aquellas con el menor F1-Score).
    """
    print("\n" + "="*50)
    print(" ANÁLISIS DE CONFIABILIDAD: PEORES CLASES ")
    print("="*50)
    
    # Ordenamos el DataFrame de menor a mayor f1-score y cogemos las primeras 3 (head(3))
    print("\n--- CNN: Top 3 clases MÁS PROBLEMÁTICAS (Menor F1-Score) ---")
    print(df_cnn.sort_values(by="f1-score")[["f1-score", "precision", "recall", "support"]].head(3))
    
    print("\n--- TRANSFER LEARNING: Top 3 clases MÁS PROBLEMÁTICAS (Menor F1-Score) ---")
    print(df_tf.sort_values(by="f1-score")[["f1-score", "precision", "recall", "support"]].head(3))

    # Breve chuleta para entender la consola
    print("\n[Interpretación]")
    print("- Precision bajo: Da falsas alarmas (confunde otras cosas con esta clase).")
    print("- Recall bajo: No la detecta cuando suena (se la traga y dice que es otra cosa).")

if __name__ == "__main__":
    df_cnn, df_tf = load_reports()
    plot_metrics(df_cnn, df_tf)
    analizar_problemas(df_cnn, df_tf)
