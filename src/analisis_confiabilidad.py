import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def load_reports():
    df_cnn = pd.read_csv(RESULTS_DIR / "classification_report_cnn.csv", index_col=0)
    df_tf = pd.read_csv(RESULTS_DIR / "classification_report_transfer.csv", index_col=0)
    
    # Filtrar solo clases reales
    clases = ["rotura_cristal", "disparo", "ladrido_perro", "sirena", "bebe_llorando", "llamar_puerta", "grito", "fondo"]
    df_cnn = df_cnn.loc[clases]
    df_tf = df_tf.loc[clases]
    return df_cnn, df_tf

def plot_metrics(df_cnn, df_tf):
    metrics = ["precision", "recall", "f1-score"]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    x = np.arange(len(df_cnn.index))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        rects1 = ax.bar(x - width/2, df_cnn[metric], width, label='CNN', color='#1f77b4')
        rects2 = ax.bar(x + width/2, df_tf[metric], width, label='Transfer', color='#ff7f0e')
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Comparativa de {metric.capitalize().replace("F1-score", "F1-Score")} por Clase')
        ax.set_xticks(x)
        ax.set_xticklabels(df_cnn.index, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    plt.tight_layout()
    output_path = RESULTS_DIR / "analisis_confiabilidad_clases.png"
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica guardada en {output_path}")

def analizar_problemas(df_cnn, df_tf):
    print("\n" + "="*50)
    print(" ANÁLISIS DE CONFIABILIDAD: PEORES CLASES ")
    print("="*50)
    
    print("\n--- CNN: Top 3 clases MÁS PROBLEMÁTICAS (Menor F1-Score) ---")
    print(df_cnn.sort_values(by="f1-score")[["f1-score", "precision", "recall", "support"]].head(3))
    
    print("\n--- TRANSFER LEARNING: Top 3 clases MÁS PROBLEMÁTICAS (Menor F1-Score) ---")
    print(df_tf.sort_values(by="f1-score")[["f1-score", "precision", "recall", "support"]].head(3))

    print("\n[Interpretación]")
    print("- Precision bajo: Da falsas alarmas (confunde otras cosas con esta clase).")
    print("- Recall bajo: No la detecta cuando suena (se la traga y dice que es otra cosa).")

if __name__ == "__main__":
    df_cnn, df_tf = load_reports()
    plot_metrics(df_cnn, df_tf)
    analizar_problemas(df_cnn, df_tf)
