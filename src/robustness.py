"""
robustness.py
-------------
Evalúa la robustez frente al ruido (AWGN) de los modelos entrenados.
Genera espectrogramas de ejemplo, matrices de confusión y curvas de degradación.
"""

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR, SR
from features import preprocess_audio, compute_logmel
from model_cnn import AudioCNN
from model_transfer import TransferHead
from precompute_embeddings import CHECKPOINT_PATH
from panns_inference import AudioTagging

def add_awgn(signal, snr_db):
    """Añade ruido gaussiano blanco (AWGN) a un nivel SNR dado."""
    if snr_db is None: return signal
    p_noise = np.mean(signal**2) / (10 ** (snr_db / 10))
    return signal + np.random.randn(len(signal)) * np.sqrt(p_noise)

def plot_spectrograms(df, out_path):
    """Genera ejemplos visuales de espectrogramas con y sin ruido."""
    clases = ['glass_breaking', 'gun_shot', 'screaming', 'siren']
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    snrs = [None, 10, 0]
    
    for col, cls in enumerate(clases):
        ruta = df[df['label_name'] == cls].iloc[0]['filepath']
        sig = preprocess_audio(ruta)
        
        for row, snr in enumerate(snrs):
            logmel = compute_logmel(add_awgn(sig, snr))
            ax = axes[row, col]
            librosa.display.specshow(logmel, sr=SR, hop_length=512, x_axis='time', ax=ax, cmap='magma')
            if row == 0: ax.set_title(cls.replace('_', ' ').title(), fontweight='bold')
            if col == 0: ax.set_ylabel(["Limpio", "10 dB", "0 dB"][row], fontweight='bold')
            ax.set_xlabel('')
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_cm(targets, preds, class_names, title, path):
    """Guarda una matriz de confusión en formato heatmap."""
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print("=" * 50 + "\n  ANÁLISIS DE ROBUSTEZ FRENTE A RUIDO\n" + "=" * 50)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    
    # 1. Cargar datos
    df = pd.read_csv(BASE_DIR / "dataset_index_features.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    clases = test_df.sort_values("label_id")["label_name"].unique().tolist()
    
    # 2. Espectrogramas de ejemplo
    print("\n[1/4] Generando espectrogramas de ejemplo...")
    plot_spectrograms(test_df, out_dir / "spectrograms_example.png")
    
    # 3. Cargar Modelos
    print("[2/4] Cargando modelos...")
    cnn_c = AudioCNN(len(clases)).to(dev).eval()
    cnn_c.load_state_dict(torch.load(BASE_DIR / "models/audio_cnn_best.pt", weights_only=True, map_location=dev))
    
    cnn_a = AudioCNN(len(clases)).to(dev).eval()
    cnn_a.load_state_dict(torch.load(BASE_DIR / "models/audio_cnn_noise_best.pt", weights_only=True, map_location=dev))
    
    tf_head = TransferHead(len(clases)).to(dev).eval()
    tf_head.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", weights_only=True, map_location=dev))
    
    panns = AudioTagging(checkpoint_path=str(CHECKPOINT_PATH), device=dev)
    
    # 4. Evaluación Conjunta (para no recalcular audios repetidamente)
    print("\n[3/4] Evaluando degradación en los niveles de SNR...")
    snr_levels = [None, 20, 15, 10, 5, 0]
    snr_labels = ["Limpio", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]
    resultados, tf_preds_10, tf_preds_0 = [], None, None
    
    for snr, label in zip(snr_levels, snr_labels):
        print(f"  -> SNR: {label}")
        targets, p_cnn_c, p_cnn_a, p_tf = [], [], [], []
        
        total = len(test_df)
        with torch.no_grad():
            for i, (_, row) in enumerate(test_df.iterrows()):
                if (i + 1) % max(1, total // 5) == 0:
                    print(f"\r     {100*(i+1)//total}%", end="", flush=True)
                try:
                    # Señal para Transfer (32kHz)
                    sig_32k, _ = librosa.load(row['filepath'], sr=32000, mono=True)
                    sig_32k = add_awgn(sig_32k, snr)
                    _, emb = panns.inference(sig_32k[None, :])
                    p_tf.append(tf_head(torch.from_numpy(emb[0]).float().unsqueeze(0).to(dev)).argmax(1).item())
                    
                    # Señal para CNN (22.05kHz)
                    sig_22k = add_awgn(preprocess_audio(row['filepath']), snr)
                    X_cnn = torch.from_numpy(compute_logmel(sig_22k)).float().unsqueeze(0).unsqueeze(0).to(dev)
                    p_cnn_c.append(cnn_c(X_cnn).argmax(1).item())
                    p_cnn_a.append(cnn_a(X_cnn).argmax(1).item())
                    
                    targets.append(row['label_id'])
                except Exception:
                    continue
                    
        # Guardar métricas
        val_x = 25 if snr is None else snr
        for model_name, preds in [("CNN (Limpio)", p_cnn_c), ("CNN (Ruido)", p_cnn_a), ("Transfer", p_tf)]:
            resultados.append({
                "modelo": model_name, "snr_label": label, "snr_x": val_x,
                "f1": f1_score(targets, preds, average='macro', zero_division=0)
            })
            
        # Guardar predicciones de Transfer para las matrices de confusión
        if snr == 10: tf_preds_10 = (targets, p_tf)
        if snr == 0:  tf_preds_0 = (targets, p_tf)

    # 5. Gráficas y CSV
    print("\n[4/4] Guardando resultados...")
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(out_dir / "robustness_snr.csv", index=False)
    
    # Gráfica estrella (F1 vs SNR)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modelos = [
        ("CNN (Limpio)", '#3b82f6', 'o-', "CNN (sin augment.)"),
        ("CNN (Ruido)",  '#f59e0b', 's-', "CNN (con augment.)"),
        ("Transfer",     '#10b981', '^-', "Transfer Learning"),
    ]
    for mod, col, fmt, etiqueta in modelos:
        d = df_res[df_res["modelo"] == mod].sort_values("snr_x", ascending=False)
        ax.plot(d["snr_x"], d["f1"], fmt, color=col, lw=2.5, ms=8,
                markeredgecolor="white", markeredgewidth=1.5, label=etiqueta)
        # Anotar F1 en limpio y en 0 dB
        for _, r in d.iterrows():
            if r["snr_x"] in (25, 0):
                dy = 12 if r["snr_x"] == 25 else -14
                ax.annotate(f"{r['f1']:.2f}", (r["snr_x"], r["f1"]),
                            textcoords="offset points", xytext=(0, dy),
                            fontsize=9, fontweight="bold", color=col, ha="center")
    
    ax.axhline(0.5, color="#94a3b8", ls="--", lw=1, alpha=0.6, label="F1 = 0.50")
    ax.set_xticks([25, 20, 15, 10, 5, 0])
    ax.set_xticklabels(snr_labels)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Nivel de Ruido (SNR)")
    ax.set_ylabel("F1-Score Macro")
    ax.set_title("Robustez ante Ruido Ambiental (AWGN)", fontweight='bold')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_dir / "robustness_curves.png", dpi=150)
    plt.close(fig)
    
    plot_cm(*tf_preds_10, clases, "Transfer - SNR 10 dB", out_dir / "confusion_matrix_transfer_10dB.png")
    plot_cm(*tf_preds_0, clases, "Transfer - SNR 0 dB", out_dir / "confusion_matrix_transfer_0dB.png")
    print("¡COMPLETADO!")

if __name__ == "__main__":
    main()
