"""
generar_todas_las_graficas.py
-----------------------------
Script unificado y simplificado para generar todos los gráficos del proyecto.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, torch
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F, librosa
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from datos_y_configuracion import CLASS_MAP, add_awgn
from entrenar_cnn import AudioCNN, DatasetEspectrogramas
from entrenar_transfer import TransferHead, DatasetEmbeddings

RESULTS_DIR, MODELS_DIR = BASE_DIR / "results", BASE_DIR / "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_all():
    # 1. Barras de métricas
    for m, a, t in [("cnn", "classification_report_cnn.csv", "Métricas CNN"), ("tf", "classification_report_transfer.csv", "Métricas Transfer")]:
        p = RESULTS_DIR / a
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            df.loc[[c for c in df.index if c not in ["accuracy", "macro avg", "weighted avg"]]].plot(kind='bar', y=['precision', 'recall', 'f1-score'], figsize=(10, 6), colormap='Set2', title=t)
            plt.tight_layout(); plt.savefig(RESULTS_DIR / f"classification_bars_{m}.png"); plt.close()

    # 2. Curvas de aprendizaje
    for m, a, t in [("cnn", "history_cnn_base.json", "CNN Base"), ("tf", "history_transfer_head.json", "Transfer Learning")]:
        p = MODELS_DIR / a
        if p.exists():
            h = json.load(open(p))
            ep = range(1, len(h['train_loss']) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(ep, h['train_loss'], label='Train'); ax1.plot(ep, h['val_loss'], label='Val'); ax1.set_title(f"{t} - Loss"); ax1.legend()
            ax2.plot(ep, h['train_acc'], label='Train'); ax2.plot(ep, h['val_acc'], label='Val'); ax2.set_title(f"{t} - Acc"); ax2.legend()
            plt.tight_layout(); plt.savefig(RESULTS_DIR / f"curvas_aprendizaje_{m}.png"); plt.close()

    # 3. Robustez
    p = RESULTS_DIR / "robustness_snr.csv"
    if p.exists():
        plt.figure(figsize=(9, 6)); sns.lineplot(data=pd.read_csv(p), x='snr_x', y='f1', hue='modelo', marker='o', lw=2)
        plt.title("Robustez (AWGN)"); plt.gca().invert_xaxis(); plt.grid(True, ls='--'); plt.tight_layout()
        plt.savefig(RESULTS_DIR / "robustness_curves.png"); plt.close()

def plot_inferencia():
    df_cnn = pd.read_csv(BASE_DIR / "dataset_index_features.csv").dropna(subset=["feature_path"])
    df_emb = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    t_cnn, t_emb = df_cnn[df_cnn["split"] == "test"].reset_index(drop=True), df_emb[df_emb["split"] == "test"].reset_index(drop=True)
    nc = len(CLASS_MAP)
    
    cnn = AudioCNN(nc).to(device).eval()
    cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_base_best.pt", weights_only=True, map_location=device))
    tf = TransferHead(nc).to(device).eval()
    tf.load_state_dict(torch.load(MODELS_DIR / "transfer_head_best.pt", weights_only=True, map_location=device))

    def get_preds(m, ds):
        targs, probs = [], []
        with torch.no_grad():
            for X, y in DataLoader(ds, batch_size=32):
                probs.append(F.softmax(m(X.to(device)), dim=1).cpu().numpy())
                targs.extend(y.numpy())
        return np.array(targs), np.concatenate(probs)

    y_cnn, p_cnn = get_preds(cnn, DatasetEspectrogramas(t_cnn))
    y_tf, p_tf = get_preds(tf, DatasetEmbeddings(t_emb))

    for y_true, probs, n_arch, t in [(y_cnn, p_cnn, "cnn", "CNN"), (y_tf, p_tf, "transfer", "Transfer")]:
        plt.figure(figsize=(9, 7))
        for i in range(nc):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
            plt.plot(fpr, tpr, label=f"{CLASS_MAP[i]} (AUC = {auc(fpr, tpr):.2f})")
        plt.plot([0, 1], [0, 1], 'k--'); plt.title(f"ROC - {t}"); plt.legend(); plt.savefig(RESULTS_DIR / f"detector_curves_{n_arch}.png"); plt.close()

        umbrales, f1s = np.linspace(0.1, 0.95, 30), []
        for u in umbrales:
            preds = np.argmax(probs, axis=1)
            preds[np.max(probs, axis=1) < u] = 7
            f1s.append(precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)[2])
        plt.figure(figsize=(8, 5)); plt.plot(umbrales, f1s, marker='.', color='purple', lw=2); plt.grid(True)
        plt.title(f"F1 vs Umbral - {t}"); plt.savefig(RESULTS_DIR / f"threshold_analysis_{n_arch}.png"); plt.close()

    from panns_inference import AudioTagging
    panns = AudioTagging(checkpoint_path=str(MODELS_DIR / "Cnn14_mAP=0.431.pth"), device=device)
    for snr in [10, 0]:
        preds, targs = [], []
        for _, r in t_emb.iterrows():
            try:
                preds.append(tf(torch.from_numpy(panns.inference(add_awgn(librosa.load(r['filepath'], sr=32000, mono=True)[0], snr)[None, :])[1][0]).float().unsqueeze(0).to(device)).argmax(1).item())
                targs.append(r['label_id'])
            except Exception: pass
        cm = confusion_matrix(targs, preds, labels=list(range(nc)))
        plt.figure(figsize=(10, 8)); sns.heatmap(cm.astype(float)/np.maximum(cm.sum(axis=1, keepdims=True), 1), annot=cm, fmt="d", cmap="Reds", xticklabels=[CLASS_MAP[i] for i in range(nc)], yticklabels=[CLASS_MAP[i] for i in range(nc)])
        plt.title(f"CM SNR {snr}dB"); plt.tight_layout(); plt.savefig(RESULTS_DIR / f"confusion_matrix_transfer_{snr}dB.png"); plt.close()

if __name__ == "__main__":
    plot_all()
    plot_inferencia()
    print("¡Gráficas guardadas en 'results/'!")
