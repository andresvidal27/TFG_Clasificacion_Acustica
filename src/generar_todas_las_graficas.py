"""
generar_todas_las_graficas.py
-----------------------------
Script unificado y simplificado para generar automáticamente todas las figuras
visuales (gráficos de barras, curvas de pérdida, gráficas ROC, etc.) a partir
de los CSV y JSON guardados previamente en la carpeta 'results'.

Soporta tanto modelos individuales como ensembles de 5 fases.
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
from entrenar_cnn import AudioCNN, DatasetEspectrogramas, ENSEMBLE_SEEDS
from entrenar_transfer import TransferHead, DatasetEmbeddings

RESULTS_DIR, MODELS_DIR = BASE_DIR / "results", BASE_DIR / "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cargar_ensemble(ModelClass, num_classes, pattern):
    """Carga los modelos ensemble si existen."""
    modelos = []
    for i in range(len(ENSEMBLE_SEEDS)):
        ruta = MODELS_DIR / f"{pattern}_{i}_best.pt"
        if ruta.exists():
            m = ModelClass(num_classes=num_classes).to(device).eval()
            m.load_state_dict(torch.load(ruta, weights_only=True, map_location=device))
            modelos.append(m)
    return modelos if len(modelos) == len(ENSEMBLE_SEEDS) else []

def plot_all():
    """Genera las gráficas estáticas leyendo de los CSV/JSON."""
    
    # 1. BARRAS DE MÉTRICAS (Classification Report) — Individuales y Ensemble
    reportes = [
        ("cnn", "classification_report_cnn.csv", "Métricas CNN (Individual)"),
        ("tf", "classification_report_transfer.csv", "Métricas Transfer (Individual)"),
        ("cnn_ensemble", "classification_report_cnn_ensemble.csv", "Métricas CNN Ensemble (5 fases)"),
        ("tf_ensemble", "classification_report_transfer_ensemble.csv", "Métricas Transfer Ensemble (5 fases)"),
    ]
    for m, a, t in reportes:
        p = RESULTS_DIR / a
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            # Ignoramos métricas agregadas globales y graficamos cada clase
            df.loc[[c for c in df.index if c not in ["accuracy", "macro avg", "weighted avg"]]].plot(
                kind='bar', y=['precision', 'recall', 'f1-score'], figsize=(10, 6), colormap='Set2', title=t)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"classification_bars_{m}.png")
            plt.close()

    # 2. CURVAS DE APRENDIZAJE (Train vs Val Loss y Acc)
    # Permiten diagnosticar si hubo Sobreajuste (Overfitting) o Subajuste (Underfitting)
    
    # Modelos individuales
    for m, a, t in [("cnn", "history_cnn_base.json", "CNN Base (Individual)"), 
                    ("tf", "history_transfer_head.json", "Transfer Learning (Individual)")]:
        p = MODELS_DIR / a
        if p.exists():
            h = json.load(open(p))
            ep = range(1, len(h['train_loss']) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Subplot Izquierdo: Loss (Pérdida/Error). Lo ideal es que Train y Val bajen parejos.
            ax1.plot(ep, h['train_loss'], label='Train')
            ax1.plot(ep, h['val_loss'], label='Val')
            ax1.set_title(f"{t} - Loss"); ax1.legend()
            
            # Subplot Derecho: Accuracy (Precisión). Lo ideal es que ambos suban parejos.
            ax2.plot(ep, h['train_acc'], label='Train')
            ax2.plot(ep, h['val_acc'], label='Val')
            ax2.set_title(f"{t} - Acc"); ax2.legend()
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"curvas_aprendizaje_{m}.png")
            plt.close()
    
    # Curvas de aprendizaje del Ensemble (5 folds superpuestos)
    for prefix, pattern, titulo_base in [("cnn", "history_cnn_ensemble_fold", "CNN Ensemble"),
                                          ("tf", "history_transfer_ensemble_fold", "Transfer Ensemble")]:
        historiales = []
        for i in range(len(ENSEMBLE_SEEDS)):
            p = MODELS_DIR / f"{pattern}_{i}.json"
            if p.exists():
                historiales.append(json.load(open(p)))
        
        if historiales:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            colores = plt.cm.tab10(np.linspace(0, 0.5, len(historiales)))
            
            for i, h in enumerate(historiales):
                ep = range(1, len(h['train_loss']) + 1)
                ax1.plot(ep, h['val_loss'], color=colores[i], alpha=0.6, label=f'Fold {i} (seed {ENSEMBLE_SEEDS[i]})')
                ax2.plot(ep, h['val_acc'], color=colores[i], alpha=0.6, label=f'Fold {i} (seed {ENSEMBLE_SEEDS[i]})')
            
            ax1.set_title(f"{titulo_base} - Val Loss (5 folds)"); ax1.legend(fontsize=8); ax1.grid(True, ls='--')
            ax2.set_title(f"{titulo_base} - Val Acc (5 folds)"); ax2.legend(fontsize=8); ax2.grid(True, ls='--')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"curvas_aprendizaje_{prefix}_ensemble.png")
            plt.close()

    # 3. CURVAS DE ROBUSTEZ FRENTE AL RUIDO BLANCO (SNR)
    p = RESULTS_DIR / "robustness_snr.csv"
    if p.exists():
        plt.figure(figsize=(10, 6))
        # Graficamos el F1 Macro en función del SNR. A menor SNR (más ruido), esperamos caídas de rendimiento.
        sns.lineplot(data=pd.read_csv(p), x='snr_x', y='f1', hue='modelo', marker='o', lw=2)
        plt.title("Robustez frente a Ruido (AWGN)")
        plt.gca().invert_xaxis() # Invertimos el eje X porque mayor SNR = Menos ruido = Mejor.
        plt.grid(True, ls='--'); plt.tight_layout()
        plt.savefig(RESULTS_DIR / "robustness_curves.png"); plt.close()

def plot_inferencia():
    """Genera gráficas complejas que requieren inferencia (pasar datos por los modelos)."""
    
    df_cnn = pd.read_csv(BASE_DIR / "dataset_index_features.csv").dropna(subset=["feature_path"])
    df_emb = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    t_cnn = df_cnn[df_cnn["split"] == "test"].reset_index(drop=True)
    # Filtrar solo muestras originales (sin augmentation) para evitar duplicados en test
    t_emb = df_emb[df_emb["split"] == "test"]
    if "augmentation" in t_emb.columns:
        t_emb = t_emb[t_emb["augmentation"] == "none"]
    t_emb = t_emb.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    nc = len(CLASS_MAP)
    
    # Cargar Modelos individuales
    cnn = AudioCNN(nc).to(device).eval()
    cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_base_best.pt", weights_only=True, map_location=device))
    tf = TransferHead(nc).to(device).eval()
    tf.load_state_dict(torch.load(MODELS_DIR / "transfer_head_best.pt", weights_only=True, map_location=device))
    
    # Cargar Ensembles
    ensemble_cnn = _cargar_ensemble(AudioCNN, nc, "cnn_ensemble_fold")
    ensemble_tf = _cargar_ensemble(TransferHead, nc, "transfer_ensemble_fold")

    def get_preds(m, ds):
        """Helper local para obtener predicciones Softmax de un modelo individual."""
        targs, probs = [], []
        with torch.no_grad():
            for X, y in DataLoader(ds, batch_size=32):
                probs.append(F.softmax(m(X.to(device)), dim=1).cpu().numpy())
                targs.extend(y.numpy())
        return np.array(targs), np.concatenate(probs)
    
    def get_preds_ensemble(modelos, ds):
        """Helper local para obtener predicciones promediadas del ensemble."""
        targs, probs_list = [], []
        with torch.no_grad():
            for X, y in DataLoader(ds, batch_size=32):
                X_dev = X.to(device)
                batch_probs = np.zeros((X.size(0), nc))
                for m in modelos:
                    batch_probs += F.softmax(m(X_dev), dim=1).cpu().numpy()
                batch_probs /= len(modelos)
                probs_list.append(batch_probs)
                targs.extend(y.numpy())
        return np.array(targs), np.concatenate(probs_list)

    # Predicciones individuales
    y_cnn, p_cnn = get_preds(cnn, DatasetEspectrogramas(t_cnn))
    y_tf, p_tf = get_preds(tf, DatasetEmbeddings(t_emb))
    
    # Predicciones ensemble (si existen)
    modelos_y_preds = [
        (y_cnn, p_cnn, "cnn", "CNN (Individual)"),
        (y_tf, p_tf, "transfer", "Transfer (Individual)")
    ]
    if ensemble_cnn:
        y_ecnn, p_ecnn = get_preds_ensemble(ensemble_cnn, DatasetEspectrogramas(t_cnn))
        modelos_y_preds.append((y_ecnn, p_ecnn, "cnn_ensemble", "CNN Ensemble (5 fases)"))
    if ensemble_tf:
        y_etf, p_etf = get_preds_ensemble(ensemble_tf, DatasetEmbeddings(t_emb))
        modelos_y_preds.append((y_etf, p_etf, "transfer_ensemble", "Transfer Ensemble (5 fases)"))

    # 4. CURVAS ROC (Receiver Operating Characteristic) y UMBRALES
    # Una curva ROC muestra el balance entre verdaderos positivos y falsos positivos a distintos umbrales
    for y_true, probs, n_arch, t in modelos_y_preds:
        plt.figure(figsize=(9, 7))
        for i in range(nc):
            # Convertir problema muticlase en uno contra el resto (One-Vs-Rest)
            try:
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
                # AUC (Area Under Curve) mide qué tan capaz es el modelo de distinguir esa clase
                plt.plot(fpr, tpr, label=f"{CLASS_MAP[i]} (AUC = {auc(fpr, tpr):.2f})")
            except ValueError:
                # Puede fallar si una clase no tiene muestras en test
                plt.plot([], [], label=f"{CLASS_MAP[i]} (sin datos)")
        plt.plot([0, 1], [0, 1], 'k--'); plt.title(f"ROC - {t}"); plt.legend()
        plt.savefig(RESULTS_DIR / f"detector_curves_{n_arch}.png"); plt.close()

        # Análisis de impacto de Threshold (Umbral Theta de decisión)
        # Probamos forzar la respuesta a "fondo" (ID 7) si la confianza no supera el umbral
        umbrales, f1s = np.linspace(0.1, 0.95, 30), []
        for u in umbrales:
            preds = np.argmax(probs, axis=1)
            preds[np.max(probs, axis=1) < u] = 7 # Rechazo / Asignar a clase Background (Fondo)
            f1s.append(precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)[2])
        
        plt.figure(figsize=(8, 5))
        plt.plot(umbrales, f1s, marker='.', color='purple', lw=2); plt.grid(True)
        plt.title(f"F1 vs Umbral - {t}"); plt.savefig(RESULTS_DIR / f"threshold_analysis_{n_arch}.png"); plt.close()

    # 5. MATRICES DE CONFUSIÓN CON RUIDO INTENSO (Para Transfer Learning — ensemble si disponible)
    from panns_inference import AudioTagging
    panns = AudioTagging(checkpoint_path=str(MODELS_DIR / "Cnn14_mAP=0.431.pth"), device=str(device))
    
    # Usar ensemble si disponible, sino individual
    tf_modelo_label = "Transfer Ensemble" if ensemble_tf else "Transfer Individual"
    
    for snr in [10, 0]: # Casos de ruido bastante molesto y ruido extremo (señal=ruido)
        preds, targs = [], []
        for _, r in t_emb.iterrows():
            try:
                audio_ruidoso = add_awgn(librosa.load(r['filepath'], sr=32000, mono=True)[0], snr)
                emb_tensor = torch.from_numpy(panns.inference(audio_ruidoso[None, :])[1][0]).float().unsqueeze(0).to(device)
                
                if ensemble_tf:
                    # Inferencia ensemble: promediar softmax de los 5 modelos
                    probs_avg = np.zeros(nc)
                    for m in ensemble_tf:
                        probs_avg += F.softmax(m(emb_tensor), dim=1).cpu().numpy()[0]
                    probs_avg /= len(ensemble_tf)
                    preds.append(np.argmax(probs_avg))
                else:
                    preds.append(tf(emb_tensor).argmax(1).item())
                
                targs.append(r['label_id'])
            except Exception: pass
        
        if not targs:
            print(f"  [Aviso] Sin datos válidos para SNR {snr}dB, saltando matriz de confusión.")
            continue
            
        cm = confusion_matrix(targs, preds, labels=list(range(nc)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.astype(float)/np.maximum(cm.sum(axis=1, keepdims=True), 1), annot=cm, fmt="d", cmap="Reds", 
                    xticklabels=[CLASS_MAP[i] for i in range(nc)], yticklabels=[CLASS_MAP[i] for i in range(nc)])
        plt.title(f"CM {tf_modelo_label} sometido a ruido SNR {snr}dB")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"confusion_matrix_transfer_{snr}dB.png"); plt.close()

if __name__ == "__main__":
    plot_all()
    plot_inferencia()
    print("¡Gráficas guardadas en 'results/'!")
