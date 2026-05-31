"""
evaluacion_completa_v2.py
-------------------------
Evalúa la precisión y robustez del modelo Transfer Learning V2.
Guarda los resultados en la carpeta resultsV2/.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, librosa, seaborn as sns, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn.functional as F

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from datos_y_configuracion import SR, preprocess_audio, add_awgn, CLASS_MAP
from entrenar_transfer import TransferHead, DatasetEmbeddings
from entrenar_transfer_v2 import DatasetEmbeddingsV2

def eval_model(model, loader, device):
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for X, y in loader:
            probs.append(F.softmax(model(X.to(device)), dim=1).cpu().numpy())
            targets.extend(y.numpy())
    probs = np.concatenate(probs)
    return np.array(targets), probs.argmax(1), probs

def plot_cm(y_true, y_pred, clases, titulo, ruta):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.astype(float)/np.maximum(cm.sum(axis=1, keepdims=True), 1), annot=cm, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
    plt.title(titulo); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(ruta); plt.close()

def main():
    print("Iniciando Evaluación del Modelo V2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = BASE_DIR / "resultsV2"
    out.mkdir(exist_ok=True)
    
    df_emb = pd.read_csv(BASE_DIR / "dataset_index_emb_v2.csv").dropna(subset=["embedding_path"])
    # Solo tomamos las muestras de test
    test_emb = df_emb[df_emb["split"] == "test"].reset_index(drop=True)
    clases = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]
    
    tf = TransferHead(len(clases)).to(device)
    tf.load_state_dict(torch.load(BASE_DIR / "models_v2/transfer_mejorado_best.pt", weights_only=True, map_location=device))
    
    print("Evaluando en test limpio (V2)...")
    y_true_tf, y_pred_tf, _ = eval_model(tf, DataLoader(DatasetEmbeddingsV2(test_emb), batch_size=32), device)
    plot_cm(y_true_tf, y_pred_tf, clases, "Matriz Confusión Transfer V2", out / "confusion_matrix_transfer_v2.png")
    pd.DataFrame(classification_report(y_true_tf, y_pred_tf, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_transfer_v2.csv")

    # Robustez
    print("Analizando robustez SNR para Transfer V2...")
    from panns_inference import AudioTagging
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=device)
    res = []
    
    for snr, lbl in zip([None, 20, 15, 10, 5, 0], ["Limpio", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]):
        print(f" -> Evaluando {lbl}...")
        targs, p_tf = [], []
        with torch.no_grad():
            for _, r in test_emb.iterrows():
                try:
                    s_32 = add_awgn(librosa.load(r['filepath'], sr=32000, mono=True)[0], snr)
                    p_tf.append(tf(torch.from_numpy(panns.inference(s_32[None, :])[1][0]).float().unsqueeze(0).to(device)).argmax(1).item())
                    targs.append(r['label_id'])
                except Exception: pass
        res.extend([
            {"modelo": "Transfer Learning V2", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_tf, average='macro', zero_division=0)}
        ])
    pd.DataFrame(res).to_csv(out / "robustness_snr_v2.csv", index=False)
    print("¡Evaluación V2 completada con éxito!")

if __name__ == "__main__": main()
