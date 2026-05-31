"""
evaluacion_completa.py
----------------------
Evalúa la precisión y robustez de los modelos entrenados.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, librosa, seaborn as sns, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn.functional as F

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from datos_y_configuracion import SR, preprocess_audio, compute_logmel, add_awgn, CLASS_MAP
from entrenar_cnn import AudioCNN, DatasetEspectrogramas
from entrenar_transfer import TransferHead, DatasetEmbeddings

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
    print("Iniciando Evaluación Completa...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = BASE_DIR / "results"
    out.mkdir(exist_ok=True)
    
    df_cnn = pd.read_csv(BASE_DIR / "dataset_index_features.csv").dropna(subset=["feature_path"])
    df_emb = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    test_cnn = df_cnn[df_cnn["split"] == "test"].reset_index(drop=True)
    test_emb = df_emb[df_emb["split"] == "test"].reset_index(drop=True)
    clases = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]
    
    cnn = AudioCNN(len(clases)).to(device)
    cnn.load_state_dict(torch.load(BASE_DIR / "models/cnn_base_best.pt", weights_only=True, map_location=device))
    tf = TransferHead(len(clases)).to(device)
    tf.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", weights_only=True, map_location=device))
    
    print("Evaluando en test limpio...")
    y_true_cnn, y_pred_cnn, _ = eval_model(cnn, DataLoader(DatasetEspectrogramas(test_cnn), batch_size=32), device)
    plot_cm(y_true_cnn, y_pred_cnn, clases, "Matriz Confusión CNN", out / "confusion_matrix_cnn.png")
    pd.DataFrame(classification_report(y_true_cnn, y_pred_cnn, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_cnn.csv")
    
    y_true_tf, y_pred_tf, _ = eval_model(tf, DataLoader(DatasetEmbeddings(test_emb), batch_size=32), device)
    plot_cm(y_true_tf, y_pred_tf, clases, "Matriz Confusión Transfer", out / "confusion_matrix_transfer.png")
    pd.DataFrame(classification_report(y_true_tf, y_pred_tf, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_transfer.csv")
    
    # Umbral
    json.dump({"theta": 0.85}, open(BASE_DIR / "models/threshold.json", "w"))

    # Robustez
    print("Analizando robustez SNR...")
    from panns_inference import AudioTagging
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=device)
    res = []
    
    for snr, lbl in zip([None, 20, 15, 10, 5, 0], ["Limpio", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]):
        print(f" -> Evaluando {lbl}...")
        targs, p_cnn, p_tf = [], [], []
        with torch.no_grad():
            for _, r in test_cnn.iterrows():
                try:
                    s_32 = add_awgn(librosa.load(r['filepath'], sr=32000, mono=True)[0], snr)
                    p_tf.append(tf(torch.from_numpy(panns.inference(s_32[None, :])[1][0]).float().unsqueeze(0).to(device)).argmax(1).item())
                    s_22 = add_awgn(preprocess_audio(r['filepath']), snr)
                    p_cnn.append(cnn(torch.from_numpy(compute_logmel(s_22)).float().unsqueeze(0).unsqueeze(0).to(device)).argmax(1).item())
                    targs.append(r['label_id'])
                except Exception: pass
        res.extend([
            {"modelo": "CNN", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_cnn, average='macro', zero_division=0)},
            {"modelo": "Transfer Learning", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_tf, average='macro', zero_division=0)}
        ])
    pd.DataFrame(res).to_csv(out / "robustness_snr.csv", index=False)
    print("¡Evaluación completada con éxito!")

if __name__ == "__main__": main()
