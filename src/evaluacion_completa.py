"""
evaluacion_completa.py
----------------------
Este script carga los pesos finales entrenados de ambos modelos (CNN y Transfer Learning)
y los evalúa frente al conjunto de TEST (datos que la red JAMÁS ha visto durante el entrenamiento).
Genera reportes de clasificación detallados (F1, Precision, Recall) y Matrices de Confusión.
Además, pone a prueba los modelos añadiéndoles diferentes niveles de ruido (Robustez).

Soporta tanto modelos individuales como ensembles de 5 fases (promediado de predicciones).
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
from entrenar_cnn import AudioCNN, DatasetEspectrogramas, ENSEMBLE_SEEDS
from entrenar_transfer import TransferHead, DatasetEmbeddings

def _cargar_ensemble(ModelClass, num_classes, pattern, device):
    """Intenta cargar los 5 modelos ensemble. Si no existen todos, devuelve lista vacía."""
    modelos = []
    for i in range(len(ENSEMBLE_SEEDS)):
        ruta = BASE_DIR / "models" / f"{pattern}_{i}_best.pt"
        if ruta.exists():
            m = ModelClass(num_classes=num_classes).to(device).eval()
            m.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
            modelos.append(m)
    return modelos if len(modelos) == len(ENSEMBLE_SEEDS) else []

def eval_model(model, loader, device):
    """
    Pasa todos los datos de un Dataloader (habitualmente TEST) por un modelo
    y recupera sus predicciones y probabilidades absolutas.
    """
    model.eval() # Modo evaluación (apaga Dropout, etc)
    targets, probs = [], []
    with torch.no_grad():
        for X, y in loader:
            # Las salidas del modelo pasan por Softmax para convertirse en porcentajes de 0 a 1
            probs.append(F.softmax(model(X.to(device)), dim=1).cpu().numpy())
            targets.extend(y.numpy())
    probs = np.concatenate(probs)
    # Devuelve (Las Etiquetas reales, Las Predicciones Elegidas, Las Probabilidades de las 8 clases)
    return np.array(targets), probs.argmax(1), probs

def eval_ensemble(modelos, loader, device):
    """
    Pasa todos los datos por los N modelos del ensemble y promedia las probabilidades softmax.
    El promediado reduce la varianza de predicciones individuales y da resultados más estables.
    """
    targets, probs_acum = [], None
    with torch.no_grad():
        for X, y in loader:
            X_dev = X.to(device)
            batch_probs = np.zeros((X.size(0), modelos[0].classifier[-1].out_features if hasattr(modelos[0], 'classifier') else modelos[0].fc[-1].out_features))
            for modelo in modelos:
                modelo.eval()
                batch_probs += F.softmax(modelo(X_dev), dim=1).cpu().numpy()
            batch_probs /= len(modelos)
            
            if probs_acum is None:
                probs_acum = batch_probs
            else:
                probs_acum = np.concatenate([probs_acum, batch_probs])
            targets.extend(y.numpy())
    
    return np.array(targets), probs_acum.argmax(1), probs_acum

def plot_cm(y_true, y_pred, clases, titulo, ruta):
    """
    Genera y guarda una Matriz de Confusión (Confusion Matrix) en formato imagen (PNG).
    Permite ver visualmente qué clase se está confundiendo con qué otra.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Normalizamos por fila (para ver los porcentajes de acierto sobre el total real de esa clase)
    sns.heatmap(cm.astype(float)/np.maximum(cm.sum(axis=1, keepdims=True), 1), 
                annot=cm, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()

def main():
    print("Iniciando Evaluación Completa...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = BASE_DIR / "results"
    out.mkdir(exist_ok=True)
    
    # Cargamos EXCLUSIVAMENTE los splits marcados como "test" en los índices precomputados
    df_cnn = pd.read_csv(BASE_DIR / "dataset_index_features.csv").dropna(subset=["feature_path"])
    df_emb = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    test_cnn = df_cnn[df_cnn["split"] == "test"].reset_index(drop=True)
    test_emb = df_emb[df_emb["split"] == "test"].reset_index(drop=True)
    clases = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]
    nc = len(clases)
    
    # ==============================================================================
    # 1. EVALUACIÓN DE MODELOS INDIVIDUALES
    # ==============================================================================
    # Instanciamos los modelos vacíos en memoria
    cnn = AudioCNN(nc).to(device)
    tf = TransferHead(nc).to(device)
    
    # Cargamos los pesos guardados en disco por el Early Stopping de la fase de entrenamiento
    cnn.load_state_dict(torch.load(BASE_DIR / "models/cnn_base_best.pt", weights_only=True, map_location=device))
    tf.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", weights_only=True, map_location=device))
    
    print("Evaluando modelos individuales en test limpio...")
    # CNN Base individual
    y_true_cnn, y_pred_cnn, _ = eval_model(cnn, DataLoader(DatasetEspectrogramas(test_cnn), batch_size=32), device)
    plot_cm(y_true_cnn, y_pred_cnn, clases, "Matriz Confusión CNN (Individual)", out / "confusion_matrix_cnn.png")
    pd.DataFrame(classification_report(y_true_cnn, y_pred_cnn, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_cnn.csv")
    
    # Transfer Learning individual
    y_true_tf, y_pred_tf, _ = eval_model(tf, DataLoader(DatasetEmbeddings(test_emb), batch_size=32), device)
    plot_cm(y_true_tf, y_pred_tf, clases, "Matriz Confusión Transfer (Individual)", out / "confusion_matrix_transfer.png")
    pd.DataFrame(classification_report(y_true_tf, y_pred_tf, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_transfer.csv")
    
    # ==============================================================================
    # 2. EVALUACIÓN DE ENSEMBLES (5 FASES)
    # ==============================================================================
    ensemble_cnn = _cargar_ensemble(AudioCNN, nc, "cnn_ensemble_fold", device)
    ensemble_tf = _cargar_ensemble(TransferHead, nc, "transfer_ensemble_fold", device)
    
    if ensemble_cnn:
        print(f"Evaluando Ensemble CNN ({len(ensemble_cnn)} modelos) en test limpio...")
        y_true_ecnn, y_pred_ecnn, _ = eval_ensemble(ensemble_cnn, DataLoader(DatasetEspectrogramas(test_cnn), batch_size=32), device)
        plot_cm(y_true_ecnn, y_pred_ecnn, clases, f"Matriz Confusión CNN Ensemble ({len(ensemble_cnn)} modelos)", out / "confusion_matrix_cnn_ensemble.png")
        pd.DataFrame(classification_report(y_true_ecnn, y_pred_ecnn, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_cnn_ensemble.csv")
        
        # Comparar individual vs ensemble
        acc_ind = np.mean(y_pred_cnn == y_true_cnn)
        acc_ens = np.mean(y_pred_ecnn == y_true_ecnn)
        print(f"  CNN Individual Acc: {acc_ind:.4f} | CNN Ensemble Acc: {acc_ens:.4f} | Mejora: {(acc_ens - acc_ind)*100:+.2f}%")
    else:
        print("[Info] No se encontraron modelos CNN ensemble. Ejecuta entrenar_cnn_ensemble() primero.")
    
    if ensemble_tf:
        print(f"Evaluando Ensemble Transfer ({len(ensemble_tf)} modelos) en test limpio...")
        y_true_etf, y_pred_etf, _ = eval_ensemble(ensemble_tf, DataLoader(DatasetEmbeddings(test_emb), batch_size=32), device)
        plot_cm(y_true_etf, y_pred_etf, clases, f"Matriz Confusión Transfer Ensemble ({len(ensemble_tf)} modelos)", out / "confusion_matrix_transfer_ensemble.png")
        pd.DataFrame(classification_report(y_true_etf, y_pred_etf, target_names=clases, output_dict=True)).T.to_csv(out / "classification_report_transfer_ensemble.csv")
        
        # Comparar individual vs ensemble
        acc_ind = np.mean(y_pred_tf == y_true_tf)
        acc_ens = np.mean(y_pred_etf == y_true_etf)
        print(f"  Transfer Individual Acc: {acc_ind:.4f} | Transfer Ensemble Acc: {acc_ens:.4f} | Mejora: {(acc_ens - acc_ind)*100:+.2f}%")
    else:
        print("[Info] No se encontraron modelos Transfer ensemble. Ejecuta entrenar_transfer_ensemble() primero.")
    
    # Guardamos Umbrales (se mantiene la lógica original de calibración)
    threshold_data = {
        "theta": 0.85, # Subimos la exigencia general probabilística (señales más fuertes)
        "thresholds_por_clase": {
            "sirena": 0.95,          # Mucha seguridad
            "ladrido_perro": 0.95,   
            "rotura_cristal": 0.85,
            "disparo": 0.60,         # Muy sensible (menos estricto)
            "bebe_llorando": 0.85,
            "llamar_puerta": 0.45,   # Muy reducido para captar bien los golpes
            "grito": 0.80            # Reducido un poco para que sea más fácil detectar gritos
        },
        "min_rms_general": 0.025,    # Todo sonido debe tener al menos esta energía para ser analizado
        "min_rms_por_clase": {
            "grito": 0.065,          # Bajado el límite físico de grito para que salte sin tener que reventar el micro
            "sirena": 0.040,         # Exigencia alta
            "disparo": 0.035,        # Un disparo lejano puede tener menos energía física (menos estricto)
            "rotura_cristal": 0.035,
            "ladrido_perro": 0.030,
            "llamar_puerta": 0.010   # Exigencia casi nula de volumen, los golpes son sutiles
        }
    }
    json.dump(threshold_data, open(BASE_DIR / "models/threshold.json", "w"))

    # ==============================================================================
    # 3. ANÁLISIS DE ROBUSTEZ FRENTE A RUIDO AMBIENTAL (AWGN)
    # ==============================================================================
    print("Analizando robustez SNR...")
    from panns_inference import AudioTagging
    # Re-cargamos PANNs para procesar audios en crudo, porque les meteremos ruido artificial
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    res = []
    
    # Simulamos degradación de audio. 20 dB = ruido muy leve, 0 dB = ruido igual de fuerte que la señal
    for snr, lbl in zip([None, 20, 15, 10, 5, 0], ["Limpio", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]):
        print(f" -> Evaluando {lbl}...")
        targs, p_cnn, p_tf = [], [], []
        # Para ensemble, acumulamos las probabilidades y las promediamos
        p_ecnn_probs, p_etf_probs = [], []
        
        with torch.no_grad():
            for _, r in test_cnn.iterrows():
                try:
                    # --- Transfer Learning (PANNs -> Head) ---
                    s_32 = add_awgn(librosa.load(r['filepath'], sr=32000, mono=True)[0], snr)
                    emb_tensor = torch.from_numpy(panns.inference(s_32[None, :])[1][0]).float().unsqueeze(0).to(device)
                    
                    # Individual
                    p_tf.append(tf(emb_tensor).argmax(1).item())
                    
                    # Ensemble Transfer
                    if ensemble_tf:
                        probs_avg = np.zeros(nc)
                        for m in ensemble_tf:
                            probs_avg += F.softmax(m(emb_tensor), dim=1).cpu().numpy()[0]
                        probs_avg /= len(ensemble_tf)
                        p_etf_probs.append(np.argmax(probs_avg))
                    
                    # --- CNN Base ---
                    s_22 = add_awgn(preprocess_audio(r['filepath']), snr)
                    logmel_tensor = torch.from_numpy(compute_logmel(s_22)).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Individual
                    p_cnn.append(cnn(logmel_tensor).argmax(1).item())
                    
                    # Ensemble CNN
                    if ensemble_cnn:
                        probs_avg = np.zeros(nc)
                        for m in ensemble_cnn:
                            probs_avg += F.softmax(m(logmel_tensor), dim=1).cpu().numpy()[0]
                        probs_avg /= len(ensemble_cnn)
                        p_ecnn_probs.append(np.argmax(probs_avg))
                    
                    targs.append(r['label_id'])
                except Exception: pass
        
        # Almacenamos el F1-Macro resultante en esa situación de ruido particular
        res.extend([
            {"modelo": "CNN", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_cnn, average='macro', zero_division=0)},
            {"modelo": "Transfer Learning", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_tf, average='macro', zero_division=0)}
        ])
        if ensemble_cnn:
            res.append({"modelo": "CNN Ensemble", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_ecnn_probs, average='macro', zero_division=0)})
        if ensemble_tf:
            res.append({"modelo": "Transfer Ensemble", "snr_label": lbl, "snr_x": 25 if snr is None else snr, "f1": f1_score(targs, p_etf_probs, average='macro', zero_division=0)})
    
    # Guardamos el análisis de degradación acústica en un CSV para luego graficarlo
    pd.DataFrame(res).to_csv(out / "robustness_snr.csv", index=False)
    print("¡Evaluación completada con éxito!")

if __name__ == "__main__": main()
