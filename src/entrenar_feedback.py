"""
entrenar_feedback.py
--------------------
Script para realizar aprendizaje continuo con el feedback recopilado en el dashboard.
Aplica data augmentation sobre el audio original y protege el modelo usando un test guard.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import f1_score
from panns_inference import AudioTagging

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from datos_y_configuracion import CLASS_MAP, add_awgn
from entrenar_transfer import TransferHead, calcular_pesos_clases, DatasetEmbeddings

def extract_augmented_embeddings(filepath, panns, device):
    """Genera versiones aumentadas del audio y extrae sus embeddings."""
    y, sr = librosa.load(filepath, sr=32000, mono=True)
    embeddings = []
    
    # 4 variaciones: ruido (AWGN) + shift temporal aleatorio
    snrs = [20, 15, 10, 5]
    for snr in snrs:
        y_aug = add_awgn(y, snr)
        
        # Desplazamiento temporal aleatorio entre -0.5s y 0.5s
        shift = int(np.random.uniform(-0.5, 0.5) * sr)
        y_aug = np.roll(y_aug, shift)
        
        # Asegurar longitud exacta de 5s para PANNs (32000 * 5 = 160000)
        target_len = 5 * sr
        if len(y_aug) > target_len:
            y_aug = y_aug[:target_len]
        else:
            y_aug = np.pad(y_aug, (0, max(0, target_len - len(y_aug))))
            
        with torch.no_grad():
            _, emb = panns.inference(y_aug[None, :])
        embeddings.append(emb[0])
        
    return embeddings

class FeedbackMixedDataset(Dataset):
    """Dataset que combina audios antiguos con nuevos embeddings de feedback en memoria."""
    def __init__(self, emb_viejos, lbl_viejos, emb_nuevos, lbl_nuevos):
        self.x = np.vstack([emb_viejos, emb_nuevos]) if len(emb_nuevos) > 0 else emb_viejos
        self.y = np.concatenate([lbl_viejos, lbl_nuevos]) if len(lbl_nuevos) > 0 else lbl_viejos

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

def evaluar_modelo(model, df_test, device):
    """Evalúa el modelo sobre el split de test y devuelve el F1-Macro."""
    model.eval()
    y_true, y_pred = [], []
    loader = DataLoader(DatasetEmbeddings(df_test), batch_size=32, shuffle=False)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return f1_score(y_true, y_pred, average='macro')

def main():
    print("="*50)
    print(" ENTRENAMIENTO CONTINUO CON FEEDBACK ")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_MAP)
    
    # 5. COHERENCIA DE CLASES
    print("Verificando coherencia de clases...")
    print(f"Clases definidas en CLASS_MAP: {CLASS_MAP}")
    
    # 1. Cargar el feedback
    feedback_csv = BASE_DIR / "data_feedback/feedback_index.csv"
    if not feedback_csv.exists():
        print("[Info] No hay datos de feedback todavía.")
        sys.exit(0)

    df_feedback = pd.read_csv(feedback_csv)
    if len(df_feedback) == 0:
        print("[Info] El archivo de feedback está vacío.")
        sys.exit(0)
        
    # Verificar que los label_ids están dentro de rango y coinciden
    for idx, row in df_feedback.iterrows():
        l_id = row['label_id']
        l_name = row['label_name']
        if l_id not in CLASS_MAP:
            print(f"[ERROR] El label_id {l_id} no existe en CLASS_MAP. Abortando.")
            sys.exit(1)
        if CLASS_MAP[l_id] != l_name:
            print(f"[ERROR] Incoherencia en fila {idx}: ID {l_id} es '{CLASS_MAP[l_id]}' pero el CSV dice '{l_name}'. Abortando.")
            sys.exit(1)

    # 2. Cargar embeddings base y aumentados
    print("\nCargando PANNs para data augmentation del feedback...")
    panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
    
    emb_nuevos, lbl_nuevos = [], []
    
    # Listas para guardar SOLO la muestra original de cada audio, para poder diagnosticar su predicción
    diag_emb_base = []
    diag_lbl_base = []
    
    print(f"Procesando {len(df_feedback)} muestras de feedback...")
    for _, row in df_feedback.iterrows():
        try:
            # PROBLEMA 3: Verificación del segmento
            y_check, sr_check = librosa.load(row["filepath"], sr=32000, mono=True)
            duracion = len(y_check) / sr_check
            rms = np.sqrt(np.mean(y_check**2))
            if duracion < 1.0:
                print(f"[Advertencia] Audio muy corto ({duracion:.1f}s): {Path(row['filepath']).name}")
            if rms < 0.001:
                print(f"[Advertencia] Audio casi en silencio (RMS={rms:.4f}): {Path(row['filepath']).name}")
                
            # PROBLEMA 1: Usar el embedding_path si existe para el base
            if "embedding_path" in row and pd.notna(row["embedding_path"]) and Path(row["embedding_path"]).exists():
                emb_base = np.load(row["embedding_path"])
                emb_nuevos.append(emb_base)
            else:
                # Fallback si no hay embedding precalculado
                y_base = y_check
                if len(y_base) > 5 * 32000: y_base = y_base[:5 * 32000]
                else: y_base = np.pad(y_base, (0, max(0, 5 * 32000 - len(y_base))))
                with torch.no_grad():
                    _, emb_base = panns.inference(y_base[None, :])
                emb_base = emb_base[0]
                emb_nuevos.append(emb_base)
                
            lbl_nuevos.append(row["label_id"])
            
            # Guardamos la muestra original para evaluarla luego
            diag_emb_base.append(emb_base)
            diag_lbl_base.append(row["label_id"])
            
            # PROBLEMA 2: Data augmentation REAL sobre el audio
            embs_aug = extract_augmented_embeddings(row["filepath"], panns, device)
            for emb_aug in embs_aug:
                emb_nuevos.append(emb_aug)
                lbl_nuevos.append(row["label_id"])
                
        except Exception as e:
            print(f"Error procesando {row['filepath']}: {e}")
            
    emb_nuevos = np.array(emb_nuevos)
    lbl_nuevos = np.array(lbl_nuevos)
    
    # Liberar memoria de PANNs
    del panns
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Cargar datos históricos
    print("Cargando memoria histórica y set de test...")
    df_old = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    df_old_train = df_old[df_old["split"] == "train"].reset_index(drop=True)
    df_test = df_old[df_old["split"] == "test"].reset_index(drop=True)
    
    # Seleccionamos una muestra aleatoria equitativa del dataset original (ej. 200 muestras por clase)
    df_sample = df_old_train.groupby("label_id").sample(n=min(200, len(df_old_train)), replace=True).drop_duplicates()
    
    emb_viejos = np.array([np.load(p) for p in df_sample["embedding_path"].values])
    lbl_viejos = df_sample["label_id"].values

    # Dataset mixto
    dataset = FeedbackMixedDataset(emb_viejos, lbl_viejos, emb_nuevos, lbl_nuevos)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. Cargar y evaluar Modelo Actual (PROBLEMA 3a)
    ruta_modelo = BASE_DIR / "models/transfer_head_best.pt"
    if not ruta_modelo.exists():
        print("[Error] No se encontró el modelo original.")
        sys.exit(1)
        
    model = TransferHead(num_classes).to(device)
    model.load_state_dict(torch.load(ruta_modelo, map_location=device, weights_only=True))
    
    print("\nEvaluando modelo actual en test set...")
    f1_antes = evaluar_modelo(model, df_test, device)
    print(f"-> F1-Macro global ANTES: {f1_antes:.4f}")
    
    # Evaluar en el feedback antes de reentrenar
    print("\n[DIAGNÓSTICO] Predicciones sobre el feedback ANTES del reentrenamiento:")
    model.eval()
    with torch.no_grad():
        for i, (emb, lbl_id) in enumerate(zip(diag_emb_base, diag_lbl_base)):
            out = model(torch.from_numpy(emb).float().unsqueeze(0).to(device))
            probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
            pred_id = np.argmax(probs)
            marcador = "[OK]" if lbl_id == pred_id else "[FALLO]"
            print(f"Muestra {i+1} | Real: {CLASS_MAP[lbl_id]} | Predice: {CLASS_MAP[pred_id]} (Prob: {probs[pred_id]:.2f}) {marcador}")

    # 5. Entrenar sobre una COPIA del modelo (PROBLEMA 3b)
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.train()
    
    pesos_clase = calcular_pesos_clases(lbl_viejos, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 12
    print(f"\nIniciando Fine-Tuning por {epochs} épocas...")
    for epoch in range(1, epochs + 1):
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            salida = model_copy(X)
            loss = criterion(salida, y)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * X.size(0)
            t_correct += (salida.argmax(1) == y).sum().item()
            t_total += y.size(0)
            
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {t_loss/t_total:.4f} | Acc: {t_correct/t_total:.4f}")

    # 6. Evaluar copia y decidir (Guard) (PROBLEMA 3c y 3d)
    print("\nEvaluando modelo actualizado en test set...")
    f1_despues = evaluar_modelo(model_copy, df_test, device)
    print(f"-> F1-Macro global DESPUÉS: {f1_despues:.4f}")
    
    # Evaluar en el feedback después de reentrenar
    print("\n[DIAGNÓSTICO] Predicciones sobre el feedback DESPUÉS del reentrenamiento:")
    model_copy.eval()
    with torch.no_grad():
        for i, (emb, lbl_id) in enumerate(zip(diag_emb_base, diag_lbl_base)):
            out = model_copy(torch.from_numpy(emb).float().unsqueeze(0).to(device))
            probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
            pred_id = np.argmax(probs)
            marcador = "[OK]" if lbl_id == pred_id else "[FALLO]"
            print(f"Muestra {i+1} | Real: {CLASS_MAP[lbl_id]} | Predice: {CLASS_MAP[pred_id]} (Prob: {probs[pred_id]:.2f}) {marcador}")
    
    guardado = False
    tolerancia = 0.01
    
    print("\n" + "="*50)
    print(f" RESULTADO FINAL: F1_antes = {f1_antes:.4f} | F1_después = {f1_despues:.4f}")
    if f1_despues >= f1_antes:
        torch.save(model_copy.state_dict(), ruta_modelo)
        print("[DECISIÓN] EL MODELO MEJORÓ. CAMBIOS GUARDADOS.")
        guardado = True
    elif f1_despues >= f1_antes - tolerancia:
        torch.save(model_copy.state_dict(), ruta_modelo)
        print(f"[DECISIÓN] EL MODELO EMPEORÓ LEVEMENTE (dentro de tolerancia {tolerancia}). CAMBIOS GUARDADOS.")
        guardado = True
    else:
        print(f"[DECISIÓN] EL MODELO EMPEORÓ DEMASIADO (cayó más de {tolerancia}). DESCARTANDO CAMBIOS para proteger el sistema.")
    print("="*50 + "\n")
        
    # Guardar historial (PROBLEMA 3e)
    historial_csv = BASE_DIR / "data_feedback/historial_reentrenamientos.csv"
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    df_hist = pd.DataFrame([{
        "timestamp": timestamp_str,
        "n_muestras_feedback": len(df_feedback),
        "f1_antes": f1_antes,
        "f1_despues": f1_despues,
        "guardado": guardado
    }])
    df_hist.to_csv(historial_csv, mode='a', header=not historial_csv.exists(), index=False)
    print("Historial actualizado.")
    
    # Retornar código de éxito incluso si no se guardó, porque el proceso terminó correctamente
    sys.exit(0)

if __name__ == "__main__":
    main()
