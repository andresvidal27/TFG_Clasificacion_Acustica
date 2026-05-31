"""
entrenar_transfer_v2.py
-----------------------
Define la arquitectura de la cabeza clasificadora (TransferHead) y la entrena
usando el dataset enriquecido (dataset_index_emb_v2.csv) que contiene muestras
con aumento de datos. Guarda el modelo en models_v2/.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Importar configuración original
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR
from entrenar_cnn import calcular_pesos_clases
from entrenar_transfer import TransferHead

# ==============================================================================
# DATASET DE EMBEDDINGS (V2)
# ==============================================================================
class DatasetEmbeddingsV2(Dataset):
    """Carga los vectores de características 1D (.npy) de PANNs CNN14."""
    def __init__(self, df: pd.DataFrame):
        self.rutas = df["embedding_path"].values
        self.etiquetas = df["label_id"].values

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        emb = np.load(self.rutas[idx])  # Array (2048,)
        x = torch.from_numpy(emb).float()
        y = torch.tensor(self.etiquetas[idx], dtype=torch.long)
        return x, y

# ==============================================================================
# BUCLE DE ENTRENAMIENTO
# ==============================================================================
def entrenar_transfer_v2():
    print("Iniciando entrenamiento Transfer Learning V2 (con Data Augmentation)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = BASE_DIR / "dataset_index_emb_v2.csv"
    
    if not csv_path.exists():
        print(f"Error: No se encontró {csv_path}. Ejecuta preparar_datos_v2.py primero.")
        return

    df = pd.read_csv(csv_path).dropna(subset=["embedding_path"])
    df = df[df["embedding_path"].notna()]
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = df["label_id"].nunique()

    print(f"Muestras de Entrenamiento: {len(train_df)}")
    print(f"Muestras de Validación: {len(val_df)}")

    # DataLoaders
    train_loader = DataLoader(DatasetEmbeddingsV2(train_df), batch_size=32, shuffle=True)
    val_loader = DataLoader(DatasetEmbeddingsV2(val_df), batch_size=32, shuffle=False)

    # Inicializar modelo y pérdida
    model = TransferHead(num_classes).to(device)
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    # Reducimos un poco el learning rate para estabilizar el aprendizaje con más datos
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 12, 0
    
    # NUEVA CARPETA PARA EL MODELO
    models_v2_dir = BASE_DIR / "models_v2"
    models_v2_dir.mkdir(exist_ok=True)
    ruta_guardado = models_v2_dir / "transfer_mejorado_best.pt"

    for epoch in range(1, 101):
        # --- Fase Entrenamiento ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            salida = model(X)
            loss = criterion(salida, y)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * X.size(0)
            t_correct += (salida.argmax(1) == y).sum().item()
            t_total += y.size(0)
            
        historial["train_loss"].append(t_loss / t_total)
        historial["train_acc"].append(t_correct / t_total)

        # --- Fase Validación ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                salida = model(X)
                v_loss += criterion(salida, y).item() * X.size(0)
                v_correct += (salida.argmax(1) == y).sum().item()
                v_total += y.size(0)
                
        historial["val_loss"].append(v_loss / v_total)
        historial["val_acc"].append(v_correct / v_total)

        print(f"Epoch {epoch:03d} | Train Loss: {historial['train_loss'][-1]:.4f} Acc: {historial['train_acc'][-1]:.4f} | Val Loss: {historial['val_loss'][-1]:.4f} Acc: {historial['val_acc'][-1]:.4f}")

        # Early Stopping
        if historial["val_loss"][-1] < mejor_loss:
            mejor_loss = historial["val_loss"][-1]
            torch.save(model.state_dict(), ruta_guardado)
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia:
                print(f"Early stopping en epoch {epoch}.")
                break

    # Guardar historial
    with open(models_v2_dir / "history_transfer_mejorado.json", "w") as f:
        json.dump(historial, f)
    print(f"\n[OK] ¡Modelo V2 guardado exitosamente en {ruta_guardado}!")

if __name__ == "__main__":
    entrenar_transfer_v2()
