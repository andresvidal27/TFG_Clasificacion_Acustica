"""
entrenar_transfer.py
--------------------
Define la arquitectura de la cabeza clasificadora (TransferHead) y el código
para entrenarla usando los embeddings extraídos por el modelo CNN14 (Transfer Learning).
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Importar configuración
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR
from entrenar_cnn import calcular_pesos_clases

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO TRANSFER LEARNING
# ==============================================================================
class TransferHead(nn.Module):
    """Cabeza densa para clasificar a partir de embeddings (2048 dimensiones)."""
    def __init__(self, num_classes=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# ==============================================================================
# 2. DATASET DE EMBEDDINGS
# ==============================================================================
class DatasetEmbeddings(Dataset):
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
# 3. ENTRENAMIENTO
# ==============================================================================
def entrenar_transfer():
    """Bucle principal de entrenamiento para el clasificador Transfer Learning."""
    print("Iniciando entrenamiento del modelo Transfer Learning (CNN14 Embeddings)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(BASE_DIR / "dataset_index_emb.csv")
    df = df[df["embedding_path"].notna()]
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = df["label_id"].nunique()

    # DataLoaders
    train_loader = DataLoader(DatasetEmbeddings(train_df), batch_size=32, shuffle=True)
    val_loader = DataLoader(DatasetEmbeddings(val_df), batch_size=32, shuffle=False)

    # Inicializar modelo y pérdida
    model = TransferHead(num_classes).to(device)
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 10, 0
    nombre_modelo = "transfer_head"
    ruta_guardado = BASE_DIR / "models" / f"{nombre_modelo}_best.pt"
    ruta_guardado.parent.mkdir(exist_ok=True)

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
    with open(BASE_DIR / "models" / f"history_{nombre_modelo}.json", "w") as f:
        json.dump(historial, f)
    print(f"[OK] Modelo guardado en {ruta_guardado}\n")

if __name__ == "__main__":
    entrenar_transfer()
