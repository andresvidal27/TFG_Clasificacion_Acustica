"""
entrenar_cnn.py
---------------
Define la arquitectura de la red convolucional (AudioCNN) y el código
necesario para entrenarla desde cero usando los espectrogramas Log-Mel.
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

# Importar configuración
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO CNN
# ==============================================================================
class AudioCNN(nn.Module):
    """Red Convolucional de 3 bloques para clasificación de espectrogramas."""
    def __init__(self, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

# ==============================================================================
# 2. DATASET DE ESPECTROGRAMAS
# ==============================================================================
class DatasetEspectrogramas(Dataset):
    """Carga los espectrogramas Log-Mel (.npy) guardados previamente."""
    def __init__(self, df: pd.DataFrame, aplicar_ruido=False, ruido_std=0.01):
        self.rutas = df["feature_path"].values
        self.etiquetas = df["label_id"].values
        self.aplicar_ruido = aplicar_ruido
        self.ruido_std = ruido_std

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        # Cargar numpy array y añadir dimension de canal: (1, 128, 216)
        logmel = np.load(self.rutas[idx])[np.newaxis, ...]
        if self.aplicar_ruido:
            # 1. Ruido Gaussiano
            ruido = np.random.normal(0, self.ruido_std, logmel.shape)
            logmel = logmel + ruido.astype(np.float32)
            
            # 2. Frequency Masking (enmascarar hasta 10 bandas de frecuencia)
            f_mask = np.random.randint(0, 15)
            f_0 = np.random.randint(0, max(1, logmel.shape[1] - f_mask))
            logmel[:, f_0:f_0+f_mask, :] = 0
            
            # 3. Time Masking (enmascarar hasta 20 frames de tiempo)
            t_mask = np.random.randint(0, 30)
            t_0 = np.random.randint(0, max(1, logmel.shape[2] - t_mask))
            logmel[:, :, t_0:t_0+t_mask] = 0
            
        x = torch.from_numpy(logmel).float()
        y = torch.tensor(self.etiquetas[idx], dtype=torch.long)
        return x, y

# ==============================================================================
# 3. ENTRENAMIENTO
# ==============================================================================
def calcular_pesos_clases(etiquetas, num_classes, device):
    """Calcula pesos para balancear las clases durante el entrenamiento."""
    conteos = np.bincount(etiquetas, minlength=num_classes).astype(float)
    pesos = len(etiquetas) / (num_classes * conteos)
    return torch.FloatTensor(pesos).to(device)

def entrenar_cnn(con_ruido=False):
    """Bucle principal de entrenamiento para la CNN."""
    print(f"Iniciando entrenamiento de la CNN {'(Con Aumento de Ruido)' if con_ruido else '(Base)'}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(BASE_DIR / "dataset_index_features.csv")
    df = df[df["feature_path"].notna()]
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = df["label_id"].nunique()

    # DataLoaders
    train_loader = DataLoader(DatasetEspectrogramas(train_df, aplicar_ruido=con_ruido), batch_size=32, shuffle=True)
    val_loader = DataLoader(DatasetEspectrogramas(val_df, aplicar_ruido=False), batch_size=32, shuffle=False)

    # Inicializar modelo y pérdida
    model = AudioCNN(num_classes).to(device)
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 10, 0
    nombre_modelo = "cnn_base"
    ruta_guardado = BASE_DIR / "models" / f"{nombre_modelo}_best.pt"
    ruta_guardado.parent.mkdir(exist_ok=True)

    for epoch in range(1, 101): # Máximo 100 epochs
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
    entrenar_cnn(con_ruido=True)
