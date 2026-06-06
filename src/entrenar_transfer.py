"""
entrenar_transfer.py
--------------------
Define la arquitectura de la cabeza clasificadora (TransferHead) y el código
para entrenarla usando los embeddings extraídos por el modelo PANNs CNN14.
Esto es Transfer Learning: aprovechamos un modelo ya preentrenado con millones de 
audios (PANNs) que nos da un vector matemático (embedding) de 2048 dimensiones.
Nosotros solo entrenamos esta pequeña red (cabeza) para que traduzca ese vector a nuestras 8 clases.

Incluye también la función de entrenamiento Ensemble (5 fases) que entrena 5 cabezas
con semillas diferentes y permite promediar sus predicciones para mayor robustez.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Importar configuración global y utilidades del otro script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR
from entrenar_cnn import calcular_pesos_clases, ENSEMBLE_SEEDS

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO TRANSFER LEARNING (LA "CABEZA")
# ==============================================================================
class TransferHead(nn.Module):
    """
    Cabeza densa (Perceptrón Multicapa) para clasificar a partir de embeddings.
    A diferencia de la CNN base, aquí no hay convoluciones, porque PANNs ya hizo 
    todo el trabajo pesado de entender el sonido.
    """
    def __init__(self, num_classes=8):
        super().__init__()
        self.fc = nn.Sequential(
            # Capa 1: Recibe el embedding de PANNs (2048 dimensiones) y lo comprime a 256.
            nn.Linear(2048, 256),
            nn.ReLU(),           # Activación no lineal
            nn.Dropout(0.3),     # Apaga un 30% de las conexiones aleatoriamente para evitar memorizar
            # Capa 2: De las 256 dimensiones extraídas, predice una de las 8 clases finales.
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Pasa el embedding 'x' por las capas lineales."""
        return self.fc(x)

# ==============================================================================
# 2. DATASET DE EMBEDDINGS (CARGA DE DATOS)
# ==============================================================================
class DatasetEmbeddings(Dataset):
    """
    Dataset optimizado para Transfer Learning. En lugar de procesar audios pesados
    o imágenes (espectrogramas), simplemente lee matrices 1D precalculadas (.npy).
    Esto permite que el entrenamiento dure segundos en vez de horas.
    """
    def __init__(self, df: pd.DataFrame):
        self.rutas = df["embedding_path"].values
        self.etiquetas = df["label_id"].values

    def __len__(self):
        """PyTorch necesita saber cuántos datos hay para organizar las épocas."""
        return len(self.rutas)

    def __getitem__(self, idx):
        """Devuelve un par (Datos, Etiqueta) para el índice solicitado."""
        emb = np.load(self.rutas[idx])  # Array (2048,)
        x = torch.from_numpy(emb).float() # Lo convierte a tensor de PyTorch
        y = torch.tensor(self.etiquetas[idx], dtype=torch.long)
        return x, y

# ==============================================================================
# 3. BUCLE DE ENTRENAMIENTO PRINCIPAL
# ==============================================================================
def _entrenar_transfer_una_vez(seed=42, nombre_modelo="transfer_head"):
    """
    Bucle principal de entrenamiento para una sola ejecución de la cabeza Transfer Learning.
    Parámetros:
        seed: Semilla para reproducibilidad (distintas semillas = distintos modelos ensemble).
        nombre_modelo: Prefijo para guardar el archivo .pt y el historial JSON.
    """
    # Fijar semilla para reproducibilidad de esta fase particular
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\nIniciando entrenamiento de '{nombre_modelo}' (seed={seed})...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargamos el archivo CSV que apunta a los embeddings precomputados (.npy)
    df = pd.read_csv(BASE_DIR / "dataset_index_emb.csv").dropna(subset=["embedding_path"])
    df = df[df["embedding_path"].notna()]
    
    # Separamos en los mismos splits garantizando la reproducibilidad
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = df["label_id"].nunique()

    # DataLoaders: Generan lotes de 32 muestras
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(DatasetEmbeddings(train_df), batch_size=32, shuffle=True, generator=g)
    val_loader = DataLoader(DatasetEmbeddings(val_df), batch_size=32, shuffle=False)

    # 1. Instanciamos la cabeza en la GPU (o CPU)
    model = TransferHead(num_classes).to(device)
    
    # 2. Penalizamos los errores en clases menos representadas calculando pesos
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    
    # 3. Optimizador Adam. Usamos lr=5e-5 porque transfer learning es sensible y 
    # queremos ajustes muy finos. Weight_decay añade regularización L2 (evita sobreajuste).
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Diccionario para trazar la curva de aprendizaje
    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 12, 0 # Permitimos 12 épocas sin mejora antes de abortar
    ruta_guardado = BASE_DIR / "models" / f"{nombre_modelo}_best.pt"
    ruta_guardado.parent.mkdir(exist_ok=True)

    for epoch in range(1, 101):
        # --- Fase de Entrenamiento ---
        model.train() # Activa el Dropout
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad() # Limpia gradientes
            
            salida = model(X) # Forward Pass
            loss = criterion(salida, y) # Calcular pérdida
            loss.backward() # Backpropagation (derivar)
            optimizer.step() # Actualizar los pesos de TransferHead
            
            t_loss += loss.item() * X.size(0)
            t_correct += (salida.argmax(1) == y).sum().item()
            t_total += y.size(0)
            
        historial["train_loss"].append(t_loss / t_total)
        historial["train_acc"].append(t_correct / t_total)

        # --- Fase de Validación (Prueba ciega) ---
        model.eval() # Congela el Dropout
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

        print(f"  Epoch {epoch:03d} | Train Loss: {historial['train_loss'][-1]:.4f} Acc: {historial['train_acc'][-1]:.4f} | Val Loss: {historial['val_loss'][-1]:.4f} Acc: {historial['val_acc'][-1]:.4f}")

        # --- Lógica de Early Stopping ---
        # Guardar el modelo en disco SOLAMENTE si bate el récord de pérdida mínima en Validación
        if historial["val_loss"][-1] < mejor_loss:
            mejor_loss = historial["val_loss"][-1]
            torch.save(model.state_dict(), ruta_guardado)
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia:
                print(f"  Early stopping en epoch {epoch}.")
                break

    # Guardar historial en un JSON para poder graficarlo después
    with open(BASE_DIR / "models" / f"history_{nombre_modelo}.json", "w") as f:
        json.dump(historial, f)
    print(f"  [OK] Modelo guardado en {ruta_guardado}")
    
    return historial

def entrenar_transfer():
    """Entrena un solo modelo Transfer Learning (comportamiento original)."""
    print("Iniciando entrenamiento del modelo Transfer Learning (con Data Augmentation)...")
    _entrenar_transfer_una_vez(seed=42, nombre_modelo="transfer_head")
    print("[OK] Entrenamiento Transfer Learning individual completado.\n")

def entrenar_transfer_ensemble():
    """
    Entrena 5 cabezas Transfer Learning con semillas diferentes (Ensemble de 5 fases).
    Cada modelo se guarda como 'transfer_ensemble_fold_0_best.pt' ... 'transfer_ensemble_fold_4_best.pt'.
    Al promediar las predicciones softmax de los 5 modelos se obtiene un resultado más robusto
    y estable, reduciendo la varianza de las predicciones individuales.
    """
    print("=" * 60)
    print("ENTRENAMIENTO ENSEMBLE TRANSFER LEARNING (5 FASES)")
    print("=" * 60)
    
    historiales = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n{'─' * 40}")
        print(f"FASE {i+1}/5 (Seed: {seed})")
        print(f"{'─' * 40}")
        h = _entrenar_transfer_una_vez(seed=seed, nombre_modelo=f"transfer_ensemble_fold_{i}")
        historiales.append(h)
    
    # Guardar resumen del ensemble
    resumen = {
        "seeds": ENSEMBLE_SEEDS,
        "num_folds": 5,
        "best_val_acc_per_fold": [max(h["val_acc"]) for h in historiales],
        "mean_best_val_acc": float(np.mean([max(h["val_acc"]) for h in historiales])),
    }
    with open(BASE_DIR / "models" / "ensemble_transfer_summary.json", "w") as f:
        json.dump(resumen, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE TRANSFER LEARNING COMPLETADO")
    print(f"  Mejor Val Acc por fold: {[f'{a:.4f}' for a in resumen['best_val_acc_per_fold']]}")
    print(f"  Media Best Val Acc:     {resumen['mean_best_val_acc']:.4f}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    # Primero entrenamos el modelo individual (compatible con el comportamiento original)
    entrenar_transfer()
    # Luego entrenamos el ensemble de 5 fases
    entrenar_transfer_ensemble()
