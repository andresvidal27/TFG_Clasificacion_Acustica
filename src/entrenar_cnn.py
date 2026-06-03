"""
entrenar_cnn.py
---------------
Define la arquitectura de la red convolucional (AudioCNN) y el código
necesario para entrenarla desde cero (scratch) usando espectrogramas Log-Mel 2D.
Esta es nuestra red "sencilla" que nos sirve como baseline frente a Transfer Learning.
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

# Importar configuración global
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_y_configuracion import BASE_DIR

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO CNN (ARQUITECTURA)
# ==============================================================================
class AudioCNN(nn.Module):
    """
    Red Convolucional Clásica (VGG-style). Toma una "imagen" en escala de grises
    (1 canal de profundidad, 128 de alto(frecuencias), 216 de ancho(tiempo)).
    """
    def __init__(self, num_classes=8):
        super().__init__()
        
        # El extractor de características usa 3 bloques de convolución apilados.
        self.features = nn.Sequential(
            # Bloque 1: Expande de 1 a 32 filtros. El BatchNorm estabiliza, ReLU activa, MaxPool reduce el tamaño espacial a la mitad.
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Bloque 2: Expande a 64 filtros.
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Bloque 3: Expande a 128 filtros.
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # AdaptiveAvgPool2d colapsa lo que quede del eje X e Y en un solo píxel (1x1) de 128 canales.
        # Esto nos independiza de la longitud exacta del audio en segundos.
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # El clasificador final toma el vector 128, pasa por una capa densa y acaba en el número de clases.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Dropout(0.5), # Apaga el 50% de las neuronas aleatoriamente para evitar sobreajuste (memorización)
            nn.Linear(64, num_classes) # Salida cruda (logits), sin softmax aún
        )

    def forward(self, x):
        """Define cómo fluye la información de la entrada 'x' hacia adelante."""
        return self.classifier(self.pool(self.features(x)))

# ==============================================================================
# 2. DATASET DE ESPECTROGRAMAS (DATALOADER CUSTOM)
# ==============================================================================
class DatasetEspectrogramas(Dataset):
    """
    Dataset para PyTorch. Carga los espectrogramas Log-Mel (.npy) precalculados por
    'datos_y_configuracion.py' directamente desde el disco duro en demanda para ahorrar RAM.
    """
    def __init__(self, df: pd.DataFrame, aplicar_ruido=False, ruido_std=0.01):
        self.rutas = df["feature_path"].values
        self.etiquetas = df["label_id"].values
        self.aplicar_ruido = aplicar_ruido
        self.ruido_std = ruido_std

    def __len__(self):
        """PyTorch necesita saber la cantidad total de datos para organizar las iteraciones."""
        return len(self.rutas)

    def __getitem__(self, idx):
        """Recupera un solo ejemplo de entrenamiento en el índice 'idx'."""
        # Se carga el numpy array y se le añade una dimensión falsa [np.newaxis] 
        # para que pase de (128, 216) a (1, 128, 216), representando el "canal de color"
        logmel = np.load(self.rutas[idx])[np.newaxis, ...]
        
        # Data Augmentation dinámico sobre el Espectrograma (SpecAugment y Ruido)
        # Esto ocurre SOBRE LA MARCHA, por lo que cada época la red ve variaciones diferentes
        if self.aplicar_ruido:
            # 1. Ruido Gaussiano
            ruido = np.random.normal(0, self.ruido_std, logmel.shape)
            logmel = logmel + ruido.astype(np.float32)
            
            # 2. Frequency Masking (Bórra bandas horizontales simulando pérdida de una banda de frecuencia)
            f_mask = np.random.randint(0, 15)
            f_0 = np.random.randint(0, max(1, logmel.shape[1] - f_mask))
            logmel[:, f_0:f_0+f_mask, :] = 0
            
            # 3. Time Masking (Borra bandas verticales simulando un pequeño corte en el audio)
            t_mask = np.random.randint(0, 30)
            t_0 = np.random.randint(0, max(1, logmel.shape[2] - t_mask))
            logmel[:, :, t_0:t_0+t_mask] = 0
            
        x = torch.from_numpy(logmel).float()
        y = torch.tensor(self.etiquetas[idx], dtype=torch.long)
        return x, y

# ==============================================================================
# 3. LÓGICA DEL BUCLE DE ENTRENAMIENTO
# ==============================================================================
def calcular_pesos_clases(etiquetas, num_classes, device):
    """
    Como tenemos clases con menos muestras que otras (desbalanceo), calculamos un
    'multiplicador' de castigo. Si la red se equivoca en una clase poco frecuente, el error
    se multiplica por un número mayor.
    """
    conteos = np.bincount(etiquetas, minlength=num_classes).astype(float)
    # Suavizado de pesos usando raíz cuadrada para evitar inestabilidad en desbalances extremos
    pesos = len(etiquetas) / (num_classes * np.sqrt(conteos))
    return torch.FloatTensor(pesos).to(device)

def entrenar_cnn(con_ruido=False):
    """Bucle principal de entrenamiento (Fine Tuning y Early Stopping)."""
    print(f"Iniciando entrenamiento de la CNN {'(Con Aumento de Ruido)' if con_ruido else '(Base)'}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargamos el CSV que indica qué archivo .npy usar
    df = pd.read_csv(BASE_DIR / "dataset_index_features.csv").dropna(subset=["feature_path"])
    df = df[df["feature_path"].notna()]
    
    # Separamos en los Splits que definimos inicialmente
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = df["label_id"].nunique()

    # DataLoaders (Iteradores que amontonan los datos en 'batches' o paquetes de 32 para procesar en paralelo en GPU)
    # Solo se baraja y se aumenta el conjunto Train.
    train_loader = DataLoader(DatasetEspectrogramas(train_df, aplicar_ruido=con_ruido), batch_size=32, shuffle=True)
    val_loader = DataLoader(DatasetEspectrogramas(val_df, aplicar_ruido=False), batch_size=32, shuffle=False)

    # 1. Instanciar la arquitectura en la VRAM de la gráfica
    model = AudioCNN(num_classes).to(device)
    # 2. Función de Error / Pérdida (Cross Entropy) ponderada con nuestros pesos de desbalanceo
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    # 3. Optimizador (Adam) que actualizará los pesos usando descenso de gradiente
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Variables para EARLY STOPPING (Cortar el entrenamiento antes si vemos que ya no aprende)
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 10, 0
    nombre_modelo = "cnn_base"
    ruta_guardado = BASE_DIR / "models" / f"{nombre_modelo}_best.pt"
    ruta_guardado.parent.mkdir(exist_ok=True)

    # Un Epoch significa que la red ha visto TODOS los datos de entrenamiento exactamente una vez.
    for epoch in range(1, 101): # Límite estricto superior: 100
    
        # --- A. Fase de Entrenamiento (La red APRENDE) ---
        model.train() # Activa el Dropout y el BatchNorm
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad() # Limpiar los gradientes de la iteración anterior
            
            salida = model(X) # Forward Pass (Predecir)
            loss = criterion(salida, y) # Calcular cuánto nos hemos equivocado
            loss.backward() # Calcular derivadas (Backward Pass)
            optimizer.step() # Actualizar los pesos de la red neuronal
            
            # Llevar la cuenta estadística
            t_loss += loss.item() * X.size(0)
            t_correct += (salida.argmax(1) == y).sum().item()
            t_total += y.size(0)
            
        historial["train_loss"].append(t_loss / t_total)
        historial["train_acc"].append(t_correct / t_total)

        # --- B. Fase de Validación (La red se PONE A PRUEBA) ---
        model.eval() # Congelar pesos, desactivar Dropout
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad(): # Desactivar motor matemático de gradientes (ahorra memoria)
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                salida = model(X)
                
                v_loss += criterion(salida, y).item() * X.size(0)
                v_correct += (salida.argmax(1) == y).sum().item()
                v_total += y.size(0)
                
        historial["val_loss"].append(v_loss / v_total)
        historial["val_acc"].append(v_correct / v_total)

        print(f"Epoch {epoch:03d} | Train Loss: {historial['train_loss'][-1]:.4f} Acc: {historial['train_acc'][-1]:.4f} | Val Loss: {historial['val_loss'][-1]:.4f} Acc: {historial['val_acc'][-1]:.4f}")

        # --- C. Lógica de Early Stopping ---
        # Si este epoch consiguió un error MENOR en los datos ocultos (Val), el modelo es objetivamente mejor
        if historial["val_loss"][-1] < mejor_loss:
            mejor_loss = historial["val_loss"][-1]
            torch.save(model.state_dict(), ruta_guardado) # ¡Se sobreescribe el checkpoint en disco!
            epochs_sin_mejora = 0 # Reiniciamos contador
        else:
            # Si se equivoca más o igual que la última vez...
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia: # Si llevamos 10 epochs seguidos estancados
                print(f"Early stopping en epoch {epoch}. El modelo lleva 10 epochs sin mejorar en Validación.")
                break

    # Guardar un registro JSON para que luego generar_todas_las_graficas.py pueda dibujarlo
    with open(BASE_DIR / "models" / f"history_{nombre_modelo}.json", "w") as f:
        json.dump(historial, f)
    print(f"[OK] Modelo guardado en {ruta_guardado}\n")

if __name__ == "__main__":
    entrenar_cnn(con_ruido=True)
