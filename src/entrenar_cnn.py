"""
entrenar_cnn.py
---------------
Define la arquitectura de la red convolucional (AudioCNN) y el código
necesario para entrenarla desde cero (scratch) usando espectrogramas Log-Mel 2D.
Esta es nuestra red "sencilla" que nos sirve como baseline frente a Transfer Learning.

Incluye también la función de entrenamiento Ensemble (5 fases) que entrena 5 modelos
con semillas diferentes y permite promediar sus predicciones para mayor robustez.
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

# Semillas para el ensemble de 5 fases
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO CNN MEJORADO (ARQUITECTURA)
# ==============================================================================
class AudioCNN(nn.Module):
    """
    Red Convolucional Mejorada (VGG-style profunda). Toma una "imagen" en escala de grises
    (1 canal de profundidad, 128 de alto(frecuencias), 216 de ancho(tiempo)).
    
    Mejoras respecto a la versión anterior:
    - 5 bloques convolucionales (32→64→128→256→256) en lugar de 3
    - Dropout2d entre bloques para regularización espacial
    - Dual Pooling (Avg + Max) concatenado para capturar tanto la media como los picos
    - Clasificador más profundo con BatchNorm1d para estabilizar el entrenamiento
    """
    def __init__(self, num_classes=8):
        super().__init__()
        
        # El extractor de características usa 5 bloques de convolución apilados.
        self.features = nn.Sequential(
            # Bloque 1: Expande de 1 a 32 filtros.
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # Apaga mapas de características completos (regularización espacial leve)
            
            # Bloque 2: Expande a 64 filtros.
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Bloque 3: Expande a 128 filtros.
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Bloque 4: Expande a 256 filtros.
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Bloque 5: Se mantiene en 256 filtros para añadir profundidad sin duplicar parámetros
            # (subir a 512 con un dataset de este tamaño provocaría sobreajuste).
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        
        # Dual Pooling: Captura tanto la respuesta media (patrones generales) como el pico máximo
        # (características más salientes). Se concatenan para obtener un vector de 256*2 = 512.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # El clasificador final toma el vector concatenado de 512 dimensiones.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),       # 512 (256 avg + 256 max) → 256
            nn.BatchNorm1d(256),       # Normalización para estabilizar gradientes
            nn.ReLU(), 
            nn.Dropout(0.5),           # Apaga el 50% de las neuronas para evitar sobreajuste
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # Salida cruda (logits), sin softmax aún
        )

    def forward(self, x):
        """Define cómo fluye la información de la entrada 'x' hacia adelante."""
        feats = self.features(x)
        # Concatenamos el avg pooling y el max pooling en la dimensión de canales
        pooled = torch.cat([self.avg_pool(feats), self.max_pool(feats)], dim=1)
        return self.classifier(pooled)

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
            # 1. Ruido Gaussiano con intensidad variable (entre 0.5x y 1.5x del std base)
            ruido_intensidad = self.ruido_std * np.random.uniform(0.5, 1.5)
            ruido = np.random.normal(0, ruido_intensidad, logmel.shape)
            logmel = logmel + ruido.astype(np.float32)
            
            # 2. Frequency Masking más agresivo (Borra bandas horizontales simulando pérdida de frecuencia)
            # Ahora hasta 20 bandas (antes 15) para forzar a la red a no depender de una sola frecuencia
            f_mask = np.random.randint(0, 20)
            f_0 = np.random.randint(0, max(1, logmel.shape[1] - f_mask))
            logmel[:, f_0:f_0+f_mask, :] = 0
            
            # 3. Time Masking más agresivo (Borra bandas verticales simulando cortes de audio)
            # Ahora hasta 40 bandas (antes 30) para mayor robustez temporal
            t_mask = np.random.randint(0, 40)
            t_0 = np.random.randint(0, max(1, logmel.shape[2] - t_mask))
            logmel[:, :, t_0:t_0+t_mask] = 0
            
            # 4. Segundo Frequency Masking (doble enmascaramiento, probabilidad 50%)
            if np.random.random() > 0.5:
                f_mask2 = np.random.randint(0, 10)
                f_02 = np.random.randint(0, max(1, logmel.shape[1] - f_mask2))
                logmel[:, f_02:f_02+f_mask2, :] = 0
            
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

def _entrenar_cnn_una_vez(con_ruido=False, seed=42, nombre_modelo="cnn_base"):
    """
    Bucle principal de entrenamiento para una sola ejecución de la CNN.
    Parámetros:
        con_ruido: Si True, aplica Data Augmentation sobre los espectrogramas.
        seed: Semilla para reproducibilidad (distintas semillas = distintos modelos ensemble).
        nombre_modelo: Prefijo para guardar el archivo .pt y el historial JSON.
    """
    # Fijar semilla para reproducibilidad de esta fase particular
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\nIniciando entrenamiento de '{nombre_modelo}' (seed={seed}, ruido={con_ruido})...")
    
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
    # El generator con la semilla asegura que el barajado sea diferente para cada fase del ensemble
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(DatasetEspectrogramas(train_df, aplicar_ruido=con_ruido), batch_size=32, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(DatasetEspectrogramas(val_df, aplicar_ruido=False), batch_size=32, shuffle=False)

    # 1. Instanciar la arquitectura en la VRAM de la gráfica
    model = AudioCNN(num_classes).to(device)
    # 2. Función de Error / Pérdida (Cross Entropy) ponderada con nuestros pesos de desbalanceo
    pesos_clase = calcular_pesos_clases(train_df["label_id"].values, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    # 3. Optimizador AdamW (Adam con Weight Decay desacoplado) para regularización L2 más efectiva
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 4. Scheduler: Reduce el Learning Rate automáticamente si val_loss se estanca durante 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Variables para EARLY STOPPING (Cortar el entrenamiento antes si vemos que ya no aprende)
    mejor_loss = float("inf")
    paciencia, epochs_sin_mejora = 15, 0  # 15 epochs de paciencia (más que antes por el scheduler)
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
            
            # Gradient Clipping: Recorta los gradientes si son demasiado grandes (estabiliza el entrenamiento)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Actualizar el Learning Rate en base a la pérdida de validación
        scheduler.step(historial["val_loss"][-1])

        print(f"  Epoch {epoch:03d} | Train Loss: {historial['train_loss'][-1]:.4f} Acc: {historial['train_acc'][-1]:.4f} | Val Loss: {historial['val_loss'][-1]:.4f} Acc: {historial['val_acc'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- C. Lógica de Early Stopping ---
        # Si este epoch consiguió un error MENOR en los datos ocultos (Val), el modelo es objetivamente mejor
        if historial["val_loss"][-1] < mejor_loss:
            mejor_loss = historial["val_loss"][-1]
            torch.save(model.state_dict(), ruta_guardado) # ¡Se sobreescribe el checkpoint en disco!
            epochs_sin_mejora = 0 # Reiniciamos contador
        else:
            # Si se equivoca más o igual que la última vez...
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia: # Si llevamos 15 epochs seguidos estancados
                print(f"  Early stopping en epoch {epoch}. El modelo lleva {paciencia} epochs sin mejorar en Validación.")
                break

    # Guardar un registro JSON para que luego generar_todas_las_graficas.py pueda dibujarlo
    with open(BASE_DIR / "models" / f"history_{nombre_modelo}.json", "w") as f:
        json.dump(historial, f)
    print(f"  [OK] Modelo guardado en {ruta_guardado}")
    
    return historial

def entrenar_cnn(con_ruido=True):
    """Entrena un solo modelo CNN (comportamiento original, compatible con scripts existentes)."""
    print(f"Iniciando entrenamiento de la CNN {'(Con Aumento de Ruido)' if con_ruido else '(Base)'}...")
    _entrenar_cnn_una_vez(con_ruido=con_ruido, seed=42, nombre_modelo="cnn_base")
    print("[OK] Entrenamiento CNN individual completado.\n")

def entrenar_cnn_ensemble(con_ruido=True):
    """
    Entrena 5 modelos CNN con semillas diferentes (Ensemble de 5 fases).
    Cada modelo se guarda como 'cnn_ensemble_fold_0_best.pt' ... 'cnn_ensemble_fold_4_best.pt'.
    Al promediar las predicciones softmax de los 5 modelos se obtiene un resultado más robusto
    y estable, reduciendo la varianza de las predicciones individuales.
    """
    print("=" * 60)
    print("ENTRENAMIENTO ENSEMBLE CNN (5 FASES)")
    print("=" * 60)
    
    historiales = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n{'─' * 40}")
        print(f"FASE {i+1}/5 (Seed: {seed})")
        print(f"{'─' * 40}")
        h = _entrenar_cnn_una_vez(con_ruido=con_ruido, seed=seed, nombre_modelo=f"cnn_ensemble_fold_{i}")
        historiales.append(h)
    
    # Guardar resumen del ensemble
    resumen = {
        "seeds": ENSEMBLE_SEEDS,
        "num_folds": 5,
        "best_val_acc_per_fold": [max(h["val_acc"]) for h in historiales],
        "mean_best_val_acc": float(np.mean([max(h["val_acc"]) for h in historiales])),
    }
    with open(BASE_DIR / "models" / "ensemble_cnn_summary.json", "w") as f:
        json.dump(resumen, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE CNN COMPLETADO")
    print(f"  Mejor Val Acc por fold: {[f'{a:.4f}' for a in resumen['best_val_acc_per_fold']]}")
    print(f"  Media Best Val Acc:     {resumen['mean_best_val_acc']:.4f}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    # Primero entrenamos el modelo individual (compatible con el comportamiento original)
    entrenar_cnn(con_ruido=True)
    # Luego entrenamos el ensemble de 5 fases
    entrenar_cnn_ensemble(con_ruido=True)
