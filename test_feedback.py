"""
test_feedback.py
----------------
Script de prueba para validar el clasificador con datos de retroalimentación (feedback).
Carga el modelo Transfer Learning (TransferHead) y evalúa las predicciones sobre los
audios registrados en el histórico de feedback de los usuarios, comparándolas con las etiquetas reales.
"""

import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import warnings

# Desactivar advertencias molestas de librerías externas (como librosa/pytorch)
warnings.filterwarnings("ignore")

# Configuración de la ruta base y adición del directorio 'src' para importar módulos locales
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# pyrefly: ignore [missing-import]
from entrenar_transfer import TransferHead
# pyrefly: ignore [missing-import]
from datos_y_configuracion import CLASS_MAP
from panns_inference import AudioTagging
import librosa

# Configurar el dispositivo de computación (GPU si está disponible, o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASS_MAP)

# ==============================================================================
# CARGAR MODELO CLASIFICADOR (TRANSFER HEAD)
# ==============================================================================
print("[Info] Cargando clasificador final...")
model = TransferHead(num_classes).to(device)
# Cargar los pesos guardados del mejor entrenamiento de la cabeza de transferencia
model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
model.eval() # Poner en modo de evaluación para desactivar dropout/batchnorm

# Cargar PANNs de forma perezosa (lazy loading) solo si hay audios sin embedding precomputado
panns = None

# Cargar el archivo de índice del feedback de los usuarios
print("[Info] Leyendo datos de feedback...")
df = pd.read_csv(BASE_DIR / "data_feedback/feedback_index.csv")

print("\n--- INICIANDO PRUEBA DE EVALUACIÓN DE FEEDBACK ---\n")

# Iterar sobre cada registro del conjunto de datos de feedback
for _, row in df.iterrows():
    # 1. OPTIMIZACIÓN: Cargar el embedding precomputado (.npy) si existe la ruta y el archivo
    if "embedding_path" in row and pd.notna(row["embedding_path"]) and Path(row["embedding_path"]).exists():
        emb_val = np.load(row["embedding_path"])
    else:
        # FALLBACK: Si no existe el embedding precomputado, extraerlo sobre la marcha
        if panns is None:
            print("[PANNs] Inicializando extractor de características robustas (Cnn14)...")
            panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
            
        # Cargar el archivo de audio nativo a 32kHz (frecuencia esperada por PANNs)
        y, _ = librosa.load(row["filepath"], sr=32000, mono=True)
        
        # Ajustar la duración a exactamente 5 segundos (recortar o añadir padding de ceros)
        if len(y) > 5 * 32000:
            y = y[:5 * 32000]
        else:
            y = np.pad(y, (0, 5 * 32000 - len(y)))
        
        # Extraer el embedding usando la red neuronal pre-entrenada PANNs (sin gradientes)
        with torch.no_grad():
            _, emb = panns.inference(y[None, :])
            emb_val = emb[0]
            
    # 2. INFERENCIA CON EL MODELO PERSONALIZADO (TRANSFERHEAD)
    with torch.no_grad():
        # Convertir a tensor, añadir dimensión de batch, pasar por el clasificador y aplicar Softmax
        emb_tensor = torch.from_numpy(emb_val).float().unsqueeze(0).to(device)
        probs = F.softmax(model(emb_tensor), dim=1).cpu().numpy()[0]
        
    # Obtener el índice con la probabilidad máxima y decodificar el nombre de la clase
    pred_idx = np.argmax(probs)
    pred_class = CLASS_MAP[pred_idx]
    
    # 3. MOSTRAR RESULTADOS
    print(f"Audio: {Path(row['filepath']).name}")
    print(f"True: {row['label_name']} (ID: {row['label_id']})")
    print(f"Pred: {pred_class} (Prob: {probs[pred_idx]:.2f})")
    print("-" * 30)
