import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# pyrefly: ignore [missing-import]
from entrenar_transfer import TransferHead
# pyrefly: ignore [missing-import]
from datos_y_configuracion import CLASS_MAP
from panns_inference import AudioTagging
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASS_MAP)

# Load model
model = TransferHead(num_classes).to(device)
model.load_state_dict(torch.load(BASE_DIR / "models/transfer_head_best.pt", map_location=device, weights_only=True))
model.eval()

# Cargar PANNs solo si es estrictamente necesario (lazy loading)
panns = None

# Load feedback
df = pd.read_csv(BASE_DIR / "data_feedback/feedback_index.csv")

for _, row in df.iterrows():
    # Intentar cargar embedding precomputado si existe
    if "embedding_path" in row and pd.notna(row["embedding_path"]) and Path(row["embedding_path"]).exists():
        emb_val = np.load(row["embedding_path"])
    else:
        # Fallback: cargar audio y pasarlo por PANNs
        if panns is None:
            print("Cargando PANNs para audios sin precomputar...")
            panns = AudioTagging(checkpoint_path=str(BASE_DIR / "models/Cnn14_mAP=0.431.pth"), device=str(device))
            
        y, _ = librosa.load(row["filepath"], sr=32000, mono=True)
        if len(y) > 5 * 32000: y = y[:5 * 32000]
        else: y = np.pad(y, (0, 5 * 32000 - len(y)))
        
        with torch.no_grad():
            _, emb = panns.inference(y[None, :])
            emb_val = emb[0]
            
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(emb_val).float().unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
        
    pred_idx = np.argmax(probs)
    pred_class = CLASS_MAP[pred_idx]
    
    print(f"Audio: {Path(row['filepath']).name}")
    print(f"True: {row['label_name']} (ID: {row['label_id']})")
    print(f"Pred: {pred_class} (Prob: {probs[pred_idx]:.2f})")
    print("-" * 30)
