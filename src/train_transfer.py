"""
train_transfer.py
-----------------
Script para entrenar la cabeza de clasificacion (TransferHead) sobre los
embeddings precomputados de CNN14.
Reutiliza la logica de entrenamiento generica de src/train.py.

Uso:
    python src/train_transfer.py
    python src/train_transfer.py --lr 1e-4 --epochs 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader

# Asegurar que src/ este en el path para importaciones relativas
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR
from train import train_model, compute_class_weights
from model_transfer import TransferHead


class EmbeddingDataset(Dataset):
    """Dataset que carga embeddings precomputados (.npy).

    Cada muestra se devuelve como un tensor 1D de 2048 dimensiones.
    """

    def __init__(self, df: pd.DataFrame):
        self.filepaths = df["embedding_path"].values
        self.labels = df["label_id"].values

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        emb = np.load(self.filepaths[idx])         # (2048,)
        x = torch.from_numpy(emb).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento TransferHead sobre embeddings CNN14"
    )
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximo de epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Paciencia del early stopping (default: 10)")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Cargar datos ─────────────────────────────────────────────────────
    csv_path = BASE_DIR / "dataset_index_emb.csv"
    if not csv_path.exists():
        print(f"[ERROR] No se encuentra {csv_path}.")
        print("Ejecuta primero: python src/precompute_embeddings.py")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)

    # Filtrar solo aquellos que tienen embedding (por si alguno fallo)
    df = df[df["embedding_path"].notna() & (df["embedding_path"] != "")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    num_classes = df["label_id"].nunique()
    print(f"Clases: {num_classes} | Train: {len(train_df)} | Val: {len(val_df)}")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_ds = EmbeddingDataset(train_df)
    val_ds = EmbeddingDataset(val_df)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Pesos de clase ───────────────────────────────────────────────────
    class_weights = compute_class_weights(
        train_df["label_id"].values, num_classes, device
    )
    print(f"Pesos de clase: {class_weights.cpu().numpy().round(3)}")

    # ── Modelo ───────────────────────────────────────────────────────────
    model = TransferHead(num_classes=num_classes)
    model_name = "transfer_head"

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {model_name} | "
          f"Parametros: {total_params:,} (entrenables: {trainable:,})")

    # ── Entrenar ─────────────────────────────────────────────────────────
    model, history = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    print("\nEntrenamiento completado.")
    final_epoch = len(history["val_acc"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    print(f"  Epochs ejecutados: {final_epoch}")
    print(f"  Mejor epoch: {best_epoch} "
          f"(val_loss={history['val_loss'][best_epoch-1]:.4f}, "
          f"val_acc={history['val_acc'][best_epoch-1]:.4f})")

if __name__ == "__main__":
    main()
