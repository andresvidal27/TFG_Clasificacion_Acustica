"""
train.py
--------
Script generico de entrenamiento para modelos de clasificacion acustica.
Usa espectrogramas log-mel precomputados (.npy).

Parametrizable para reutilizar con distintos modelos (CNN propia,
transfer learning) y con/sin augmentation de ruido.

Uso:
    python src/train.py --model cnn
    python src/train.py --model cnn --noise
    python src/train.py --model cnn --noise --lr 5e-4 --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader

# Asegurar que src/ este en el path para importaciones relativas
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioFeatureDataset(Dataset):
    """Dataset que carga espectrogramas log-mel precomputados (.npy).

    Cada muestra se devuelve como un tensor (1, 128, 216) con canal unico.
    Opcionalmente aplica augmentation de ruido gaussiano.
    """

    def __init__(self, df: pd.DataFrame,
                 augment_noise: bool = False,
                 noise_std: float = 0.01):
        self.filepaths = df["feature_path"].values
        self.labels = df["label_id"].values
        self.augment_noise = augment_noise
        self.noise_std = noise_std

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        logmel = np.load(self.filepaths[idx])      # (128, 216)
        logmel = logmel[np.newaxis, ...]             # (1, 128, 216)

        if self.augment_noise:
            noise = np.random.normal(0, self.noise_std, logmel.shape)
            logmel = logmel + noise.astype(np.float32)

        x = torch.from_numpy(logmel).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── Utilidades ───────────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray,
                          num_classes: int,
                          device: torch.device) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada clase.

    weight_c = N_total / (num_classes * N_c)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = len(labels)
    weights = total / (num_classes * counts)
    return torch.FloatTensor(weights).to(device)


# ── Entrenamiento generico ───────────────────────────────────────────────────

def train_model(model: nn.Module,
                model_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                class_weights: torch.Tensor,
                device: torch.device,
                lr: float = 1e-3,
                max_epochs: int = 100,
                patience: int = 10,
                save_dir: Path | None = None) -> tuple:
    """Entrena un modelo con early stopping y guarda el mejor checkpoint.

    Parametros
    ----------
    model : nn.Module
        Cualquier modelo que acepte (batch, 1, 128, 216) y devuelva logits.
    model_name : str
        Nombre base para los archivos de checkpoint e historial.
    train_loader, val_loader : DataLoader
        DataLoaders de entrenamiento y validacion.
    class_weights : Tensor
        Pesos de clase para CrossEntropyLoss.
    device : torch.device
    lr : float
        Learning rate inicial para Adam.
    max_epochs : int
    patience : int
        Paciencia del early stopping (sobre val_loss).
    save_dir : Path
        Directorio donde guardar checkpoint e historial.

    Devuelve
    --------
    (model, history) donde model tiene los pesos del mejor epoch cargados,
    y history es un dict con listas de train_loss, val_loss, train_acc, val_acc.
    """
    if save_dir is None:
        save_dir = BASE_DIR / "models"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
    }

    checkpoint_path = save_dir / f"{model_name}_best.pt"
    history_path = save_dir / f"{model_name}_history.json"

    for epoch in range(1, max_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_batches = len(train_loader)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += X.size(0)
            
            # Simple progress
            if (i + 1) % max(1, train_batches // 4) == 0:
                pct = 100 * (i + 1) / train_batches
                print(f"\r  Epoch {epoch} - Train: {pct:.0f}%...", end="", flush=True)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        val_batches = len(val_loader)
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += X.size(0)
                
                if (i + 1) % max(1, val_batches // 2) == 0:
                    pct = 100 * (i + 1) / val_batches
                    print(f"\r  Epoch {epoch} - Val:   {pct:.0f}%...", end="", flush=True)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        # ── Registro ─────────────────────────────────────────────────────
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\rEpoch {epoch:3d}/{max_epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
              f"lr={current_lr:.1e}")

        # ── Early stopping ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Mejor modelo guardado ({checkpoint_path.name})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping en epoch {epoch} "
                      f"(sin mejora en {patience} epochs)")
                break

    # ── Guardar historial ────────────────────────────────────────────────
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistorial guardado en {history_path}")

    # ── Cargar mejor modelo ──────────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"Mejor modelo cargado desde {checkpoint_path}")

    return model, history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos de clasificacion acustica"
    )
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn",
                        help="Modelo a entrenar (default: cnn)")
    parser.add_argument("--noise", action="store_true",
                        help="Activar augmentation con ruido gaussiano")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Desviacion estandar del ruido (default: 0.01)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
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
    csv_path = BASE_DIR / "dataset_index_features.csv"
    df = pd.read_csv(csv_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    num_classes = df["label_id"].nunique()
    print(f"Clases: {num_classes} | Train: {len(train_df)} | Val: {len(val_df)}")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_ds = AudioFeatureDataset(
        train_df, augment_noise=args.noise, noise_std=args.noise_std
    )
    val_ds = AudioFeatureDataset(val_df, augment_noise=False)

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
    if args.model == "cnn":
        from model_cnn import AudioCNN
        model = AudioCNN(num_classes=num_classes)
        model_name = "audio_cnn"
    else:
        raise NotImplementedError(
            f"Modelo '{args.model}' aun no implementado. "
            f"Usa --model cnn por ahora."
        )

    if args.noise:
        model_name += "_noise"

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
