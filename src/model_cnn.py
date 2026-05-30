"""
model_cnn.py
------------
CNN propia para clasificacion de audio a partir de espectrogramas log-mel.

Arquitectura:
  3 bloques conv: Conv2d(3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)
  Canales: 1 -> 32 -> 64 -> 128
  AdaptiveAvgPool2d(1) (global average pooling)
  Flatten -> Linear(128,64) -> ReLU -> Dropout(0.5) -> Linear(64, num_classes)
  Sin softmax final (lo maneja CrossEntropyLoss).

Entrada esperada: (batch, 1, 128, 216)
"""

# pyrefly: ignore [missing-import]
import torch.nn as nn


class AudioCNN(nn.Module):
    """Red convolucional para clasificacion acustica."""

    def __init__(self, num_classes: int = 12):
        super().__init__()

        # ── 3 bloques convolucionales ────────────────────────────────────
        self.features = nn.Sequential(
            # Bloque 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ── Global average pooling ───────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Clasificador ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Parametros
        ----------
        x : Tensor de forma (batch, 1, 128, 216)

        Devuelve
        --------
        Tensor de forma (batch, num_classes) con logits sin activar.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
