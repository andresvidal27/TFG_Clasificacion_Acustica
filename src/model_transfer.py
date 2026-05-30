"""
model_transfer.py
-----------------
Define la cabeza de clasificacion (TransferHead) para el modelo de transfer learning.
El modelo base (CNN14) se utiliza solo como extractor de caracteristicas (embeddings)
y no se define aqui para evitar dependencias innecesarias en el entrenamiento.
"""

# pyrefly: ignore [missing-import]
import torch.nn as nn

class TransferHead(nn.Module):
    """Cabeza densa para clasificacion sobre embeddings de CNN14.
    
    Toma como entrada el embedding de 2048 dimensiones extraido de CNN14,
    lo reduce a 256, aplica dropout y devuelve logits para `num_classes`.
    """
    def __init__(self, num_classes=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Parametros
        ----------
        x : Tensor de forma (batch, 2048)

        Devuelve
        --------
        Tensor de forma (batch, num_classes) con logits sin activar.
        """
        return self.fc(x)
