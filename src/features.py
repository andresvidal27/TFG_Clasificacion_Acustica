"""
features.py
-----------
Funciones de extracción de características de audio:
  - preprocess_audio: carga y normaliza un clip de audio a N_SAMPLES muestras.
  - compute_logmel: calcula el espectrograma log-mel de forma (N_MELS, 216).
"""

import numpy as np
import librosa

from config import SR, N_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX


def preprocess_audio(filepath: str) -> np.ndarray:
    """Carga un clip de audio y lo devuelve normalizado con longitud fija.

    Pasos:
      1. Carga el audio a SR Hz mono con librosa.
      2. Ajusta a exactamente N_SAMPLES muestras:
         - Recorta si es más largo.
         - Rellena con ceros al final si es más corto.
      3. Normaliza la amplitud al rango [-1, 1] dividiendo por el máximo
         absoluto (si la señal no es silencio).

    Parámetros
    ----------
    filepath : str
        Ruta absoluta o relativa al archivo de audio.

    Devuelve
    --------
    np.ndarray
        Array de forma (N_SAMPLES,) con dtype float32.
    """
    signal, _ = librosa.load(filepath, sr=SR, mono=True)

    # Ajustar longitud
    if len(signal) > N_SAMPLES:
        signal = signal[:N_SAMPLES]
    elif len(signal) < N_SAMPLES:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)))

    # Normalizar amplitud a [-1, 1]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal.astype(np.float32)


def compute_logmel(signal: np.ndarray) -> np.ndarray:
    """Calcula el espectrograma log-mel de una señal de audio.

    Usa los parámetros definidos en config.py:
      SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX.

    La transformación aplicada es: log(S + 1e-6) para evitar log(0).

    Parámetros
    ----------
    signal : np.ndarray
        Señal de audio de forma (N_SAMPLES,) con dtype float32.

    Devuelve
    --------
    np.ndarray
        Espectrograma log-mel de forma (N_MELS, 216) con dtype float32.
        Con la configuración por defecto: (128, 216).
    """
    S = librosa.feature.melspectrogram(
        y=signal,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    log_S = np.log(S + 1e-6)
    return log_S.astype(np.float32)
