from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ESC50_DIR = BASE_DIR / "data_esc50"
DATA_URBAN_DIR = BASE_DIR / "data_urban"

# Audio parameters
SR = 22050
DURATION = 5.0
N_SAMPLES = 110250
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 11025
