"""
precompute_features.py
----------------------
Recorre dataset_index.csv, extrae el espectrograma log-mel de cada clip,
lo guarda en features/{label_id}_{indice}.npy y genera dataset_index_features.csv
con la columna adicional feature_path.

Uso:
    python src/precompute_features.py

El directorio features/ se crea automáticamente si no existe.
Al final se verifica que todos los .npy tengan forma (N_MELS, 216).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path


# Asegurar que src/ esté en el path para las importaciones relativas
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR, N_MELS
from features import preprocess_audio, compute_logmel

# ── Rutas ────────────────────────────────────────────────────────────────────
INDEX_CSV    = BASE_DIR / "dataset_index.csv"
OUTPUT_CSV   = BASE_DIR / "dataset_index_features.csv"
FEATURES_DIR = BASE_DIR / "features"

EXPECTED_SHAPE = (N_MELS, 216)

# ── Preparar directorio de salida ────────────────────────────────────────────
FEATURES_DIR.mkdir(exist_ok=True)

# ── Leer índice ───────────────────────────────────────────────────────────────
df = pd.read_csv(INDEX_CSV)
print(f"Clips a procesar: {len(df)}")

# ── Calcular y guardar features ──────────────────────────────────────────────
feature_paths: list[str] = []
errors: list[int] = []

total = len(df)
for idx, row in df.iterrows():
    if (idx + 1) % max(1, total // 10) == 0:
        print(f"  {100*(idx+1)//total}% ({idx+1}/{total})", flush=True)
    out_name = f"{row['label_id']}_{idx}.npy"
    out_path = FEATURES_DIR / out_name

    try:
        signal  = preprocess_audio(row["filepath"])
        log_mel = compute_logmel(signal)
        np.save(out_path, log_mel)
        feature_paths.append(str(out_path))
    except Exception as e:
        print(f"\n[ERROR] idx={idx} | {row['filepath']}\n  → {e}")
        feature_paths.append("")
        errors.append(idx)

# ── Guardar CSV actualizado ───────────────────────────────────────────────────
df["feature_path"] = feature_paths
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV guardado en {OUTPUT_CSV}")

# ── Verificación de formas ────────────────────────────────────────────────────
print("\n=== Verificando formas de los .npy ===")
wrong_shape: list[str] = []
for path in df["feature_path"]:
    if not path:
        continue
    arr = np.load(path)
    if arr.shape != EXPECTED_SHAPE:
        wrong_shape.append(f"{path}  →  {arr.shape}")

if wrong_shape:
    print(f"[ADVERTENCIA] {len(wrong_shape)} archivo(s) con forma incorrecta:")
    for s in wrong_shape:
        print(f"  {s}")
else:
    print(f"Todos los {len(df) - len(errors)} .npy tienen la forma correcta {EXPECTED_SHAPE} [OK]")

if errors:
    print(f"\n[ADVERTENCIA] {len(errors)} clip(s) fallaron durante el procesado:")
    for i in errors:
        print(f"  idx={i}  {df.loc[i, 'filepath']}")
else:
    print("Sin errores de procesado [OK]")
