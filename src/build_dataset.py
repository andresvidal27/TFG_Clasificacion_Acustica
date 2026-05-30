"""
build_dataset.py
----------------
Lee data_esc50/esc50.csv y data_urban/UrbanSound8K.csv, construye un DataFrame
unificado con 11 clases de audio, realiza un split estratificado 70/15/15
y guarda el resultado en dataset_index.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import BASE_DIR, DATA_ESC50_DIR, DATA_URBAN_DIR

SEED = 42

# ── Mapa de clases ──────────────────────────────────────────────────────────
CLASS_MAP = {
    0: "glass_breaking",
    1: "gun_shot",
    2: "dog_bark",
    3: "siren",
    4: "car_horn",
    5: "crying_baby",
    6: "thunderstorm",
    7: "fireworks",
    8: "clock_alarm",
    9: "door_knock",
    10: "background",
    11: "screaming",
}

# Mapeo inverso: nombre -> id
NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}

# ── Correspondencia de nombres entre datasets ───────────────────────────────
# ESC-50 usa nombres ligeramente distintos para algunas categorías
ESC50_TO_LABEL = {
    "glass_breaking": "glass_breaking",
    "crying_baby":    "crying_baby",
    "thunderstorm":   "thunderstorm",
    "fireworks":      "fireworks",
    "clock_alarm":    "clock_alarm",
    "door_wood_knock": "door_knock",   # ESC-50 llama "door_wood_knock"
    "dog":            "dog_bark",      # ESC-50 llama "dog"
    "siren":          "siren",
    "car_horn":       "car_horn",
}

# Clases de UrbanSound8K que contribuyen como "background"
URBAN_BG_CLASSES = {
    "air_conditioner", "children_playing", "drilling",
    "engine_idling", "jackhammer", "street_music",
}

# ── Lectura de CSVs ─────────────────────────────────────────────────────────
esc50 = pd.read_csv(DATA_ESC50_DIR / "esc50.csv")
urban = pd.read_csv(DATA_URBAN_DIR / "UrbanSound8K.csv")


def esc50_filepath(filename: str) -> str:
    """Ruta relativa desde BASE_DIR para un archivo de ESC-50."""
    return str(DATA_ESC50_DIR / "audio" / filename)


def urban_filepath(fold: int, filename: str) -> str:
    """Ruta relativa desde BASE_DIR para un archivo de UrbanSound8K."""
    return str(DATA_URBAN_DIR / f"fold{fold}" / filename)


# ── Construcción del DataFrame unificado ────────────────────────────────────
rows: list[dict] = []

# 1) Clases solo-ESC-50 (40 muestras cada una):
#    glass_breaking, crying_baby, thunderstorm, fireworks, clock_alarm, door_knock
esc_only_cats = {
    "glass_breaking", "crying_baby", "thunderstorm",
    "fireworks", "clock_alarm", "door_wood_knock",
}
for _, r in esc50[esc50["category"].isin(esc_only_cats)].iterrows():
    label_name = ESC50_TO_LABEL[r["category"]]
    rows.append({
        "filepath":   esc50_filepath(r["filename"]),
        "label_id":   NAME_TO_ID[label_name],
        "label_name": label_name,
        "source":     "esc50",
    })

# 2) gun_shot: hasta 300 muestras de UrbanSound8K
urban_gunshot = urban[urban["class"] == "gun_shot"].copy()
if len(urban_gunshot) > 300:
    urban_gunshot = urban_gunshot.sample(n=300, random_state=SEED)
for _, r in urban_gunshot.iterrows():
    rows.append({
        "filepath":   urban_filepath(r["fold"], r["slice_file_name"]),
        "label_id":   NAME_TO_ID["gun_shot"],
        "label_name": "gun_shot",
        "source":     "urban",
    })

# 3) Clases combinadas ESC-50 + UrbanSound8K: dog_bark, siren, car_horn
#    40 de ESC-50 + completar con Urban hasta 300 (muestrear si > 300)
combined_cats = {
    "dog":      ("dog_bark",  "dog_bark"),   # (esc50_cat, urban_class)
    "siren":    ("siren",     "siren"),
    "car_horn": ("car_horn",  "car_horn"),
}
for esc_cat, (label_name, urban_class) in combined_cats.items():
    # a) 40 muestras de ESC-50
    esc_subset = esc50[esc50["category"] == esc_cat]
    for _, r in esc_subset.iterrows():
        rows.append({
            "filepath":   esc50_filepath(r["filename"]),
            "label_id":   NAME_TO_ID[label_name],
            "label_name": label_name,
            "source":     "esc50",
        })

    # b) Muestras de UrbanSound8K para completar hasta 300
    urban_subset = urban[urban["class"] == urban_class].copy()
    needed = 300 - len(esc_subset)
    if len(urban_subset) > needed:
        urban_subset = urban_subset.sample(n=needed, random_state=SEED)
    for _, r in urban_subset.iterrows():
        rows.append({
            "filepath":   urban_filepath(r["fold"], r["slice_file_name"]),
            "label_id":   NAME_TO_ID[label_name],
            "label_name": label_name,
            "source":     "urban",
        })

# 4) background (clase 10): ESC-50 no-peligro + Urban BG → muestrear 700
#    ESC-50: todas las categorías que NO están en nuestro mapa de peligro
danger_esc_cats = set(ESC50_TO_LABEL.keys())
esc_bg = esc50[~esc50["category"].isin(danger_esc_cats)]
bg_rows: list[dict] = []
for _, r in esc_bg.iterrows():
    bg_rows.append({
        "filepath":   esc50_filepath(r["filename"]),
        "label_id":   NAME_TO_ID["background"],
        "label_name": "background",
        "source":     "esc50",
    })

#    UrbanSound8K: clases de fondo
urban_bg = urban[urban["class"].isin(URBAN_BG_CLASSES)]
for _, r in urban_bg.iterrows():
    bg_rows.append({
        "filepath":   urban_filepath(r["fold"], r["slice_file_name"]),
        "label_id":   NAME_TO_ID["background"],
        "label_name": "background",
        "source":     "urban",
    })

bg_df = pd.DataFrame(bg_rows)
if len(bg_df) > 700:
    bg_df = bg_df.sample(n=700, random_state=SEED)
rows.extend(bg_df.to_dict("records"))

# 5) screaming (clase 11): gritos de data_esc50 → muestrear hasta 300
gritos_dir = DATA_ESC50_DIR / "gritos"
if gritos_dir.exists():
    gritos_files = sorted(list(gritos_dir.glob("*.wav")))
    gritos_rows = []
    for filepath in gritos_files:
        gritos_rows.append({
            "filepath":   str(filepath),
            "label_id":   NAME_TO_ID["screaming"],
            "label_name": "screaming",
            "source":     "esc50",
        })
    gritos_df = pd.DataFrame(gritos_rows)
    if len(gritos_df) > 300:
        gritos_df = gritos_df.sample(n=300, random_state=SEED)
    rows.extend(gritos_df.to_dict("records"))
else:
    print(f"Warning: No se encontró el directorio de gritos en {gritos_dir}")

# ── DataFrame final ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Split estratificado 70/15/15 ────────────────────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label_id"], random_state=SEED,
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED,
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# ── Guardar ─────────────────────────────────────────────────────────────────
output_path = BASE_DIR / "dataset_index.csv"
df.to_csv(output_path, index=False)
print(f"Dataset guardado en {output_path}")

# ── Resumen ─────────────────────────────────────────────────────────────────
print("\n=== Clips por clase ===")
print(df.groupby(["label_id", "label_name"]).size().to_string())

print("\n=== Clips por clase y split ===")
summary = df.groupby(["label_name", "split"]).size().unstack(fill_value=0)
summary = summary.reindex(columns=["train", "val", "test"])
print(summary.to_string())

print(f"\nTotal clips: {len(df)}")
