import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Configuración de estilo académico
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13
})

import csv

# 2. Las 8 clases de tu TFG y su mapeo al CSV
clases = [
    'Ladrido de perro', 'Sirena', 'Bebé llorando', 'Golpe en la puerta',
    'Rotura de cristal', 'Disparo', 'Grito', 'Fondo'
]

clase_to_csv_label = {
    'Ladrido de perro': 'ladrido_perro',
    'Sirena': 'sirena',
    'Bebé llorando': 'bebe_llorando',
    'Golpe en la puerta': 'llamar_puerta',
    'Rotura de cristal': 'rotura_cristal',
    'Disparo': 'disparo',
    'Grito': 'grito',
    'Fondo': 'fondo'
}

# 3. Intentar buscar audios reales que existan usando dataset_index.csv
rutas_audios = []
csv_path = 'dataset_index.csv'
rutas_encontradas = {}

if os.path.exists(csv_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row['label_name']
                filepath = row['filepath']
                if os.path.exists(filepath):
                    if lbl not in rutas_encontradas:
                        rutas_encontradas[lbl] = filepath
                if len(rutas_encontradas) == 8:
                    break
    except Exception as e:
        print(f"Error al leer dataset_index.csv: {e}")

# Mapear cada clase a su ruta real encontrada (o None si no existe)
for clase in clases:
    label_csv = clase_to_csv_label[clase]
    ruta = rutas_encontradas.get(label_csv, None)
    rutas_audios.append(ruta)

# 4. Crear la figura (Cuadrícula de 2 filas x 4 columnas)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7), sharex=True, sharey=True)
axes = axes.flatten() # Aplanamos el array de ejes para iterar fácilmente

# 5. Parámetros exactos de tu extracción (Capítulo 3)
n_fft = 1024
hop_length = 512
n_mels = 128
sr_target = 22050

# 6. Bucle para procesar y dibujar cada clase
for i, (clase, ruta) in enumerate(zip(clases, rutas_audios)):
    
    # Intenta cargar el audio real. Si no lo encuentra, genera ruido de prueba.
    try:
        if ruta and os.path.exists(ruta):
            y, sr = librosa.load(ruta, sr=sr_target, duration=5.0)
        else:
            raise FileNotFoundError
    except Exception:
        # Fallback de seguridad: Ruido aleatorio (para probar el script)
        sr = sr_target
        y = np.random.randn(sr * 5) * 0.05
        print(f"Aviso: No se encontró un audio real para '{clase}'. Generando audio simulado.")

    # Calcular el log-Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Dibujar el espectrograma (magma)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, 
                                   x_axis='time', y_axis='mel', 
                                   cmap='magma', ax=axes[i])
    axes[i].set_title(clase, fontweight='bold')
    
    # Limpiar etiquetas redundantes para que quede elegante
    if i >= 4: # Solo la fila de abajo tiene la etiqueta del tiempo
        axes[i].set_xlabel('Tiempo (s)')
    else:
        axes[i].set_xlabel('')
        
    if i % 4 == 0: # Solo la primera columna tiene la etiqueta de frecuencias
        axes[i].set_ylabel('Bandas Mel (Hz)')
    else:
        axes[i].set_ylabel('')

# 7. Ajustar los márgenes para que no se pisen los títulos y guardar
plt.tight_layout()
nombre_archivo = 'figura_4_2_espectrogramas_clases.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"\n¡Imagen generada con éxito como '{nombre_archivo}'!")

plt.show()