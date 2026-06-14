import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# 1. Configuración de estilo académico
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12
})

# 2. Intentar buscar un audio real que exista de Ladrido de perro (ladrido_perro)
ruta_audio = None
csv_path = 'dataset_index.csv'
if os.path.exists(csv_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['label_name'] == 'ladrido_perro':
                    filepath = row['filepath']
                    if os.path.exists(filepath):
                        ruta_audio = filepath
                        break
    except Exception as e:
        print(f"Error al leer dataset_index.csv: {e}")

# Fallback si no se encuentra
if not ruta_audio or not os.path.exists(ruta_audio):
    # Generar señal sintética clara de tipo transitorio
    print("Aviso: No se encontró audio real de ladrido_perro. Generando señal sintética transitoria.")
    sr_target = 22050
    t = np.linspace(0, 3.0, int(sr_target * 3.0), endpoint=False)
    # Pulso transitorio
    y = np.sin(2 * np.pi * 440 * t) * np.exp(-5 * (t - 1.0)**2)
else:
    sr_target = 22050
    y, sr_target = librosa.load(ruta_audio, sr=sr_target, duration=3.0) # 3 segundos es ideal para visualizar un ladrido completo

# 3. Parámetros de extracción
n_fft = 1024
hop_length = 512
n_mels = 128

# Calcular el log-Mel original
S = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_dB = librosa.power_to_db(S, ref=np.max)
spec_std_base = np.std(S_dB)

# 4. Funciones de aumento de datos inteligentes (aseguran que la máscara caiga sobre el sonido)
def apply_gauss_noise(spec, intensity, base_std):
    # Incrementamos la desviación estándar del ruido para hacerlo más evidente
    noise = np.random.normal(0, intensity * 0.25 * abs(base_std), spec.shape)
    spec_noisy = spec + noise
    return np.clip(spec_noisy, -80, 0)

def apply_masking(spec, num_masks, mask_size, axis):
    spec_masked = spec.copy()
    num_axis = spec.shape[axis]
    if axis == 0: # FreqMask
        # Colocamos dos máscaras de frecuencia anchas y separadas
        for i in range(num_masks):
            mask_start = np.random.randint(10 + i * 40, min(num_axis - mask_size - 10, 10 + i * 40 + 25))
            spec_masked[mask_start:mask_start+mask_size, :] = -80.0
    else: # TimeMask
        col_means = np.mean(spec, axis=0)
        # Buscar el índice del máximo de energía
        max_idx = np.argmax(col_means)
        
        # Máscara 1 (centrada en el pico principal de energía del ladrido)
        start1 = max(0, min(num_axis - mask_size, max_idx - mask_size // 2))
        spec_masked[:, start1:start1+mask_size] = -80.0
        
        # Máscara 2 (en otra zona activa pero lo bastante alejada para que se distingan ambos cortes)
        if num_masks > 1:
            min_dist = int(1.5 * mask_size)
            left_r = range(0, max(0, start1 - min_dist))
            right_r = range(min(num_axis - mask_size, start1 + mask_size + min_dist), num_axis - mask_size)
            valid_starts = list(left_r) + list(right_r)
            if valid_starts:
                start2 = np.random.choice(valid_starts)
                spec_masked[:, start2:start2+mask_size] = -80.0
            else:
                start2 = np.random.randint(0, num_axis - mask_size)
                spec_masked[:, start2:start2+mask_size] = -80.0
    return spec_masked

# Aplicar las técnicas por separado con valores más agresivos para que se note mucho el cambio
S_dB_noise = apply_gauss_noise(S_dB, intensity=2.5, base_std=spec_std_base)
S_dB_freq = apply_masking(S_dB, num_masks=2, mask_size=20, axis=0)  # Dos franjas de 20 bandas mel
S_dB_time = apply_masking(S_dB, num_masks=2, mask_size=20, axis=1)  # Dos franjas de 20 tramas temporales (sin solapamiento)

# 5. Graficar en una cuadrícula 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

# (a) Original
img_orig = librosa.display.specshow(S_dB, sr=sr_target, hop_length=hop_length, 
                                   x_axis='time', y_axis='mel', cmap='magma', ax=axes[0, 0])
axes[0, 0].set_title('(a) Espectrograma Log-Mel Original (Ladrido)')
axes[0, 0].set_ylabel('Bandas Mel (Hz)')

# (b) Ruido Gaussiano
librosa.display.specshow(S_dB_noise, sr=sr_target, hop_length=hop_length, 
                         x_axis='time', y_axis='mel', cmap='magma', ax=axes[0, 1])
axes[0, 1].set_title('(b) Con Aumento de Ruido Gaussiano')
axes[0, 1].set_ylabel('')

# (c) Enmascaramiento de Frecuencia
librosa.display.specshow(S_dB_freq, sr=sr_target, hop_length=hop_length, 
                         x_axis='time', y_axis='mel', cmap='magma', ax=axes[1, 0])
axes[1, 0].set_title('(c) Con Enmascaramiento de Frecuencia (FreqMask)')
axes[1, 0].set_ylabel('Bandas Mel (Hz)')
axes[1, 0].set_xlabel('Tiempo (s)')

# (d) Enmascaramiento de Tiempo
librosa.display.specshow(S_dB_time, sr=sr_target, hop_length=hop_length, 
                         x_axis='time', y_axis='mel', cmap='magma', ax=axes[1, 1])
axes[1, 1].set_title('(d) Con Enmascaramiento de Tiempo (TimeMask)')
axes[1, 1].set_ylabel('')
axes[1, 1].set_xlabel('Tiempo (s)')

# Añadir barra de color común a la derecha
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(img_orig, cax=cbar_ax, format='%+2.0f dB')
cbar_ax.set_ylabel('Energía logarítmica (dB)')

# Guardar la imagen en alta resolución
nombre_archivo = 'figura_4_3_data_augmentation.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"\n¡Imagen de Data Augmentation generada estáticamente como '{nombre_archivo}'!")

plt.show()