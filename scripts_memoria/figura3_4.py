import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuración de estilo (coherente con la plantilla ETSIT)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
})

# 2. Cargar tu propio archivo de audio
# SUSTITUYE 'ejemplo_disparo.wav' por la ruta a un audio real de tu dataset
# Si no tienes uno a mano ahora, librosa.ex('trumpet') carga un audio de prueba
ruta_audio = librosa.ex('trumpet') 
y, sr = librosa.load(ruta_audio, sr=22050, duration=5.0)

# 3. Parámetros exactos de tu TFG (sección 3.2.2)
n_fft = 1024
hop_length = 512
n_mels = 128

# 4. Cálculo del Espectrograma Mel y compresión logarítmica
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length, n_mels=n_mels)
S_dB = librosa.power_to_db(S, ref=np.max)

# 5. Visualización
fig, ax = plt.subplots(figsize=(8, 4.5))

# Usamos un mapa de color como 'magma' para visualizar mejor la energía
img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, 
                               x_axis='time', y_axis='mel', 
                               cmap='magma', ax=ax)

# Añadimos la barra de color lateral para la energía en dB
cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
cbar.set_label('Energía logarítmica (dB)')

# Etiquetas en español
ax.set_title('Espectrograma log-Mel de 5 segundos')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Bandas Mel (Hz)')

# Guardar la imagen en alta resolución
plt.tight_layout()
nombre_archivo = 'figura_3_5_espectrograma_logmel.png'
plt.savefig(nombre_archivo, dpi=300)
print(f"¡Imagen generada con éxito como '{nombre_archivo}'!")

plt.show()