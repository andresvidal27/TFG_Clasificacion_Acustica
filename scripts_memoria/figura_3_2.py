import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de estilo
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# 2. Señal analógica original (200 Hz)
tiempo_total = 0.01
t_continua = np.linspace(0, tiempo_total, 1000)
y_continua = np.sin(2 * np.pi * 200 * t_continua)

# 3. Función auxiliar para graficar cada caso
def plot_muestreo(ax, fs, color, titulo):
    t_muestras = np.arange(0, tiempo_total, 1/fs)
    y_muestras = np.sin(2 * np.pi * 200 * t_muestras)
    
    ax.plot(t_continua, y_continua, 'gray', linestyle='--', alpha=0.7, label='Señal analógica')
    ax.stem(t_muestras, y_muestras, linefmt=f'{color}-', markerfmt=f'{color}o', basefmt='k-', label=f'Muestras ({fs} Hz)')
    
    ax.set_title(titulo)
    ax.set_ylabel('Amplitud')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

# 4. Creación de la figura
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

plot_muestreo(ax1, 800, 'C0', 'Digitalización con baja frecuencia de muestreo')
plot_muestreo(ax2, 4000, 'C3', 'Digitalización con alta frecuencia de muestreo')
ax2.set_xlabel('Tiempo (s)')

plt.tight_layout()

# Obtenemos la ruta a la carpeta "imagenes memoria" relativa a este script
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_dir = os.path.join(base_dir, "imagenes_memoria")
os.makedirs(img_dir, exist_ok=True)

ruta_imagen = os.path.join(img_dir, 'figura_3_2_muestreo.png')
plt.savefig(ruta_imagen, dpi=300)
print(f"¡Imagen generada con éxito en '{ruta_imagen}'!")
plt.show()