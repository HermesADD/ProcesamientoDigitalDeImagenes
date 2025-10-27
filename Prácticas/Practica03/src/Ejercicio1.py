import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from PIL import Image
import random


def histograma(imagen: np.ndarray):
    imagen256 = imagen.astype(np.uint8)

    histograma_dict = {i: 0 for i in range(256)}
    y, x = imagen.shape
    for j in range (0,y):
        for i in range(0,x):
            histograma_dict[imagen256[j][i]] += 1
    return histograma_dict

def mostrar_imagen_e_histograma(imagen: np.ndarray, hist_dict: dict, titulo: str = "Imagen"):
    # Crear figura con 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    # Mostrar la imagen
    ax1.imshow(imagen, cmap='gray')
    ax1.set_title('Imagen')
    ax1.axis('off')
    
    # Mostrar el histograma
    niveles = list(hist_dict.keys())
    frecuencias = list(hist_dict.values())
    ax2.bar(niveles, frecuencias, width=1, color='blue', alpha=0.7)
    ax2.set_title('Histograma')
    ax2.set_xlabel('Nivel de intensidad')
    ax2.set_ylabel('Frecuencia')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def mostrar_comparacion_ecualizacion(imagen_original: np.ndarray, imagen_ecualizada: np.ndarray, hist_original: dict, hist_ecualizado: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ecualización del Histograma', fontsize=16, fontweight='bold')
    
    # Imagen original
    axes[0, 0].imshow(imagen_original, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    # Histograma original
    niveles_orig = list(hist_original.keys())
    frecuencias_orig = list(hist_original.values())
    axes[0, 1].bar(niveles_orig, frecuencias_orig, width=1, color='blue', alpha=0.7)
    axes[0, 1].set_title('Histograma Original')
    axes[0, 1].set_xlabel('Nivel de intensidad')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Imagen ecualizada
    axes[1, 0].imshow(imagen_ecualizada, cmap='gray')
    axes[1, 0].set_title('Imagen Ecualizada')
    axes[1, 0].axis('off')
    
    # Histograma ecualizado
    niveles_ecua = list(hist_ecualizado.keys())
    frecuencias_ecua = list(hist_ecualizado.values())
    axes[1, 1].bar(niveles_ecua, frecuencias_ecua, width=1, color='green', alpha=0.7)
    axes[1, 1].set_title('Histograma Ecualizado')
    axes[1, 1].set_xlabel('Nivel de intensidad')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def negativo(imagen: np.ndarray):
    imagen_negativo = np.zeros(imagen.shape, dtype=np.uint8)
    y,x = imagen_negativo.shape
    for j in range (0,y):
        for i in range(0,x):
            imagen_negativo[j][i] = 255 - imagen[j][i]
    return imagen_negativo

def logaritmica(imagen: np.ndarray):
    """
    Transformación logarítmica: s = c * log(1 + r)
    Expande valores bajos y comprime valores altos
    """
    # Asegurar que sea uint8
    imagen = imagen.astype(np.uint8)
    
    imagen_logaritmica = np.zeros(imagen.shape, dtype=np.float64)
    
    # Convertir a float64 ANTES de hacer operaciones
    img_max = float(np.max(imagen))  # ← Convertir a float primero
    
    # Evitar división por cero
    if img_max == 0:
        return imagen.copy()
    
    # Usar floats para evitar overflow
    c = 255.0 / np.log(1.0 + img_max)  # ← Todo en float
    
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            # Convertir pixel a float antes de sumar
            imagen_logaritmica[j][i] = c * np.log(1.0 + float(imagen[j][i]))
    
    return imagen_logaritmica.astype(np.uint8)

def exponencial(imagen: np.ndarray):
    imagen_exponencial = np.zeros(imagen.shape, dtype=np.float32)
    # Normalizar imagen a [0,1]
    imagen_norm = imagen / 255.0
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            # Aplicar exponencial y reescalar a [0, 255]
            imagen_exponencial[j][i] = (np.exp(imagen_norm[j][i]) - 1) / (np.e - 1) * 255
    return imagen_exponencial.astype(np.uint8)

def potencia(imagen: np.ndarray, gamma: float, c: float = 1.0):
    imagen_potencia = np.zeros(imagen.shape, dtype=np.float32)
    # Normalizar a [0,1]
    imagen_norm = imagen / 255.0
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            # Aplicar transformación de potencia
            imagen_potencia[j][i] = c * np.power(imagen_norm[j][i], gamma) * 255
    return imagen_potencia.astype(np.uint8)

def ecualizar_histograma(imagen: np.ndarray):
    imagen = imagen.astype(np.uint8)
    y, x = imagen.shape
    total_pixeles = y * x

    histo, _ = np.histogram(imagen.flatten(), bins=256, range=[0, 256])
    
    # Calcular probabilidades
    probabilidad = histo / total_pixeles
    
    # Calcular CDF (Cumulative Distribution Function - suma acumulativa)
    cdf = np.cumsum(probabilidad)
    
    # Escalar CDF a rango [0, 255]
    cdf_escalado = (cdf * 255).astype(np.uint8)
    
    # Mapear cada pixel usando el CDF como tabla de lookup
    imagen_ecualizada = cdf_escalado[imagen]
    
    return imagen_ecualizada

def ejercicio1():
    rutas = [
        "../imagenes/mamografia.tiff", 
        "../imagenes/brain.tiff", "../imagenes/granos.png", 
        "../imagenes/resonancia.tiff", 
        "../imagenes/lena.tiff"
    ]
    ruta_imagen = random.choice(rutas)
    print("Rango de valores y tipo de dato de la imagen seleccionada:")
    print("Minimo, Maximo, Tipo de Dato")
    print("-----------------------------------")
    print(np.min(io.imread(ruta_imagen)), np.max(io.imread(ruta_imagen)), io.imread(ruta_imagen).dtype)
    print(f"Imagen seleccionada aleatoriamente: {ruta_imagen}")
    
    imagen = io.imread(ruta_imagen)
    hist_dict = histograma(imagen)
    
    # Transformación negativa
    imagen_negativo = negativo(imagen)
    hist_dict_negativo = histograma(imagen_negativo)

    # Transformación logarítmica
    imagen_logaritmica = logaritmica(imagen)
    hist_dict_logaritmica = histograma(imagen_logaritmica)
    
    # Transformación exponencial
    imagen_exponencial = exponencial(imagen)
    hist_dict_exponencial = histograma(imagen_exponencial)
    
    # Transformación de potencia con gamma < 1 (aclara)
    imagen_gamma_menor = potencia(imagen, gamma=0.4)
    hist_dict_gamma_menor = histograma(imagen_gamma_menor)
    
    # Transformación de potencia con gamma > 1 (oscurece)
    imagen_gamma_mayor = potencia(imagen, gamma=2.5)
    hist_dict_gamma_mayor = histograma(imagen_gamma_mayor)
    
    imagen_ecualizada = ecualizar_histograma(imagen)
    hist_dict_ecualizada = histograma(imagen_ecualizada)
    # Mostrar todas las transformaciones
    mostrar_imagen_e_histograma(imagen, hist_dict, "Original")
    mostrar_imagen_e_histograma(imagen_negativo, hist_dict_negativo, "Negativo (s = 255 - r)")
    mostrar_imagen_e_histograma(imagen_logaritmica, hist_dict_logaritmica, "Logarítmica (s = c*log(1+r))")
    mostrar_imagen_e_histograma(imagen_exponencial, hist_dict_exponencial, "Exponencial (inversa logarítmica)")
    mostrar_imagen_e_histograma(imagen_gamma_menor, hist_dict_gamma_menor, "Potencia γ=0.4 (aclara)")
    mostrar_imagen_e_histograma(imagen_gamma_mayor, hist_dict_gamma_mayor, "Potencia γ=2.5 (oscurece)")
    mostrar_imagen_e_histograma(imagen_ecualizada, hist_dict_ecualizada, "Ecualización del Histograma")
    mostrar_comparacion_ecualizacion(imagen, imagen_ecualizada, hist_dict, hist_dict_ecualizada)
    

    

if __name__ == "__main__":
    ejercicio1()