"""
Módulo: Transformaciones y Ecualización de Histograma
Autor: Hermes Alberto Delgado Díaz
319258613

Descripción:
    Implementa diversas transformaciones de intensidad para imágenes en escala de grises,
    incluyendo negativo, logarítmica, exponencial, potencia y ecualización del histograma.

Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
    - random
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import random


def histograma(imagen: np.ndarray) -> dict:
    """
    Calcula el histograma de una imagen en escala de grises.
    
    El histograma representa la distribución de frecuencias de los niveles
    de intensidad (0-255) presentes en la imagen.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises como array de NumPy.
                            Puede ser de cualquier tipo numérico.
    
    Returns:
        dict: Diccionario donde las claves son niveles de intensidad (0-255)
              y los valores son la frecuencia (cantidad de píxeles) de cada nivel.
    """
    imagen256 = imagen.astype(np.uint8)

    histograma_dict = {i: 0 for i in range(256)}
    y, x = imagen.shape
    for j in range (0,y):
        for i in range(0,x):
            histograma_dict[imagen256[j][i]] += 1
    return histograma_dict

def mostrar_imagen_e_histograma(imagen: np.ndarray, hist_dict: dict, titulo: str = "Imagen"):
    """
    Visualiza una imagen junto con su histograma en una figura de dos paneles.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises a visualizar.
        hist_dict (dict): Diccionario con el histograma de la imagen.
        titulo (str, optional): Título general de la figura. Por defecto "Imagen".
    
    Returns:
        None: Muestra la figura usando matplotlib.pyplot.show()
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    ax1.imshow(imagen, cmap='gray')
    ax1.set_title('Imagen')
    ax1.axis('off')
    
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
    """
    Visualiza la comparación entre una imagen original y su versión ecualizada.
    
    Crea una figura con 4 paneles que muestra:
    - Superior izquierda: Imagen original
    - Superior derecha: Histograma original
    - Inferior izquierda: Imagen ecualizada
    - Inferior derecha: Histograma ecualizado
    
    Args:
        imagen_original (np.ndarray): Imagen original en escala de grises.
        imagen_ecualizada (np.ndarray): Imagen después de ecualización.
        hist_original (dict): Histograma de la imagen original.
        hist_ecualizado (dict): Histograma de la imagen ecualizada.
    
    Returns:
        None: Muestra la figura comparativa.
    """

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

def negativo(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica transformación negativa a la imagen.
    
    Esta transformación invierte los niveles de intensidad, convirtiendo
    píxeles oscuros en claros y viceversa.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises (0-255).
    
    Returns:
        np.ndarray: Imagen negativa con tipo uint8.
    """

    imagen_negativo = np.zeros(imagen.shape, dtype=np.uint8)
    y,x = imagen_negativo.shape
    for j in range (0,y):
        for i in range(0,x):
            imagen_negativo[j][i] = 255 - imagen[j][i]
    return imagen_negativo

def logaritmica(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica transformación logarítmica a la imagen.
    
    Fórmula: s = c * log(1 + r)
    donde:
    - r: intensidad original
    - s: intensidad transformada
    - c: constante de escalamiento = 255 / log(1 + max(r))
    
    Esta transformación expande los valores de intensidad bajos (oscuros)
    y comprime los valores altos (claros). Útil para mejorar detalles
    en regiones oscuras.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises.
    
    Returns:
        np.ndarray: Imagen transformada con tipo uint8.
    """

    imagen = imagen.astype(np.uint8)
    
    imagen_logaritmica = np.zeros(imagen.shape, dtype=np.float64)
    
    img_max = float(np.max(imagen))  
    
    if img_max == 0:
        return imagen.copy()
    
    c = 255.0 / np.log(1.0 + img_max) 
    
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            imagen_logaritmica[j][i] = c * np.log(1.0 + float(imagen[j][i]))
    
    return imagen_logaritmica.astype(np.uint8)

def exponencial(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica transformación exponencial a la imagen (inversa de logarítmica).
    
    Esta transformación comprime valores bajos y expande valores altos.
    Es la inversa de la transformación logarítmica.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises.
    
    Returns:
        np.ndarray: Imagen transformada con tipo uint8.
    """

    imagen_exponencial = np.zeros(imagen.shape, dtype=np.float32)
    
    imagen_norm = imagen / 255.0
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            
            imagen_exponencial[j][i] = (np.exp(imagen_norm[j][i]) - 1) / (np.e - 1) * 255
    return imagen_exponencial.astype(np.uint8)

def potencia(imagen: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Aplica transformación de potencia (ley gamma) a la imagen.

    El parámetro gamma controla el comportamiento:
    - gamma < 1: Aclara la imagen, expande tonos oscuros
    - gamma = 1: No hay cambio (transformación identidad)
    - gamma > 1: Oscurece la imagen, comprime tonos oscuros
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises.
        gamma (float): Exponente gamma. Valores típicos: 0.3-3.0
        c (float, optional): Constante de escalamiento. Por defecto 1.0
    
    Returns:
        np.ndarray: Imagen transformada con tipo uint8.
    """

    imagen_potencia = np.zeros(imagen.shape, dtype=np.float32)
   
    imagen_norm = imagen / 255.0
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
           
            imagen_potencia[j][i] = c * np.power(imagen_norm[j][i], gamma) * 255
    return imagen_potencia.astype(np.uint8)

def ecualizar_histograma(imagen: np.ndarray) -> np.ndarray:
    """
    Ecualiza el histograma de una imagen para mejorar el contraste.
    
    La ecualización redistribuye las intensidades para que el histograma
    sea aproximadamente uniforme, utilizando la función de distribución
    acumulativa (CDF) como función de transformación.
    
    Proceso:
    1. Calcular histograma normalizado (probabilidades)
    2. Calcular CDF (suma acumulativa de probabilidades)
    3. Escalar CDF al rango [0, 255]
    4. Mapear cada píxel usando CDF como tabla de lookup
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises.
    
    Returns:
        np.ndarray: Imagen con histograma ecualizado (uint8).
    """

    imagen = imagen.astype(np.uint8)
    y, x = imagen.shape
    total_pixeles = y * x

    hist_dict = histograma(imagen)

    histo = np.array([hist_dict[i] for i in range(256)])
    # Calcular probabilidades
    probabilidad = histo / total_pixeles
    
    # Calcular CDF (Cumulative Distribution Function - suma acumulativa)
    cdf = np.cumsum(probabilidad)
    
    # Escalar CDF a rango [0, 255]
    cdf_escalado = (cdf * 255).astype(np.uint8)
    
    # Mapear cada pixel usando el CDF como tabla de lookup
    imagen_ecualizada = cdf_escalado[imagen]
    
    return imagen_ecualizada

def inicia_Ejercicio1():
    """
    Función principal que demuestra todas las transformaciones de imagen.
    
    Esta función:
    1. Selecciona aleatoriamente una imagen de un conjunto predefinido
    2. Muestra información sobre el rango y tipo de datos
    3. Aplica diversas transformaciones de intensidad
    4. Visualiza cada transformación con su histograma
    
    Transformaciones aplicadas:
    - Negativo (inversión de intensidades)
    - Logarítmica (expande oscuros)
    - Exponencial (expande claros)
    - Potencia con gamma=0.4 (aclara)
    - Potencia con gamma=2.5 (oscurece)
    - Ecualización de histograma (mejora contraste)
    
    Returns:
        None: Genera múltiples visualizaciones usando matplotlib.
    """
    
    rutas = [
        "../imagenes/lena.tiff",
        "../imagenes/mamografia.tiff",
        "../imagenes/brain.tiff",
        "../imagenes/granos.png",
        "../imagenes/resonancia.tiff",
        "../imagenes/tungsten_1.jpg",
        "../imagenes/tungsten_2.jpg"
    ]
    print("Iniciando Ejercicio 1: Transformaciones y Ecualización de Histograma")

    ruta_imagen = random.choice(rutas)
    print(f"\nImagen seleccionada: {ruta_imagen}")
    
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
    mostrar_imagen_e_histograma(imagen_negativo, hist_dict_negativo, "Negativo")
    mostrar_imagen_e_histograma(imagen_logaritmica, hist_dict_logaritmica, "Logarítmica")
    mostrar_imagen_e_histograma(imagen_exponencial, hist_dict_exponencial, "Exponencial")
    mostrar_imagen_e_histograma(imagen_gamma_menor, hist_dict_gamma_menor, "Potencia gamma=0.4 (aclara)")
    mostrar_imagen_e_histograma(imagen_gamma_mayor, hist_dict_gamma_mayor, "Potencia gamma=2.5 (oscurece)")
    mostrar_imagen_e_histograma(imagen_ecualizada, hist_dict_ecualizada, "Ecualización del Histograma")
    mostrar_comparacion_ecualizacion(imagen, imagen_ecualizada, hist_dict, hist_dict_ecualizada)