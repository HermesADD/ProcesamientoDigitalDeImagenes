"""
Módulo: Filtros y Operadores en Imágenes de Escala de Grises
Autor: Hermes Alberto Delgado Díaz
319258613

Descripción:
    Este módulo implementa diversas funciones para el procesamiento de imágenes en escala de grises,
    incluyendo la adición de ruido, aplicación de filtros (promedio, mediana), detección de bordes
    (Prewitt, Sobel), operadores Laplacianos y técnicas de realce como Unsharp Masking.

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

def convolucion(imagen: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Realiza la operación de convolución 2D entre una imagen y un kernel.
    
    La convolución es una operación fundamental en procesamiento de imágenes
    que aplica un kernel (máscara) sobre cada píxel de la imagen, calculando
    la suma ponderada de los píxeles vecinos.
    
    Proceso:
    1. Agregar padding a la imagen para manejar bordes
    2. Para cada píxel, extraer región del tamaño del kernel
    3. Multiplicar elemento a elemento y sumar
    4. Almacenar resultado en la posición correspondiente
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
        kernel (np.ndarray): Kernel/máscara de convolución (generalmente 3x3, 5x5, etc).
    
    Returns:
        np.ndarray: Imagen resultante después de la convolución (float64).
    """

    img_h, img_w = imagen.shape
    kernel_h, kernel_w = kernel.shape
     
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    imagen_padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    imagen_padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = imagen

    resultado = np.zeros_like(imagen, dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            region = imagen_padded[i:i + kernel_h, j:j + kernel_w]
            valor = np.sum(region * kernel)
            resultado[i, j] = valor
    
    return resultado

def agregar_ruido_gaussiano(imagen: np.ndarray, media: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Agrega ruido gaussiano a una imagen.
    
    El ruido gaussiano simula variaciones aleatorias en la intensidad de
    los píxeles, común en sensores de cámaras digitales debido a factores
    como temperatura y condiciones de iluminación.
    
    Args:
        imagen (np.ndarray): Imagen original en escala de grises.
        media (float, optional): Media de la distribución gaussiana. Por defecto 0.
        sigma (float, optional): Desviación estándar del ruido. Por defecto 25.
    
    Returns:
        np.ndarray: Imagen con ruido gaussiano añadido (uint8).
    """

    imagen = imagen.astype(np.float64)

    ruido = np.random.normal(media, sigma, imagen.shape)
    imagen_ruidosa = imagen + ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 255)

    return imagen_ruidosa.astype(np.uint8)

def agregar_ruido_sal_pimienta(imagen: np.ndarray, probabilidad: float = 0.05) -> np.ndarray:
    """
    Agrega ruido de sal y pimienta a una imagen.
    
    Este tipo de ruido aparece como píxeles aleatorios completamente blancos
    (sal, valor 255) o completamente negros (pimienta, valor 0). Es común en
    transmisiones de imágenes con errores de bits o sensores defectuosos.
    
    Args:
        imagen (np.ndarray): Imagen original en escala de grises.
        probabilidad (float, optional): Probabilidad total de ruido (se divide
                                       50/50 entre sal y pimienta). Por defecto 0.05.
    
    Returns:
        np.ndarray: Imagen con ruido sal y pimienta añadido (mismo tipo que entrada).
    """
    imagen_ruidosa = imagen.copy()

    num_sal = int(probabilidad * imagen.size * 0.5)
    coords_sal = [np.random.randint(0,i,num_sal) for i in imagen.shape]
    imagen_ruidosa[coords_sal[0], coords_sal[1]] = 255

    num_pimienta = int(probabilidad * imagen.size * 0.5)
    coords_pimienta = [np.random.randint(0,i,num_pimienta) for i in imagen.shape]
    imagen_ruidosa[coords_pimienta[0], coords_pimienta[1]] = 0

    return imagen_ruidosa

def filtro_promedio_estandar(imagen: np.ndarray, tam: int) -> np.ndarray:
    """
    Aplica filtro de promedio estándar (box filter) a la imagen.
    
    Este filtro reemplaza cada píxel por el promedio aritmético de sus vecinos
    en una ventana cuadrada de tamaño tam x tam. Todos los pesos son iguales (1/n²).
    
    Kernel: matriz tam x tam con todos los elementos = 1/(tam²)
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
        tam (int): Tamaño del kernel (ventana cuadrada). Debe ser impar.
    
    Returns:
        np.ndarray: Imagen suavizada (uint8).
    """
    kernel = np.ones((tam, tam)) / (tam * tam)

    resultado = convolucion(imagen.astype(np.float64), kernel)
    
    return np.clip(resultado, 0, 255).astype(np.uint8)

def filtro_promedio_ponderado(imagen: np.ndarray, tam: int) -> np.ndarray:
    """
    Aplica filtro de promedio ponderado (weighted average) a la imagen.
    
    Similar al filtro promedio, pero asigna mayor peso a píxeles cercanos
    al centro y menor peso a píxeles lejanos. Esto preserva mejor los bordes
    que el promedio estándar.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
        tam (int): Tamaño del kernel. Soporta 3, 5, 7, 11 o cualquier impar.
    
    Returns:
        np.ndarray: Imagen suavizada con preservación de bordes (uint8).
    """

    if tam == 3:
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype= np.float64) / 16.0
    elif tam == 5:
        kernel = np.array([[1, 2, 3, 2, 1],
                          [2, 4, 6, 4, 2],
                          [3, 6, 9, 6, 3],
                          [2, 4, 6, 4, 2],
                          [1, 2, 3, 2, 1]], dtype=np.float64) / 81.0
    elif tam == 7:
        kernel = np.ones((tam, tam), dtype=np.float64)
        centro = tam // 2
        for i in range(tam):
            for j in range(tam):
                dist = abs(i - centro) + abs(j - centro)
                kernel[i, j] = max(1, tam - dist)
        
        kernel /= kernel.sum()

    else: 
        kernel = np.ones((tam, tam), dtype=np.float64)
        centro = tam // 2
        for i in range(tam):
            for j in range(tam):
                dist = abs(i - centro) + abs(j - centro)
                kernel[i, j] = max(1, tam - dist)
        kernel = kernel / kernel.sum()

    resultado = convolucion(imagen.astype(np.float64), kernel)
    return np.clip(resultado, 0, 255).astype(np.uint8)

def filtro_mediana(imagen: np.ndarray, tam: int) -> np.ndarray:
    """
    Aplica filtro de mediana a la imagen.
    
    Reemplaza cada píxel por la mediana estadística de los píxeles en su
    vecindad. Es un filtro no lineal especialmente efectivo contra ruido
    impulsivo (sal y pimienta) mientras preserva bordes.
    
    Proceso:
    1. Para cada píxel, extraer ventana tam x tam
    2. Ordenar valores de la ventana
    3. Seleccionar valor central (mediana)
    4. Asignar mediana al píxel de salida
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
        tam (int): Tamaño de la ventana (debe ser impar). 
    
    Returns:
        np.ndarray: Imagen filtrada (uint8).
    """

    img_h, img_w = imagen.shape
    pad = tam // 2

    imagen_padded = np.zeros((img_h + 2 * pad, img_w + 2 * pad))
    imagen_padded[pad:pad + img_h, pad:pad + img_w] = imagen

    resultado = np.zeros_like(imagen)

    for i in range(img_h):
        for j in range(img_w):
            region = imagen_padded[i:i + tam, j:j + tam]

            valores = region.flatten()
            valores_sort = np.sort(valores)
            mediana = valores_sort[len(valores_sort) // 2]
            resultado[i, j] = mediana

    return resultado.astype(np.uint8)

def gradiente_prewitt(imagen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica el operador de Prewitt para detección de bordes.
    
    El operador de Prewitt calcula el gradiente de la imagen usando dos
    kernels direccionales (horizontal y vertical) que aproximan las
    derivadas parciales.
    
    Kernels:
    Gx (horizontal):     Gy (vertical):
    [-1  0  1]           [-1 -1 -1]
    [-1  0  1]           [ 0  0  0]
    [-1  0  1]           [ 1  1  1]
    
    Magnitud: |G| = √(Gx² + Gy²)
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
    
    Returns:
        tuple: (gx, gy, magnitud)
            - gx: Gradiente en dirección X (uint8)
            - gy: Gradiente en dirección Y (uint8)
            - magnitud: Magnitud del gradiente (uint8)
    """

    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float64)
    
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0], 
                        [1, 1, 1]], dtype=np.float64)
    
    gx = convolucion(imagen.astype(np.float64), kernel_x)
    gy = convolucion(imagen.astype(np.float64), kernel_y)

    magnitud = np.zeros_like(gx)

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            magnitud[i, j] = np.sqrt(gx[i, j]**2 + gy[i, j]**2)

    gx = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    magnitud = np.clip(magnitud, 0, 255).astype(np.uint8)

    return gx, gy, magnitud

def gradiente_sobel(imagen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica el operador de Sobel para detección de bordes.
    
    Similar a Prewitt, pero con mayor énfasis en píxeles centrales,
    resultando en mejor aproximación a la derivada y menor sensibilidad
    al ruido.
    
    Kernels:
    Gx (horizontal):     Gy (vertical):
    [-1  0  1]           [-1 -2 -1]
    [-2  0  2]           [ 0  0  0]
    [-1  0  1]           [ 1  2  1]
    
    Magnitud: |G| = √(Gx² + Gy²)
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
    
    Returns:
        tuple: (gx, gy, magnitud)
            - gx: Gradiente en dirección X (uint8)
            - gy: Gradiente en dirección Y (uint8)
            - magnitud: Magnitud del gradiente (uint8)
    """

    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
   
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    gx = convolucion(imagen.astype(np.float64), kernel_x)
    gy = convolucion(imagen.astype(np.float64), kernel_y)
    
    magnitud = np.zeros_like(gx)
    for i in range(magnitud.shape[0]):
        for j in range(magnitud.shape[1]):
            magnitud[i, j] = np.sqrt(gx[i, j]**2 + gy[i, j]**2)
    
    gx = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    magnitud = np.clip(magnitud, 0, 255).astype(np.uint8)
    
    return gx, gy, magnitud

def laplaciano90(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica el operador Laplaciano con conectividad de 90 grados (4-vecinos).
    
    El Laplaciano es un operador de segunda derivada que detecta cambios
    de intensidad en todas direcciones simultáneamente. Es isotrópico
    (invariante a rotación en 90°).
    
    Kernel (4-conectividad):
    [ 0  1  0]
    [ 1 -4  1]
    [ 0  1  0]
    
    Detecta regiones donde ocurren cambios rápidos de intensidad (bordes).
    Valores negativos indican transición oscuro→claro, positivos claro→oscuro.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
    
    Returns:
        np.ndarray: Imagen del Laplaciano (float64, puede tener valores negativos).
    """
    
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)
    
    resultado = convolucion(imagen.astype(np.float64), kernel)

    return resultado

def laplaciano45(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica el operador Laplaciano con conectividad de 45 grados (8-vecinos).
    
    Variante del Laplaciano que considera vecinos diagonales además de
    ortogonales, proporcionando mayor sensibilidad a bordes en todas
    direcciones, especialmente diagonales.
    
    Kernel (8-conectividad):
    [ 1  0  1]
    [ 0 -8  0]
    [ 1  0  1]
    
    También conocido como Laplaciano diagonal o Laplaciano isotrópico.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
    
    Returns:
        np.ndarray: Imagen del Laplaciano (float64, puede tener valores negativos).
    """

    kernel = np.array([[1, 0, 1],
                       [0, -8, 0],
                       [1, 0, 1]], dtype=np.float64)
    
    resultado = convolucion(imagen.astype(np.float64), kernel)

    return resultado

def unsharp_masking(imagen: np.ndarray, filtro_blur: str, tam: int) -> np.ndarray:
    """
    Aplica técnica de Unsharp Masking para realce de bordes.
    
    Unsharp Masking es una técnica clásica de realce que funciona
    substrayendo una versión suavizada de la imagen de la original,
    y luego sumando esta "máscara" de alta frecuencia a la imagen.
    
    Proceso:
    1. Suavizar imagen -> imagen_blur
    2. Calcular máscara: mask = imagen - imagen_blur (detalles)
    3. Realzar: resultado = imagen + mask
    
    Fórmula: resultado = 2 x imagen - imagen_blur
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises.
        filtro_blur (str): Tipo de filtro para suavizado: 'promedio' o 'ponderado'.
        tam (int): Tamaño del kernel de suavizado.
    
    Returns:
        np.ndarray: Imagen realzada (uint8).
    """

    if filtro_blur == 'promedio':
        imagen_blur = filtro_promedio_estandar(imagen, tam)
    else: 
        imagen_blur = filtro_promedio_ponderado(imagen, tam)
    
    imagen_sharp = imagen.astype(np.float64) - imagen_blur.astype(np.float64)

    resultado = imagen.astype(np.float64) + imagen_sharp
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    return resultado    

def mostrar_comparacion(imagenes: list, titulos: list, filas: int = 2, cols: int = 4, tam_fig: tuple = (16,8)):
    """
    Visualiza múltiples imágenes en una grilla para comparación lado a lado.
    
    Función auxiliar para mostrar resultados de diferentes filtros o
    transformaciones aplicadas a una o varias imágenes.
    
    Args:
        imagenes (list): Lista de imágenes (np.ndarray) a mostrar.
        titulos (list): Lista de títulos para cada imagen.
        filas (int, optional): Número de filas en la grilla. Por defecto 2.
        cols (int, optional): Número de columnas en la grilla. Por defecto 4.
        tam_fig (tuple, optional): Tamaño de la figura (ancho, alto). Por defecto (16,8).
    
    Returns:
        None: Muestra la figura usando matplotlib.
    """
    fig, axes = plt.subplots(filas, cols, figsize=tam_fig)
    axes = axes.flatten()

    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')

    for i in range(len(imagenes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def ejercicio2a(imagen_sin_ruido: np.ndarray, imagen_con_ruido: np.ndarray):
    """
    Ejercicio 2a: Compara el efecto del filtro promedio estándar.
    
    Aplica filtros de promedio de diferentes tamaños (3x3, 5x5, 7x7, 11x11)
    tanto a imágenes limpias como ruidosas, mostrando el trade-off entre
    reducción de ruido y pérdida de detalles.
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original sin ruido.
        imagen_con_ruido (np.ndarray): Imagen con ruido gaussiano.
    """

    print(f"\nEjercicio 2a: Filtro Promedio Estándar")

    tamanos = [3, 5, 7, 11]
    imagenes = []
    titulos = []   

    imagenes.append(imagen_sin_ruido)
    titulos.append("Original sin ruido")

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen sin ruido")
        imagen_filtrada = filtro_promedio_estandar(imagen_sin_ruido, tam)
        imagenes.append(imagen_filtrada)
        titulos.append(f"Filtro {tam}x{tam} sin ruido")

    imagenes.append(imagen_con_ruido)
    titulos.append("Original con ruido")

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen con ruido")
        imagen_filtrada = filtro_promedio_estandar(imagen_con_ruido, tam)
        imagenes.append(imagen_filtrada)
        titulos.append(f"Filtro {tam}x{tam} con ruido")

    mostrar_comparacion(imagenes[:5], titulos[:5], filas=1, cols=5, tam_fig=(20,4))
    mostrar_comparacion(imagenes[5:], titulos[5:], filas=1, cols=5, tam_fig=(20,4))

def ejercicio2b(imagen_sin_ruido: np.ndarray, imagen_con_ruido: np.ndarray):
    """
    Ejercicio 2b: Compara el efecto del filtro promedio ponderado.
    
    Similar al ejercicio 2a pero usando filtros ponderados que dan mayor
    peso a píxeles centrales. Permite observar mejor preservación de bordes
    comparado con filtro promedio estándar.
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original sin ruido.
        imagen_con_ruido (np.ndarray): Imagen con ruido gaussiano.
    """

    print(f"\nEjercicio 2b: Filtro Promedio Ponderado")
    tamanos = [3, 5, 7, 11]
    imagenes = []
    titulos = []   

    imagenes.append(imagen_sin_ruido)
    titulos.append("Original sin ruido")

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen sin ruido")
        imagen_filtrada = filtro_promedio_ponderado(imagen_sin_ruido, tam)
        imagenes.append(imagen_filtrada)
        titulos.append(f"Filtro {tam}x{tam} sin ruido")

    imagenes.append(imagen_con_ruido)
    titulos.append("Original con ruido")   

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen con ruido")
        imagen_filtrada = filtro_promedio_ponderado(imagen_con_ruido, tam)
        imagenes.append(imagen_filtrada)
        titulos.append(f"Filtro {tam}x{tam} con ruido")
    
    mostrar_comparacion(imagenes[:5], titulos[:5], filas=1, cols=5, tam_fig=(20,4))
    mostrar_comparacion(imagenes[5:], titulos[5:], filas=1, cols=5, tam_fig=(20,4))

def ejercicio2c(imagen_sin_ruido: np.ndarray):
    """
    Ejercicio 2c: Evalúa el filtro de mediana contra diferentes tipos de ruido.
    
    Compara la efectividad del filtro de mediana en dos escenarios:
    1. Ruido sal y pimienta 
    2. Ruido gaussiano
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original sin ruido.
    """

    print(f"\nEjercicio 2c: Filtro Mediana con ruido sal y pimienta y gaussiano")

    print("Apicando ruido sal y pimienta a la imagen...")
    img_sal_pimienta = agregar_ruido_sal_pimienta(imagen_sin_ruido, probabilidad=0.05)

    print("Aplicando ruido gaussiano a la imagen...")
    img_gaussiano = agregar_ruido_gaussiano(imagen_sin_ruido)

    tamanos = [3, 5, 7, 11]

    print("Filtrando imagen con ruido sal y pimienta...")
    imagenes_sp = [imagen_sin_ruido, img_sal_pimienta]
    titulos_sp = ["Original sin ruido", "Imagen con ruido sal y pimienta"]

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen con ruido sal y pimienta")
        imagen_filtrada = filtro_mediana(img_sal_pimienta, tam)
        imagenes_sp.append(imagen_filtrada)
        titulos_sp.append(f"Filtro mediana {tam}x{tam}")
    
    mostrar_comparacion(imagenes_sp, titulos_sp, filas=2, cols=3, tam_fig=(15,10))

    print("Filtrando imagen con ruido gaussiano...")
    imagenes_gauss = [imagen_sin_ruido, img_gaussiano]
    titulos_gauss = ["Original sin ruido", "Imagen con ruido gaussiano"]

    for tam in tamanos:
        print(f"  Aplicando filtro: {tam}x{tam} a imagen con ruido gaussiano")
        imagen_filtrada = filtro_mediana(img_gaussiano, tam)
        imagenes_gauss.append(imagen_filtrada)
        titulos_gauss.append(f"Filtro mediana {tam}x{tam}")
    
    mostrar_comparacion(imagenes_gauss, titulos_gauss, filas=2, cols=3, tam_fig=(15,10))

def ejercicio2d(imagen_sin_ruido: np.ndarray, imagen_con_ruido: np.ndarray):
    """
    Ejercicio 2d: Compara detectores de bordes Prewitt y Sobel.
    
    Aplica ambos operadores a imágenes limpias y ruidosas, mostrando:
    - Gradientes direccionales (X e Y)
    - Magnitud del gradiente (detección de bordes)
    - Efecto del ruido en la detección
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original sin ruido.
        imagen_con_ruido (np.ndarray): Imagen con ruido gaussiano.
    """

    print(f"\nEjercicio 2d: Detectores de bordes Prewitt y Sobel")

    print("Aplicando gradiente Prewitt a imagen sin ruido...")
    px, py, pmag = gradiente_prewitt(imagen_sin_ruido)
    imagenes_p = [imagen_sin_ruido, px, py, pmag]
    titulos_p = ["Original sin ruido", "Gradiente Prewitt X", "Gradiente Prewitt Y", "Magnitud Prewitt"]
    mostrar_comparacion(imagenes_p, titulos_p, filas=1, cols=4, tam_fig=(16,4))

    print("Aplicando gradiente Sobel a imagen sin ruido...")
    sx, sy, smag = gradiente_sobel(imagen_sin_ruido)
    imagenes_s = [imagen_sin_ruido, sx, sy, smag]       
    titulos_s = ["Original sin ruido", "Gradiente Sobel X", "Gradiente Sobel Y", "Magnitud Sobel"]
    mostrar_comparacion(imagenes_s, titulos_s, filas=1, cols=4, tam_fig=(16,4))

    print("Aplicando gradiente Prewitt a imagen con ruido...")
    px_r, py_r, pmag_r = gradiente_prewitt(imagen_con_ruido)
    imagenes_pr = [imagen_con_ruido, px_r, py_r, pmag_r]
    titulos_pr = ["Original con ruido", "Gradiente Prewitt X", "Gradiente Prewitt Y", "Magnitud Prewitt"]
    mostrar_comparacion(imagenes_pr, titulos_pr, filas=1, cols=4, tam_fig=(16,4))

    print("Aplicando gradiente Sobel a imagen con ruido...")
    sx_r, sy_r, smag_r = gradiente_sobel(imagen_con_ruido)
    imagenes_sr = [imagen_con_ruido, sx_r, sy_r, smag_r]       
    titulos_sr = ["Original con ruido", "Gradiente Sobel X", "Gradiente Sobel Y", "Magnitud Sobel"]
    mostrar_comparacion(imagenes_sr, titulos_sr, filas=1, cols=4, tam_fig=(16,4))

def ejercicio2e(imagen_sin_ruido: np.ndarray, imagen_con_ruido: np.ndarray):
    """
    Ejercicio 2e: Demuestra técnicas de realce con Laplaciano y Unsharp Masking.
    
    Aplica:
    1. Laplaciano (90° y 45°) para detectar bordes
    2. Realce mediante resta del Laplaciano
    3. Unsharp Masking con diferentes parámetros
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original sin ruido.
        imagen_con_ruido (np.ndarray): Imagen con ruido gaussiano.
    """

    print(f"\nEjercicio 2e: Filtro Laplaciano y Unsharp Masking")
    print("Difuminando imágenes con filtro 5x5...")
    img_difuminada = filtro_promedio_ponderado(imagen_sin_ruido, tam=5)
    img_con_ruido_difuminada = filtro_promedio_ponderado(imagen_con_ruido, tam=5)

    print("Aplicando Laplaciano 90º")
    lap90 = laplaciano90(img_difuminada)
    lap90_norm = np.clip((lap90 - lap90.min()) * 255 / (lap90.max() - lap90.min()), 0, 255).astype(np.uint8)
    realce90 = np.clip(imagen_sin_ruido.astype(np.float64) - lap90, 0, 255).astype(np.uint8)

    print("Aplicando Laplaciano 45º")
    lap45 = laplaciano45(imagen_sin_ruido)
    lap45_norm = np.clip((lap45 - lap45.min()) * 255 / (lap45.max() - lap45.min()), 0, 255).astype(np.uint8)
    realce45 = np.clip(imagen_sin_ruido.astype(np.float64) - lap45, 0, 255).astype(np.uint8)

    imagenes_lap = [imagen_sin_ruido, lap90_norm, realce90, lap45_norm, realce45]
    titulos_lap = ["Original", "Laplaciano 90º", "Realce Laplaciano 90º", "Laplaciano 45º", "Realce Laplaciano 45º"]
    mostrar_comparacion(imagenes_lap, titulos_lap, filas=1, cols=5, tam_fig=(20,4))

    print("Aplicando Unsharp Masking a imagen sin ruido...")
    unsharp_3_prom = unsharp_masking(img_difuminada, 'promedio', 3)
    unsharp_7_prom = unsharp_masking(img_difuminada, 'promedio', 7)
    unsharp_3_pond = unsharp_masking(img_difuminada, 'ponderado', 3)
    unsharp_7_pond = unsharp_masking(img_difuminada, 'ponderado', 7)

    imagenes_um = [img_difuminada, unsharp_3_prom, unsharp_7_prom, unsharp_3_pond, unsharp_7_pond]
    titulos_um = ["Original", "Unsharp Masking Promedio 3x3", "Unsharp Masking Promedio 7x7",
                  "Unsharp Masking Ponderado 3x3", "Unsharp Masking Ponderado 7x7"]
    mostrar_comparacion(imagenes_um, titulos_um, filas=1, cols=5, tam_fig=(20,4))

    print("Aplicando Unsharp Masking a imagen con ruido...")
    unsharp_3_prom_r = unsharp_masking(img_con_ruido_difuminada, 'promedio', 3)
    unsharp_7_prom_r = unsharp_masking(img_con_ruido_difuminada, 'promedio', 7)     
    unsharp_3_pond_r = unsharp_masking(img_con_ruido_difuminada, 'ponderado', 3)
    unsharp_7_pond_r = unsharp_masking(img_con_ruido_difuminada , 'ponderado', 7)
    imagenes_um_r = [img_con_ruido_difuminada, unsharp_3_prom_r, unsharp_7_prom_r, unsharp_3_pond_r, unsharp_7_pond_r]
    titulos_um_r = ["Original con ruido", "Unsharp Masking Promedio 3x3", "Unsharp Masking Promedio 7x7",
                    "Unsharp Masking Ponderado 3x3", "Unsharp Masking Ponderado 7x7"]
    mostrar_comparacion(imagenes_um_r, titulos_um_r, filas=1, cols=5, tam_fig=(20,4))

def inicia_Ejercicio2():
    """
    Función principal que ejecuta todos los ejercicios de filtrado espacial.
    
    Esta función:
    1. Selecciona una imagen aleatoriamente del conjunto disponible
    2. La normaliza a rango [0, 255] si es necesario
    3. Genera versión con ruido gaussiano
    4. Ejecuta secuencialmente ejercicios 2a-2e
    
    Ejercicios ejecutados:
    - 2a: Filtro promedio estándar
    - 2b: Filtro promedio ponderado
    - 2c: Filtro de mediana vs diferentes ruidos
    - 2d: Detectores de bordes (Prewitt y Sobel)
    - 2e: Realce con Laplaciano y Unsharp Masking
    
    Returns:
        None: Genera múltiples visualizaciones interactivas.
    """
    print("Iniciando Ejercicio 2: Filtrado Espacial")
    rutas = [
        "../imagenes/lena.tiff",
        "../imagenes/mamografia.tiff",
        "../imagenes/brain.tiff",
        "../imagenes/granos.png",
        "../imagenes/resonancia.tiff",
        "../imagenes/tungsten_1.jpg",
        "../imagenes/tungsten_2.jpg"
    ]

    ruta = random.choice(rutas)
    print(f"\nImagen seleccionada: {ruta}")

    imagen_sin_ruido = io.imread(ruta)

    if imagen_sin_ruido.dtype != np.uint8:
        imagen_sin_ruido = ((imagen_sin_ruido - imagen_sin_ruido.min()) / 
                           (imagen_sin_ruido.max() - imagen_sin_ruido.min()) * 255).astype(np.uint8)
    
    imagen_con_ruido = agregar_ruido_gaussiano(imagen_sin_ruido, sigma=25)

    ejercicio2a(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2b(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2c(imagen_sin_ruido)
    ejercicio2d(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2e(imagen_sin_ruido, imagen_con_ruido)




