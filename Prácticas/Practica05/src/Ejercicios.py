"""
Práctica 5: Restauración de Imágenes
Autor: Hermes Alberto Delgado Díaz
319258613

Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# =====================================
# FUNCIONES DE RUIDO
# =====================================

def ruidoSalPimienta(imagen, pa, pb):
    """
    Aplica ruido de sal y pimienta a una imagen.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises
        pa (float): Probabilidad de ruido tipo pimienta (píxeles negros), valor entre 0 y 1
        pb (float): Probabilidad de ruido tipo sal (píxeles blancos), valor entre 0 y 1
    
    Returns:
        np.ndarray: Imagen con ruido de sal y pimienta aplicado (tipo uint8)
    """
    
    imagen_ruidosa = imagen.copy().astype(np.float64)
    
    num_pimienta = int(pa * imagen.size)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    imagen_ruidosa[coords_pimienta[0], coords_pimienta[1]] = 0
    
    num_sal = int(pb * imagen.size)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    imagen_ruidosa[coords_sal[0], coords_sal[1]] = 255
    
    return imagen_ruidosa.astype(np.uint8)

def ruidoGaussiano(imagen, media, desviacion_estandar):
    """
    Aplica ruido gaussiano aditivo a una imagen.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises
        media (float): Media de la distribución gaussiana (típicamente 0)
        desviacion_estandar (float): Desviación estándar del ruido
    
    Returns:
        np.ndarray: Imagen con ruido gaussiano aplicado (tipo uint8)
    """
    
    imagen_float = imagen.astype(np.float64)
    ruido = np.random.normal(media, desviacion_estandar, imagen.shape)
    imagen_ruidosa = imagen_float + ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 255)
    return imagen_ruidosa.astype(np.uint8)

# =====================================
# FILTRO PROMEDIO ARITMÉTICO
# =====================================

def convolucionAritmetica(n, vecinos):
    """
    Calcula el promedio aritmético de un conjunto de valores.
    
    Args:
        n (int): Número total de elementos en el vecindario
        vecinos (np.ndarray): Array con los valores de los píxeles vecinos
    
    Returns:
        float: Promedio aritmético de los valores
    """
    suma = 0.0
    for v in vecinos:
        suma += float(v)
    promedio = suma / n
    return promedio

def filtroAritmetico(orden, imagen):
    """
    Aplica un filtro de promedio aritmético a una imagen.
    
    Args:
        orden (int): Tamaño de la ventana del filtro (debe ser impar, ej: 3, 5, 7)
        imagen (np.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        np.ndarray: Imagen filtrada (tipo uint8)
    """
    filas, columnas = imagen.shape
    pad = orden // 2
    
    imagen_padded = np.pad(imagen, pad, mode = 'edge')
    resultado = np.zeros_like(imagen, dtype = np.float64)
    
    for i in range(filas):
        for j in range(columnas):
            ventana = imagen_padded[i: i + orden, j: j + orden]
            vecinos = ventana.flatten()
            
            n = orden * orden
            resultado[i, j] = convolucionAritmetica(n,vecinos)
            
    return np.clip(resultado, 0, 255).astype(np.uint8)
    
# =====================================
# FILTRO PROMEDIO GEOMÉTRICO
# =====================================

def convolucionGeometrica(n, vecinos):
    """
    Calcula el promedio geométrico de un conjunto de valores.
    
    Args:
        n (int): Número total de elementos en el vecindario
        vecinos (np.ndarray): Array con los valores de los píxeles vecinos
    
    Returns:
        float: Promedio geométrico de los valores
    """
    producto = 1.0
    
    for v in vecinos:
        if v != 0:
            producto *= v
        else:
            producto *= 0.001
    exponente = 1.0 / n
    resultado  = producto ** exponente
    return resultado

def filtroGeometrico(orden, imagen):
    """
    Aplica un filtro de promedio geométrico a una imagen.
    
    Args:
        orden (int): Tamaño de la ventana del filtro (debe ser impar, ej: 3, 5, 7)
        imagen (np.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        np.ndarray: Imagen filtrada (tipo uint8)
    """
    filas, columnas = imagen.shape
    pad = orden // 2
    
    imagen_padded = np.pad(imagen.astype(np.float64), pad, mode = 'edge')
    resultado = np.zeros_like(imagen, dtype = np.float64)
    
    for i in range(filas):
        for j in range(columnas):
            ventana = imagen_padded[i: i + orden, j: j + orden]
            vecinos = ventana.flatten()
            
            n = orden * orden
            resultado[i, j] = convolucionGeometrica(n, vecinos)
    
    return np.clip(resultado, 0, 255).astype(np.uint8)
    
# =====================================
# FILTRO ADAPTATIVO 
# =====================================

def mediaLocal(vecinos):
    """
    Calcula la media local de un vecindario de píxeles.
    
    Args:
        vecinos (np.ndarray): Array con los valores de los píxeles vecinos
    
    Returns:
        float: Media de los valores del vecindario
    """
    return np.mean(vecinos)

def varianzaLocal(vecinos):
    """
    Calcula la varianza local de un vecindario de píxeles.
    
    Args:
        vecinos (np.ndarray): Array con los valores de los píxeles vecinos
    
    Returns:
        float: Varianza de los valores del vecindario
    """
    return np.var(vecinos)

def varianzaGeneral(imagen):
    """
    Calcula la varianza global de toda la imagen.
    
    Args:
        imagen (np.ndarray): Imagen completa en escala de grises
    
    Returns:
        float: Varianza de toda la imagen
    """
    return np.var(imagen)

def filtroAdaptativo(orden, imagen):
    """
    Aplica un filtro adaptativo de reducción de ruido.
    
    Args:
        orden (int): Tamaño de la ventana del filtro (debe ser impar)
        imagen (np.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        np.ndarray: Imagen filtrada adaptativamente (tipo uint8)
    """
    filas, cols = imagen.shape
    pad = orden // 2
    
    imagen_padded = np.pad(imagen.astype(np.float64), pad, mode='edge')
    resultado = np.zeros_like(imagen, dtype=np.float64)
    
    varianzaN = varianzaGeneral(imagen)
    
    for i in range(filas):
        for j in range(cols):
            # Extraer vecindad
            ventana = imagen_padded[i:i+orden, j:j+orden]
            vecinos = ventana.flatten()
            
            mediaL = mediaLocal(vecinos)
            varianzaL = varianzaLocal(vecinos)
            
            pixel = imagen_padded[i+pad, j+pad]
            
            if varianzaL != 0:
                frac = varianzaN / varianzaL
                frac = min(frac, 1.0)
                resta = pixel - mediaL
                resultado[i, j] = pixel - (frac * resta)
            else:
                resultado[i, j] = mediaL
    
    return np.clip(resultado, 0, 255).astype(np.uint8)

# =====================================
# FILTRO MEDIANA 
# =====================================

def filtroMediana(orden, imagen):
    """
    Aplica un filtro de mediana a una imagen.
    
    Args:
        orden (int): Tamaño de la ventana del filtro (debe ser impar, ej: 3, 5, 7)
        imagen (np.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        np.ndarray: Imagen filtrada (tipo uint8)
    """
    img_h, img_w = imagen.shape
    pad = orden // 2

    imagen_padded = np.zeros((img_h + 2 * pad, img_w + 2 * pad))
    imagen_padded[pad:pad + img_h, pad:pad + img_w] = imagen

    resultado = np.zeros_like(imagen)

    for i in range(img_h):
        for j in range(img_w):
            region = imagen_padded[i:i + orden, j:j + orden]

            valores = region.flatten()
            valores_sort = np.sort(valores)
            mediana = valores_sort[len(valores_sort) // 2]
            resultado[i, j] = mediana

    return resultado.astype(np.uint8)

# =====================================
# FILTRO MEDIANA ADAPTATIVO
# =====================================

def mediana_pixeles(pixeles):
    """
    Calcula la mediana de un conjunto de valores de píxeles.
    
    Args:
        pixeles (list o np.ndarray): Lista de valores de píxeles
    
    Returns:
        float: Valor de la mediana
    """
    listaSort = sorted(pixeles)
    n = len(listaSort)
    
    if n % 2 == 0:
        mediaIzq = listaSort[n // 2 - 1]
        mediaDer = listaSort[n // 2]
        mediana = (mediaIzq + mediaDer) / 2
    else:
        mediana = listaSort[n // 2]
    return mediana

def nivelA(zMed, zMax, zMin):
    """
    Calcula las diferencias para el Nivel A del filtro de mediana adaptativo.
    Args:
        zMed (float): Valor de la mediana del vecindario
        zMax (float): Valor máximo del vecindario
        zMin (float): Valor mínimo del vecindario
    
    Returns:
        tuple: (A1, A2) donde:
            - A1 = z_med - z_min
            - A2 = z_med - z_max
    """
    A1 = zMed - zMin
    A2 = zMed - zMax
    
    return A1, A2

def nivelB(zXY, zMax, zMin):
    """
    Calcula las diferencias para el Nivel B del filtro de mediana adaptativo.
    
    Args:
        zXY (float): Valor del píxel actual
        zMax (float): Valor máximo del vecindario
        zMin (float): Valor mínimo del vecindario
    
    Returns:
        tuple: (B1, B2) donde:
            - B1 = z_xy - z_min
            - B2 = z_xy - z_max
    """
    B1 = zXY - zMin
    B2 = zXY - zMax
    return B1, B2

def filtroMedianaAdaptativo(imagen, smax):
    """
    Aplica un filtro de mediana adaptativo a una imagen.
    
    Este filtro mejora el filtro de mediana tradicional ajustando dinámicamente
    el tamaño de la ventana. El algoritmo opera en dos niveles:
    
    Nivel A: Determina si la mediana es ruido
        - Si la mediana NO es ruido → ir al Nivel B
        - Si la mediana ES ruido → incrementar tamaño de ventana
    
    Nivel B: Determina si el píxel actual es ruido
        - Si el píxel NO es ruido → mantener valor original
        - Si el píxel ES ruido → reemplazar por la mediana
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises
        smax (int): Tamaño máximo permitido para la ventana (debe ser impar)
    
    Returns:
        np.ndarray: Imagen filtrada adaptativamente (tipo uint8)
    """
    filas, cols = imagen.shape
    resultado = np.zeros_like(imagen, dtype=np.float64)
    
    pad_max = smax // 2
    imagen_padded = np.pad(imagen.astype(np.float64), pad_max, mode='edge')
    
    for i in range(filas):
        for j in range(cols):
            pi = i + pad_max
            pj = j + pad_max
            
            zXY = imagen_padded[pi, pj]
            
            orden_actual = 3
            
            while orden_actual <= smax:
                pad = orden_actual // 2
                
                ventana = imagen_padded[pi-pad:pi+pad+1, pj-pad:pj+pad+1]
                vecinos = ventana.flatten()
                
                zMin = np.min(vecinos)
                zMax = np.max(vecinos)
                zMed = mediana_pixeles(vecinos)
                
                A1, A2 = nivelA(zMed, zMax, zMin)
                
                if A1 > 0 and A2 < 0:
                    B1, B2 = nivelB(zXY, zMax, zMin)
                    
                    if B1 > 0 and B2 < 0:
                        resultado[i, j] = zXY
                    else:
                        resultado[i, j] = zMed
                    break
                else:
                    orden_actual += 2
                    
                    if orden_actual > smax:
                        resultado[i, j] = zXY
                        break
    
    return np.clip(resultado, 0, 255).astype(np.uint8)

# =====================================
# FILTRO PROMEDIO PONDERADO
# =====================================
def kernel_promedio_ponderado(tam):
    """
    Genera un kernel para filtro de promedio ponderado.
    
    Args:
        tam (int): Tamaño del kernel (debe ser impar, ej: 3, 5, 7)
    
    Returns:
        np.ndarray: Kernel normalizado de tamaño tam x tam
    """
    if tam == 3:
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float64) / 16.0
    elif tam == 5:
        kernel = np.array([[1, 2, 3, 2, 1],
                           [2, 4, 6, 4, 2],
                           [3, 6, 9, 6, 3],
                           [2, 4, 6, 4, 2],
                           [1, 2, 3, 2, 1]], dtype=np.float64) / 81.0
    else:
        # Para tamaños 7, 9, 11, ... usamos una máscara ponderada
        kernel = np.ones((tam, tam), dtype=np.float64)
        centro = tam // 2
        for i in range(tam):
            for j in range(tam):
                dist = abs(i - centro) + abs(j - centro)
                kernel[i, j] = max(1, tam - dist)
        kernel /= kernel.sum()

    return kernel

def convolucion(imagen, kernel):
    """
    Aplica una operación de convolución entre una imagen y un kernel.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises
        kernel (np.ndarray): Kernel o máscara de convolución
    
    Returns:
        np.ndarray: Resultado de la convolución
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

def filtro_promedio_ponderado(imagen, tam):
    """
    Aplica un filtro de promedio ponderado a una imagen.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en escala de grises
        tam (int): Tamaño del kernel (debe ser impar)
    
    Returns:
        np.ndarray: Imagen suavizada (tipo uint8)
    """
    kernel = kernel_promedio_ponderado(tam)
    resultado = convolucion(imagen.astype(np.float64), kernel)
    return np.clip(resultado, 0, 255).astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 1
# =====================================

def filtroWienerCaso1(imagenOriginal, imagenDegradada):
    """
    Aplica el filtro de Wiener para el Caso 1: Ruido Aditivo.
    
    Args:
        imagenOriginal (np.ndarray): Imagen original
        imagenDegradada (np.ndarray): Imagen con ruido 
    Returns:
        np.ndarray: Imagen restaurada (tipo uint8)
    """
    ruido_muestra = np.zeros_like(imagenOriginal, dtype=np.float64)
    ruido = ruidoGaussiano(ruido_muestra, 0, 25)
    
    Snn = np.abs(np.fft.fft2(ruido))**2 
    
    Sff = np.abs(np.fft.fft2(imagenOriginal))**2
    
    denominador = Sff + Snn
    epsilon = 1e-10
    W = Sff / (denominador + epsilon)
    
    G = np.fft.fft2(imagenDegradada)
    
    F_frecuencia = W * G
    
    F_restaurada = np.abs(np.fft.ifft2(F_frecuencia))
    
    return np.clip(F_restaurada, 0, 255).astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 2
# =====================================

def filtroWienerCaso2(imagenOriginal, imagenDegradada):
    """
    Aplica el filtro de Wiener para el Caso 2: Pérdida de Nitidez (sin ruido).
    
    Args:
        imagenOriginal (np.ndarray): Imagen original 
        imagenDegradada (np.ndarray): Imagen con pérdida de nitidez
    
    Returns:
        np.ndarray: Imagen restaurada (tipo uint8)
    """
    F = np.fft.fft2(imagenOriginal)
    G = np.fft.fft2(imagenDegradada)
    
    epsilon = 1e-10
    H = G / (F + epsilon)
    
    W = 1 / (H + epsilon)
    
    F_estimada = W * G
    F_restaurada = np.abs(np.fft.ifft2(F_estimada))
    return np.clip(F_restaurada, 0, 255).astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 3
# =====================================

def filtroWienerCaso3(imagenOriginal, imagenDegradada):
    """
    Aplica el filtro de Wiener para el Caso 3: Ruido + Pérdida de Nitidez .
    
    Args:
        imagenOriginal (np.ndarray): Imagen original 
        imagenDegradada (np.ndarray): Imagen con ruido + pérdida de nitidez
    
    Returns:
        np.ndarray: Imagen restaurada (tipo uint8)
    """
    F = np.fft.fft2(imagenOriginal)
    G = np.fft.fft2(imagenDegradada)
    
    Sff = np.abs(F)**2
    
    ruido_muestra = np.zeros_like(imagenOriginal, dtype=np.float64)
    ruido = ruidoGaussiano(ruido_muestra, 0, 25)
    
    N = np.fft.fft2(ruido)
    Snn = np.abs(N)**2 
    
    epsilon = 1e-10
    H = G / (F + N + epsilon)
    
    denominador = H * (Sff + Snn)
    
    W = Sff / (denominador + epsilon)
    
    F_estimada = G * W
    F_restaurada = np.abs(np.fft.ifft2(F_estimada))
    
    return np.clip(F_restaurada, 0, 255).astype(np.uint8)
# =====================================
# FILTRO DE WIENER - CASO 4
# =====================================

def filtroWienerCaso4(imagenOriginal, imagenDegradada):
    """
    Aplica el filtro de Wiener para el Caso 4: Pérdida de Nitidez + Ruido .
    
    Args:
        imagenOriginal (np.ndarray): Imagen original 
        imagenDegradada (np.ndarray): Imagen con pérdida de nitidez + ruido
    
    Returns:
        np.ndarray: Imagen restaurada (tipo uint8)
    """
    F = np.fft.fft2(imagenOriginal)
    G = np.fft.fft2(imagenDegradada)
    
    Sff = np.abs(F)**2
    
    ruido_muestra = np.zeros_like(imagenOriginal, dtype=np.float64)
    ruido = ruidoGaussiano(ruido_muestra, 0, 25)
    
    N = np.fft.fft2(ruido)
    Snn = np.abs(N)**2
    
    epsilon = 1e-10
    H = (G - N) / (F + epsilon)
    
    H_conj = np.conj(H)       
    H_mag2 = np.abs(H)**2     
    
    numerador = Sff * H_conj
    
    denominador = (H_mag2 * Sff) + Snn
    
    W = numerador / (denominador + epsilon)
    
    F_estimada = W * G
    img_restaurada = np.abs(np.fft.ifft2(F_estimada))
    
    return np.clip(img_restaurada, 0, 255).astype(np.uint8)

# =====================================
# FUNCIONES AUXILIARES
# =====================================
def cargarImagen(ruta_imagen):
    print("Cargando imagen de prueba...")
    
    try:
        imagen = io.imread(ruta_imagen)
        print(f"Imagen cargada. Shape: {imagen.shape}, Dtype: {imagen.dtype}")
    except Exception as e:
        print(f"No se pudo cargar la imagen: {e}")
        print("Creando imagen de prueba...")
        imagen = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    
    if len(imagen.shape) == 3:
        print(f"Imagen a color detectada con shape: {imagen.shape}")
        print("Convirtiendo a escala de grises...")
        imagen = np.mean(imagen, axis=2).astype(np.uint8)
        
        print(f"Nueva shape: {imagen.shape}")
    
    if imagen.dtype != np.uint8:
        print(f"Normalizando imagen de {imagen.dtype} a uint8...")
        imagen = ((imagen - imagen.min()) / 
                 (imagen.max() - imagen.min()) * 255).astype(np.uint8)
    
    print(f"Imagen lista. Shape final: {imagen.shape}, Dtype: {imagen.dtype}")
    
    return imagen

def ejercicio1():
    print("EJERCICIO 1:")
    print("="*60)
    imagen = cargarImagen('../imagenes/circuit2.png')
    media = 0
    desviacion_estandar = 25
    img_gauss = ruidoGaussiano(imagen, media, desviacion_estandar)
    img_arit = filtroAritmetico(3, img_gauss)
    img_geo = filtroGeometrico(3, img_gauss)
    
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(imagen, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_gauss, cmap='gray')
    axes[0, 1].set_title('Con Ruido Gaussiano')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_arit, cmap='gray')
    axes[1, 0].set_title('Filtro Aritmético 3x3')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_geo, cmap='gray')
    axes[1, 1].set_title('Filtro Geométrico 3x3')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def ejercicio2():
    print("EJERCICIO 2:")
    print("="*60)
    imagen = cargarImagen('../imagenes/circuit2.png')
    media = 0
    desviacion_estandar = 25
    img_gauss = ruidoGaussiano(imagen, media, desviacion_estandar)
    img_arit = filtroAritmetico(7, img_gauss)
    img_geo = filtroGeometrico(7, img_gauss)
    img_adap = filtroAdaptativo(7, img_gauss)
    
    _, axes = plt.subplots(2, 3, figsize=(8, 8))
    axes[0, 0].imshow(imagen, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_gauss, cmap='gray')
    axes[0, 1].set_title('Con Ruido Gaussiano')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_arit, cmap='gray')
    axes[1, 0].set_title('Filtro Aritmético 7x7')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_geo, cmap='gray')
    axes[1, 1].set_title('Filtro Geométrico 7x7')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_adap, cmap='gray')
    axes[1, 2].set_title('Filtro Adaptativo 7x7')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def ejercicio3():
    print("EJERCICIO 3:")
    print("="*60)
    imagen = cargarImagen('../imagenes/circuit2.png')
    pa = 0.25
    pb = 0.25
    img_SP = ruidoSalPimienta(imagen ,pa , pb)
    img_med = filtroMediana(7, img_SP)
    img_medAdap = filtroMedianaAdaptativo(img_SP, 7)
    
    _, axes = plt.subplots(1, 4, figsize=(12, 12))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_SP, cmap='gray')
    axes[1].set_title('Con Ruido Sal y Pimienta')
    axes[1].axis('off')
    
    axes[2].imshow(img_med, cmap='gray')
    axes[2].set_title('Filtro Mediana 7x7')
    axes[2].axis('off')
    
    axes[3].imshow(img_medAdap, cmap='gray')
    axes[3].set_title('Filtro Mediana Adaptativo Smax = 7')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

def ejercicio4():
    print("EJERCICIO 4: Filtro de Wiener - Caso 1 (Ruido Aditivo)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    imagenD = ruidoGaussiano(imagen, 0, 25) 
    imagenR = filtroWienerCaso1(imagen, imagenD)
    _, axes = plt.subplots(1, 3, figsize=(12, 12))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagenD, cmap='gray')
    axes[1].set_title('Con Ruido')
    axes[1].axis('off')
    
    axes[2].imshow(imagenR, cmap='gray')
    axes[2].set_title('Restaurada')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
   
def ejercicio5():
    print("EJERCICIO 5: Filtro de Wiener - Caso 2 (Pérdida de nitidez)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    
    imagenD = filtro_promedio_ponderado(imagen, 9) 
    imagenR = filtroWienerCaso2(imagen, imagenD)
    
    _, axes = plt.subplots(1, 3, figsize=(12, 12))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagenD, cmap='gray')
    axes[1].set_title('Con Pérdida de Nitidez')
    axes[1].axis('off')
    
    axes[2].imshow(imagenR, cmap='gray')
    axes[2].set_title('Restaurada')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio6():
    print("EJERCICIO 6: Filtro de Wiener - Caso 3 (Ruido + PN)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    imagenRuido = ruidoGaussiano(imagen, 0, 25)
    imagenD = filtro_promedio_ponderado(imagenRuido, 9)
    imagenR = filtroWienerCaso3(imagen, imagenD)
    
    _, axes = plt.subplots(1, 4, figsize=(12, 12))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagenRuido, cmap='gray')
    axes[1].set_title('Con Ruido')
    axes[1].axis('off')
    
    axes[2].imshow(imagenD, cmap='gray')
    axes[2].set_title('Con Ruido + Pérdida de Nitidez')
    axes[2].axis('off')
    
    axes[3].imshow(imagenR, cmap='gray')
    axes[3].set_title('Restaurada')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio7():
    print("EJERCICIO 7: Filtro de Wiener - Caso 4 (PN + Ruido)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    imagenPN = filtro_promedio_ponderado(imagen, 9)
    imagenD = ruidoGaussiano(imagenPN, 0, 25)
    imagenR = filtroWienerCaso3(imagen, imagenD)
    
    _, axes = plt.subplots(1, 4, figsize=(12, 12))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagenPN, cmap='gray')
    axes[1].set_title('Con Pérdida de Nitidez')
    axes[1].axis('off')
    
    axes[2].imshow(imagenD, cmap='gray')
    axes[2].set_title('Con Pérdida de Nitidez + Ruido')
    axes[2].axis('off')
    
    axes[3].imshow(imagenR, cmap='gray')
    axes[3].set_title('Restaurada')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    