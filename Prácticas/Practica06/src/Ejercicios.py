"""
Práctica 6: Representación del color.
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
# FUNCIONES AUXILIARES
# =====================================

def cargarImagen(ruta_imagen: str) -> np.ndarray:
    """
    Carga una imagen ingresando su ruta.

    Args:
        ruta_imagen (str): Ruta de la imagen

    Returns:
        np.ndarray: Imagen localizada
    """
    print("Cargando imagen...")
    imagen = io.imread(ruta_imagen)
    print(f"Imagen cargada. Shape: {imagen.shape}, Dtype: {imagen.dtype}")
    return imagen

def normalizar(imagen: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen con valores de intensidad entre [0,1]

    Args:
        imagen (np.ndarray): Imagen a normalizar

    Returns:
        np.ndarray: Imagen normalizada con rango [0,1]
    """
    return imagen.astype(np.float64) / 255.0

def desnormalizar(imagen: np.ndarray) -> np.ndarray:
    """
    Desnormaliza una imagen con valores de intensidad de [0,1] a [0,255]

    Args:
        imagen (np.ndarray): Imagen a desnormalizar

    Returns:
        np.ndarray: Imagen con rango de intensidad entre [0,255]
    """
    
    return np.clip(imagen * 255, 0, 255).astype(np.uint8)

def histograma(imagen: np.ndarray) -> np.ndarray:
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
    for j in range(0, y):
        for i in range(0, x):
            histograma_dict[imagen256[j][i]] += 1
    return histograma_dict


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
    es_normalizada = imagen.dtype in [np.float32, np.float64] and imagen.max() <= 1.0
    
    if es_normalizada:
        imagen_uint8 = (imagen * 255).astype(np.uint8)
    else:
        imagen_uint8 = imagen.astype(np.uint8)
    
    y, x = imagen_uint8.shape
    total_pixeles = y * x

    hist_dict = histograma(imagen_uint8)
    histo = np.array([hist_dict[i] for i in range(256)])
    
    probabilidad = histo / total_pixeles
    
    cdf = np.cumsum(probabilidad)
    
    cdf_escalado = (cdf * 255).astype(np.uint8)
    
    imagen_ecualizada = cdf_escalado[imagen_uint8]
    
    if es_normalizada:
        return imagen_ecualizada.astype(np.float64) / 255.0
    else:
        return imagen_ecualizada
    
# =====================================
# FUNCION RGB a HSI
# =====================================

def rgb_a_hsi(imagen: np.ndarray) -> tuple:
    """
    Convierte una imagen RGB al espacio de color HSI (Hue, Saturation, Intensity).
    
    Args:
        imagen (np.ndarray): Imagen RGB con valores en rango [0, 255] o [0, 1].
                            Shape esperado: (altura, ancho, 3)
    
    Returns:
        tuple: Tupla con tres arrays de NumPy (H, S, I):
            - H (np.ndarray): Componente de matiz en grados [0, 360°]
            - S (np.ndarray): Componente de saturación normalizada [0, 1]
            - I (np.ndarray): Componente de intensidad normalizada [0, 1]
    """
    img_norm = normalizar(imagen)
    
    #Extracción de canales R, G, B
    r = img_norm[:,:,0]
    g = img_norm[:,:,1]
    b = img_norm[:,:,2]
    
    num = 0.5 * ((r-g)+(r-b))
    dem = np.sqrt((r-g)**2 + (r-b)*(g-b))
    dem[dem == 0] = 1e-10
    
    theta = np.arccos(np.clip(num/dem,-1, 1))
    theta_grados = np.degrees(theta)
    
    H = np.where(b <= g, theta_grados, 360-theta_grados)
    
    #Componente S
    
    min_rgb = np.minimum(np.minimum(r,g),b)
    suma_rgb = r + g + b
    suma_rgb[suma_rgb == 0] = 1e-10
    
    S = 1 - (3/suma_rgb) * min_rgb
    
    #Componente I
    
    I = (r + g + b) / 3.0
    
    return H, S, I

# =====================================
# FUNCIONES HSI a RGB
# =====================================

def hsi_a_rgb(H: np.ndarray,S: np.ndarray ,I: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen del espacio de color HSI a RGB.
    
    Args:
        H (np.ndarray): Componente de matiz en grados [0, 360°]
        S (np.ndarray): Componente de saturación [0, 1]
        I (np.ndarray): Componente de intensidad [0, 1]
    
    Returns:
        np.ndarray: Imagen RGB con valores en rango [0, 255] tipo uint8.
                   Shape: (altura, ancho, 3)
    """
    altura, ancho = H.shape
    
    #Creación de canales R, G, B
    r = np.zeros((altura, ancho))
    g = np.zeros((altura, ancho))
    b = np.zeros((altura, ancho))
    
    # Sector RG: 0° <= H < 120°
    sector_rg = (H >= 0) & (H < 120)
    H_rad = np.radians(H[sector_rg])
    b[sector_rg] = I[sector_rg] * (1 - S[sector_rg])
    r[sector_rg] = I[sector_rg] * (1 + (S[sector_rg] * np.cos(H_rad)) / 
                                np.cos(np.radians(60) - H_rad))
    g[sector_rg] = 3 * I[sector_rg] - (r[sector_rg] + b[sector_rg])
    
    # Sector GB: 120° <= H < 240°
    sector_gb = (H >= 120) & (H < 240)
    H_mod = H[sector_gb] - 120
    H_rad = np.radians(H_mod)
    r[sector_gb] = I[sector_gb] * (1 - S[sector_gb])
    g[sector_gb] = I[sector_gb] * (1 + (S[sector_gb] * np.cos(H_rad)) / 
                                np.cos(np.radians(60) - H_rad))
    b[sector_gb] = 3 * I[sector_gb] - (r[sector_gb] + g[sector_gb])
    
    # Sector BR: 240° <= H <= 360°
    sector_br = (H >= 240) & (H <= 360)
    H_mod = H[sector_br] - 240
    H_rad = np.radians(H_mod)
    g[sector_br] = I[sector_br] * (1 - S[sector_br])
    b[sector_br] = I[sector_br] * (1 + (S[sector_br] * np.cos(H_rad)) / 
                                np.cos(np.radians(60) - H_rad))
    r[sector_br] = 3 * I[sector_br] - (g[sector_br] + b[sector_br])
    
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    imagen_rgb = np.stack([r, g, b], axis=2)
    
    return desnormalizar(imagen_rgb)
# =====================================
# FUNCIONES PSEUDOCOLOR
# =====================================

def colores(num_colores: int) -> np.ndarray:
    """
    Genera n colores distribuidos uniformemente en RGB.
    
    Args:
        num_colores (int): Número de colores a generar
    
    Returns:
        np.ndarray: Array de forma (num_colores, 3) con valores RGB en [0, 255]
    """
    if num_colores <= 0:
        return np.array([])
    
    colores_rgb = []
    
    for i in range(num_colores):
        # Generar colores usando el espacio HSL para mejor distribución
        hue = i / num_colores  # Distribuir uniformemente el matiz
        
        # Convertir HSL a RGB (fórmula estándar)
        h = hue * 6.0  # Escalar a [0, 6]
        c = 1.0  # Chroma (saturación máxima)
        x = c * (1 - abs((h % 2) - 1))
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convertir a [0, 255]
        colores_rgb.append([int(r * 255), int(g * 255), int(b * 255)])
    
    return np.array(colores_rgb, dtype=np.uint8)

def pseudocolor(imagen: np.ndarray, num_colores: int) -> np.ndarray:
    """
    Asigna pseudocolores a una imagen en escala de grises dividiendo
    el rango de intensidades en intervalos y asignando un color a cada uno.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises (2D)
        num_colores (int): Número de colores a utilizar
    
    Returns:
        np.ndarray: Imagen en pseudocolor (RGB) con forma (altura, ancho, 3)
    """
    # Asegurar que la imagen está en uint8 [0, 255]
    if imagen.dtype in [np.float32, np.float64]:
        imagen_uint8 = (imagen * 255).astype(np.uint8)
    else:
        imagen_uint8 = imagen.astype(np.uint8)
    
    # Generar la paleta de colores
    paleta = colores(num_colores)
    
    # Crear imagen RGB de salida
    altura, ancho = imagen_uint8.shape
    imagen_pseudocolor = np.zeros((altura, ancho, 3), dtype=np.uint8)
    
    # Dividir el rango [0, 255] en num_colores intervalos
    tamaño_intervalo = 256 / num_colores
    
    # Asignar colores según el intervalo de intensidad
    for i in range(num_colores):
        # Definir límites del intervalo
        limite_inferior = int(i * tamaño_intervalo)
        limite_superior = int((i + 1) * tamaño_intervalo)
        
        # Para el último intervalo, incluir 255
        if i == num_colores - 1:
            limite_superior = 256
        
        # Crear máscara para píxeles en este rango
        mascara = (imagen_uint8 >= limite_inferior) & (imagen_uint8 < limite_superior)
        
        # Asignar el color correspondiente
        imagen_pseudocolor[mascara] = paleta[i]
    
    return imagen_pseudocolor
    
# =====================================
# FUNCIONES EJECUCIÓN
# =====================================

def ejercicio1():
    """
    Ejercicio 1: Conversión RGB-HSI y realce mediante ecualización de intensidad.
    
    Realiza las siguientes operaciones:
    1. Carga una imagen RGB (flowers2.bmp)
    2. Convierte la imagen de RGB a HSI
    3. Convierte de vuelta a RGB para verificar la transformación
    4. Ecualiza la componente I (intensidad) para mejorar el contraste
    5. Convierte la imagen con I ecualizada de vuelta a RGB
    6. Muestra las 4 imágenes resultantes en una cuadrícula 2x2
    
    La ecualización de la componente I permite mejorar el contraste de la
    imagen sin alterar la información de color (H y S), resultando en una
    imagen con colores preservados pero mejor distribución de intensidades.
    """
    
    imagen =cargarImagen('../imagenes/flowers2.bmp')
    
    print("\n Convirtiendo RGB a HSI")
    H, S, I = rgb_a_hsi(imagen)
    H_vis = H / 360.0
    imagenHSI = np.stack([H_vis, S, I ], axis=2)
    imagenHSI = desnormalizar(imagenHSI)
    print(f"  - Imagen HSI: H={H.shape}, S={S.shape}, I={I.shape}")
    imagenRGB = hsi_a_rgb(H,S,I)
    print("")
    print("Ecualizando componente I...")
    I_ecualizada = ecualizar_histograma(I)
    
    print("Convirtiendo HSI realzada a RGB...")
    imagen_realzada = hsi_a_rgb(H, S, I_ecualizada)
    
    _, axes = plt.subplots(2,2,figsize= (12,6))
    axes[0,0].imshow(imagen)
    axes[0,0].set_title('Imagen Original', fontsize=14, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(imagenHSI)
    axes[0,1].set_title('Imagen RGB -> HSI', fontsize=14, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(imagenRGB)
    axes[1,0].set_title('Imagen HSI -> RGB', fontsize=14, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(imagen_realzada)
    axes[1,1].set_title('Imagen HSI con I ecualizada -> RGB', fontsize=14, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

def ejercicio2():
    """
    Ejercicio 2: Aplicación de pseudocolor a imágenes médicas.
    
    Proceso:
    1. Carga imágenes médicas (radiografías, tomografías, etc.)
    2. Convierte a escala de grises si es necesario
    3. Aplica pseudocolor con diferentes números de niveles (4, 8, 12)
    4. Muestra la imagen original y las versiones con pseudocolor
    """
    # Lista de imágenes a procesar (ajusta las rutas según tu estructura)
    imagenes_rutas = [
        '../imagenes/cadera.jpg',
        '../imagenes/mano.jpeg',
        '../imagenes/medtest.png',
        '../imagenes/rodilla.jpg'
    ]
    
    # Diferentes niveles de color a probar
    niveles_color = [4,8,12]
    
    for ruta in imagenes_rutas:
        imagen = cargarImagen(ruta)  
        # Si la imagen es RGB, convertir a escala de grises
        if len(imagen.shape) == 3:
                # Conversión simple a escala de grises
            imagen_gray = np.mean(imagen, axis=2).astype(np.uint8)
        else:
            imagen_gray = imagen
                
        num_resultados = len(niveles_color) + 1
        _, axes = plt.subplots(1, num_resultados, figsize=(5 * num_resultados, 5))
                
        axes[0].imshow(imagen_gray, cmap='gray')
        axes[0].set_title('Original\n(Escala de grises)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
                
        for i, num_niveles in enumerate(niveles_color):
            print(f"  Aplicando pseudocolor con {num_niveles} niveles...")
            imagen_pseudo = pseudocolor(imagen_gray, num_niveles)
                    
            axes[i + 1].imshow(imagen_pseudo)
            axes[i + 1].set_title(f'Pseudocolor\n{num_niveles} niveles', 
                                        fontsize=12, fontweight='bold')
            axes[i + 1].axis('off')
                
        plt.tight_layout()
        plt.show()