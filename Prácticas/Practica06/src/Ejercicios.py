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
def cargarImagen(ruta_imagen):
    print("Cargando imagen de prueba...")
    imagen = io.imread(ruta_imagen)
    print(f"Imagen cargada. Shape: {imagen.shape}, Dtype: {imagen.dtype}")
    return imagen

def normalizar(imagen):
    return imagen.astype(np.float64) / 255.0

def desnormalizar(imagen):
    return np.clip(imagen * 255, 0, 255).astype(np.uint8)

def histograma(imagen):
    imagen256 = imagen.astype(np.uint8)
    histograma_dict = {i: 0 for i in range(256)}
    y, x = imagen.shape
    for j in range(0, y):
        for i in range(0, x):
            histograma_dict[imagen256[j][i]] += 1
    return histograma_dict

def ecualizar_histograma(imagen):
    es_normalizada = imagen.dtype in [np.float32, np.float64] and imagen.max() <= 1.0
    
    if es_normalizada:
        imagen_uint8 = (imagen * 255).astype(np.uint8)
    else:
        imagen_uint8 = imagen.astype(np.uint8)
    
    y, x = imagen_uint8.shape
    total_pixeles = y * x

    hist_dict = histograma(imagen_uint8)
    histo = np.array([hist_dict[i] for i in range(256)])
    
    # Calcular probabilidades
    probabilidad = histo / total_pixeles
    
    # Calcular CDF
    cdf = np.cumsum(probabilidad)
    
    # Escalar CDF a rango [0, 255]
    cdf_escalado = (cdf * 255).astype(np.uint8)
    
    # Mapear cada pixel usando el CDF
    imagen_ecualizada = cdf_escalado[imagen_uint8]
    
    # Devolver en el mismo formato que la entrada
    if es_normalizada:
        return imagen_ecualizada.astype(np.float64) / 255.0
    else:
        return imagen_ecualizada
    
    
# =====================================
# FUNCION RGB a HSI
# =====================================

def rgb_a_hsi(imagen):
    img_norm = normalizar(imagen)
    
    r = img_norm[:,:,0]
    g = img_norm[:,:,1]
    b = img_norm[:,:,2]
    
    #Componente H
    
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

def hsi_a_rgb(H, S, I):
    height, width = H.shape
    r = np.zeros((height, width))
    g = np.zeros((height, width))
    b = np.zeros((height, width))
    
    # Sector RG: 0° <= H < 120°
    sector_rg = (H >= 0) & (H < 120)
    H_rad = np.radians(H[sector_rg])
    b[sector_rg] = I[sector_rg] * (1 - S[sector_rg])
    r[sector_rg] = I[sector_rg] * (1 + (S[sector_rg] * np.cos(H_rad)) / 
                                np.cos(np.radians(60) - H_rad))
    g[sector_rg] = 3 * I[sector_rg] - (r[sector_rg] + b[sector_rg])
    
    # Sector GB: 120° <= H < 240°
    sector_gb = (H >= 120) & (H < 240)
    H_adjusted = H[sector_gb] - 120
    H_rad = np.radians(H_adjusted)
    r[sector_gb] = I[sector_gb] * (1 - S[sector_gb])
    g[sector_gb] = I[sector_gb] * (1 + (S[sector_gb] * np.cos(H_rad)) / 
                                np.cos(np.radians(60) - H_rad))
    b[sector_gb] = 3 * I[sector_gb] - (r[sector_gb] + g[sector_gb])
    
    # Sector BR: 240° <= H <= 360°
    sector_br = (H >= 240) & (H <= 360)
    H_adjusted = H[sector_br] - 240
    H_rad = np.radians(H_adjusted)
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

def rebanado_intensidad(imagen, num_niveles):
    """
    Representa una imagen en escala de grises en pseudocolor usando
    el método de rebanado de intensidad.
    
    Args:
        imagen: numpy array 2D en escala de grises (puede ser uint8 o normalizada)
        num_niveles: número de niveles de color (P+1 intervalos)
    
    Returns:
        imagen_pseudocolor: numpy array con shape (height, width, 3) en RGB
    """
    # Asegurar que la imagen esté en uint8 [0, 255]
    if imagen.dtype in [np.float32, np.float64]:
        imagen_gray = (imagen * 255).astype(np.uint8)
    else:
        imagen_gray = imagen.astype(np.uint8)
    
    height, width = imagen_gray.shape
    
    # Crear imagen de pseudocolor
    imagen_pseudocolor = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calcular los límites de los intervalos (planos de rebanado)
    # Dividimos el rango [0, 255] en num_niveles intervalos
    limites = np.linspace(0, 256, num_niveles + 1, dtype=int)
    
    # Generar colores distintivos para cada nivel
    # Usamos el espacio de color HSV para obtener colores bien distribuidos
    colores = []
    for i in range(num_niveles):
        # Distribuir el Hue uniformemente en el círculo de colores
        hue = (i * 360.0) / num_niveles
        
        # Convertir HSV a RGB (S=1, V=1 para colores saturados)
        h_norm = hue / 360.0
        
        # Conversión HSV a RGB (caso simple con S=1, V=1)
        sector = int(h_norm * 6)
        f = (h_norm * 6) - sector
        
        if sector == 0:
            r, g, b = 1, f, 0
        elif sector == 1:
            r, g, b = 1 - f, 1, 0
        elif sector == 2:
            r, g, b = 0, 1, f
        elif sector == 3:
            r, g, b = 0, 1 - f, 1
        elif sector == 4:
            r, g, b = f, 0, 1
        else:
            r, g, b = 1, 0, 1 - f
        
        # Convertir a uint8
        color_rgb = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
        colores.append(color_rgb)
    
    # Asignar colores según el intervalo de intensidad
    for k in range(num_niveles):
        # Crear máscara para el k-ésimo intervalo V_k
        if k == num_niveles - 1:
            # Último intervalo incluye el límite superior
            mascara = (imagen_gray >= limites[k]) & (imagen_gray <= limites[k + 1])
        else:
            mascara = (imagen_gray >= limites[k]) & (imagen_gray < limites[k + 1])
        
        # Asignar el color c_k a todos los píxeles en el intervalo V_k
        imagen_pseudocolor[mascara] = colores[k]
    
    return imagen_pseudocolor

def ejercicio1():
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
    axes[0,1].set_title('Imagen RGB -> HSI)', fontsize=14, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(imagenRGB)
    axes[1,0].set_title('Imagen HSI -> RGB)', fontsize=14, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(imagen_realzada)
    axes[1,1].set_title('Imagen HSI con I ecualizada -> RGB)', fontsize=14, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio2():
    """
    Aplica pseudocolor a imágenes en escala de grises usando
    el método de rebanado de intensidad.
    """
    # Lista de imágenes a procesar (ajusta las rutas según tu estructura)
    imagenes_rutas = [
        '../imagenes/cadera.jpg',
        '../imagenes/mano.jpeg',
        '../imagenes/medtest.png',
        '../imagenes/rodilla.jgp'
    ]
    
    # Diferentes niveles de color a probar
    niveles_color = [3, 4, 8, 16]
    
    for ruta in imagenes_rutas:
        try:
            print(f"\nProcesando: {ruta}")
            imagen = cargarImagen(ruta)
            
            # Si la imagen es RGB, convertir a escala de grises
            if len(imagen.shape) == 3:
                # Conversión simple a escala de grises
                imagen_gray = np.mean(imagen, axis=2).astype(np.uint8)
            else:
                imagen_gray = imagen
            
            # Crear figura para mostrar resultados
            num_resultados = len(niveles_color) + 1
            fig, axes = plt.subplots(1, num_resultados, figsize=(5 * num_resultados, 5))
            
            # Mostrar imagen original
            axes[0].imshow(imagen_gray, cmap='gray')
            axes[0].set_title('Original\n(Escala de grises)', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Aplicar pseudocolor con diferentes niveles
            for i, num_niveles in enumerate(niveles_color):
                print(f"  Aplicando pseudocolor con {num_niveles} niveles...")
                imagen_pseudo = rebanado_intensidad(imagen_gray, num_niveles)
                
                axes[i + 1].imshow(imagen_pseudo)
                axes[i + 1].set_title(f'Pseudocolor\n{num_niveles} niveles', 
                                     fontsize=12, fontweight='bold')
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except FileNotFoundError:
            print(f"  ⚠ Archivo no encontrado: {ruta}")
            print(f"  Verifica la ruta de la imagen.")
    
    print("\n✓ Ejercicio 2 completado.")