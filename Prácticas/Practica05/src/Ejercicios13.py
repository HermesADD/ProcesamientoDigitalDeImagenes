"""
Práctica 5: Restauración de Imágenes
Autor: Hermes Alberto Delgado Díaz
319258613

Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
    - scipy.fft
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# =====================================
# FUNCIONES DE RUIDO
# =====================================

def ruidoSalPimienta(imagen, pa, pb):
    imagen_ruidosa = imagen.copy().astype(np.float64)
    
    num_pimienta = int(pa * imagen.size)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    imagen_ruidosa[coords_pimienta[0], coords_pimienta[1]] = 0
    
    num_sal = int(pb * imagen.size)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    imagen_ruidosa[coords_sal[0], coords_sal[1]] = 255
    
    return imagen_ruidosa.astype(np.uint8)

def ruidoGaussiano(imagen, media, desviacion_estandar):
    imagen_float = imagen.astype(np.float64)
    ruido = np.random.normal(media, desviacion_estandar, imagen.shape)
    imagen_ruidosa = imagen_float + ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 255)
    return imagen_ruidosa.astype(np.uint8)

# =====================================
# FILTRO PROMEDIO ARITMÉTICO
# =====================================

def convolucionAritmetica(n, vecinos):
    suma = 0.0
    for v in vecinos:
        suma += float(v)
    promedio = suma / n
    return promedio

def filtroAritmetico(orden, imagen):
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
    return np.mean(vecinos)

def varianzaLocal(vecinos):
    return np.var(vecinos)

def varianzaGeneral(imagen):
    return np.var(imagen)

def filtroAdaptativo(imagen, orden, varianzaN):
    filas, cols = imagen.shape
    pad = orden // 2
    
    # Agregar padding
    imagen_padded = np.pad(imagen.astype(np.float64), pad, mode='edge')
    resultado = np.zeros_like(imagen, dtype=np.float64)
    
    # Si no se proporciona varianza del ruido, estimarla
    if varianzaN is None:
        varianzaN = varianzaGeneral(imagen)
    
    # Aplicar filtro
    for i in range(filas):
        for j in range(cols):
            # Extraer vecindad
            ventana = imagen_padded[i:i+orden, j:j+orden]
            vecinos = ventana.flatten()
            
            # Calcular estadísticas locales
            mediaL = mediaLocal(vecinos)
            varianzaL = varianzaLocal(vecinos)
            
            # Valor del píxel actual
            pixel = imagen_padded[i+pad, j+pad]
            
            # Aplicar fórmula adaptativa
            if varianzaL != 0:
                frac = varianzaN / varianzaL
                # Limitar la fracción para evitar sobre-corrección
                frac = min(frac, 1.0)
                resta = pixel - mediaL
                resultado[i, j] = pixel - (frac * resta)
            else:
                resultado[i, j] = mediaL
    
    return np.clip(resultado, 0, 255).astype(np.uint8)

# =====================================
# FILTRO MEDIANA ADAPTATIVO
# =====================================

def mediana_pixeles(pixeles):
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
    A1 = zMed - zMin
    A2 = zMed - zMax
    
    return A1, A2

def nivelB(zXY, zMax, zMin):
    B1 = zXY - zMin
    B2 = zXY - zMax
    return B1, B2

def filtroMedianaAdaptativo(imagen, smax):
    filas, cols = imagen.shape
    resultado = np.zeros_like(imagen, dtype=np.float64)
    
    # Agregar padding para el tamaño máximo
    pad_max = smax // 2
    imagen_padded = np.pad(imagen.astype(np.float64), pad_max, mode='edge')
    
    for i in range(filas):
        for j in range(cols):
            # Coordenadas en imagen con padding
            pi = i + pad_max
            pj = j + pad_max
            
            # Valor del píxel actual
            zXY = imagen_padded[pi, pj]
            
            # Empezar con ventana de tamaño 3
            orden_actual = 3
            
            while orden_actual <= smax:
                pad = orden_actual // 2
                
                # Extraer ventana actual
                ventana = imagen_padded[pi-pad:pi+pad+1, pj-pad:pj+pad+1]
                vecinos = ventana.flatten()
                
                # Calcular estadísticas
                zMin = np.min(vecinos)
                zMax = np.max(vecinos)
                zMed = mediana_pixeles(vecinos)
                
                # NIVEL A
                A1, A2 = nivelA(zMed, zMax, zMin)
                
                if A1 > 0 and A2 < 0:
                    # z_med no es impulso, ir al nivel B
                    # NIVEL B
                    B1, B2 = nivelB(zXY, zMax, zMin)
                    
                    if B1 > 0 and B2 < 0:
                        # z_xy no es impulso
                        resultado[i, j] = zXY
                    else:
                        # z_xy es impulso
                        resultado[i, j] = zMed
                    break
                else:
                    # z_med es impulso, incrementar ventana
                    orden_actual += 2
                    
                    if orden_actual > smax:
                        # Alcanzó tamaño máximo
                        resultado[i, j] = zXY
                        break
    
    return np.clip(resultado, 0, 255).astype(np.uint8)

def probar_filtros():
    """
    Función para probar todos los filtros implementados.
    """
    print("Cargando imagen de prueba...")
    
    # Cargar imagen (ajusta la ruta según tu estructura)
    try:
        imagen = io.imread('../imagenes/circuit2.png')
        print(f"Imagen cargada. Shape: {imagen.shape}, Dtype: {imagen.dtype}")
    except Exception as e:
        print(f"No se pudo cargar la imagen: {e}")
        print("Creando imagen de prueba...")
        imagen = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    
    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3:
        print(f"Imagen a color detectada con shape: {imagen.shape}")
        print("Convirtiendo a escala de grises...")
        # Método 1: Promedio de canales
        imagen = np.mean(imagen, axis=2).astype(np.uint8)
        # Método 2 (alternativo): Conversión estándar RGB a gris
        # imagen = (0.299 * imagen[:,:,0] + 0.587 * imagen[:,:,1] + 0.114 * imagen[:,:,2]).astype(np.uint8)
        print(f"Nueva shape: {imagen.shape}")
    
    # Normalizar si no es uint8
    if imagen.dtype != np.uint8:
        print(f"Normalizando imagen de {imagen.dtype} a uint8...")
        imagen = ((imagen - imagen.min()) / 
                 (imagen.max() - imagen.min()) * 255).astype(np.uint8)
    
    print(f"Imagen lista. Shape final: {imagen.shape}, Dtype: {imagen.dtype}")
    print("\n1. Probando Filtro Aritmético...")
    # Agregar ruido gaussiano
    img_gauss = ruidoGaussiano(imagen, 0, 25)
    img_arit = filtroAritmetico(7, img_gauss)
    
    print("\n2. Probando Filtro Geométrico...")
    img_geom = filtroGeometrico(7, img_gauss)
    
    print("\n3. Probando Filtro Adaptativo...")
    # Estimar varianza del ruido (sigma^2 = 25^2 = 625)
    varianza_ruido = 625
    img_adapt = filtroAdaptativo(img_gauss, 7, varianza_ruido)
    
    print("\n4. Probando Filtro Mediana Adaptativo...")
    # Agregar ruido sal y pimienta
    img_sp = ruidoSalPimienta(imagen, 0.25, 0.25)
    img_med_adapt = filtroMedianaAdaptativo(img_sp, 7)
    
    # Mostrar resultados
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Fila 1: Filtros con ruido gaussiano
    axes[0, 0].imshow(imagen, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_gauss, cmap='gray')
    axes[0, 1].set_title('Con Ruido Gaussiano')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_arit, cmap='gray')
    axes[0, 2].set_title('Filtro Aritmético 7x7')
    axes[0, 2].axis('off')
    
    # Fila 2: Continuación filtros gaussiano
    axes[1, 0].imshow(img_geom, cmap='gray')
    axes[1, 0].set_title('Filtro Geométrico 7x7')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_adapt, cmap='gray')
    axes[1, 1].set_title('Filtro Adaptativo 7x7')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    # Fila 3: Filtro mediana adaptativo
    axes[2, 0].imshow(imagen, cmap='gray')
    axes[2, 0].set_title('Original')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(img_sp, cmap='gray')
    axes[2, 1].set_title('Con Ruido Sal y Pimienta')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(img_med_adapt, cmap='gray')
    axes[2, 2].set_title('Filtro Mediana Adaptativo')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n¡Pruebas completadas!")

# Ejecutar pruebas si se ejecuta directamente
if __name__ == "__main__":
    probar_filtros()
    
    