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

def filtroAdaptativo(orden, imagen):
    filas, cols = imagen.shape
    pad = orden // 2
    
    # Agregar padding
    imagen_padded = np.pad(imagen.astype(np.float64), pad, mode='edge')
    resultado = np.zeros_like(imagen, dtype=np.float64)
    
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
# FILTRO MEDIANA 
# =====================================

def filtroMediana(orden, imagen):
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

# =====================================
# FILTRO PROMEDIO PONDERADO
# =====================================
def kernel_promedio_ponderado(tam):
    
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
    kernel = kernel_promedio_ponderado(tam)
    resultado = convolucion(imagen.astype(np.float64), kernel)
    return np.clip(resultado, 0, 255).astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 1
# =====================================

def filtroWienerCaso1(imagen_degradada, varianza_ruido):
    g = imagen_degradada.astype(np.float64)
    filas, columnas = g.shape
    
    G = np.fft.fft2(g)
    
    S_gg = np.abs(G) ** 2
    
    S_nn = varianza_ruido * filas * columnas
    
    S_ff = np.maximum(S_gg - S_nn, 1e-10)
    
    W = S_ff / (S_ff + S_nn)
    
    # Paso 6: Aplicar el filtro en frecuencia
    F_restaurada = W * G
    
    # Paso 7: Transformada inversa de Fourier
    f_restaurada = np.fft.ifft2(F_restaurada)
    f_restaurada = np.real(f_restaurada)
    
    # Paso 8: Clip y convertir a uint8
    f_restaurada = np.clip(f_restaurada, 0, 255)
    
    return f_restaurada.astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 2
# =====================================

def obtener_H(kernel, shape_imagen):
    kernel = kernel / np.sum(kernel)
    
    kh, kw = kernel.shape
    
    # Padding del kernel al tamaño de la imagen
    padded = np.zeros(shape_imagen, dtype=np.float64)
    
    # Colocar kernel en la esquina superior izquierda
    padded[:kh, :kw] = kernel
    
    # Hacer shift circular: mover el centro del kernel a (0,0)
    padded = np.roll(padded, -kh // 2, axis=0)
    padded = np.roll(padded, -kw // 2, axis=1)
    
    # FFT
    H = np.fft.fft2(padded)
    
    return H

def filtroWienerCaso2(imagen_degradada, kernel):
    g = imagen_degradada.astype(np.float64)
    G = np.fft.fft2(g)

    H = obtener_H(kernel, g.shape)

    # Evitar divisiones por cero
    eps = 1e-6
    H_seguro = np.where(np.abs(H) < eps, eps, H)

    F_rest = G / H_seguro
    f_rest = np.fft.ifft2(F_rest)
    f_rest = np.real(f_rest)
    f_rest = np.clip(f_rest, 0, 255)
    return f_rest.astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 3
# =====================================
def filtroWienerCaso3(imagen_degradada, kernel, varianza_ruido):
    g = imagen_degradada.astype(np.float64)
    G = np.fft.fft2(g)

    H = obtener_H(kernel, g.shape)
    
    eps = 1e-10
    H_seguro = np.where(np.abs(H) < eps, eps, H)
    
    S_gg = np.abs(G) ** 2
    
    filas, columnas = g.shape
    S_nn = varianza_ruido * filas * columnas
    
    H_abs2 = np.abs(H) ** 2
    H_abs2_seguro = np.maximum(H_abs2, eps)
    
    S_ff_estimado = np.maximum((S_gg / H_abs2_seguro) - S_nn, eps)
    
    numerador = S_ff_estimado
    denominador = H_seguro * (S_ff_estimado + S_nn)
    
    denominador = np.where(np.abs(denominador) < eps, eps, denominador)
    
    W = numerador / denominador
    F_rest = W * G
    
    f_rest = np.fft.ifft2(F_rest)
    f_rest = np.real(f_rest)
    f_rest = np.clip(f_rest, 0, 255)
    
    return f_rest.astype(np.uint8)

# =====================================
# FILTRO DE WIENER - CASO 4
# =====================================
def filtroWienerCaso4(imagen_degradada, kernel, varianza_ruido):
    g = imagen_degradada.astype(np.float64)
    G = np.fft.fft2(g)
    
    H = obtener_H(kernel, g.shape)
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    
    eps = 1e-10
    H_abs2 = np.maximum(H_abs2, eps)
    
    S_gg = np.abs(G) ** 2
    
    filas, columnas = g.shape
    S_nn = varianza_ruido * filas * columnas
    
    S_ff = np.maximum(S_gg - S_nn, S_nn * 0.1)
    
    numerador = H_conj * S_ff
    denominador = H_abs2 * S_ff + S_nn
    denominador = np.maximum(denominador, eps)
    
    W = numerador / denominador
    F_rest = W * G
    
    f_rest = np.fft.ifft2(F_rest)
    f_rest = np.real(f_rest)
    
    f_min, f_max = np.percentile(f_rest, [0.5, 99.5])
    f_rest = np.clip(f_rest, f_min, f_max)
    
    if f_max > f_min:
        f_rest = ((f_rest - f_min) / (f_max - f_min)) * 255
    
    return np.clip(f_rest, 0, 255).astype(np.uint8)

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
    
    media = 0
    desviacion_estandar = 50
    varianza_ruido = desviacion_estandar ** 2
    print("\n2. Agregando ruido gaussiano...")
    imagen_ruidosa = ruidoGaussiano(imagen, media, desviacion_estandar)
    
    print("\n3. Aplicando filtro de Wiener (Caso 1)...")
    imagen_restaurada = filtroWienerCaso1(imagen_ruidosa, varianza_ruido)
    print("   Restauración completada")
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('(a) Imagen Original\n(Lena)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(imagen_ruidosa, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'(b) Con Ruido Gaussiano', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(imagen_restaurada, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'(c) Restaurada con Wiener\n' +
                      f'Caso 1', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio5():
    print("EJERCICIO 5: Filtro de Wiener - Caso 2 (Pérdida de nitidez)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    
    kernel9 = kernel_promedio_ponderado(9)
    imagen_borrosa = filtro_promedio_ponderado(imagen, 9)
    
    imagen_restaurada = filtroWienerCaso2(imagen_borrosa, kernel9)
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('(a) Imagen Original (Lena)')
    axes[0].axis('off')
    
    axes[1].imshow(imagen_borrosa, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('(b) Degradada\n(Pérdida de nitidez 9x9)')
    axes[1].axis('off')
    
    axes[2].imshow(imagen_restaurada, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('(c) Restaurada con Wiener\nCaso 2')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio6():
    print("EJERCICIO 6: Filtro de Wiener - Caso 3 (Ruido + Blur)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    
    media = 0
    desviacion_estandar = 20
    varianza_ruido = desviacion_estandar ** 2
    imagen_ruidosa = ruidoGaussiano(imagen, media, desviacion_estandar)
    
    kernel9 = kernel_promedio_ponderado(9)
    imagen_degradada = filtro_promedio_ponderado(imagen_ruidosa, 9)
    
    imagen_restaurada = filtroWienerCaso3(imagen_degradada, kernel9, varianza_ruido)
    
    _, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('(a) Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagen_ruidosa, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('(b) Solo Ruido Gaussiano')
    axes[1].axis('off')
    
    axes[2].imshow(imagen_degradada, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('(c) Ruido + Blur 9x9')
    axes[2].axis('off')
    
    axes[3].imshow(imagen_restaurada, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('(d) Restaurada con Wiener\nCaso 3')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def ejercicio7():
    print("EJERCICIO 7: Filtro de Wiener - Caso 4 (Blur + Ruido)")
    print("="*60)
    
    imagen = cargarImagen('../imagenes/lenag.bmp')
    
    print("\n1. Aplicando pérdida de nitidez (filtro paso bajas 9x9)...")
    kernel9 = kernel_promedio_ponderado(9)
    imagen_borrosa = filtro_promedio_ponderado(imagen, 9)
    
    # 2) Luego agregar ruido gaussiano
    print("\n2. Agregando ruido gaussiano...")
    media = 0
    desviacion_estandar = 20
    varianza_ruido = desviacion_estandar ** 2
    imagen_degradada = ruidoGaussiano(imagen_borrosa, media, desviacion_estandar)
    
    # 3) Restaurar con Wiener - Caso 4
    print("\n3. Aplicando filtro de Wiener (Caso 4)...")
    imagen_restaurada = filtroWienerCaso4(imagen_degradada, kernel9, varianza_ruido)
    print("   Restauración completada")
    
    # 4) Mostrar resultados
    _, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('(a) Imagen Original\n(Lena)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(imagen_borrosa, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('(b) Solo Pérdida de Nitidez\n(Blur 9x9)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(imagen_degradada, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('(c) Blur + Ruido Gaussiano\n(Imagen Degradada)', fontsize=12)
    axes[2].axis('off')
    
    axes[3].imshow(imagen_restaurada, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('(d) Restaurada con Wiener\nCaso 4', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("EJERCICIO 7 COMPLETADO")
    print("="*60)
    