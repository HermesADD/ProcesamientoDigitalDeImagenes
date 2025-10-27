import matplotlib.pyplot as plt
import numpy as np
from skimage import io, util
from scipy import ndimage
from scipy.ndimage import median_filter
import random


def agregar_ruido_gaussiano(imagen, media=0, sigma=25):
    """
    Agrega ruido gaussiano a una imagen
    """
    imagen = imagen.astype(np.float64)
    ruido = np.random.normal(media, sigma, imagen.shape)
    imagen_ruidosa = imagen + ruido
    # Recortar valores fuera del rango [0, 255]
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 255)
    return imagen_ruidosa.astype(np.uint8)


def agregar_ruido_sal_pimienta(imagen, cantidad=0.05):
    """
    Agrega ruido sal y pimienta a una imagen
    cantidad: proporción de píxeles afectados (0 a 1)
    """
    imagen_ruidosa = imagen.copy()
    # Sal (blanco)
    num_sal = int(cantidad * imagen.size * 0.5)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    imagen_ruidosa[coords_sal[0], coords_sal[1]] = 255
    
    # Pimienta (negro)
    num_pimienta = int(cantidad * imagen.size * 0.5)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    imagen_ruidosa[coords_pimienta[0], coords_pimienta[1]] = 0
    
    return imagen_ruidosa


def filtro_promedio_estandar(imagen, tamano):
    """
    Filtro paso bajas promedio estándar
    """
    kernel = np.ones((tamano, tamano)) / (tamano * tamano)
    return ndimage.convolve(imagen.astype(np.float64), kernel).astype(np.uint8)


def filtro_promedio_ponderado(imagen, tamano):
    """
    Filtro paso bajas promedio ponderado (más peso al centro)
    """
    if tamano == 3:
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16.0
    elif tamano == 5:
        kernel = np.array([[1, 2, 3, 2, 1],
                          [2, 4, 6, 4, 2],
                          [3, 6, 9, 6, 3],
                          [2, 4, 6, 4, 2],
                          [1, 2, 3, 2, 1]]) / 81.0
    elif tamano == 7:
        # Kernel gaussiano aproximado
        kernel = np.ones((tamano, tamano))
        centro = tamano // 2
        for i in range(tamano):
            for j in range(tamano):
                dist = abs(i - centro) + abs(j - centro)
                kernel[i, j] = max(1, tamano - dist)
        kernel = kernel / kernel.sum()
    else:  # 11x11
        kernel = np.ones((tamano, tamano))
        centro = tamano // 2
        for i in range(tamano):
            for j in range(tamano):
                dist = abs(i - centro) + abs(j - centro)
                kernel[i, j] = max(1, tamano - dist)
        kernel = kernel / kernel.sum()
    
    return ndimage.convolve(imagen.astype(np.float64), kernel).astype(np.uint8)


def filtro_mediana(imagen, tamano):
    """
    Filtro mediana
    """
    return median_filter(imagen, size=tamano)


def gradiente_prewitt(imagen):
    """
    Operador Prewitt en X, Y y magnitud
    """
    # Prewitt X
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    
    # Prewitt Y
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    
    gx = ndimage.convolve(imagen.astype(np.float64), kernel_x)
    gy = ndimage.convolve(imagen.astype(np.float64), kernel_y)
    
    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = np.clip(magnitud, 0, 255).astype(np.uint8)
    
    return gx.astype(np.uint8), gy.astype(np.uint8), magnitud


def gradiente_sobel(imagen):
    """
    Operador Sobel en X, Y y magnitud
    """
    # Sobel X
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    # Sobel Y
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    gx = ndimage.convolve(imagen.astype(np.float64), kernel_x)
    gy = ndimage.convolve(imagen.astype(np.float64), kernel_y)
    
    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = np.clip(magnitud, 0, 255).astype(np.uint8)
    
    return gx.astype(np.uint8), gy.astype(np.uint8), magnitud


def laplaciano_90(imagen):
    """
    Laplaciano isotrópico a 90 grados
    """
    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
    
    resultado = ndimage.convolve(imagen.astype(np.float64), kernel)
    return resultado


def laplaciano_45(imagen):
    """
    Laplaciano isotrópico a 45 grados (incluye diagonales)
    """
    kernel = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])
    
    resultado = ndimage.convolve(imagen.astype(np.float64), kernel)
    return resultado


def unsharp_masking(imagen, filtro_blur, tamano):
    """
    Filtro Unsharp Masking
    imagen_sharp = imagen - imagen_blur
    resultado = imagen + imagen_sharp
    """
    if filtro_blur == 'promedio':
        imagen_blur = filtro_promedio_estandar(imagen, tamano)
    else:  # ponderado
        imagen_blur = filtro_promedio_ponderado(imagen, tamano)
    
    # imagen_sharp = imagen - imagen_blur
    imagen_sharp = imagen.astype(np.float64) - imagen_blur.astype(np.float64)
    
    # resultado = imagen + imagen_sharp
    resultado = imagen.astype(np.float64) + imagen_sharp
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    
    return resultado


def mostrar_comparacion(imagenes, titulos, filas=2, cols=4, tam_fig=(16, 8)):
    """
    Muestra múltiples imágenes en una cuadrícula
    """
    fig, axes = plt.subplots(filas, cols, figsize=tam_fig)
    axes = axes.flatten()
    
    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(len(imagenes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def ejercicio2a_promedio_estandar(imagen_sin_ruido, imagen_con_ruido):
    """
    a) Filtros promedio estándar 3x3, 5x5, 7x7, 11x11
    """
    print("=== Ejercicio 2a: Filtro Promedio Estándar ===")
    
    tamanos = [3, 5, 7, 11]
    imagenes = []
    titulos = []
    
    # Sin ruido
    imagenes.append(imagen_sin_ruido)
    titulos.append("Original sin ruido")
    
    for tam in tamanos:
        img_filtrada = filtro_promedio_estandar(imagen_sin_ruido, tam)
        imagenes.append(img_filtrada)
        titulos.append(f"Sin ruido {tam}x{tam}")
    
    # Con ruido
    imagenes.append(imagen_con_ruido)
    titulos.append("Original con ruido")
    
    for tam in tamanos:
        img_filtrada = filtro_promedio_estandar(imagen_con_ruido, tam)
        imagenes.append(img_filtrada)
        titulos.append(f"Con ruido {tam}x{tam}")
    
    mostrar_comparacion(imagenes[:5], titulos[:5], filas=1, cols=5, tam_fig=(20, 4))
    mostrar_comparacion(imagenes[5:], titulos[5:], filas=1, cols=5, tam_fig=(20, 4))


def ejercicio2b_promedio_ponderado(imagen_sin_ruido, imagen_con_ruido):
    """
    b) Filtros promedio ponderado 3x3, 5x5, 7x7, 11x11
    """
    print("=== Ejercicio 2b: Filtro Promedio Ponderado ===")
    
    tamanos = [3, 5, 7, 11]
    imagenes = []
    titulos = []
    
    # Sin ruido
    imagenes.append(imagen_sin_ruido)
    titulos.append("Original sin ruido")
    
    for tam in tamanos:
        img_filtrada = filtro_promedio_ponderado(imagen_sin_ruido, tam)
        imagenes.append(img_filtrada)
        titulos.append(f"Sin ruido {tam}x{tam}")
    
    # Con ruido
    imagenes.append(imagen_con_ruido)
    titulos.append("Original con ruido")
    
    for tam in tamanos:
        img_filtrada = filtro_promedio_ponderado(imagen_con_ruido, tam)
        imagenes.append(img_filtrada)
        titulos.append(f"Con ruido {tam}x{tam}")
    
    mostrar_comparacion(imagenes[:5], titulos[:5], filas=1, cols=5, tam_fig=(20, 4))
    mostrar_comparacion(imagenes[5:], titulos[5:], filas=1, cols=5, tam_fig=(20, 4))


def ejercicio2c_mediana(imagen_sin_ruido):
    """
    c) Filtro mediana con ruido sal y pimienta y gaussiano
    """
    print("=== Ejercicio 2c: Filtro Mediana ===")
    
    # Ruido sal y pimienta
    img_sal_pimienta = agregar_ruido_sal_pimienta(imagen_sin_ruido, cantidad=0.05)
    
    # Ruido gaussiano
    img_gaussiano = agregar_ruido_gaussiano(imagen_sin_ruido, sigma=25)
    
    tamanos = [3, 5, 7, 11]
    
    # Sal y pimienta
    imagenes_sp = [imagen_sin_ruido, img_sal_pimienta]
    titulos_sp = ["Original", "Sal y pimienta"]
    
    for tam in tamanos:
        img_filtrada = filtro_mediana(img_sal_pimienta, tam)
        imagenes_sp.append(img_filtrada)
        titulos_sp.append(f"Mediana {tam}x{tam}")
    
    mostrar_comparacion(imagenes_sp, titulos_sp, filas=2, cols=3, tam_fig=(15, 10))
    
    # Gaussiano
    imagenes_g = [imagen_sin_ruido, img_gaussiano]
    titulos_g = ["Original", "Gaussiano"]
    
    for tam in tamanos:
        img_filtrada = filtro_mediana(img_gaussiano, tam)
        imagenes_g.append(img_filtrada)
        titulos_g.append(f"Mediana {tam}x{tam}")
    
    mostrar_comparacion(imagenes_g, titulos_g, filas=2, cols=3, tam_fig=(15, 10))


def ejercicio2d_gradientes(imagen_sin_ruido, imagen_con_ruido):
    """
    d) Detectores de borde: Prewitt y Sobel
    """
    print("=== Ejercicio 2d: Detectores de Borde ===")
    
    # Prewitt sin ruido
    px, py, pmag = gradiente_prewitt(imagen_sin_ruido)
    imagenes_p = [imagen_sin_ruido, px, py, pmag]
    titulos_p = ["Original sin ruido", "Prewitt X", "Prewitt Y", "Prewitt Magnitud"]
    mostrar_comparacion(imagenes_p, titulos_p, filas=1, cols=4, tam_fig=(16, 4))
    
    # Sobel sin ruido
    sx, sy, smag = gradiente_sobel(imagen_sin_ruido)
    imagenes_s = [imagen_sin_ruido, sx, sy, smag]
    titulos_s = ["Original sin ruido", "Sobel X", "Sobel Y", "Sobel Magnitud"]
    mostrar_comparacion(imagenes_s, titulos_s, filas=1, cols=4, tam_fig=(16, 4))
    
    # Prewitt con ruido
    px_r, py_r, pmag_r = gradiente_prewitt(imagen_con_ruido)
    imagenes_pr = [imagen_con_ruido, px_r, py_r, pmag_r]
    titulos_pr = ["Original con ruido", "Prewitt X", "Prewitt Y", "Prewitt Magnitud"]
    mostrar_comparacion(imagenes_pr, titulos_pr, filas=1, cols=4, tam_fig=(16, 4))
    
    # Sobel con ruido
    sx_r, sy_r, smag_r = gradiente_sobel(imagen_con_ruido)
    imagenes_sr = [imagen_con_ruido, sx_r, sy_r, smag_r]
    titulos_sr = ["Original con ruido", "Sobel X", "Sobel Y", "Sobel Magnitud"]
    mostrar_comparacion(imagenes_sr, titulos_sr, filas=1, cols=4, tam_fig=(16, 4))


def ejercicio2e_laplaciano(imagen_sin_ruido, imagen_con_ruido):
    """
    e) Laplaciano 45° y 90°, y Unsharp masking
    """
    print("=== Ejercicio 2e: Laplaciano y Unsharp Masking ===")
    
    # Difuminar imágenes con filtro 5x5
    img_difuminada = filtro_promedio_estandar(imagen_sin_ruido, 5)
    img_ruido_difuminada = filtro_promedio_estandar(imagen_con_ruido, 5)
    
    # Laplaciano 90 grados
    lap90 = laplaciano_90(imagen_sin_ruido)
    lap90_norm = np.clip((lap90 - lap90.min()) * 255 / (lap90.max() - lap90.min()), 0, 255).astype(np.uint8)
    realce90 = np.clip(imagen_sin_ruido.astype(np.float64) - lap90, 0, 255).astype(np.uint8)
    
    # Laplaciano 45 grados
    lap45 = laplaciano_45(imagen_sin_ruido)
    lap45_norm = np.clip((lap45 - lap45.min()) * 255 / (lap45.max() - lap45.min()), 0, 255).astype(np.uint8)
    realce45 = np.clip(imagen_sin_ruido.astype(np.float64) - lap45, 0, 255).astype(np.uint8)
    
    imagenes_lap = [imagen_sin_ruido, lap90_norm, realce90, lap45_norm, realce45]
    titulos_lap = ["Original", "Laplaciano 90°", "Realce 90°", "Laplaciano 45°", "Realce 45°"]
    mostrar_comparacion(imagenes_lap, titulos_lap, filas=1, cols=5, tam_fig=(20, 4))
    
    # Unsharp masking - Sin ruido
    print("\nUnsharp masking - Imagen sin ruido:")
    unsharp_3_prom = unsharp_masking(img_difuminada, 'promedio', 3)
    unsharp_7_prom = unsharp_masking(img_difuminada, 'promedio', 7)
    unsharp_3_pond = unsharp_masking(img_difuminada, 'ponderado', 3)
    unsharp_7_pond = unsharp_masking(img_difuminada, 'ponderado', 7)
    
    imagenes_um = [img_difuminada, unsharp_3_prom, unsharp_7_prom, unsharp_3_pond, unsharp_7_pond]
    titulos_um = ["Difuminada 5x5", "Unsharp Prom 3x3", "Unsharp Prom 7x7", 
                  "Unsharp Pond 3x3", "Unsharp Pond 7x7"]
    mostrar_comparacion(imagenes_um, titulos_um, filas=1, cols=5, tam_fig=(20, 4))
    
    # Unsharp masking - Con ruido
    print("\nUnsharp masking - Imagen con ruido:")
    unsharp_3_prom_r = unsharp_masking(img_ruido_difuminada, 'promedio', 3)
    unsharp_7_prom_r = unsharp_masking(img_ruido_difuminada, 'promedio', 7)
    unsharp_3_pond_r = unsharp_masking(img_ruido_difuminada, 'ponderado', 3)
    unsharp_7_pond_r = unsharp_masking(img_ruido_difuminada, 'ponderado', 7)
    
    imagenes_um_r = [img_ruido_difuminada, unsharp_3_prom_r, unsharp_7_prom_r, 
                     unsharp_3_pond_r, unsharp_7_pond_r]
    titulos_um_r = ["Ruido difum 5x5", "Unsharp Prom 3x3", "Unsharp Prom 7x7", 
                    "Unsharp Pond 3x3", "Unsharp Pond 7x7"]
    mostrar_comparacion(imagenes_um_r, titulos_um_r, filas=1, cols=5, tam_fig=(20, 4))


def main():
    # Cargar imagen
    rutas = [
        "../imagenes/lena.tiff",
        "../imagenes/mamografia.tiff",
        "../imagenes/brain.tiff",
        "../imagenes/granos.png"
    ]
    
    ruta = random.choice(rutas)
    print(f"Imagen seleccionada: {ruta}\n")
    
    imagen_sin_ruido = io.imread(ruta)
    if imagen_sin_ruido.dtype != np.uint8:
        imagen_sin_ruido = ((imagen_sin_ruido - imagen_sin_ruido.min()) / 
                           (imagen_sin_ruido.max() - imagen_sin_ruido.min()) * 255).astype(np.uint8)
    
    # Generar imagen con ruido gaussiano
    imagen_con_ruido = agregar_ruido_gaussiano(imagen_sin_ruido, sigma=25)
    
    # Ejecutar ejercicios
    ejercicio2a_promedio_estandar(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2b_promedio_ponderado(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2c_mediana(imagen_sin_ruido)
    ejercicio2d_gradientes(imagen_sin_ruido, imagen_con_ruido)
    ejercicio2e_laplaciano(imagen_sin_ruido, imagen_con_ruido)
    
    print("\n✓ Ejercicio 2 completado")


if __name__ == "__main__":
    main()   
    