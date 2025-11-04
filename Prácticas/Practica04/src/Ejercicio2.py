"""
Módulo: Filtros Butterworth en el Dominio de la Frecuencia
Autor: Hermes Alberto Delgado Díaz
319258613
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def ruido_gaussiano(imagen: np.ndarray, media: float = 0, std: float = 25):
    """
    Agrega ruido gaussiano a la imagen.
    
    Args:
        imagen (np.ndarray): Imagen original
        media (float): Media del ruido gaussiano
        std (float): Desviación estándar del ruido
    
    Returns:
        np.ndarray: Imagen con ruido gaussiano
    """
    ruido = np.random.normal(media, std, imagen.shape)
    imagen_ruidosa = imagen + ruido
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)


def ruido_sal_pimienta(imagen: np.ndarray, probabilidad: float = 0.05):
    """
    Agrega ruido sal y pimienta a la imagen.
    
    Args:
        imagen (np.ndarray): Imagen original
        probabilidad (float): Probabilidad total de ruido
    
    Returns:
        np.ndarray: Imagen con ruido sal y pimienta
    """
    imagen_ruidosa = imagen.copy()
    
    num_sal = int(probabilidad * imagen.size * 0.5)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    imagen_ruidosa[coords_sal[0], coords_sal[1]] = 255
    
    num_pimienta = int(probabilidad * imagen.size * 0.5)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    imagen_ruidosa[coords_pimienta[0], coords_pimienta[1]] = 0
    
    return imagen_ruidosa


def paso_bajas_butterworth(imagen: np.ndarray, D0: float = 50, n: int = 2):
    """
    Aplica el Filtro Pasa Bajas Butterworth (BLPF) para el suavizado de la imagen.
    
    Args:
        imagen (np.ndarray): Imagen de entrada.
        D0 (float): Frecuencia de corte.
        n (int): Orden del filtro.
    
    Returns:
        tuple[np.ndarray, np.ndarray]
            - np.ndarray: Imagen filtrada (suavizada).
            - np.ndarray: Matriz de la función de transferencia H(u,v).
    """
    c, r = imagen.shape
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    
    # Calcular distancia desde el centro
    D = np.sqrt((u - r/2)**2 + (v - c/2)**2)
    
    # Filtro Butterworth paso bajas
    H = 1 / (1 + (D / D0)**(2 * n))
    
    # Transformada de Fourier
    F = np.fft.fftshift(np.fft.fft2(imagen))
    F_magnitud = np.abs(F)
    F_fase = np.angle(F)
    
    # Aplicar filtro
    G_magnitud = F_magnitud * H
    
    # Reconstruir
    G = G_magnitud * np.exp(1j * F_fase)
    imagen_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    
    return imagen_filtrada.astype(np.uint8), H


def paso_altas_butterworth(imagen: np.ndarray, D0: float = 50, n: int = 2):
    """
    Aplica el Filtro Pasa Altas Butterworth (BHPF) para el realce de bordes.
    
    Args:
        imagen (np.ndarray): Imagen de entrada.
        D0 (float): Frecuencia de corte.
        n (int): Orden del filtro.
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - np.ndarray: Imagen filtrada (realzada).
            - np.ndarray: Matriz de la función de transferencia H(u,v).
    """
    c, r = imagen.shape
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    
    # Calcular distancia desde el centro
    D = np.sqrt((u - r/2)**2 + (v - c/2)**2)
    
    # Evitar división por cero en el centro
    D[D == 0] = 1e-10
    
    # Filtro Butterworth paso altas
    H = 1 / (1 + (D0 / D)**(2 * n))
    
    # Transformada de Fourier
    F = np.fft.fftshift(np.fft.fft2(imagen))
    F_magnitud = np.abs(F)
    F_fase = np.angle(F)
    
    # Aplicar filtro
    G_magnitud = F_magnitud * H
    
    # Reconstruir
    G = G_magnitud * np.exp(1j * F_fase)
    imagen_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    
    return imagen_filtrada.astype(np.uint8), H


def ejercicio2a_paso_bajas(imagen_sin_ruido, imagen_con_ruido, tipo):
    """
    Ejercicio 2a: Prueba filtros paso bajas Butterworth con diferentes
    frecuencias de corte D0 y órdenes n.
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original limpia
        imagen_con_ruido (np.ndarray): Imagen con ruido
    """
    print("\n" + "=" * 70)
    print("EJERCICIO 2a: FILTRO PASO BAJAS BUTTERWORTH")
    print("=" * 70)
    
    valores_D0 = [10, 50, 100]
    valores_n = [2, 4, 8]
    
    print("\nPrueba 1: Variando D0 (frecuencia de corte) con n=2")
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 12))
    fig1.suptitle('Filtro Paso Bajas Butterworth - Variando D0 (n=2) Sin Ruido', 
                  fontsize=12, fontweight='bold')
    
    for idx, D0 in enumerate(valores_D0):
        img_filtrada_sin, filtro_H = paso_bajas_butterworth(imagen_sin_ruido, D0=D0, n=2)
        
        axes1[0, idx].imshow(filtro_H, cmap='gray')
        axes1[0, idx].set_title(f'Filtro H(u,v)\nD0={D0}, n=2')
        axes1[0, idx].axis('off')
        
        axes1[1, idx].imshow(img_filtrada_sin, cmap='gray')
        axes1[1, idx].set_title(f'Sin ruido\nD0={D0}')
        axes1[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 12))
    fig1.suptitle(f'Filtro Paso Bajas Butterworth - Variando D0 (n=2) Con Ruido {tipo}', 
                  fontsize=12, fontweight='bold')
    
    for idx, D0 in enumerate(valores_D0):
        img_filtrada_con, filtro_H = paso_bajas_butterworth(imagen_con_ruido, D0=D0, n=2)
        axes1[0, idx].imshow(filtro_H, cmap='gray')
        axes1[0, idx].set_title(f'Filtro H(u,v)\nD0={D0}, n=2')
        axes1[0, idx].axis('off')
        
        axes1[1, idx].imshow(img_filtrada_con, cmap='gray')
        axes1[1, idx].set_title(f'Con ruido\nD0={D0}')
        axes1[1, idx].axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\nPrueba 2: Variando n (orden) con D0=50")
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 12))
    fig2.suptitle('Filtro Paso Bajas Butterworth - Variando n (D0=50) Sin Ruido', 
                  fontsize=12, fontweight='bold')
    
    for idx, n in enumerate(valores_n):
        img_filtrada_sin, filtro_H = paso_bajas_butterworth(imagen_sin_ruido, D0=50, n=n)
        
        axes2[0, idx].imshow(filtro_H, cmap='gray')
        axes2[0, idx].set_title(f'Filtro H(u,v)\nD0=50, n={n}')
        axes2[0, idx].axis('off')
        
        axes2[1, idx].imshow(img_filtrada_sin, cmap='gray')
        axes2[1, idx].set_title(f'Sin ruido\nn={n}')
        axes2[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 12))
    fig2.suptitle(f'Filtro Paso Bajas Butterworth - Variando n (D0=50) Con Ruido {tipo}', 
                  fontsize=12, fontweight='bold')
    
    for idx, n in enumerate(valores_n):
        img_filtrada_con, filtro_H = paso_bajas_butterworth(imagen_con_ruido, D0=50, n=n)
        axes2[0, idx].imshow(filtro_H, cmap='gray')
        axes2[0, idx].set_title(f'Filtro H(u,v)\nD0=50, n={n}')
        axes2[0, idx].axis('off')
        
        axes2[1, idx].imshow(img_filtrada_con, cmap='gray')
        axes2[1, idx].set_title(f'Con ruido\nn={n}')
        axes2[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def ejercicio2b_paso_altas(imagen_sin_ruido, imagen_con_ruido, tipo):
    """
    Ejercicio 2b: Prueba filtros paso altas Butterworth para realce de bordes.
    
    Args:
        imagen_sin_ruido (np.ndarray): Imagen original limpia
        imagen_con_ruido (np.ndarray): Imagen con ruido
    """
    print("\n" + "=" * 70)
    print("EJERCICIO 2b: FILTRO PASO ALTAS BUTTERWORTH")
    print("=" * 70)
    
    valores_D0 = [10, 50, 100]
    valores_n = [2, 4, 8]
    
    print("\nPrueba 1: Variando D0 (frecuencia de corte) con n=2")
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 12))
    fig1.suptitle('Filtro Paso Altas Butterworth - Variando D0 (n=2) Sin Ruido', 
                  fontsize=12, fontweight='bold')
    
    for idx, D0 in enumerate(valores_D0):
        img_filtrada_sin, filtro_H = paso_altas_butterworth(imagen_sin_ruido, D0=D0, n=2)
        
        axes1[0, idx].imshow(filtro_H, cmap='gray')
        axes1[0, idx].set_title(f'Filtro H(u,v)\nD0={D0}, n=2')
        axes1[0, idx].axis('off')
        
        axes1[1, idx].imshow(img_filtrada_sin, cmap='gray')
        axes1[1, idx].set_title(f'Sin ruido\nD0={D0}')
        axes1[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 12))
    fig1.suptitle(f'Filtro Paso Altas Butterworth - Variando D0 (n=2) Con Ruido {tipo}', 
                  fontsize=12, fontweight='bold')
    
    for idx, D0 in enumerate(valores_D0):
        img_filtrada_con, filtro_H = paso_altas_butterworth(imagen_con_ruido, D0=D0, n=2)
        
        axes1[0, idx].imshow(filtro_H, cmap='gray')
        axes1[0, idx].set_title(f'Filtro H(u,v)\nD0={D0}, n=2')
        axes1[0, idx].axis('off')
        
        axes1[1, idx].imshow(img_filtrada_con, cmap='gray')
        axes1[1, idx].set_title(f'Con ruido\nD0={D0}')
        axes1[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPrueba 2: Variando n (orden) con D0=30")
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 12))
    fig2.suptitle('Filtro Paso Altas Butterworth - Variando n (D0=30) Sin Ruido', 
                  fontsize=12, fontweight='bold')
    
    for idx, n in enumerate(valores_n):
        img_filtrada_sin, filtro_H = paso_altas_butterworth(imagen_sin_ruido, D0=30, n=n)
        
        axes2[0, idx].imshow(filtro_H, cmap='gray')
        axes2[0, idx].set_title(f'Filtro H(u,v)\nD0=30, n={n}')
        axes2[0, idx].axis('off')
        
        axes2[1, idx].imshow(img_filtrada_sin, cmap='gray')
        axes2[1, idx].set_title(f'Sin ruido\nn={n}')
        axes2[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 12))
    fig2.suptitle(f'Filtro Paso Altas Butterworth - Variando n (D0=30) Con Ruido {tipo}', 
                  fontsize=12, fontweight='bold')
    
    for idx, n in enumerate(valores_n):
        img_filtrada_con, filtro_H = paso_altas_butterworth(imagen_con_ruido, D0=30, n=n)
        
        axes2[0, idx].imshow(filtro_H, cmap='gray')
        axes2[0, idx].set_title(f'Filtro H(u,v)\nD0=30, n={n}')
        axes2[0, idx].axis('off')
        
        axes2[1, idx].imshow(img_filtrada_con, cmap='gray')
        axes2[1, idx].set_title(f'Sin ruido\nn={n}')
        axes2[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def iniciaEjercicio(ruta_imagen: str = "../imagenes/saturn_bw.tif"):
    """
    Función principal que ejecuta el Ejercicio 2 completo.
    """
    print("=" * 70)
    print("PRÁCTICA 4 - EJERCICIO 2")
    print("FILTRADO EN EL DOMINIO DE LA FRECUENCIA")
    print("=" * 70)
  
    imagen = io.imread(ruta_imagen)
   
    
    if imagen.dtype != np.uint8:
        imagen = ((imagen - imagen.min()) / (imagen.max() - imagen.min()) * 255).astype(np.uint8)
    
    print(f"\nImagen cargada: {imagen.shape}")
    print(f"Tipo: {imagen.dtype}")
    print(f"Rango: [{imagen.min()}, {imagen.max()}]")
    
    print("\nGenerando imagen con ruido sal y pimienta...")
    imagenSP = ruido_sal_pimienta(imagen, probabilidad=0.05)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Imágenes de Entrada', fontsize=14, fontweight='bold')
    
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Imagen sin ruido')
    axes[0].axis('off')
    
    axes[1].imshow(imagenSP, cmap='gray')
    axes[1].set_title('Imagen con ruido sal y pimienta')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    ejercicio2a_paso_bajas(imagen, imagenSP, "Sal y Pimienta")
    
    ejercicio2b_paso_altas(imagen, imagenSP, "Sal y Pimienta")

    #print("\nGenerando imagen con ruido gaussiano...")
    #imagenG = ruido_gaussiano(imagen)

    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #fig.suptitle('Imágenes de Entrada', fontsize=14, fontweight='bold')
    
    #axes[0].imshow(imagen, cmap='gray')
    #axes[0].set_title('Imagen sin ruido')
    #axes[0].axis('off')
    
    #axes[1].imshow(imagenG, cmap='gray')
    #axes[1].set_title('Imagen con ruido gaussiano')
    #axes[1].axis('off')
    
    #plt.tight_layout()
    #plt.show()

    #ejercicio2a_paso_bajas(imagen, imagenG, "Gaussiano")
    
    #ejercicio2b_paso_altas(imagen, imagenG, "Gaussiano")