"""
Módulo: Transformada de Fourier
Autor: Hermes Alberto Delgado Díaz
319258613
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def fourier(imagen : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        Calcula la Transformada de Fourier y obtiene el espectro de magnitud y fase sin corrimiento.

        Args:
            imagen(np.ndarray): Imagen de entrada

        Returns:
            tuple[np.ndarray, np.ndarray]
            - Espectro de magnitud
            - Espectro de fase.
    """

    tf1 = np.fft.fft2(imagen)

    tf1_magnitud = abs(tf1)
    tf1_fase = np.angle(tf1)

    esp_mag = np.log(1 + tf1_magnitud)

    return esp_mag, tf1_fase

def fourier_corrimiento(imagen : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        Calcula la Transformada de Fourier y obtiene el espectro de magnitud y fase con corrimiento.

        Args:
            imagen(np.ndarray): Imagen de entrada.
        
        Returns:
            tuple[np.ndarray, np.ndarray]
            - Espectro de magnitud centrado.
            - Espectro de fase centrado.
    """

    tf1 = np.fft.fft2(imagen)

    tfc1 = np.fft.fftshift(tf1)

    tfc1_magnitud = abs(tfc1)
    tfc1_fase = np.angle(tfc1)

    esp_mag = np.log(1 + tfc1_magnitud)

    return esp_mag, tfc1_fase

def fourier_inv_comp(imagen: np.ndarray) -> np.ndarray:
    """
        Calcula la Transformada Inversa de Fourier utilizando 
        la magnitud y fase originales, reconstruyendo la imagen completa.

        Args:
            imagen(np.ndarray): Imagen de entrada

        Returns:
            np.ndarray: Imagen reconstruida completa.
    """

    tf1 = np.fft.fft2(imagen)

    tf1_magnitud = abs(tf1)
    tf1_fase = np.angle(tf1)

    itf1_completa = np.fft.ifft2(tf1_magnitud * np.exp(1j * tf1_fase))

    itf1_completa_magnitud = abs(itf1_completa)

    return itf1_completa_magnitud

def fourier_inv_amp(imagen: np.ndarray) -> np.ndarray:
    """
        Reconstruye la imagen utilizando solo la magnitud, forzando la fase a 0. 

        Args:
            imagen(np.ndarray): Imagen de entrada

        Returns:
            np.ndarray: Imagen reconstruida solo con magnitud
    """
    tf1 = np.fft.fft2(imagen)
    tf1_magnitud = abs(tf1)
    itf1_amplitud = np.fft.ifft2(tf1_magnitud)
    itf1_amplitud_magnitud = abs(itf1_amplitud)

    return np.uint8(itf1_amplitud_magnitud)

def fourier_inv_fase(imagen: np.ndarray) -> np.ndarray:
    """
        Reconstruye la imagen utilizando solo la fase del espectro, 
        forzando la amplitud a uno. 

        Args:
            imagen(np.ndarray): Imagen de entrada

        Returns:
            np.ndarray: Imagen reconstruida solo con fase.
    """
    tf1 = np.fft.fft2(imagen)
    tfc1 = np.fft.fftshift(tf1)
    tfc1_fase = np.angle(tfc1)
    itf1_fase = np.fft.ifft2(np.exp(1j * tfc1_fase))
    itf1_fase_magnitud = abs(itf1_fase)

    return itf1_fase_magnitud

def iniciaEjercicio(ruta_imagen = "../imagenes/fourier.bmp"):
    """
    Función principal que ejecuta el Ejercicio 1, cargando la imagen
    y mostrando los resultados en dos ventanas.
    """
    imagen = io.imread(ruta_imagen)

    print("\n" + "=" * 70)
    print("EJERCICIO 1: TRANSFORMADA DE FOURIER")
    print("=" * 70)
    esp_mag_sin, esp_fase_sin = fourier(imagen)

    esp_mag_con, esp_fase_con = fourier_corrimiento(imagen)

    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Análisis de Fourier - Espectros', fontsize=16, fontweight='bold')

    axes1[0, 0].imshow(imagen, cmap='gray')
    axes1[0, 0].set_title('Imagen Original', fontsize=12)
    axes1[0, 0].axis('off')

    axes1[0, 1].imshow(esp_mag_sin, cmap='gray')
    axes1[0, 1].set_title('Espectro Amplitud\n(sin corrimiento)', fontsize=12)
    axes1[0, 1].axis('off')

    axes1[0, 2].imshow(esp_fase_sin, cmap='gray')
    axes1[0, 2].set_title('Espectro Fase\n(sin corrimiento)', fontsize=12)
    axes1[0, 2].axis('off')

    axes1[1, 0].axis('off')

    axes1[1, 1].imshow(esp_mag_con, cmap='gray')
    axes1[1, 1].set_title('Espectro Amplitud\n(con corrimiento)', fontsize=12)
    axes1[1, 1].axis('off')

    axes1[1, 2].imshow(esp_fase_con, cmap='gray')
    axes1[1, 2].set_title('Espectro Fase\n(con corrimiento)', fontsize=12)
    axes1[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    inv_completa = fourier_inv_comp(imagen)
    inv_amplitud = fourier_inv_amp(imagen)
    inv_fase = fourier_inv_fase(imagen)

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
    fig2.suptitle('Transformadas Inversas de Fourier', fontsize=16, fontweight='bold')

    axes2[0].imshow(imagen, cmap='gray')
    axes2[0].set_title('Imagen Original', fontsize=12)
    axes2[0].axis('off')

    axes2[1].imshow(inv_completa, cmap='gray')
    axes2[1].set_title('Transformada Completa\n(Amplitud + Fase)', fontsize=12)
    axes2[1].axis('off')

    axes2[2].imshow(inv_amplitud, cmap='gray')
    axes2[2].set_title('Solo Amplitud', fontsize=12)
    axes2[2].axis('off')

    axes2[3].imshow(inv_fase, cmap='gray')
    axes2[3].set_title('Solo Fase', fontsize=12)
    axes2[3].axis('off')

    plt.tight_layout()
    plt.show()