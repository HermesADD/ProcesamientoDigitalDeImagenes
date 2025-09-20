"""
Módulo: EscalarImagen
Práctica 2 
Hermes Alberto Delgado Díaz
319258613

Descripción:
    Este módulo implementa funciones para modificar los niveles de intensidad
    (escala de grises) de una imagen. El proceso consiste en aplicar una
    cuantización, reduciendo los niveles de gris mediante la operación:

        nuevo_pixel = pixel - (pixel % nivel)

    De esta manera, se eliminan detalles finos y se conserva únicamente un
    subconjunto de intensidades. Además cuenta con una función para iniciar el problema 4 y que muestra
    la imagen original y sus versiones con niveles de intensidad escalados

    Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
"""

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

import numpy as np

from skimage import io

def imagenIntensidad(imagen, nivel):
    """
    Aplica cuantización de niveles de intensidad a una imagen en escala de grises.

    Parámetros
    ----------
    imagen : np.ndarray
        Imagen de entrada en formato 2D (escala de grises).
    nivel : int
        Factor de cuantización. Mientras mayor sea el valor,
        menos niveles de gris conserva la imagen.

    Retorna
    -------
    np.ndarray
        Imagen con niveles de intensidad reducidos.
    """

    y, x = imagen.shape
    imagenNueva = np.ones((y,x), dtype=np.uint8)

    print(y,x)
    print(nivel)
    for j in range(0, y):
        for i in range(0, x):
            px = imagen[j][i]
            px = px - (px % nivel)
            
            imagenNueva[j][i] = px
    return imagenNueva

def iniciaProblema(ruta_imagen: str="../imagenes/Atleta.png"):
    """
    Genera versiones de una imagen con distintos niveles de intensidad.

    - Convierte la imagen a escala de grises si es necesario.
    - Aplica cuantización en 4 pasos, duplicando el nivel cada vez (2, 4, 8, 16).
    - Muestra la imagen original y las versiones cuantizadas.

    Parámetros
    ----------
    ruta_imagen : str
        Ruta de la imagen a procesar.
    """

    img = io.imread(ruta_imagen)
    if img.ndim == 3:
        img256 = img[:, :, 0]  
    else:  
        img256 = img
          
    nivel = 2
    imgs = [img256]
    for _ in range(4):
        imgs.append(imagenIntensidad(imgs[-1], nivel))
        nivel  = nivel * 2

    escala = 256
    for i, im in enumerate(imgs, 1):
        plt.figure(i)
        plt.imshow(im)
        plt.axis('on')
        plt.title(f"Escala de grises {escala}")
        escala = escala//2

    plt.show()
