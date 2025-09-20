"""
Módulo: EscalarImagen
Práctica 2 
Hermes Alberto Delgado Díaz
319258613

Descripción:
    Este modulo contiene funciones para realizar operaciones de escalado de imagenes. 
    En particular, se implementa un método para incrementar la escala de una imagen al doble, 
    así como una función para iniciar el problema 3 y que muestra la imagen original y 
    sus versiones escaladas sucesivamente.

Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
"""

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

import numpy as np

from skimage import io

def imagenEscalaDoble(imagen):
    """
    Escala una imagen al doble de su tamaño en ambos ejes (alto y ancho).

    Parámetros:
    -----------
    imagen : np.ndarray
        Imagen de entrada en formato de matriz (escala de grises).

    Retorna:
    --------
    np.ndarray
        Imagen escalada al doble del tamaño original.

    Detalles:
    ---------
    Para cada píxel (j, i) de la imagen resultante, se toma el píxel
    imagen[j//2, i//2] de la imagen original (replicación 2x2).
    """
    y, x = imagen.shape
    imagenDoble = np.ones((y*2,x*2), dtype=np.uint8)
    y_d , x_d = imagenDoble.shape

    print(y,x)
    print(y_d, x_d)

    for j in range(0, y_d):
        for i in range(0, x_d):
            imagenDoble[j][i] = imagen[j//2][i//2]

    return imagenDoble

def iniciaProblema(ruta_imagen: str = "../imagenes/Atleta64.png"):
    """
    Inicia el problema 3:
    - Carga una imagen.
    - Convierte a escala de grises si es necesario.
    - Genera versiones escaladas de la imagen hasta incrementar su tamaño a 1024x1024.
    - Muestra la imagen original y las escaladas en distintas figuras.

    Parámetros:
    -----------
    ruta_imagen : str
        Ruta relativa o absoluta de la imagen a cargar.
    """
    img64 = io.imread(ruta_imagen)
    escala = 64

    imgs = [img64]
    
    while imgs[-1].shape[0] < 1024 and imgs[-1].shape[1] < 1024:
        imgs.append(imagenEscalaDoble(imgs[-1]))

    for i, im in enumerate(imgs, 1):
        plt.figure(i)
        plt.imshow(im)
        plt.axis('on')
        plt.title(f"Escala {escala}")
        escala = escala*2

    plt.show()