"""
Módulo: EscalarImagen
Práctica 2 
Hermes Alberto Delgado Díaz
319258613

Descripción:
    Este modulo contiene funciones para realizar operaciones de escalado de imagenes. 
    En particular, se implementa un método para reducir la escala de una imagen a la mitad, 
    así como una función para iniciar el problema 1 y 2 que muestra la imagen original y 
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

def imagenEscalaMitad(imagen):
    """
    Escala una imagen a la mitad de su tamaño en ambos ejes (alto y ancho).

    Parámetros:
    -----------
    imagen : np.ndarray
        Imagen de entrada en formato de matriz (escala de grises).

    Retorna:
    --------
    np.ndarray
        Imagen escalada a la mitad del tamaño original.

    Detalles:
    ---------
    La función toma un pixel representativo de cada bloque de 2x2,
    reduciendo así el tamaño de la imagen en un factor de 2.
    """
    y, x = imagen.shape

    imagenEscalada = np.ones((y//2,x//2), dtype= np.uint8)
    print(y,x)
    print(imagenEscalada.shape)
    for j in range (0,y,2):
        for i in range(0,x,2):
            imagenEscalada[j//2][i//2]= imagen[j][i]
    return imagenEscalada

def iniciaProblema(ruta_imagen: str = "../imagenes/Atleta.png"):
    """
    Inicia el problema 1 y 2:
    - Carga una imagen.
    - Convierte a escala de grises si es necesario.
    - Genera versiones escaladas de la imagen hasta reducir su tamaño 4 veces.
    - Muestra la imagen original y las escaladas en distintas figuras.

    Parámetros:
    -----------
    ruta_imagen : str
        Ruta relativa o absoluta de la imagen a cargar.
    """
    img = io.imread(ruta_imagen)

    if img.ndim == 3:
        img1024 = img[:, :, 0]  
    else:  
        img1024 = img  
        
    escala = 1024
    imgs = [img1024]
    for _ in range(4):
        imgs.append(imagenEscalaMitad(imgs[-1]))

    for i, im in enumerate(imgs, 1):
        plt.figure(i)
        plt.imshow(im)
        plt.axis('on')
        plt.title(f"Escala {escala}")
        escala = escala//2

    plt.show()