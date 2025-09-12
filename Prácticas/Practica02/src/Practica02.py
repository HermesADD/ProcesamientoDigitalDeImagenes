# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt

#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#tambien importamos numpy ya que lo usamos para crear y manipular matrices
import numpy as np

from skimage import io

def iniciaProblema1():
    img1024 = io.imread("../imagenes/Panda.PNG",'r')

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
    
    # Importante: una sola llamada
    plt.show()


def imagenEscalaMitad(imagen):
    y, x = imagen.shape

    imagenEscalada = np.ones((y//2,x//2), dtype= np.uint8)
    print(y,x)
    print(imagenEscalada.shape)
    for j in range (0,y,2):
        for i in range(0,x,):
            imagenEscalada[j//2][i//2]= imagen[j][i]
    return imagenEscalada

def iniciaProblema3():
    img64 = io.imread("../imagenes/mandala64.jpg", 'r')
    escala = 64

    imgs = [img64]
    for _ in range(4):
        imgs.append(imagenEscalaDoble(imgs[-1]))

    for i, im in enumerate(imgs, 1):
        plt.figure(i)
        plt.imshow(im)
        plt.axis('on')
        plt.title(f"Escala {escala}")
        escala = escala*2

    # Importante: una sola llamada
    plt.show()

def imagenEscalaDoble(imagen):
    y, x = imagen.shape
    imagenDoble = np.ones((y*2,x*2), dtype=np.uint8)
    y_d , x_d = imagenDoble.shape

    print(y,x)
    print(y_d, x_d)

    for j in range(0, y_d):
        for i in range(0, x_d):
            imagenDoble[j][i] = imagen[j//2][i//2]

    return imagenDoble

def iniciaProblema5():


if __name__ == "__main__":
    valor = input(int)
    if valor ==1 :
        iniciaProblema1()
    else:
        iniciaProblema3()