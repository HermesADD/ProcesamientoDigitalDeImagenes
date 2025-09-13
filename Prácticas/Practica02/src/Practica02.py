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
    img64 = io.imread("../imagenes/panda64.png", 'r')
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
    img256 = io.imread("../imagenes/Panda.PNG",'r')

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

    # Importante: una sola llamada
    plt.show()

def imagenIntensidad(imagen, nivel):
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

if __name__ == "__main__":
    valor  = input()
    if valor == "1":
        iniciaProblema1()
    elif valor == "3":
        iniciaProblema3()
    elif valor == "5":
        iniciaProblema5()
    else:
        print("Valor no encontrado")