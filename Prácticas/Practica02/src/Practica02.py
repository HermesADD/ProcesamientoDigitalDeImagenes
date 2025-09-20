# importamos el modulo pyplot, y lo llamamos plt
import matplotlib.pyplot as plt

#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

#tambien importamos numpy ya que lo usamos para crear y manipular matrices
import numpy as np

from skimage import io

from PIL import Image

def iniciaProblema1():
    img = io.imread("../imagenes/Atleta.png")

    if img.ndim == 3:
        img1024 = img[:, :, 0]  
    else:  # Si ya es en escala de grises
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

    # Importante: una sola llamada
    plt.show()


def imagenEscalaMitad(imagen):
    y, x = imagen.shape

    imagenEscalada = np.ones((y//2,x//2), dtype= np.uint8)
    print(y,x)
    print(imagenEscalada.shape)
    for j in range (0,y,2):
        for i in range(0,x,2):
            imagenEscalada[j//2][i//2]= imagen[j][i]
    return imagenEscalada

def iniciaProblema3():
    img64 = io.imread("../imagenes/Atleta64.png")
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
    img = io.imread("../imagenes/Atleta.png")
    if img.ndim == 3:
        img256 = img[:, :, 0]  
    else:  # Si ya es en escala de grises
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

def vecinos4(y,x,H,W):
    posiblesVecinos = [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
    return [(a,b) for (a,b) in posiblesVecinos if 0 <= a < H and 0 <= b < W]
    #print(f"Los pixeles 4-adyacentes de {y,x} son: {vecinos}")

def vecinos8(y,x,H,W):
    posiblesVecinos = [(y+1,x+1),(y+1,x-1),(y-1,x+1),(y-1,x-1),(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
    return [(a,b) for (a,b) in posiblesVecinos if 0 <= a < H and 0 <= b < W]
    #print(f"Los pixeles 8-adyacentes de {y,x} son: {vecinos}")

def imagenAdyacencia(imagen,n_pixeles):
    H,W = imagen.shape
    print(H,W)

    generadorRandom = np.random.default_rng()
    
    totalPixeles = H * W

    n = min(n_pixeles, totalPixeles)

    indices = generadorRandom.choice(totalPixeles, size=n, replace=False)

    coordenadas = [(int(idx // W),int(idx % W)) for idx in indices]

    for (y,x) in coordenadas:
        print(f"Pixel seleccionado: ({y},{x})")
        v4 = vecinos4(y,x,H,W)
        v8 = vecinos8(y,x,H,W)
        print(f"4-adyacencia: {v4}")
        print(f"8-adyacencia: {v8}")
    return coordenadas

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def _pinta_bloque(img, x, y, color, r=0):
    """Pinta un bloque centrado en (x,y) de tamaño (2r+1)x(2r+1)."""
    W, H = img.size
    px = img.load()
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            xx = _clip(x+dx, 0, W-1)
            yy = _clip(y+dy, 0, H-1)
            px[xx, yy] = color

def marcarVecinosColores(ruta, coordenadas,
                         pintar_centro=True, r=0):
    """
    Colorea para cada coordenada (y,x):
      - Vecinos 4 (ortogonales) en ROJO
      - Diagonales (8-ady - 4-ady) en AZUL
      - (opcional) el píxel central en AMARILLO
    r controla el “grosor”: 0=1px, 1=3x3, 2=5x5.
    """
    # Abrimos en RGB para poder colorear
    img = Image.open(ruta).convert("RGB")
    W, H = img.size
    Hn, Wn = H, W  # para nuestras funciones (H=filas, W=cols)

    rojo = (255, 0, 0)
    azul = (0, 102, 255)
    amarillo = (255, 255, 0)

    for (y,x) in coordenadas:
        # calcular vecinos con límites
        v4 = vecinos4(y, x, Hn, Wn)
        v8 = vecinos8(y, x, Hn, Wn)
        # diagonales = 8-adyacentes que NO estén en 4-adyacentes
        diag = [p for p in v8 if p not in v4]

        # Primero azules (diagonales) y luego rojos (4-ady) para que el rojo prevalezca si se superponen
        for (yy, xx) in diag:
            _pinta_bloque(img, xx, yy, azul, r=r)
        for (yy, xx) in v4:
            _pinta_bloque(img, xx, yy, rojo, r=r)

        if pintar_centro:
            _pinta_bloque(img, x, y, amarillo, r=r)

    img.save("../imagenes/cuadroPixeles.png")

def distanciaCityBlock(p, q):
    (y1, x1), (y2, x2) = p, q
    return abs(x1 - x2) + abs(y1 - y2)

def distanciaChessbord(p, q):
    (y1, x1), (y2, x2) = p, q
    return max(abs(x1 - x2), abs(y1 - y2))

def distanciaEntrePuntos(coordenadas):
    if len(coordenadas) < 2:
        print("\n Se necesitan al menos dos pixeles.")
        return []

    ref = coordenadas[0]
    print(f"\nPíxel de elegido como referencia: {ref}")

    resultados = []
    for punto in coordenadas[1:]:
        d_cb = distanciaCityBlock(ref, punto)
        d_ch = distanciaChessbord(ref, punto)
        print(f"  -> hacia {punto}: CityBlock={d_cb}, Chessboard={d_ch}")

def iniciaEjercicio7():
    ruta = "../imagenes/cuadro.png"
    img = io.imread(ruta)
    if img.ndim == 3:
        imgG = img[:, :, 0]  
    else:  # Si ya es en escala de grises
        imgG = img
    
    coordenadas = imagenAdyacencia(imgG, 5)
    distanciaEntrePuntos(coordenadas)
    
    marcarVecinosColores(ruta, coordenadas)

    imgPintada =io.imread("../imagenes/cuadroPixeles.png")
    
    plt.figure(1); plt.imshow(img); plt.axis('on'); plt.title("Original")
    plt.figure(2); plt.imshow(imgPintada); plt.axis('on'); plt.title("Marcada 4 y 8 adyacencias")
    plt.show()

if __name__ == "__main__":
    valor  = input()
    if valor == "1":
        iniciaProblema1()
    elif valor == "3":
        iniciaProblema3()
    elif valor == "5":
        iniciaProblema5()
    elif valor == "7":
        iniciaEjercicio7()
    else:
        print("Valor no encontrado")