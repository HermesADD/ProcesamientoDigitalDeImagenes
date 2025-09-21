"""
Módulo: adyacencias
Autor: Hermes Alberto Delgado Díaz
Descripción:
    Utilidades para seleccionar píxeles al azar en una imagen, calcular sus
    vecinos 4 y 8-adyacentes, medir distancias City-Block y Chessboard entre
    puntos, y colorear en una copia de la imagen los vecinos usando PIL.

Dependencias:
    - numpy
    - matplotlib.pyplot
    - skimage.io
    - PIL (Pillow)
"""

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

import numpy as np

from skimage import io

from PIL import Image

def vecinos4(y,x,H,W):
    """
    Devuelve los vecinos 4-adyacentes (arriba, abajo, izquierda, derecha)
    del píxel (y, x) acotados al rectángulo [0..H-1] x [0..W-1].

    Parámetros
    ----------
    y, x : int
        Coordenadas del píxel central (fila, columna).
    H, W : int
        Alto (número de filas) y ancho (número de columnas) de la imagen.

    Retorna
    -------
    List[Coord]
        Lista de coordenadas válidas (y, x) adyacentes ortogonales.
    """
    posiblesVecinos = [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
    return [(a,b) for (a,b) in posiblesVecinos if 0 <= a < H and 0 <= b < W]

def vecinos8(y,x,H,W):
    """
    Devuelve los vecinos 8-adyacentes (incluye diagonales) de (y, x)
    acotados al rectángulo [0..H-1] x [0..W-1].

    Parámetros
    ----------
    y, x : int
        Coordenadas del píxel central (fila, columna).
    H, W : int
        Alto (número de filas) y ancho (número de columnas) de la imagen.

    Retorna
    -------
    List[Coord]
        Lista de coordenadas válidas (y, x) adyacentes (ortogonales y diagonales).
    """
    posiblesVecinos = [(y+1,x+1),(y+1,x-1),(y-1,x+1),(y-1,x-1),(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
    return [(a,b) for (a,b) in posiblesVecinos if 0 <= a < H and 0 <= b < W]

def imagenAdyacencia(imagen,n_pixeles):
    """
    Elige aleatoriamente hasta n_pixeles de la imagen (sin reemplazo) y
    regresa sus coordenadas (y, x). También imprime sus vecinos 4 y 8-adyacentes.

    Parámetros
    ----------
    imagen : np.ndarray
        Imagen 2D (escala de grises) con shape (H, W).
    n_pixeles : int
        Número de píxeles a seleccionar (se ajusta a H*W si excede).

    Retorna
    -------
    List[Coord]
        Lista de coordenadas seleccionadas [(y, x), ...].
    """
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
    """
    Restringe el valor v al rango [lo, hi].

    Parámetros
    ----------
    v : int
        Valor a recortar.
    lo, hi : int
        Límite inferior y superior.

    Retorna
    -------
    int
        Valor recortado.
    """
    return max(lo, min(hi, v))

def pinta_bloque(img, x, y, color, r=0):
    """
    Pinta un bloque centrado en (x, y) de tamaño (2r+1) x (2r+1) con un color RGB.

    Parámetros
    ----------
    img : PIL.Image.Image
        Imagen en modo RGB sobre la que se pintará.
    x, y : int
        Coordenadas (columna, fila) del centro del bloque en coordenadas de PIL.
    color : (int, int, int)
        Color RGB (0..255).
    r : int
        Radio del bloque (0 = 1x1, 1 = 3x3, 2 = 5x5, ...).
    """
    W, H = img.size
    px = img.load()
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            xx = _clip(x+dx, 0, W-1)
            yy = _clip(y+dy, 0, H-1)
            px[xx, yy] = color

def marcarVecinosColores(ruta, coordenadas, r=0, salida: str = "../imagenes/cuadroPixeles.png"):
    """
    Colorea vecinos en una copia de la imagen:

      - Vecinos 4 (ortogonales) en ROJO
      - Diagonales (8-adyacentes que no son 4-adyacentes) en AZUL
      - Píxel central en AMARILLO

    El resultado se guarda en disco.

    Parámetros
    ----------
    ruta : str
        Ruta del archivo de imagen original (se abrirá en RGB).
    coordenadas : List[Coord]
        Lista de píxeles (y, x) a procesar.
    r : int
        “Grosor” del pintado (0=1px, 1=3x3, 2=5x5, ...).
    salida : str
        Ruta del archivo de salida donde se guarda la imagen marcada.
    """
    img = Image.open(ruta).convert("RGB")
    W, H = img.size 

    rojo = (255, 0, 0)
    azul = (0, 102, 255)
    amarillo = (255, 255, 0)

    for (y,x) in coordenadas:
        v4 = vecinos4(y, x, H, W)
        v8 = vecinos8(y, x, H, W)

        diag = [p for p in v8 if p not in v4]

        for (yy, xx) in diag:
            pinta_bloque(img, xx, yy, azul, r=r)
        for (yy, xx) in v4:
            pinta_bloque(img, xx, yy, rojo, r=r)

        pinta_bloque(img, x, y, amarillo, r=r)

    img.save(salida)

def distanciaCityBlock(p, q):
    """
    Distancia City-Block entre dos puntos (y, x).

    Parámetros
    ----------
    p, q : Coord
        Puntos (y, x).

    Retorna
    -------
    int
        |x1 - x2| + |y1 - y2|
    """
    (y1, x1), (y2, x2) = p, q
    return abs(x1 - x2) + abs(y1 - y2)

def distanciaChessboard(p, q):
    """
    Distancia Chessboard entre dos puntos (y, x).

    Parámetros
    ----------
    p, q : Coord
        Puntos (y, x).

    Retorna
    -------
    int
        max(|x1 - x2|, |y1 - y2|)
    """
    (y1, x1), (y2, x2) = p, q
    return max(abs(x1 - x2), abs(y1 - y2))

def distanciaEntrePuntos(coordenadas):
    """
    Imprime y retorna las distancias City-Block y Chessboard desde el primer
    punto de la lista hacia cada uno de los demás.

    Parámetros
    ----------
    coordenadas : List[Coord]
        Lista de coordenadas [(y, x), ...]. Debe contener al menos 2 puntos.
    """
    if len(coordenadas) < 2:
        print("\n Se necesitan al menos dos pixeles.")
        return []

    ref = coordenadas[0]
    print(f"\nPíxel de elegido como referencia: {ref}")

    for punto in coordenadas[1:]:
        d_cb = distanciaCityBlock(ref, punto)
        d_ch = distanciaChessboard(ref, punto)
        print(f"  -> hacia {punto}: CityBlock={d_cb}, Chessboard={d_ch}")

def iniciaEjercicio(ruta_imagen: str = "../imagenes/cuadro.png", ruta_imgPintada: str = "../imagenes/cuadroPixeles.png"):
    """
    Flujo completo del ejercicio 7:
      1) Carga '../imagenes/cuadro.png' y la pasa a grises si hace falta.
      2) Selecciona 5 píxeles al azar y muestra sus adyacencias.
      3) Calcula distancias desde el primer punto a los demás.
      4) Colorea vecinos y centro en una copia y guarda '../imagenes/cuadroPixeles.png'.
      5) Muestra la imagen original y la marcada con matplotlib.
    """
    ruta = ruta_imagen
    img = io.imread(ruta)
    if img.ndim == 3:
        imgG = img[:, :, 0]  
    else: 
        imgG = img
    
    coordenadas = imagenAdyacencia(imgG, 5)
    distanciaEntrePuntos(coordenadas)
    
    marcarVecinosColores(ruta, coordenadas)

    imgPintada =io.imread(ruta_imgPintada)
    
    plt.figure(1); plt.imshow(img); plt.axis('on'); plt.title("Original")
    plt.figure(2); plt.imshow(imgPintada); plt.axis('on'); plt.title("Marcada 4 y 8 adyacencias")
    plt.show()