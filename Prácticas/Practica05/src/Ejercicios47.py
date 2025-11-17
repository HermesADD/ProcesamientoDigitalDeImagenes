"""
Práctica 5: Restauración de Imágenes
Ejercicios 4-7: Filtro de Wiener

Implementación de los 4 casos del filtro de Wiener:
- Caso 1 (Ej. 4): Ruido aditivo gaussiano
- Caso 2 (Ej. 5): Pérdida de nitidez (blur)
- Caso 3 (Ej. 6): Ruido + Blur (en ese orden)
- Caso 4 (Ej. 7): Blur + Ruido (en ese orden)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


