import numpy as np
from convolution import convolution

def calcular_gradiente(imagen_suavizada):
    # Kernels de Sobel (para las derivadas en X y Y)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    # Aplicar convolución para obtener las derivadas
    I_x = convolution(imagen_suavizada, sobel_x)
    I_y = convolution(imagen_suavizada, sobel_y)
    
    # Magnitud del gradiente: M = sqrt(I_x^2 + I_y^2)
    magnitud = np.sqrt(I_x**2 + I_y**2)
    
    # Dirección (Ángulo) del gradiente
    angulos = np.arctan2(I_y, I_x) * (180 / np.pi) 
    angulos[angulos < 0] += 180 # Normalizar ángulos a [0, 180] grados
    
    return magnitud, angulos