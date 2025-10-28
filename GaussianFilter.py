import numpy as np
from convolution import convolution

def rgb2gray(image):
    # Conversión a Escala de Grises 
    gray_image = (0.299 * image[:, :, 0] + 
                  0.587 * image[:, :, 1] + 
                  0.114 * image[:, :, 2])
    
    return gray_image

def gaussian_filter_image(image, sigma):
    
    # Conversión a Escala de Grises (Luma)
    gray_image = rgb2gray(image)
    
    # --- Generación del Kernel Gaussiano ---
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    
    for x in range(-m, m + 1):
        for y in range(-m, m + 1):
            x1 = 2 * np.pi * (sigma**2)
            x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_filter[x + m, y + m] = (1 / x1) * x2
    
    # Normalización del Kernel
    gaussian_filter /= gaussian_filter.sum()
    
    # Aplicar Convolución
    im_filtered = convolution(gray_image, gaussian_filter)
    
    return im_filtered
