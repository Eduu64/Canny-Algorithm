import numpy as np


def convolution(oldimage, kernel):
    image_h = oldimage.shape[0]
    image_w = oldimage.shape[1]
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    # h y w son la cantidad de píxeles a acolchar en cada lado (para kernel impar)
    h = kernel_h // 2
    w = kernel_w // 2
    
    # Acolchado de la imagen (Padding)
    if len(oldimage.shape) == 3:
        # Para imágenes a color (necesita acolchado en el tercer eje con (0,0))
        image_pad = np.pad(oldimage, pad_width=(
            (h, h), (w, w), (0, 0)), mode='constant', 
            constant_values=0).astype(np.float32)
    else: # Imágenes en escala de grises
        image_pad = np.pad(oldimage, pad_width=(
            (h, h), (w, w)), mode='constant', constant_values=0).astype(np.float32)
    
    H_pad, W_pad = image_pad.shape[:2]
    
    # Si la imagen es a color, la convolución debe aplicarse canal por canal. 
    # Aquí asumimos que esta función solo será llamada con una imagen en escala de grises (2D) 
    # o ya lo estamos manejando externamente.
    if len(oldimage.shape) > 2:
        raise ValueError("Esta convolución debe ser llamada con arrays 2D (canales individuales).")

    image_conv = np.zeros((H_pad, W_pad), dtype=np.float32)

    # Convolución
    for i in range(h, H_pad - h):
        for j in range(w, W_pad - w):
            ventana = image_pad[i-h : i-h + kernel_h, j-w : j-w + kernel_w]
            # Producto punto entre la ventana y el kernel
            image_conv[i, j] = (ventana * kernel).sum()
    
    # Recorte Final para devolver el tamaño original (eliminando el acolchado)
    h_end = -h if h != 0 else H_pad
    w_end = -w if w != 0 else W_pad
    
    return image_conv[h:h_end, w:w_end]