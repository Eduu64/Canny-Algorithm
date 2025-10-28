import numpy as np
from PIL import Image
from GaussianFilter import gaussian_filter_image
from Gradiente import calcular_gradiente
from NonMaximum import supresion_non_max
from Histeresis import umbralizacion_histéresis

def canny_algorithm(image_path, sigma, T_low, T_high):
    print("Iniciando Detección de Bordes Canny (Implementación Manual)...")

    try:

        # Cargar y convertir a float32
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, dtype=np.float32)
    
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {image_path}")
        return None
    
    # 1. Suavizado (Gaussian Blur)
    print("Aplicando Filtro Gaussiano...")
    imagen_suavizada = gaussian_filter_image(image, sigma)
    
    if imagen_suavizada is None:
        print("Error Aplicando Filtro Gaussiano...")
        return None
        
    # 2. Cálculo del Gradiente (Sobel)
    print("Calculando Magnitud y Dirección del Gradiente...")
    magnitud, angulos = calcular_gradiente(imagen_suavizada)
    
    # 3. Supresión de No Máximos (NMS)
    print("Aplicando Supresión de No Máximos...")
    bordes_delgados = supresion_non_max(magnitud, angulos)
    
    # 4. Umbralización con Histéresis
    print(f"Aplicando Umbralización por Histéresis (T_low={T_low}, T_high={T_high})...")
    bordes_finales = umbralizacion_histéresis(bordes_delgados, T_low, T_high)
    
    print("Detección de Bordes completado.")
    return imagen_suavizada,bordes_delgados,bordes_finales

'''
if __name__ == '__main__':
    IMAGEN_RUTA = 'Lennapng.png' 
    
    # Parámetros Comunes para Canny
    SIGMA = 1.75  
    UMBRAL_BAJO = 25
    UMBRAL_ALTO = 50
    
    mapa_de_bordes = canny_algorithm(IMAGEN_RUTA, 
                                           sigma=SIGMA, 
                                           T_low=UMBRAL_BAJO, 
                                           T_high=UMBRAL_ALTO)
    
    if mapa_de_bordes is not None:
        # Guardar el resultado como una imagen PNG
        resultado_img = Image.fromarray(mapa_de_bordes)
        resultado_img.save('Bordes.png')
        print("Resultado guardado como 'Bordes.png'")

'''