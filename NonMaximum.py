import numpy as np

def supresion_non_max(magnitud, angulos):
    
    H, W = magnitud.shape
    salida = np.zeros_like(magnitud, dtype=np.float32)
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            
            vecino1 = 0
            vecino2 = 0
            angulo = angulos[i, j]
            
            # Cuantificación de ángulos: Comparamos el píxel con sus vecinos
            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                # Ángulo 0° (Horizontal): compara con vecinos Este y Oeste
                vecino1, vecino2 = magnitud[i, j + 1], magnitud[i, j - 1]
            elif (22.5 <= angulo < 67.5):
                # Ángulo 45° (Diagonal NE/SW)
                vecino1, vecino2 = magnitud[i + 1, j - 1], magnitud[i - 1, j + 1]
            elif (67.5 <= angulo < 112.5):
                # Ángulo 90° (Vertical): compara con vecinos Norte y Sur
                vecino1, vecino2 = magnitud[i + 1, j], magnitud[i - 1, j]
            elif (112.5 <= angulo < 157.5):
                # Ángulo 135° (Diagonal NW/SE)
                vecino1, vecino2 = magnitud[i - 1, j - 1], magnitud[i + 1, j + 1]

            # Si la magnitud actual no es la más alta en esa dirección, se suprime a 0
            if magnitud[i, j] >= vecino1 and magnitud[i, j] >= vecino2:
                salida[i, j] = magnitud[i, j]
            else:
                salida[i, j] = 0
                
    return salida