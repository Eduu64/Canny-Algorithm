import numpy as np

def umbralizacion_histéresis(bordes_delgados, T_low, T_high):
    """Conecta bordes débiles a bordes fuertes usando propagación (DFS)."""
    
    H, W = bordes_delgados.shape
    salida = np.zeros((H, W), dtype=np.uint8)
    
    # 1. Clasificación inicial
    bordes_fuertes_i, bordes_fuertes_j = np.where(bordes_delgados >= T_high)
    bordes_debiles_i, bordes_debiles_j = np.where((bordes_delgados >= T_low) & (bordes_delgados < T_high))
    
    salida[bordes_fuertes_i, bordes_fuertes_j] = 255
    
    # 2. Conexión de Bordes Débiles a Fuertes (usando Pila para DFS)
    pila_fuertes = list(zip(bordes_fuertes_i, bordes_fuertes_j))
    
    # Crea un mapa de bordes débiles para una verificación rápida
    mapa_debiles = set(zip(bordes_debiles_i, bordes_debiles_j))
    
    while pila_fuertes:
        i, j = pila_fuertes.pop()
        
        # Recorre la vecindad 3x3
        for ni in range(i - 1, i + 2):
            for nj in range(j - 1, j + 2):
                
                # Comprobación de límites y si el vecino es un borde débil no procesado
                if (0 <= ni < H and 0 <= nj < W and 
                    (ni, nj) in mapa_debiles and # Verifica si es un borde débil
                    salida[ni, nj] == 0): # Verifica si no ha sido marcado ya como fuerte
                    
                    salida[ni, nj] = 255 # Conviértelo en borde fuerte
                    pila_fuertes.append((ni, nj)) # Agrégalo a la pila
                    mapa_debiles.remove((ni, nj)) # Lo eliminamos para no procesarlo dos veces
    
    return salida
