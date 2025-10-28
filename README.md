# Canny-Algorithm: Detección de Bordes con Python

Una implementación completa y modular del **Algoritmo de Detección de Bordes de Canny** en Python, diseñado para el procesamiento de imágenes y el análisis de visión por computadora.

## Muestra Visual

<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/e8092615-6f4b-4b07-bcb4-02ddd9ef3b7f" />

## Sobre el Proyecto

El Algoritmo de Canny es uno de los métodos más potentes para extraer información estructural de las imágenes. 

### Estructura Modular

El código está organizado siguiendo las etapas del algoritmo de Canny:

1.  **`GaussianFilter.py`**: Suavizado de la imagen para reducir el ruido.
2.  **`Gradiente.py`**: Cálculo del gradiente de intensidad y la dirección del borde.
3.  **`NonMaximum.py`**: Aplicación de la supresión de no-máximos para adelgazar los bordes.
4.  **`Histeresis.py`**: Aplicación del umbral de histéresis para determinar los bordes finales.
5.  **`convolution.py`**:  Implementación de la operación de convolución base para los filtros.
6.  **`canny.py`**: Módulo principal que orquesta la ejecución del algoritmo.
7.  **`GUI.py`**: Implementación de la interfaz gráfica de usuario.
8.  **`main.py`**: Punto de entrada del programa.

---

## Requisitos y Tecnologías

El proyecto está construido enteramente en Python.

**Librerías:**

* tkinter: Para la Interfaz Gráfica de Usuario (GUI).

* numpy: Procesamiento numérico y manipulación eficiente de matrices/datos.

* Pillow (PIL): Manejo y procesamiento de imágenes.

* os: Interacción con el sistema operativo (rutas de archivos).

---
