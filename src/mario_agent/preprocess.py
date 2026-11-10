"""Preprocesamiento compartido entre entrenamiento (wrappers) y agente entregable.

Solo usa NumPy y OpenCV. Si cambia aquí, cambia en ambos lados.
"""
import cv2
import numpy as np


def preprocess_frame(rgb: np.ndarray, size: int = 84) -> np.ndarray:
    """Imagen RGB (H, W, 3) uint8 -> escala de grises (size, size) uint8."""
    rgb = np.ascontiguousarray(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
