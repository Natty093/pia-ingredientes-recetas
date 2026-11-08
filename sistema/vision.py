"""
Utilidades de visión compartidas por app.py y pruebas.py.

- preparar_imagen: endereza la foto según su EXIF (las fotos de celular suelen venir
  giradas) y la convierte al tamaño que espera la CNN.
- clasificar: devuelve las 3 mejores predicciones y un estado:
    "aceptado"        -> confianza >= umbral, se agrega solo
    "confirmar"       -> confianza baja, el usuario decide
    "no_ingrediente"  -> imagen casi plana (negra, blanca, un solo color) o la CNN
                         predijo la clase "otro" (si el modelo se reentrena con ella)
"""
import numpy as np
from PIL import ImageOps

IMG_SIZE = (224, 224)
CLASE_OTRO = "otro"          # carpeta opcional data/train/otro/ al reentrenar
STD_MIN = 20.0               # desviación estándar mínima de los píxeles (0-255).
                             # Las fotos reales de ingredientes probadas tienen > 55;
                             # una imagen negra, blanca o de un solo color tiene ~0.


def preparar_imagen(img):
    """PIL.Image -> arreglo float32 (224, 224, 3) con píxeles 0-255."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img.resize(IMG_SIZE), dtype="float32")


def es_imagen_plana(x):
    # Variación dentro de cada canal de color (una foto toda roja tiene variación 0
    # en cada canal, aunque los canales R, G y B sean distintos entre sí)
    return float(x.std(axis=(0, 1)).mean()) < STD_MIN


def clasificar(modelo, clases, img, umbral=0.70, k=3):
    x = preparar_imagen(img)
    p = modelo.predict(x[None], verbose=0)[0]
    idx = np.argsort(p)[::-1][:k]
    top = [(clases[i], float(p[i])) for i in idx]
    ing, conf = top[0]
    if es_imagen_plana(x) or ing == CLASE_OTRO:
        estado = "no_ingrediente"
    elif conf >= umbral:
        estado = "aceptado"
    else:
        estado = "confirmar"
    return {"top": [t for t in top if t[0] != CLASE_OTRO] or top, "estado": estado}
