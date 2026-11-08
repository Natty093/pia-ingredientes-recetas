"""
Convierte el dataset de Kaggle "Fruits and Vegetables Image Recognition" a la
estructura crudo/<ingrediente>/ que usa dividir_dataset.py.

- Traduce los nombres de las clases al espanol (MAPA).
- Junta train/validation/test del original: luego dividir_dataset.py los vuelve
  a repartir.
- Elimina las imagenes duplicadas. En el dataset original muchas fotos de test
  y validation son copias exactas de fotos de train; si se conservan, la misma
  foto puede quedar en train y en test y la exactitud medida sale inflada.

Uso:
    python preparar_kaggle.py --origen kaggle --destino crudo
"""
import argparse
import hashlib
import shutil
from collections import Counter
from pathlib import Path

MAPA = {
    "tomato": "tomate", "onion": "cebolla", "garlic": "ajo", "potato": "papa",
    "carrot": "zanahoria", "chilli pepper": "chile", "jalepeno": "chile",
    "bell pepper": "pimiento", "capsicum": "pimiento", "paprika": "pimiento",
    "corn": "maiz", "sweetcorn": "maiz", "cucumber": "pepino", "lettuce": "lechuga",
    "spinach": "espinaca", "lemon": "limon", "orange": "naranja",
    "apple": "manzana", "banana": "platano", "mango": "mango", "pineapple": "pina",
    "beetroot": "betabel", "cabbage": "col", "cauliflower": "coliflor",
    "eggplant": "berenjena", "ginger": "jengibre", "grapes": "uvas", "kiwi": "kiwi",
    "pear": "pera", "peas": "chicharo", "pomegranate": "granada",
    "raddish": "rabano", "soy beans": "soya", "sweetpotato": "camote",
    "turnip": "nabo", "watermelon": "sandia",
}


def huella(ruta):
    return hashlib.md5(ruta.read_bytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--origen", required=True)
ap.add_argument("--destino", default="crudo")
a = ap.parse_args()

vistas = {}               # huella -> ingrediente de la primera copia encontrada
copiadas, duplicadas = Counter(), Counter()
conflictos = 0            # misma foto guardada en dos ingredientes distintos

for split in sorted(Path(a.origen).iterdir()):
    if not split.is_dir():
        continue
    for clase in sorted(split.iterdir()):
        if not clase.is_dir() or clase.name.lower() not in MAPA:
            continue
        ingrediente = MAPA[clase.name.lower()]
        salida = Path(a.destino) / ingrediente
        salida.mkdir(parents=True, exist_ok=True)
        for f in sorted(clase.iterdir()):
            if not f.is_file():
                continue
            h = huella(f)
            if h in vistas:
                duplicadas[ingrediente] += 1
                conflictos += vistas[h] != ingrediente
                continue
            vistas[h] = ingrediente
            nombre = f"{clase.name}_{split.name}_{f.name}".replace(" ", "_")
            shutil.copy2(f, salida / nombre)
            copiadas[ingrediente] += 1

for ing in sorted(copiadas):
    print(f"{ing:12s} unicas={copiadas[ing]:4d}  duplicadas descartadas={duplicadas[ing]}")
print(f"\nTotal: {sum(copiadas.values())} imagenes unicas, "
      f"{sum(duplicadas.values())} duplicadas descartadas, "
      f"{conflictos} repetidas entre ingredientes distintos")
