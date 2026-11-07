import argparse
import shutil
from pathlib import Path

MAPA = {
    "tomato": "tomate", "onion": "cebolla", "garlic": "ajo", "potato": "papa",
    "carrot": "zanahoria", "chilli pepper": "chile", "jalepeno": "chile",
    "bell pepper": "pimiento", "capsicum": "pimiento", "corn": "maiz",
    "sweetcorn": "maiz", "cucumber": "pepino", "lettuce": "lechuga",
    "spinach": "espinaca", "lemon": "limon", "orange": "naranja",
    "apple": "manzana", "banana": "platano", "mango": "mango", "pineapple": "pina",
}

ap = argparse.ArgumentParser()
ap.add_argument("--origen", required=True)
ap.add_argument("--destino", default="crudo")
a = ap.parse_args()

for split in Path(a.origen).iterdir():
    if not split.is_dir():
        continue
    for clase in split.iterdir():
        if not clase.is_dir() or clase.name.lower() not in MAPA:
            continue
        salida = Path(a.destino) / MAPA[clase.name.lower()]
        salida.mkdir(parents=True, exist_ok=True)
        for f in clase.iterdir():
            if f.is_file():
                nombre = f"{clase.name}_{split.name}_{f.name}".replace(" ", "_")
                shutil.copy2(f, salida / nombre)

for c in sorted(Path(a.destino).iterdir()):
    print(c.name, len(list(c.iterdir())))
