"""
Agrega ingredientes de despensa (huevo, queso, leche, harina) desde el dataset de
Roboflow "recipes" a la carpeta crudo/ que ya llenó preparar_kaggle.py.

Dataset: https://universe.roboflow.com/thatonenerds-space/recipes-bkrot
Descárgalo en formato "Folder Structure" y descomprímelo en sistema/roboflow/
(debe quedar roboflow/train/<clase>/*.jpg, roboflow/valid/..., roboflow/test/...).

Qué hace:
- Copia solo las clases de MAPA (eggs, cheese, milk, flour) traducidas al español.
  Las demás (tomato, onion, potato...) se ignoran para no mezclar en una misma
  clase fotos de dos datasets con estilos distintos.
- Deja UNA foto por cada foto original. Roboflow genera copias aumentadas
  (volteadas, con otro brillo) con el mismo nombre base antes de ".rf.": si una
  copia cae en train y otra en test, la exactitud medida sale inflada.
- Elimina duplicados exactos (misma huella MD5), igual que preparar_kaggle.py.
- Limita las fotos por clase (--max, 120 por defecto) para que queden en una
  cantidad parecida a las clases de Kaggle (~90 fotos únicas por clase) y el
  modelo no se incline hacia los ingredientes nuevos.

Uso (después de preparar_kaggle.py y antes de dividir_dataset.py):
    python preparar_roboflow.py --origen roboflow --destino crudo
"""
import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

MAPA = {"eggs": "huevo", "cheese": "queso", "milk": "leche", "flour": "harina"}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def nombre_base(f):
    """eggs_290_png.rf.3a9c...jpg -> eggs_290_png (igual para todas sus copias aumentadas)."""
    return f.name.split(".rf.")[0] if ".rf." in f.name else f.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta del dataset de Roboflow")
    ap.add_argument("--destino", default="crudo")
    ap.add_argument("--max", type=int, default=120, help="fotos máximas por ingrediente")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    origen = Path(a.origen)
    if not origen.is_dir():
        raise SystemExit(f"No existe la carpeta '{origen}'.")

    # ingrediente -> {nombre_base: [archivos]}
    grupos = defaultdict(lambda: defaultdict(list))
    for split in sorted(p for p in origen.iterdir() if p.is_dir()):
        for clase in sorted(p for p in split.iterdir() if p.is_dir()):
            ing = MAPA.get(clase.name.lower())
            if ing is None:
                continue
            for f in sorted(clase.iterdir()):
                if f.suffix.lower() in EXT:
                    grupos[ing][nombre_base(f)].append(f)

    if not grupos:
        raise SystemExit(f"No encontré carpetas {sorted(MAPA)} dentro de '{origen}/<split>/'. "
                         "¿Lo descargaste en formato 'Folder Structure'?")

    random.seed(a.semilla)
    for ing in sorted(grupos):
        salida = Path(a.destino) / ing
        if salida.exists() and any(salida.iterdir()):
            raise SystemExit(f"'{salida}' ya tiene fotos. Bórrala si quieres volver a generarla.")
        salida.mkdir(parents=True, exist_ok=True)

        # una foto por original (la primera en orden alfabético: misma elección en cada corrida)
        unicas, vistas, copias, repetidas = [], set(), 0, 0
        for base in sorted(grupos[ing]):
            archivos = grupos[ing][base]
            copias += len(archivos) - 1
            f = archivos[0]
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in vistas:
                repetidas += 1
                continue
            vistas.add(h)
            unicas.append(f)

        random.shuffle(unicas)
        elegidas = unicas[:a.max]
        for f in elegidas:
            shutil.copy2(f, salida / f"roboflow_{nombre_base(f)}{f.suffix.lower()}")
        print(f"{ing:8s} originales={len(unicas):4d}  copias aumentadas descartadas={copias:4d}  "
              f"duplicadas={repetidas}  copiadas={len(elegidas)}")

    print(f"\nListo. Ahora ejecuta: python dividir_dataset.py --origen {a.destino} --destino data")


if __name__ == "__main__":
    main()
