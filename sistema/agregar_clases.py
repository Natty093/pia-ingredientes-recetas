"""
Agrega clases nuevas (una carpeta por ingrediente) a un dataset ya dividido en
train/val/test, sin mover las fotos que ya estaban en cada conjunto.

Sirve para sumar al modelo los ingredientes de despensa preparados con
preparar_roboflow.py y descargar_despensa.py (huevo, queso, tortilla, pollo...)
sin rehacer la división del resto: así el modelo nuevo se compara con el anterior
usando las mismas fotos de prueba.

- Elimina duplicados por huella MD5, entre sí y contra el dataset original.
- Reparte las fotos de cada clase nueva 70/15/15 con semilla fija.
- Si una clase ya existe en el dataset, sus fotos se suman a ella.

Uso:
    python agregar_clases.py --origen crudo_despensa --data data --destino data_v3
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def huella(ruta):
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta con una subcarpeta por ingrediente")
    ap.add_argument("--data", default="data", help="dataset actual (train/val/test)")
    ap.add_argument("--destino", required=True)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    origen, data, destino = Path(a.origen), Path(a.data), Path(a.destino)
    if destino.exists():
        raise SystemExit(f"{destino} ya existe; bórralo o usa otro --destino")
    shutil.copytree(data, destino)
    vistas = {huella(f) for f in data.rglob("*") if f.is_file()}
    existentes = {p.name for p in (data / "train").iterdir() if p.is_dir()}

    random.seed(a.semilla)
    duplicadas = 0
    for clase in sorted(p for p in origen.iterdir() if p.is_dir()):
        fotos = []
        for f in sorted(clase.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            h = huella(f)
            if h in vistas:
                duplicadas += 1
                continue
            vistas.add(h)
            fotos.append(f)
        random.shuffle(fotos)
        n_train, n_val = int(len(fotos) * 0.70), int(len(fotos) * 0.15)
        partes = {"train": fotos[:n_train], "val": fotos[n_train:n_train + n_val],
                  "test": fotos[n_train + n_val:]}
        for split, lista in partes.items():
            salida = destino / split / clase.name
            salida.mkdir(parents=True, exist_ok=True)
            for f in lista:
                shutil.copy2(f, salida / f"{clase.name}_{f.name}")
        tipo = "refuerzo" if clase.name in existentes else "nueva"
        print(f"{clase.name:12s} {tipo:9s} +{len(partes['train'])} train  "
              f"+{len(partes['val'])} val  +{len(partes['test'])} test")
    print("Duplicados descartados:", duplicadas)


if __name__ == "__main__":
    main()
