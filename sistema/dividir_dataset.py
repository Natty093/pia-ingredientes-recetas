"""
Divide un dataset crudo en train / val / test (70/15/15).

Entrada esperada (una carpeta por ingrediente, nombre en minusculas y sin acentos):
    crudo/tomate/img1.jpg ...
    crudo/cebolla/img1.jpg ...

Uso:
    python dividir_dataset.py --origen crudo --destino data
"""
import argparse
import random
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True)
    ap.add_argument("--destino", default="data")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.semilla)
    destino = Path(a.destino)
    # Si data/ ya tiene una division anterior, la nueva se mezclaria con ella y una
    # misma foto podria quedar en train y en test
    if any(f.is_file() for f in destino.rglob("*") if f.name != ".gitkeep"):
        raise SystemExit(f"'{destino}' ya tiene imagenes de una division anterior. "
                         "Borra train/, val/ y test/ dentro de ella y vuelve a ejecutar.")
    for clase in sorted(p for p in Path(a.origen).iterdir() if p.is_dir()):
        imgs = sorted(f for f in clase.iterdir() if f.suffix.lower() in EXT)  # mismo orden siempre
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        partes = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }
        for nombre, lista in partes.items():
            salida = destino / nombre / clase.name
            salida.mkdir(parents=True, exist_ok=True)
            for f in lista:
                shutil.copy2(f, salida / f.name)
        print(f"{clase.name:15s} total={n:4d}  train={len(partes['train'])}  "
              f"val={len(partes['val'])}  test={len(partes['test'])}")


if __name__ == "__main__":
    main()
