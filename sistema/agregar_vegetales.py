"""
Agrega fotos del dataset de Kaggle "Vegetable Image Dataset" (15 verduras,
fotos de mercado y cultivo) a un dataset ya dividido en train/val/test.

- Las clases que ya existen reciben fotos extra (refuerzo).
- Brocoli, calabaza, papaya y ejote se agregan como clases nuevas.
- No se usa "Radish": es rabano blanco (daikon), distinto del rabano rojo.
- Se eliminan duplicados por huella MD5, entre si y contra el dataset original.
- Las fotos nuevas se reparten 70/15/15; las fotos que ya estaban en
  train/val/test se quedan donde estaban, para poder comparar con el modelo
  anterior usando el mismo conjunto de prueba.

Uso:
    python agregar_vegetales.py --origen "kaggle_vegetales/Vegetable Images" --data data --destino data_v2
"""
import argparse
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

MAPA = {
    "Broccoli": "brocoli", "Pumpkin": "calabaza", "Papaya": "papaya", "Bean": "ejote",
    "Brinjal": "berenjena", "Cabbage": "col", "Capsicum": "pimiento", "Carrot": "zanahoria",
    "Cauliflower": "coliflor", "Cucumber": "pepino", "Potato": "papa", "Tomato": "tomate",
}


def huella(ruta):
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta 'Vegetable Images' del ZIP")
    ap.add_argument("--data", default="data", help="dataset actual (train/val/test)")
    ap.add_argument("--destino", default="data_v2")
    ap.add_argument("--extra", type=int, default=150, help="fotos extra por clase existente")
    ap.add_argument("--nuevas", type=int, default=250, help="fotos por clase nueva")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    origen, data, destino = Path(a.origen), Path(a.data), Path(a.destino)
    if destino.exists():
        raise SystemExit(f"{destino} ya existe; bórralo o usa otro --destino")
    shutil.copytree(data, destino)
    vistas = {huella(f) for f in data.rglob("*") if f.is_file()}
    existentes = {p.name for p in (data / "train").iterdir() if p.is_dir()}

    random.seed(a.semilla)
    resumen, duplicadas = {}, Counter()
    for carpeta, clase in MAPA.items():
        fotos = sorted(f for split in ("train", "validation", "test")
                       for f in (origen / split / carpeta).iterdir())
        random.shuffle(fotos)
        limite = a.extra if clase in existentes else a.nuevas
        elegidas = []
        for f in fotos:
            if len(elegidas) == limite:
                break
            h = huella(f)
            if h in vistas:
                duplicadas[clase] += 1
                continue
            vistas.add(h)
            elegidas.append(f)
        n_train, n_val = int(len(elegidas) * 0.70), int(len(elegidas) * 0.15)
        partes = {"train": elegidas[:n_train], "val": elegidas[n_train:n_train + n_val],
                  "test": elegidas[n_train + n_val:]}
        for split, lista in partes.items():
            salida = destino / split / clase
            salida.mkdir(parents=True, exist_ok=True)
            for f in lista:
                shutil.copy2(f, salida / f"veg_{carpeta}_{f.parent.parent.name}_{f.name}")
        resumen[clase] = (("nueva" if clase not in existentes else "refuerzo"),
                          *(len(partes[s]) for s in ("train", "val", "test")))

    for clase, (tipo, tr, va, te) in sorted(resumen.items()):
        print(f"{clase:12s} {tipo:9s} +{tr} train  +{va} val  +{te} test")
    print("Duplicados descartados:", sum(duplicadas.values()), dict(duplicadas))


if __name__ == "__main__":
    main()
