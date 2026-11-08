"""
Corre el pipeline completo sobre una carpeta de imagenes y genera una tabla
de evidencias (CSV) para el reporte.

Uso:
    python pruebas.py imagenes_prueba
Nombra las imagenes como  <ingrediente_real>_<lo_que_sea>.jpg  (ej. tomate_luz_baja.jpg)
para que se calcule si la prediccion fue correcta.
"""
import csv
import json
import sys
from pathlib import Path

import tensorflow as tf
from PIL import Image

from recommender import Recomendador, normalizar
from vision import clasificar

BASE = Path(__file__).parent


def main(carpeta):
    modelo = tf.keras.models.load_model(BASE / "modelo" / "modelo_ingredientes.keras")
    clases = [normalizar(c) for c in
              json.loads((BASE / "modelo" / "clases.json").read_text(encoding="utf-8"))]
    rec = Recomendador(BASE / "recetas.json")

    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        sys.exit(f"No existe la carpeta '{carpeta}'.")

    filas = []
    for f in sorted(carpeta.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        res = clasificar(modelo, clases, Image.open(f))
        pred, conf = res["top"][0]
        real = normalizar(f.stem.split("_")[0])
        recetas = rec.recomendar([pred], top_n=3, min_cobertura=0) \
            if res["estado"] != "no_ingrediente" else []
        filas.append({
            "archivo": f.name, "real": real, "prediccion": pred,
            "confianza": f"{conf:.3f}", "estado": res["estado"],
            "correcto": "si" if real == pred else "no",
            "recetas_top3": " | ".join(r["nombre"] for r in recetas),
        })
        print(filas[-1])

    if not filas:
        sys.exit(f"No hay imagenes (.jpg, .jpeg, .png, .webp) en '{carpeta}'. "
                 "Agrega fotos nombradas como <ingrediente>_<descripcion>.jpg")

    salida = BASE / "resultados_pruebas.csv"
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    ok = sum(r["correcto"] == "si" for r in filas)
    print(f"\nAciertos: {ok}/{len(filas)}  -> {salida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "imagenes_prueba")
