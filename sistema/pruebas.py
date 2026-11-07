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

import numpy as np
import tensorflow as tf
from PIL import Image

from recommender import Recomendador, normalizar

BASE = Path(__file__).parent


def main(carpeta):
    modelo = tf.keras.models.load_model(BASE / "modelo" / "modelo_ingredientes.keras")
    clases = [normalizar(c) for c in
              json.loads((BASE / "modelo" / "clases.json").read_text(encoding="utf-8"))]
    rec = Recomendador(BASE / "recetas.json")

    filas = []
    for f in sorted(Path(carpeta).iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img = Image.open(f).convert("RGB").resize((224, 224))
        p = modelo.predict(np.array(img, dtype="float32")[None], verbose=0)[0]
        i = int(np.argmax(p))
        real = normalizar(f.stem.split("_")[0])
        recetas = rec.recomendar([clases[i]], top_n=3, min_cobertura=0.1)
        filas.append({
            "archivo": f.name, "real": real, "prediccion": clases[i],
            "confianza": f"{p[i]:.3f}", "correcto": "si" if real == clases[i] else "no",
            "recetas_top3": " | ".join(r["nombre"] for r in recetas),
        })
        print(filas[-1])

    salida = BASE / "resultados_pruebas.csv"
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    ok = sum(r["correcto"] == "si" for r in filas)
    print(f"\nAciertos: {ok}/{len(filas)}  -> {salida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "imagenes_prueba")
