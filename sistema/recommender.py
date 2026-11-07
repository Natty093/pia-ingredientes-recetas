"""
Sistema de recomendacion de recetas basado en contenido.

Cada receta y la lista de ingredientes detectados se tratan como vectores
binarios sobre el vocabulario de ingredientes. Se combinan tres medidas:
  - Cobertura  = |R ∩ U| / |R|        (que parte de la receta ya tengo)
  - Jaccard    = |R ∩ U| / |R ∪ U|
  - Coseno     = |R ∩ U| / sqrt(|R|·|U|)   (coseno de vectores binarios)
  puntuacion = 0.6·cobertura + 0.2·jaccard + 0.2·coseno
"""
import json
import math
import unicodedata
from pathlib import Path

BASICOS = {"sal", "agua", "aceite", "pimienta", "azucar"}  # se asumen disponibles

_BONITO = {"pina": "piña", "champinon": "champiñón", "limon": "limón",
           "platano": "plátano", "azucar": "azúcar", "brocoli": "brócoli"}


def normalizar(texto):
    t = unicodedata.normalize("NFD", texto.lower().strip())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("_", " ")


def bonito(ing):
    return _BONITO.get(ing, ing).capitalize()


class Recomendador:
    def __init__(self, ruta_recetas="recetas.json"):
        recetas = json.loads(Path(ruta_recetas).read_text(encoding="utf-8"))
        self.recetas = []
        for r in recetas:
            ings = {normalizar(i) for i in r["ingredientes"]} - BASICOS
            self.recetas.append({**r, "_ings": ings})
        self.vocabulario = sorted(set().union(*(r["_ings"] for r in self.recetas)))

    def recomendar(self, ingredientes, top_n=5, min_cobertura=0.3):
        usuario = {normalizar(i) for i in ingredientes} - BASICOS
        if not usuario:
            return []
        resultados = []
        for r in self.recetas:
            rec = r["_ings"]
            inter = len(rec & usuario)
            if inter == 0:
                continue
            cobertura = inter / len(rec)
            if cobertura < min_cobertura:
                continue
            jaccard = inter / len(rec | usuario)
            coseno = inter / math.sqrt(len(rec) * len(usuario))
            puntuacion = 0.6 * cobertura + 0.2 * jaccard + 0.2 * coseno
            resultados.append({
                "nombre": r["nombre"],
                "tiempo_min": r.get("tiempo_min"),
                "pasos": r.get("pasos", []),
                "tengo": sorted(rec & usuario),
                "faltan": sorted(rec - usuario),
                "cobertura": cobertura,
                "jaccard": jaccard,
                "coseno": coseno,
                "puntuacion": puntuacion,
            })
        resultados.sort(key=lambda x: (-x["puntuacion"], len(x["faltan"])))
        return resultados[:top_n]


if __name__ == "__main__":
    rec = Recomendador()
    for r in rec.recomendar(["tomate", "cebolla", "chile", "huevo"]):
        print(f"{r['nombre']:30s} {r['puntuacion']:.2f}  faltan: {r['faltan']}")
