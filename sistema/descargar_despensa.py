"""
Descarga fotos de internet (Bing) de ingredientes de despensa mexicana y las deja
listas en crudo/<ingrediente>/ para entrenar la CNN.

Requiere:  python -m pip install icrawler

Uso (después de preparar_kaggle.py y preparar_roboflow.py, antes de dividir_dataset.py):
    python descargar_despensa.py --destino crudo

Qué hace:
1. Descarga ~30 fotos por cada búsqueda de BUSQUEDAS en descargas_web/ (sin tocar crudo/).
2. Limpia: descarta archivos que no abren como imagen, imágenes chicas (< 120 px)
   y duplicados exactos (MD5).
3. Copia hasta --max fotos por ingrediente a crudo/<ingrediente>/ como JPG.

IMPORTANTE: las fotos de internet traen basura (dibujos, logos, recetas ya cocinadas).
Antes de dividir el dataset, abre cada carpeta crudo/<ingrediente>/ y borra las fotos
que no muestren el ingrediente. 5 minutos de revisión mejoran mucho el modelo.

Las fotos son de terceros: úsalas solo para entrenar el modelo del proyecto escolar
y no las subas al repositorio (crudo/ y descargas_web/ están en .gitignore).
"""
import argparse
import hashlib
import logging
from pathlib import Path

from PIL import Image

BUSQUEDAS = {
    "tortilla": ["tortillas de maiz", "tortillas de maíz en un plato", "corn tortillas stack"],
    "jamon": ["jamon de pavo rebanado", "sliced ham", "rebanadas de jamon"],
    "pan": ["bolillo pan mexicano", "pan de caja rebanado", "sliced bread loaf"],
    "frijol": ["frijol negro crudo", "frijoles pintos crudos", "dry pinto beans"],
    "arroz": ["arroz blanco crudo", "uncooked white rice", "arroz crudo en un tazon"],
    "pollo": ["pechuga de pollo cruda", "raw chicken breast", "piezas de pollo crudo"],
}
LADO_MIN = 120


def descargar(clases, carpeta, por_busqueda):
    from icrawler.builtin import BingImageCrawler
    for ing in clases:
        for i, q in enumerate(BUSQUEDAS[ing]):
            destino = carpeta / ing / f"q{i}"
            if destino.exists() and any(destino.iterdir()):
                print(f"  ya descargado: {ing} / {q}")
                continue
            print(f"  descargando: {ing} / {q}")
            crawler = BingImageCrawler(storage={"root_dir": str(destino)},
                                       downloader_threads=4, log_level=logging.ERROR)
            crawler.crawl(keyword=q, max_num=por_busqueda)


def limpiar_y_copiar(clases, carpeta, crudo, maximo):
    for ing in clases:
        salida = crudo / ing
        if salida.exists() and any(salida.iterdir()):
            print(f"{ing:9s} '{salida}' ya tiene fotos; se omite (bórrala para regenerarla)")
            continue
        salida.mkdir(parents=True, exist_ok=True)
        vistas, copiadas, malas, chicas, repetidas = set(), 0, 0, 0, 0
        for f in sorted((carpeta / ing).rglob("*")):
            if not f.is_file() or copiadas >= maximo:
                continue
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in vistas:
                repetidas += 1
                continue
            vistas.add(h)
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    if min(im.size) < LADO_MIN:
                        chicas += 1
                        continue
                    im.save(salida / f"web_{ing}_{copiadas:03d}.jpg", quality=92)
                    copiadas += 1
            except Exception:
                malas += 1
        print(f"{ing:9s} copiadas={copiadas:3d}  no eran imagen={malas}  muy chicas={chicas}  "
              f"duplicadas={repetidas}")
        if copiadas == 0:
            salida.rmdir()   # una clase vacía rompería el entrenamiento
            print(f"          ¡No se descargó nada de {ing}! Revisa tu conexión y vuelve a ejecutar.")
        elif copiadas < 30:
            print(f"          Pocas fotos de {ing}: el modelo la reconocerá mal. "
                  "Agrega fotos propias a la carpeta si puedes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="crudo")
    ap.add_argument("--descargas", default="descargas_web")
    ap.add_argument("--clases", nargs="*", default=list(BUSQUEDAS),
                    help=f"ingredientes a descargar (por defecto todos: {' '.join(BUSQUEDAS)})")
    ap.add_argument("--por_busqueda", type=int, default=30)
    ap.add_argument("--max", type=int, default=80, help="fotos máximas por ingrediente")
    ap.add_argument("--sin_descargar", action="store_true",
                    help="solo limpiar y copiar lo ya descargado")
    a = ap.parse_args()

    otras = set(a.clases) - set(BUSQUEDAS)
    if otras:
        raise SystemExit(f"Ingredientes desconocidos: {sorted(otras)}. Opciones: {list(BUSQUEDAS)}")
    carpeta = Path(a.descargas)
    if not a.sin_descargar:
        print("Descargando fotos (unos minutos)...")
        descargar(a.clases, carpeta, a.por_busqueda)
    print("\nLimpiando y copiando a", a.destino)
    limpiar_y_copiar(a.clases, carpeta, Path(a.destino), a.max)
    print(f"\nListo. REVISA las carpetas {a.destino}/<ingrediente>/ y borra las fotos que no "
          "muestren el ingrediente antes de ejecutar dividir_dataset.py")


if __name__ == "__main__":
    main()
