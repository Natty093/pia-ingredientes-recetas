"""
Interfaz Streamlit: sube fotos -> CNN detecta ingredientes -> recomienda recetas.
Uso:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from recommender import Recomendador, bonito, normalizar

BASE = Path(__file__).parent
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Ingredientes y Recetas", page_icon="🍳", layout="wide")


@st.cache_resource
def cargar_modelo():
    import tensorflow as tf
    ruta = BASE / "modelo" / "modelo_ingredientes.keras"
    modelo = tf.keras.models.load_model(ruta)
    clases = json.loads((BASE / "modelo" / "clases.json").read_text(encoding="utf-8"))
    return modelo, [normalizar(c) for c in clases]


@st.cache_resource
def cargar_recomendador():
    return Recomendador(BASE / "recetas.json")


def predecir(modelo, clases, img, k=3):
    x = np.array(img.convert("RGB").resize(IMG_SIZE), dtype="float32")[None]
    p = modelo.predict(x, verbose=0)[0]
    idx = np.argsort(p)[::-1][:k]
    return [(clases[i], float(p[i])) for i in idx]


st.title("🍳 Reconocimiento de ingredientes y recomendación de recetas")
st.caption("CNN (MobileNetV2, transfer learning) + sistema de recomendación por similitud")

try:
    modelo, clases = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo. Ejecuta primero `python train_model.py`.")
    st.exception(e)
    st.stop()

recomendador = cargar_recomendador()

with st.sidebar:
    st.header("Parámetros")
    umbral = st.slider("Confianza mínima para aceptar un ingrediente", 0.1, 0.99, 0.60, 0.05)
    top_n = st.slider("Número de recetas", 1, 10, 5)
    min_cob = st.slider("Cobertura mínima de la receta", 0.1, 1.0, 0.3, 0.05)

col_a, col_b = st.columns(2)
with col_a:
    archivos = st.file_uploader("Sube una o varias fotos", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True)
with col_b:
    foto_cam = st.camera_input("...o toma una foto")

imagenes = [(f.name, Image.open(f)) for f in archivos or []]
if foto_cam is not None:
    imagenes.append(("camara", Image.open(foto_cam)))

detectados = []
if imagenes:
    st.subheader("1. Ingredientes detectados")
    cols = st.columns(min(4, len(imagenes)))
    for i, (nombre, img) in enumerate(imagenes):
        top = predecir(modelo, clases, img)
        with cols[i % len(cols)]:
            st.image(img, caption=nombre, use_container_width=True)
            for ing, conf in top:
                st.write(f"{bonito(ing)}: **{conf:.1%}**")
            if top[0][1] >= umbral:
                detectados.append(top[0][0])
            else:
                st.warning("Confianza baja: confirma o corrige abajo.")

st.subheader("2. Confirma o edita tus ingredientes")
opciones = sorted(set(clases) | set(recomendador.vocabulario))
seleccion = st.multiselect("Ingredientes disponibles", opciones,
                           default=sorted(set(detectados)), format_func=bonito)

if st.button("Recomendar recetas", type="primary", disabled=not seleccion):
    recetas = recomendador.recomendar(seleccion, top_n=top_n, min_cobertura=min_cob)
    st.subheader("3. Recetas recomendadas")
    if not recetas:
        st.info("No hay recetas con esos ingredientes. Prueba bajar la cobertura mínima o agregar más.")
    for r in recetas:
        with st.expander(f"{r['nombre']} — coincidencia {r['cobertura']:.0%}  "
                         f"(puntuación {r['puntuacion']:.2f})", expanded=True):
            c1, c2 = st.columns(2)
            c1.markdown("**✅ Ya tienes:** " + ", ".join(bonito(i) for i in r["tengo"]))
            c2.markdown("**🛒 Te faltan:** " +
                        (", ".join(bonito(i) for i in r["faltan"]) or "nada, ¡todo listo!"))
            st.caption(f"Tiempo aprox.: {r['tiempo_min']} min | Jaccard {r['jaccard']:.2f} | "
                       f"Coseno {r['coseno']:.2f}")
            for n, paso in enumerate(r["pasos"], 1):
                st.write(f"{n}. {paso}")
