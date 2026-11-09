"""
Interfaz Streamlit: sube fotos -> CNN detecta ingredientes -> recomienda recetas.
Uso:  streamlit run app.py
"""
import json
import os
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from recommender import Recomendador, bonito, normalizar
from vision import CLASE_OTRO, clasificar

BASE = Path(__file__).parent

EMOJI = {
    "aguacate": "🥑", "ajo": "🧄", "arroz": "🍚", "berenjena": "🍆", "brocoli": "🥦",
    "calabacita": "🥒", "camote": "🍠", "cebolla": "🧅", "champinon": "🍄", "chicharo": "🫛",
    "chile": "🌶️", "col": "🥬", "coliflor": "🥦", "crema": "🥛", "espinaca": "🥬",
    "fresa": "🍓", "frijol": "🫘", "harina": "🌾", "jamon": "🥓", "huevo": "🥚", "jengibre": "🫚", "kiwi": "🥝",
    "leche": "🥛", "lechuga": "🥬", "limon": "🍋", "maiz": "🌽", "mango": "🥭",
    "manzana": "🍎", "naranja": "🍊", "pan": "🍞", "papa": "🥔", "pasta": "🍝",
    "pepino": "🥒", "pera": "🍐", "pimiento": "🫑", "pina": "🍍", "platano": "🍌",
    "pollo": "🍗", "queso": "🧀", "sandia": "🍉", "soya": "🫘", "tomate": "🍅",
    "tortilla": "🫓", "uvas": "🍇", "zanahoria": "🥕", "calabaza": "🎃", "papaya": "🍈",
    "ejote": "🫛",
}
SUGERIDOS = ["huevo", "tomate", "cebolla", "papa", "pollo", "queso", "tortilla", "aguacate"]

st.set_page_config(page_title="Ingredientes y Recetas", page_icon="🍳", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
/* La fuente se aplica a todo menos a los íconos de Streamlit, que usan su propia fuente */
.stApp *:not([data-testid="stIconMaterial"]) { font-family: 'Nunito', sans-serif; }
.block-container { padding-top: 3.5rem; max-width: 1100px; }

.antetitulo { color: #8E2B1C; font-weight: 700; font-size: .85rem; letter-spacing: .04em;
              text-transform: uppercase; margin: 0; }
.saludo { font-size: 2.3rem; font-weight: 800; margin: .1rem 0 0 0; line-height: 1.15; }
.subsaludo { color: #7A6A64; font-size: 1.05rem; margin: .3rem 0 1rem 0; }
.seccion { font-size: 1.2rem; font-weight: 800; margin: 1.3rem 0 .5rem 0; }

/* Pestañas subrayadas como en el diseño */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 1.05rem; }

/* Botones redondos de ingredientes sugeridos */
.st-key-sugeridos button, .st-key-combina button {
    border-radius: 999px; background: #FFFFFF; border: 1px solid #E8DFDB;
    box-shadow: 0 2px 6px rgba(80, 40, 30, .08); padding: .3rem .9rem;
}
.st-key-sugeridos button:hover, .st-key-combina button:hover { border-color: #8E2B1C; color: #8E2B1C; }

/* Botón principal */
.st-key-cocinar button { border-radius: 12px; padding: .6rem 1.6rem; font-weight: 700; font-size: 1.05rem; }

/* Tarjetas */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.portada-receta {
    background: linear-gradient(135deg, #FBEDE8 0%, #F6E3D6 100%);
    border-radius: 12px; height: 110px; display: flex; align-items: center; justify-content: center;
    font-size: 2.6rem; letter-spacing: .4rem; margin-bottom: .7rem;
}
.receta-titulo { font-size: 1.15rem; font-weight: 800; margin: 0; }
.receta-datos { color: #7A6A64; font-size: .88rem; margin: .1rem 0 .5rem 0; }
.prediccion { font-size: 1.1rem; font-weight: 800; margin: .4rem 0 .1rem 0; }
.etiqueta { display: inline-block; padding: .15rem .6rem; margin: .12rem .15rem .12rem 0;
            border-radius: 999px; font-size: .85rem; line-height: 1.5; }
.tengo { background: #E6F4EA; color: #1E6B34; }
.falta { background: #FDECEA; color: #A3261B; }
.ok    { background: #E6F4EA; color: #1E6B34; }
.baja  { background: #FFF4D6; color: #8A5A00; }
.neutra { background: #F7F3F1; color: #4A3B35; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    import tensorflow as tf
    # Carpeta del modelo: "modelo" por defecto. Para probar otro sin reemplazarlo:
    #   PIA_MODELO=modelo_despensa python -m streamlit run app.py
    carpeta = BASE / os.environ.get("PIA_MODELO", "modelo")
    modelo = tf.keras.models.load_model(carpeta / "modelo_ingredientes.keras")
    clases = json.loads((carpeta / "clases.json").read_text(encoding="utf-8"))
    return modelo, [normalizar(c) for c in clases]


@st.cache_resource
def cargar_recomendador():
    return Recomendador(BASE / "recetas.json")


def con_emoji(ing):
    return f"{EMOJI.get(ing, '🥗')} {bonito(ing)}"


def etiquetas(ingredientes, clase):
    return "".join(f'<span class="etiqueta {clase}">{escape(con_emoji(i))}</span>'
                   for i in ingredientes)


def portada(ingredientes):
    iconos = [EMOJI[i] for i in sorted(ingredientes) if i in EMOJI][:3] or ["🍽️"]
    return f'<div class="portada-receta">{"".join(iconos)}</div>'


def combina_bien_con(seleccion, recetas, n=5):
    """Ingredientes que más aparecen en las recetas que comparten algo con la selección."""
    usuario = set(seleccion)
    cuenta = Counter()
    for r in recetas:
        comunes = len(r["_ings"] & usuario)
        for ing in r["_ings"] - usuario:
            cuenta[ing] += comunes
    return [ing for ing, c in cuenta.most_common(n) if c > 0]


def agregar(ing):
    if ing not in st.session_state.sel:
        st.session_state.sel = sorted(st.session_state.sel + [ing])


def tarjeta_receta(nombre, tiempo, ings, pasos, tengo=None, faltan=None, cobertura=None, datos=""):
    with st.container(border=True):
        st.markdown(portada(ings), unsafe_allow_html=True)
        t = f"⏱️ {tiempo} min" if tiempo else ""
        st.markdown(f'<div class="receta-titulo">{escape(nombre)}</div>'
                    f'<div class="receta-datos">{" · ".join(x for x in (t, datos) if x)}</div>',
                    unsafe_allow_html=True)
        if cobertura is not None:
            st.progress(cobertura, text=f"Tienes el {cobertura:.0%} de los ingredientes")
            st.markdown("**Ya tienes**<br>" + etiquetas(tengo, "tengo"), unsafe_allow_html=True)
            st.markdown("**Te faltan**<br>" + (etiquetas(faltan, "falta") or "Nada, ¡todo listo!"),
                        unsafe_allow_html=True)
        else:
            st.markdown(etiquetas(sorted(ings), "neutra"), unsafe_allow_html=True)
        with st.expander("Ver preparación"):
            for n, texto in enumerate(pasos, 1):
                st.write(f"{n}. {texto}")


hora = datetime.now().hour
saludo = "Buenos días" if 5 <= hora < 12 else "Buenas tardes" if hora < 19 else "Buenas noches"
st.markdown(f"""
<div class="antetitulo">🍳 Reconocimiento de ingredientes y recomendación de recetas</div>
<div class="saludo">{saludo} 👋</div>
<div class="subsaludo">¿Qué hay en tu cocina? Toma o sube fotos de tus ingredientes y te decimos qué cocinar.</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("Cargando el modelo..."):
        modelo, clases = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo. Ejecuta primero `python train_model.py`.")
    st.exception(e)
    st.stop()

recomendador = cargar_recomendador()
ingredientes_cnn = [c for c in clases if c != CLASE_OTRO]

with st.sidebar:
    st.header("⚙️ Parámetros")
    umbral = st.slider("Confianza mínima para aceptar un ingrediente", 0.1, 0.99, 0.70, 0.05,
                       help="Las fotos por debajo de este valor no se agregan solas: tú decides.")
    top_n = st.slider("Número de recetas", 1, 10, 5)
    min_cob = st.slider("Cobertura mínima de la receta", 0.1, 1.0, 0.3, 0.05,
                        help="Fracción de los ingredientes de la receta que ya debes tener.")
    st.divider()
    st.caption(f"La CNN reconoce **{len(ingredientes_cnn)}** ingredientes. Los que no reconoce "
               "(pollo, tortilla, arroz...) se agregan a mano.")
    st.caption("Modelo: MobileNetV2 con transfer learning · Recomendador: cobertura + Jaccard + coseno.")

tab_ing, tab_nombre = st.tabs(["Ingredientes", "Nombre de receta"])

with tab_ing:
    # ---------- Fotos ----------
    st.markdown('<div class="seccion">📸 Tus fotos</div>', unsafe_allow_html=True)
    archivos = st.file_uploader("Sube una o varias fotos", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True)
    foto_cam = None
    if st.toggle("📷 Usar la cámara"):
        foto_cam = st.camera_input("Toma una foto de un ingrediente")

    imagenes = [(f.name, Image.open(f)) for f in archivos or []]
    if foto_cam is not None:
        imagenes.append(("Cámara", Image.open(foto_cam)))

    detectados = []
    if imagenes:
        cols = st.columns(4)  # siempre 4 para que una sola foto no ocupe todo el ancho
        for i, (nombre, img) in enumerate(imagenes):
            res = clasificar(modelo, clases, img, umbral)
            top, estado = res["top"], res["estado"]
            ing, conf = top[0]
            if estado == "aceptado":
                detectados.append(ing)
            with cols[i % len(cols)]:
                with st.container(border=True):
                    # Recorte del mismo tamaño para que las tarjetas queden alineadas
                    foto = ImageOps.exif_transpose(img).convert("RGB")
                    st.image(ImageOps.fit(foto, (400, 300)), width="stretch")
                    if estado == "no_ingrediente":
                        st.markdown('<div class="prediccion">🤔 No parece un ingrediente</div>'
                                    '<span class="etiqueta falta">✗ No reconocido</span>',
                                    unsafe_allow_html=True)
                        st.caption("Prueba con una foto más cercana y con buena luz, "
                                   "o agrega el ingrediente a mano abajo.")
                        continue
                    etiqueta = ('<span class="etiqueta ok">✓ Agregado</span>' if estado == "aceptado"
                                else '<span class="etiqueta baja">⚠ Confianza baja</span>')
                    st.markdown(f'<div class="prediccion">{escape(con_emoji(ing))}</div>{etiqueta}',
                                unsafe_allow_html=True)
                    st.progress(conf, text=f"Confianza: {conf:.1%}")
                    otras = " · ".join(f"{bonito(c)} {p:.1%}" for c, p in top[1:])
                    st.caption(f"Otras opciones: {otras}")
                    if estado == "confirmar":
                        st.caption("¿Cuál es? Tócalo para agregarlo:")
                        with st.container(horizontal=True):
                            for c, _ in top:
                                st.button(con_emoji(c), key=f"conf_{i}_{nombre}_{c}",
                                          on_click=agregar, args=(c,))

    # La selección vive en session_state para poder agregar ingredientes con los botones.
    # Cuando cambian las fotos se reemplazan los ingredientes detectados y se conservan los
    # que el usuario agregó a mano.
    detectados = sorted(set(detectados))
    if "sel" not in st.session_state:
        st.session_state.sel, st.session_state.prev_det = [], []
    if detectados != st.session_state.prev_det:
        manuales = set(st.session_state.sel) - set(st.session_state.prev_det)
        st.session_state.sel = sorted(manuales | set(detectados))
        st.session_state.prev_det = detectados

    # ---------- Ingredientes ----------
    st.markdown('<div class="seccion">🧺 Ingredientes seleccionados</div>', unsafe_allow_html=True)
    opciones = sorted(set(ingredientes_cnn) | set(recomendador.vocabulario))
    seleccion = st.multiselect("Ingredientes disponibles", opciones, key="sel",
                               format_func=bonito, label_visibility="collapsed",
                               placeholder="🔍 Escribe tus ingredientes: huevo, tortilla, queso...")
    st.caption("Sal, agua, aceite, pimienta y azúcar se consideran siempre disponibles.")

    if seleccion:
        titulo, lista, clave = "Combina bien con:", combina_bien_con(seleccion, recomendador.recetas), "combina"
    else:
        titulo, lista, clave = "Sugeridos:", SUGERIDOS, "sugeridos"
    if lista:
        st.markdown(f"**{titulo}**")
        # Contenedor horizontal: los botones se acomodan en filas también en el teléfono
        with st.container(key=clave, horizontal=True):
            for ing in lista:
                st.button(con_emoji(ing), key=f"add_{ing}", on_click=agregar, args=(ing,))

    st.write("")
    with st.container(key="cocinar"):
        if st.button("🍳 Recomendar recetas", type="primary", disabled=not seleccion):
            st.session_state.mostrar = True

    # ---------- Recetas ----------
    if seleccion and st.session_state.get("mostrar"):
        recetas = recomendador.recomendar(seleccion, top_n=top_n, min_cobertura=min_cob)
        st.markdown('<div class="seccion">🍽️ Recetas para ti</div>', unsafe_allow_html=True)
        if not recetas:
            # Con uno o dos ingredientes casi ninguna receta llega a la cobertura mínima:
            # en vez de no mostrar nada, se muestran las más cercanas
            recetas = recomendador.recomendar(seleccion, top_n=top_n, min_cobertura=0)
            if recetas:
                st.info(f"Ninguna receta llega al {min_cob:.0%} de ingredientes que ya tienes. "
                        "Estas son las más cercanas; agrega más ingredientes para mejores resultados.")
            else:
                st.info("No hay recetas con esos ingredientes. Prueba agregar otros.")
        cols = st.columns(2)
        for i, r in enumerate(recetas):
            with cols[i % 2]:
                tarjeta_receta(r["nombre"], r["tiempo_min"], set(r["tengo"]) | set(r["faltan"]),
                               r["pasos"], r["tengo"], r["faltan"], r["cobertura"],
                               datos=f"Puntuación {r['puntuacion']:.2f}")

with tab_nombre:
    st.markdown('<div class="seccion">📖 Busca una receta</div>', unsafe_allow_html=True)
    texto = st.text_input("Nombre de la receta", placeholder="🔍 Escribe el nombre: tacos, ensalada, agua...",
                          label_visibility="collapsed")
    buscadas = [r for r in recomendador.recetas if normalizar(texto) in normalizar(r["nombre"])]
    buscadas.sort(key=lambda r: r["nombre"])
    st.caption(f"{len(buscadas)} de {len(recomendador.recetas)} recetas")
    cols = st.columns(2)
    for i, r in enumerate(buscadas):
        with cols[i % 2]:
            tarjeta_receta(r["nombre"], r.get("tiempo_min"), r["_ings"], r.get("pasos", []))
