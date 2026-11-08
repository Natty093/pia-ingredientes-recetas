# Reconocimiento de ingredientes y recomendación de recetas (PIA – Proyecto 11)

Sistema que identifica ingredientes en fotos con una **CNN (MobileNetV2, transfer learning)**
y recomienda recetas con un **sistema de recomendación por similitud**
(cobertura + Jaccard + coseno sobre vectores binarios de ingredientes).

## Resultados

| | |
|---|---|
| Ingredientes que reconoce la CNN | **36** frutas y verduras |
| Exactitud en prueba | **90.0 %** (806 imágenes que el modelo nunca vio) |
| Exactitud en validación | 91.8 % |
| Recetas | **101** |

El modelo incluido reconoce frutas y verduras. Los ingredientes de despensa (huevo, pollo, queso, arroz,
frijol, leche, pasta, tortilla, pan, crema…) se agregan a mano en la app. Para que la CNN también
reconozca huevo, queso, leche y harina, reentrénala con el dataset de Roboflow (ver *Reentrenar el modelo*).

Métricas por clase en `sistema/modelo/reporte_clasificacion.txt`, gráficas en
`sistema/modelo/curvas_entrenamiento.png` y `sistema/modelo/matriz_confusion.png`, y el detalle
de cada época en `sistema/modelo/entrenamiento.log`.

## Estructura
```
sistema/
├── app.py                  # Interfaz Streamlit
├── train_model.py          # Entrenamiento y evaluación de la CNN
├── recommender.py          # Motor de recomendación
├── vision.py               # Preparación de la foto (EXIF) y clasificación con estado
├── preparar_kaggle.py      # Convierte el dataset de Kaggle y elimina duplicados
├── preparar_roboflow.py    # Agrega huevo, queso, leche y harina desde Roboflow
├── dividir_dataset.py      # Divide el dataset en train/val/test (70/15/15)
├── agregar_vegetales.py    # Agrega el segundo dataset (Vegetable Image Dataset)
├── pruebas.py              # Pruebas por lote -> resultados_pruebas.csv
├── recetas.json            # Base de 101 recetas
├── requirements.txt
├── data/                   # train/ val/ test/ (se genera, no se sube a git)
├── modelo/                 # modelo entrenado, clases.json, métricas, gráficas, log
└── imagenes_prueba/        # fotos propias para las pruebas
```

## Instalación
Probado con **Python 3.13** y TensorFlow 2.21 en Windows 11, sin GPU.
```bash
python -m venv C:\venvs\pia-tf
C:\venvs\pia-tf\Scripts\python.exe -m pip install -r sistema/requirements.txt
```
> En Windows conviene crear el entorno virtual en una ruta corta (como `C:\venvs\...`):
> TensorFlow tiene rutas internas muy largas y la instalación puede fallar dentro de carpetas
> con rutas largas.

## Usar la app
El modelo entrenado ya viene en `sistema/modelo/`, así que no hace falta entrenar.
```bash
cd sistema
C:\venvs\pia-tf\Scripts\streamlit.exe run app.py
```
Se abre en el navegador en `http://localhost:8501`. Sube fotos o usa la cámara → revisa o edita
los ingredientes → "Recomendar recetas".

- La primera vez Streamlit pide un correo: es opcional, deja el campo vacío y presiona Enter.
- La terminal debe quedar abierta mientras se usa la app. Si se cierra, la página muestra
  *"Connection error"*; basta con volver a ejecutar el comando y recargar.

## Reentrenar el modelo (opcional)

### 1. Preparar el dataset
Se usa el dataset de Kaggle **Fruits and Vegetables Image Recognition**
(https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition). Descomprime
el ZIP en `sistema/kaggle/` (debe quedar `kaggle/train`, `kaggle/validation`, `kaggle/test`) y ejecuta:
```bash
cd sistema
python preparar_kaggle.py --origen kaggle --destino crudo
python dividir_dataset.py --origen crudo --destino data
```
`preparar_kaggle.py` traduce las clases al español y **elimina las imágenes duplicadas**: el
dataset original trae 910 fotos repetidas entre train, validation y test (319 de las 359 de test
también están en train). Sin este paso, la misma foto queda en entrenamiento y en prueba y la
exactitud medida sale inflada. Quedan 2,915 imágenes únicas: 2,024 de entrenamiento, 421 de
validación y 470 de prueba.

### 1b. Segundo dataset: Vegetable Image Dataset (incluido en el modelo)
Después se agrega un segundo dataset de Kaggle, **Vegetable Image Dataset**
(https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset), con fotos de verduras en
mercados y cultivos. Descomprímelo fuera del repo y ejecuta:
```bash
python agregar_vegetales.py --origen "ruta/a/Vegetable Images" --data data --destino data_v2
```
Agrega 4 clases nuevas (brócoli, calabaza, papaya y ejote, 250 fotos cada una) y 150 fotos más a
8 clases que ya existían (papa, tomate, zanahoria, pepino, pimiento, col, coliflor y berenjena).
También elimina duplicados por MD5. No usa su clase *Radish*, porque es rábano blanco (daikon),
distinto del rábano rojo. Las fotos que ya estaban en `data/` no cambian de conjunto, así que el
modelo nuevo se puede comparar con el anterior usando las mismas fotos de prueba. Quedan 5,115
imágenes: 3,564 de entrenamiento, 745 de validación y 806 de prueba.

### 1c. (Opcional) Agregar huevo, queso, leche y harina
Dataset de Roboflow **recipes** (https://universe.roboflow.com/thatonenerds-space/recipes-bkrot).
Descárgalo en formato **Folder Structure** (pide crear una cuenta gratuita; revisa su licencia
antes de usarlo), descomprímelo en `sistema/roboflow/` (debe quedar `roboflow/train/eggs/...`) y,
**después** de `preparar_kaggle.py` y **antes** de `dividir_dataset.py`, ejecuta:
```bash
python preparar_roboflow.py --origen roboflow --destino crudo
```
- Toma solo `eggs`, `cheese`, `milk` y `flour` (→ huevo, queso, leche, harina). Las demás clases
  se ignoran para no mezclar en una misma clase fotos de dos datasets.
- Deja **una foto por original**: Roboflow guarda varias copias aumentadas (volteadas, con otro
  brillo) de cada foto, y si una cae en train y otra en test la exactitud sale inflada.
- Limita cada clase a 120 fotos (`--max`) para que quede parecida a las de Kaggle.

Si ya tenías una carpeta `data/`, borra `data/train`, `data/val` y `data/test` antes de volver a
ejecutar `dividir_dataset.py` (el script se niega a mezclar una división vieja con una nueva).

Para agregar un ingrediente con fotos propias (por ejemplo pollo), crea `crudo/pollo/` con
40-60 fotos tomadas con celular antes de `dividir_dataset.py`.

> **Limitación:** las clases nuevas vienen de otro dataset con otro estilo de foto (fondo,
> iluminación). El modelo puede aprender a distinguir el estilo y no solo el ingrediente: en el
> conjunto de prueba salen bien, pero con fotos de cocina reales pueden fallar más.

### 2. Entrenar
```bash
python train_model.py --data data_v2 --epochs 10 --finetune_epochs 5
```
Dos fases: primero solo el clasificador (base congelada) y después fine-tuning de las últimas
30 capas de MobileNetV2 con las capas BatchNormalization congeladas. En CPU tarda unos 25 minutos.
Con `--salida otra_carpeta` el modelo se guarda en otra carpeta sin reemplazar el de `modelo/`.

Opciones para probar otras configuraciones: `--finetune_lr` (por defecto 1e-5) y
`--capas_finetune` (por defecto 30). Compara las configuraciones con `val_accuracy` de
`modelo/metricas.json` y deja el conjunto de prueba solo para el resultado final.

Genera en `modelo/`: `modelo_ingredientes.keras`, `clases.json`, `metricas.json`,
`curvas_entrenamiento.png`, `matriz_confusion.png`, `reporte_clasificacion.txt`.

## Pruebas para el reporte
Pon fotos propias en `sistema/imagenes_prueba/` nombradas `ingrediente_descripcion.jpg`
(p. ej. `tomate_luz_baja.jpg`, `cebolla_fondo_cocina.jpg`) y ejecuta:
```bash
cd sistema
python pruebas.py imagenes_prueba
```
La tabla `resultados_pruebas.csv` incluye la columna `estado`: `aceptado`, `confirmar`
(confianza baja) o `no_ingrediente`.

### Agregar la clase "otro" (recomendado)
La CNN siempre elige uno de sus ingredientes, aunque la foto no tenga ninguno. Para que aprenda
a decir "no es un ingrediente", crea `crudo/otro/` con fotos que no sean ingredientes (mesa vacía,
manos, platos, cocina; unas 100-200), vuelve a ejecutar `dividir_dataset.py` y reentrena.
La app y `pruebas.py` ya interpretan la clase `otro` como "no reconocido".

## Modelos y librerías usadas
- MobileNetV2 preentrenado en ImageNet (Keras Applications), reentrenado con transfer learning.
- TensorFlow/Keras, scikit-learn (métricas), Streamlit (interfaz), Pillow, NumPy, Matplotlib.

## Cómo reconoce la app una foto
1. Endereza la foto según su EXIF (las fotos de celular suelen venir giradas).
2. Si la imagen es casi plana (negra, blanca, un solo color), la marca como *No reconocido*.
3. Si la confianza es mayor o igual al umbral (0.70 por defecto), agrega el ingrediente solo.
   Si es menor, muestra las 3 opciones más probables como botones para que el usuario elija.
4. Si ninguna receta llega a la cobertura mínima (pasa con 1 o 2 ingredientes), muestra las
   más cercanas con un aviso en lugar de una lista vacía.

## Versiones
- `v1-original`: versión antes de las mejoras de robustez. Para volver a ella:
  `git checkout v1-original` (o elige la etiqueta en el selector de ramas de GitHub).

## Limitaciones y trabajo futuro
- Con fotos distintas a las del dataset (fondo blanco, otra iluminación) la exactitud baja
  bastante: en una prueba con 660 fotos del dataset Fruits-360, el modelo anterior de 32 clases
  acertó 46.5 %, y con fotos del Vegetable Image Dataset bajaba a 63.6 %. Por eso se agregó ese
  segundo dataset (el modelo actual sube a 99.5 % con esas fotos). Lo que más ayudaría ahora es
  agregar fotos propias tomadas con celular (20-30 por ingrediente) y la clase `otro`.
- Reconoce un ingrediente por foto.
- Las clases que más confunde son rábano, manzana, soya y cebolla.
- Ampliar la CNN a ingredientes que hoy se agregan a mano: de despensa (tortilla, pan, huevo,
  queso) y frutas y verduras que no vienen en los datasets (aguacate, fresa, calabacita,
  champiñón).
