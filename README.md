# Reconocimiento de ingredientes y recomendación de recetas (PIA – Proyecto 11)

Sistema que identifica ingredientes en fotos con una **CNN (MobileNetV2, transfer learning)**
y recomienda recetas con un **sistema de recomendación por similitud**
(cobertura + Jaccard + coseno sobre vectores binarios de ingredientes).

## Resultados

| | |
|---|---|
| Ingredientes que reconoce la CNN | **32** frutas y verduras |
| Exactitud en prueba | **84.9 %** (470 imágenes que el modelo nunca vio) |
| Exactitud en validación | 84.8 % |
| Recetas | **69** |

La CNN reconoce frutas y verduras. Los ingredientes de despensa (huevo, pollo, queso, arroz,
frijol, leche, pasta, tortilla, pan, crema…) se agregan a mano en la app.

Métricas por clase en `sistema/modelo/reporte_clasificacion.txt`, gráficas en
`sistema/modelo/curvas_entrenamiento.png` y `sistema/modelo/matriz_confusion.png`, y el detalle
de cada época en `sistema/modelo/entrenamiento.log`.

## Estructura
```
sistema/
├── app.py                  # Interfaz Streamlit
├── train_model.py          # Entrenamiento y evaluación de la CNN
├── recommender.py          # Motor de recomendación
├── preparar_kaggle.py      # Convierte el dataset de Kaggle y elimina duplicados
├── dividir_dataset.py      # Divide el dataset en train/val/test (70/15/15)
├── pruebas.py              # Pruebas por lote -> resultados_pruebas.csv
├── recetas.json            # Base de 69 recetas
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

### 2. Entrenar
```bash
python train_model.py --data data --epochs 10 --finetune_epochs 5
```
Dos fases: primero solo el clasificador (base congelada) y después fine-tuning de las últimas
30 capas de MobileNetV2 con las capas BatchNormalization congeladas. En CPU tarda unos 15 minutos.

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

## Modelos y librerías usadas
- MobileNetV2 preentrenado en ImageNet (Keras Applications), reentrenado con transfer learning.
- TensorFlow/Keras, scikit-learn (métricas), Streamlit (interfaz), Pillow, NumPy, Matplotlib.

## Limitaciones y trabajo futuro
- Reconoce un ingrediente por foto.
- Las clases que más confunde son rábano, papa (con manzana), manzana y pera.
- Ampliar la CNN a ingredientes que hoy se agregan a mano: de despensa (tortilla, pan, huevo,
  queso) y frutas y verduras que no vienen en el dataset (aguacate, fresa, brócoli, calabacita,
  champiñón).
