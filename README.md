# Reconocimiento de ingredientes y recomendación de recetas (PIA – Proyecto 11)

Sistema que identifica ingredientes en fotos con una **CNN (MobileNetV2, transfer learning)**
y recomienda recetas con un **sistema de recomendación por similitud**
(cobertura + Jaccard + coseno sobre vectores binarios de ingredientes).

## Estructura
```
sistema/
├── app.py                  # Interfaz Streamlit
├── train_model.py          # Entrenamiento y evaluación de la CNN
├── recommender.py          # Motor de recomendación
├── dividir_dataset.py      # Divide dataset crudo en train/val/test
├── pruebas.py              # Pruebas por lote -> resultados_pruebas.csv
├── recetas.json            # Base de 30 recetas
├── requirements.txt
├── data/                   # train/ val/ test/ (tu dataset)
├── modelo/                 # modelo entrenado, clases.json, métricas, gráficas
└── imagenes_prueba/        # fotos propias para las pruebas
```

## Instalación
Requiere Python 3.10 – 3.12.
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Paso 1: preparar el dataset
Una carpeta por ingrediente, nombre en minúsculas y sin acentos (ver `modelo/clases_sugeridas.txt`).
Fuentes: Fruits-360 (Kaggle), Fruits and Vegetables Image Recognition (Kaggle), Roboflow Universe,
y **fotos propias** (recomendado, 30+ por clase).
```bash
python dividir_dataset.py --origen crudo --destino data
```
Recomendación: usa solo los ingredientes que realmente tengas imágenes; las recetas con
ingredientes fuera de las clases de la CNN siguen funcionando porque el usuario puede
agregarlos manualmente en la app (leche, etc.).

## Paso 2: entrenar
```bash
python train_model.py --data data --epochs 10 --finetune_epochs 5
```
Genera en `modelo/`: `modelo_ingredientes.keras`, `clases.json`, `metricas.json`,
`curvas_entrenamiento.png`, `matriz_confusion.png`, `reporte_clasificacion.txt`
(todo esto va al reporte técnico).

## Paso 3: ejecutar la app
```bash
streamlit run app.py
```
Sube fotos o usa la cámara → revisa/edita ingredientes → "Recomendar recetas".

## Paso 4: pruebas para el reporte
Pon fotos propias en `imagenes_prueba/` nombradas `ingrediente_descripcion.jpg`
(p. ej. `tomate_luz_baja.jpg`, `cebolla_fondo_cocina.jpg`) y ejecuta:
```bash
python pruebas.py imagenes_prueba
```

## Modelos y librerías usadas
- MobileNetV2 preentrenado en ImageNet (Keras Applications), reentrenado con transfer learning.
- TensorFlow/Keras, scikit-learn (métricas), Streamlit (interfaz), Pillow, NumPy, Matplotlib.
