"""
Entrenamiento de la CNN (transfer learning con MobileNetV2).

Estructura esperada:
    data/train/<ingrediente>/*.jpg
    data/val/<ingrediente>/*.jpg
    data/test/<ingrediente>/*.jpg

Uso:
    python train_model.py --data data --epochs 10 --finetune_epochs 5

Genera en modelo/:
    modelo_ingredientes.keras, clases.json, metricas.json,
    curvas_entrenamiento.png, matriz_confusion.png, reporte_clasificacion.txt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
SALIDA = Path("modelo")


def cargar_datos(data_dir, batch):
    data_dir = Path(data_dir)
    kw = dict(image_size=IMG_SIZE, batch_size=batch, label_mode="int")
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", shuffle=True, seed=42, **kw)
    val = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", shuffle=False, **kw)
    test = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **kw)
    clases = train.class_names
    auto = tf.data.AUTOTUNE
    return (train.cache().prefetch(auto), val.cache().prefetch(auto),
            test.cache().prefetch(auto), clases)


def construir_modelo(n_clases):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False

    aumento = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="aumento_datos")

    entrada = layers.Input(shape=IMG_SIZE + (3,))        # pixeles 0-255
    x = aumento(entrada)                                  # solo activo al entrenar
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_clases, activation="softmax")(x)
    return models.Model(entrada, salida), base


def graficar_curvas(hist, ruta):
    plt.figure(figsize=(11, 4))
    for i, (m, t) in enumerate([("accuracy", "Exactitud"), ("loss", "Perdida")]):
        plt.subplot(1, 2, i + 1)
        plt.plot(hist[m], label="train")
        plt.plot(hist["val_" + m], label="val")
        plt.title(t)
        plt.xlabel("Epoca")
        plt.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_matriz(cm, clases, ruta):
    n = len(clases)
    plt.figure(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de confusion (test)")
    plt.xticks(range(n), clases, rotation=90)
    plt.yticks(range(n), clases)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--finetune_epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    SALIDA.mkdir(exist_ok=True)
    train, val, test, clases = cargar_datos(a.data, a.batch)
    print("Clases:", clases)

    modelo, base = construir_modelo(len(clases))
    stop = tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_accuracy")

    # Fase 1: solo la cabeza clasificadora
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = modelo.fit(train, validation_data=val, epochs=a.epochs, callbacks=[stop])

    # Fase 2: fine-tuning de las ultimas 30 capas del modelo base
    base.trainable = True
    for capa in base.layers[:-30]:
        capa.trainable = False
    # Las BatchNormalization se dejan congeladas: si se descongelan, recalculan
    # sus estadisticas con cada lote (aunque se llame con training=False) y el
    # modelo pierde exactitud al iniciar el fine-tuning
    for capa in base.layers[-30:]:
        if isinstance(capa, layers.BatchNormalization):
            capa.trainable = False
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = modelo.fit(train, validation_data=val, epochs=a.finetune_epochs,
                    callbacks=[stop])

    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    graficar_curvas(hist, SALIDA / "curvas_entrenamiento.png")

    # Evaluacion final en test
    loss, acc = modelo.evaluate(test, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = np.argmax(modelo.predict(test, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(clases)))
    graficar_matriz(cm, clases, SALIDA / "matriz_confusion.png")
    reporte = classification_report(y_true, y_pred, labels=range(len(clases)),
                                    target_names=clases, zero_division=0)
    (SALIDA / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")
    print(reporte)

    modelo.save(SALIDA / "modelo_ingredientes.keras")
    (SALIDA / "clases.json").write_text(
        json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")
    (SALIDA / "metricas.json").write_text(json.dumps(
        {"test_accuracy": float(acc), "test_loss": float(loss),
         "n_clases": len(clases), "epocas_totales": len(hist["loss"])},
        indent=2))
    print(f"\nTest accuracy: {acc:.4f}  |  Modelo guardado en {SALIDA}/")


if __name__ == "__main__":
    main()
