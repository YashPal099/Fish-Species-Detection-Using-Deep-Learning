"""
train_model.py
---------------
Trains a real fish species classifier using transfer learning
(MobileNetV2 pretrained on ImageNet) on your own fish image dataset.

WHY YOU NEED TO RUN THIS YOURSELF:
A machine-learning model needs a labelled dataset of fish images to learn
from. No such dataset is bundled with this project (image datasets are
large and licensing varies), so this script is provided so you can point
it at a dataset and produce model/fish_model.h5 + model/class_indices.json.
Once those two files exist, app.py automatically switches from demo mode
to real predictions - no other code changes needed.

RECOMMENDED FREE DATASETS:
- "A Large Scale Fish Dataset" (Kaggle)
- "Fish species image data" (Kaggle)
- Any folder of images organised as:

    dataset/
        Rohu/
            img1.jpg
            img2.jpg
        Catla/
            img1.jpg
            ...
        Tilapia/
            ...

USAGE:
    pip install -r ../requirements.txt
    python train_model.py --data_dir /path/to/dataset --epochs 15
"""

import os
import json
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for transfer learning

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset folder (class-per-subfolder)")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    train_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(BASE_DIR, "fish_model.h5"), save_best_only=True
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Optional fine-tuning pass: unfreeze the base model and train a bit more
    # with a low learning rate for better accuracy.
    base_model = model.layers[2]
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

    model.save(os.path.join(BASE_DIR, "fish_model.h5"))

    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(os.path.join(BASE_DIR, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)

    print("\nTraining complete!")
    print("Saved model/fish_model.h5 and model/class_indices.json")
    print("Restart the Flask app - it will now use real predictions instead of demo mode.")


if __name__ == "__main__":
    main()
