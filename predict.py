"""
predict.py
-----------
Loads the trained fish-species classification model (if available) and
runs inference on an uploaded image.

If no trained model is found at model/fish_model.h5, the app automatically
falls back to a clearly-labelled DEMO MODE so the website still works
end-to-end out of the box. Train a real model using train_model.py + a
fish image dataset to replace demo mode with real predictions.
"""

import os
import json
import random
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fish_model.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")
FISH_INFO_PATH = os.path.join(BASE_DIR, "fish_info.json")
IMG_SIZE = (224, 224)

with open(FISH_INFO_PATH, "r", encoding="utf-8") as f:
    FISH_INFO = json.load(f)

_model = None
_class_indices = None
_demo_mode = True


def _try_load_real_model():
    """Attempt to load a trained Keras model. Falls back to demo mode if missing."""
    global _model, _class_indices, _demo_mode

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX_PATH):
        _demo_mode = True
        return

    try:
        from tensorflow import keras  # imported lazily so demo mode has no heavy deps

        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        _class_indices = {int(k): v for k, v in idx_to_class.items()}
        _demo_mode = False
        print("[predict.py] Trained model loaded successfully. Real predictions enabled.")
    except Exception as e:  # noqa: BLE001
        print(f"[predict.py] Could not load trained model ({e}). Falling back to demo mode.")
        _demo_mode = True


_try_load_real_model()


def _preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _demo_predict(image_path):
    """
    Demo-mode predictor.

    This does NOT do real species recognition (no trained weights are shipped
    in this project because training requires a labelled fish-image dataset).
    It deterministically derives a species guess from simple image statistics
    (average colour / brightness) purely so the UI has something consistent
    and reproducible to show per image, and clearly flags itself as a demo.
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32)
    avg_color = arr.mean(axis=(0, 1))  # R, G, B
    brightness = avg_color.mean()

    species_list = list(FISH_INFO.keys())
    # Deterministic pseudo-index from image stats so the same image gives the same result
    seed_val = int(avg_color[0] + avg_color[1] * 2 + avg_color[2] * 3 + brightness)
    rng = random.Random(seed_val)
    species = rng.choice(species_list)
    confidence = round(rng.uniform(55, 78), 1)  # kept modest since this is not a real prediction

    top3 = rng.sample(species_list, k=min(3, len(species_list)))
    if species not in top3:
        top3[0] = species
    predictions = [{"species": species, "confidence": confidence}]
    remaining = [s for s in top3 if s != species]
    for s in remaining:
        predictions.append({"species": s, "confidence": round(confidence - rng.uniform(10, 30), 1)})

    return predictions


def _real_predict(image_path):
    x = _preprocess_image(image_path)
    preds = _model.predict(x, verbose=0)[0]
    top_indices = preds.argsort()[-3:][::-1]
    predictions = []
    for i in top_indices:
        species = _class_indices.get(int(i), f"Class {i}")
        confidence = round(float(preds[i]) * 100, 1)
        predictions.append({"species": species, "confidence": confidence})
    return predictions


def predict_species(image_path):
    """
    Returns a dict:
    {
        "demo_mode": bool,
        "predictions": [{"species": str, "confidence": float}, ...],
        "top_species_info": {...details from fish_info.json...} or None
    }
    """
    if _demo_mode:
        predictions = _demo_predict(image_path)
    else:
        predictions = _real_predict(image_path)

    top_species = predictions[0]["species"]
    info = FISH_INFO.get(top_species)

    return {
        "demo_mode": _demo_mode,
        "predictions": predictions,
        "top_species_info": info,
    }
