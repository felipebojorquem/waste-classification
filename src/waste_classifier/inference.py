"""Inference: single-image prediction with Grad-CAM support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from waste_classifier.config import DATA_CFG
from waste_classifier.evaluate import grad_cam

CLASS_LABELS = DATA_CFG.class_labels  # {0: "Organic", 1: "Recyclable"}


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """Load a single image and preprocess for model input.

    Returns (H, W, 3) float32 array in [0, 1].
    """
    img = tf.keras.utils.load_img(str(image_path), target_size=DATA_CFG.img_size)
    arr = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
    return arr


def predict(
    model: tf.keras.Model,
    image: np.ndarray | str | Path,
    with_gradcam: bool = False,
) -> dict:
    """Run prediction on a single image.

    Args:
        model: Trained waste classifier.
        image: (H, W, 3) array in [0, 1], or a file path.
        with_gradcam: If True, also return Grad-CAM heatmap.

    Returns:
        dict with keys: label, label_id, confidence, heatmap (optional).
    """
    if isinstance(image, (str, Path)):
        image = load_and_preprocess(image)

    prob = model.predict(image[np.newaxis], verbose=0).ravel()[0]
    label_id = int(prob >= 0.5)
    confidence = prob if label_id == 1 else 1.0 - prob

    result = {
        "label": CLASS_LABELS[label_id],
        "label_id": label_id,
        "confidence": float(confidence),
        "raw_prob": float(prob),
    }

    if with_gradcam:
        result["heatmap"] = grad_cam(model, image)

    return result
