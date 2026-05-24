"""Gradio demo — Waste Classification with Grad-CAM explainability.

Run locally:
    python app/app.py

Deploy to HuggingFace Spaces:
    Copy this file + src/ + models/ to a new HF Space (Gradio SDK).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")

# ── Resolve paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fine_tuned_vgg16.keras"
IMG_SIZE = (150, 150)
CLASS_LABELS = {0: "Organic ♻️", 1: "Recyclable 🗑️"}


# ── Load model once at startup ───────────────────────────────────────
print(f"Loading model from {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("✓ Model loaded")


def preprocess(image: np.ndarray) -> np.ndarray:
    """Resize and normalize a PIL/numpy image."""
    img = tf.image.resize(image, IMG_SIZE).numpy().astype("float32") / 255.0
    return img


def compute_gradcam(img: np.ndarray, layer_name: str = "block5_conv3") -> np.ndarray:
    """Generate Grad-CAM heatmap."""
    base = model.layers[1]
    conv_layer = base.get_layer(layer_name)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    img_tensor = tf.expand_dims(tf.cast(img, tf.float32), 0)

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_tensor)
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    cam = tf.image.resize(cam[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    return cam


def classify(image: np.ndarray) -> tuple[dict, plt.Figure]:
    """Classify waste image and return label + Grad-CAM visualization."""
    img = preprocess(image)
    prob = model.predict(img[np.newaxis], verbose=0).ravel()[0]

    label_id = int(prob >= 0.5)
    confidences = {
        CLASS_LABELS[1]: float(prob),
        CLASS_LABELS[0]: float(1 - prob),
    }

    # Grad-CAM overlay
    heatmap = compute_gradcam(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(img)
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(img)
    ax2.imshow(heatmap, cmap="jet", alpha=0.4)
    ax2.set_title(f"Grad-CAM → {CLASS_LABELS[label_id]}")
    ax2.axis("off")

    plt.tight_layout()

    return confidences, fig


# ── Gradio interface ─────────────────────────────────────────────────
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", label="Upload waste image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Classification"),
        gr.Plot(label="Grad-CAM Explainability"),
    ],
    title="♻️ Waste Classification — Organic vs Recyclable",
    description=(
        "Binary classifier using VGG16 transfer learning with fine-tuning. "
        "Upload an image of waste to classify it as **Organic** or **Recyclable**. "
        "The Grad-CAM heatmap shows which regions influenced the decision."
    ),
    examples=[],  # Add example images here after deployment
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
