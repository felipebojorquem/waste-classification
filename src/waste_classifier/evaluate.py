"""Evaluation: metrics, confusion matrix, ROC curve, Grad-CAM visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from waste_classifier.config import DATA_CFG

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})

LABEL_MAP = DATA_CFG.class_labels  # {0: "Organic", 1: "Recyclable"}


# ── Training curves ──────────────────────────────────────────────────

def plot_training_curves(
    history: tf.keras.callbacks.History,
    title_suffix: str = "",
    save_path: Path | None = None,
) -> None:
    """Plot loss and accuracy curves side by side."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history.history["loss"]) + 1)

    ax_loss.plot(epochs, history.history["loss"], label="Train")
    ax_loss.plot(epochs, history.history["val_loss"], label="Validation")
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title=f"Loss {title_suffix}")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history.history["accuracy"], label="Train")
    ax_acc.plot(epochs, history.history["val_accuracy"], label="Validation")
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy", title=f"Accuracy {title_suffix}")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.suptitle(title_suffix, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


# ── Classification report ────────────────────────────────────────────

def evaluate_model(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Run predictions and return a full evaluation dict.

    Returns dict with keys: y_true, y_pred, y_prob, report_str.
    """
    y_prob = model.predict(images, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        labels, y_pred, target_names=["Organic", "Recyclable"]
    )
    auc = roc_auc_score(labels, y_prob)

    print(f"\n{'─' * 50}")
    print(f"  {model_name}  |  AUC = {auc:.4f}")
    print(f"{'─' * 50}")
    print(report)

    return {"y_true": labels, "y_pred": y_pred, "y_prob": y_prob, "auc": auc, "report": report}


# ── Confusion matrix ─────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Organic", "Recyclable"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


# ── ROC curve ────────────────────────────────────────────────────────

def plot_roc_comparison(
    results: dict[str, dict],
    save_path: Path | None = None,
) -> None:
    """Overlay ROC curves from multiple models.

    Args:
        results: {model_name: eval_dict} from ``evaluate_model()``.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


# ── Grad-CAM ─────────────────────────────────────────────────────────

def grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    layer_name: str = "block5_conv3",
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image.

    Args:
        model: Trained classifier.
        image: (H, W, 3) float32 in [0, 1].
        layer_name: Target convolutional layer inside the VGG16 base.

    Returns:
        heatmap: (H, W) float32 in [0, 1].
    """
    # Navigate into nested base model: model = [Input, VGG16, GAP, Dense, ..., Dense]
    base = model.layers[1]  # VGG16 sub-model
    conv_layer = base.get_layer(layer_name)
    head_layers = model.layers[2:]

    # grad_model wraps the base only; the head is reapplied manually inside the tape
    # so gradients propagate from the final prediction back to the conv feature map.
    grad_model = tf.keras.Model(
        inputs=base.inputs,
        outputs=[conv_layer.output, base.output],
    )

    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(img_tensor)
        x = base_out
        for layer in head_layers:
            x = layer(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # GAP of gradients
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    # Resize to input dimensions
    cam = tf.image.resize(cam[..., np.newaxis], image.shape[:2]).numpy().squeeze()
    return cam


def plot_grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    true_label: int,
    model_name: str = "Model",
    save_path: Path | None = None,
) -> None:
    """Display original image alongside Grad-CAM overlay."""
    heatmap = grad_cam(model, image)
    y_prob = model.predict(image[np.newaxis], verbose=0).ravel()[0]
    pred_label = int(y_prob >= 0.5)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.imshow(image)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(heatmap, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap")
    ax2.axis("off")

    ax3.imshow(image)
    ax3.imshow(heatmap, cmap="jet", alpha=0.4)
    correct = true_label == pred_label
    color = "green" if correct else "red"
    ax3.set_title(
        f"True: {LABEL_MAP[true_label]} | Pred: {LABEL_MAP[pred_label]} ({y_prob:.2f})",
        color=color,
        fontweight="bold",
    )
    ax3.axis("off")

    plt.suptitle(f"{model_name} — Grad-CAM", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


# ── Sample predictions grid ──────────────────────────────────────────

def plot_prediction_grid(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
    n_samples: int = 10,
    save_path: Path | None = None,
) -> None:
    """Show a grid of predictions with color-coded correctness."""
    y_prob = model.predict(images[:n_samples], verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cols = 5
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))

    for i, ax in enumerate(axes.flat):
        if i >= n_samples:
            ax.axis("off")
            continue

        ax.imshow(images[i])
        correct = labels[i] == y_pred[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"T:{LABEL_MAP[labels[i]]}\nP:{LABEL_MAP[y_pred[i]]} ({y_prob[i]:.2f})",
            fontsize=8, color=color,
        )
        ax.axis("off")

    plt.suptitle(f"{model_name} — Sample Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)
