"""End-to-end training pipeline: download → train → evaluate → save figures.

Usage:
    python run_training.py
"""

from __future__ import annotations

import os

# ── Environment setup (must precede TF import) ──────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf  # noqa: E402

# Prevent GPU OOM on shared machines
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from waste_classifier.config import PATHS, TRAIN_CFG  # noqa: E402
from waste_classifier.data import build_datasets, download_dataset, load_test_images  # noqa: E402
from waste_classifier.evaluate import (  # noqa: E402
    evaluate_model,
    plot_confusion_matrix,
    plot_grad_cam,
    plot_prediction_grid,
    plot_roc_comparison,
    plot_training_curves,
)
from waste_classifier.model import build_model, print_trainable_summary  # noqa: E402
from waste_classifier.train import train_phase  # noqa: E402


def main() -> None:
    # ── 1. Data ──────────────────────────────────────────────────────
    download_dataset()
    train_ds, val_ds, test_ds = build_datasets()
    test_images, test_labels, _ = load_test_images()

    figs = PATHS.figures_dir
    figs.mkdir(parents=True, exist_ok=True)

    # ── 2. Feature Extraction ────────────────────────────────────────
    model_ext = build_model(trainable_base=False)
    print_trainable_summary(model_ext)

    hist_ext = train_phase(
        model_ext, train_ds, val_ds,
        learning_rate=TRAIN_CFG.extract_lr,
        checkpoint_name=TRAIN_CFG.extract_checkpoint,
        phase_name="Phase 1 — Feature Extraction",
    )
    plot_training_curves(hist_ext, "Feature Extraction", save_path=figs / "curves_extract.png")

    # ── 3. Fine-Tuning ───────────────────────────────────────────────
    model_ft = build_model(trainable_base=True, unfreeze_from=TRAIN_CFG.unfreeze_from)
    print_trainable_summary(model_ft)

    hist_ft = train_phase(
        model_ft, train_ds, val_ds,
        learning_rate=TRAIN_CFG.finetune_lr,
        checkpoint_name=TRAIN_CFG.finetune_checkpoint,
        phase_name="Phase 2 — Fine-Tuning",
    )
    plot_training_curves(hist_ft, "Fine-Tuning", save_path=figs / "curves_finetune.png")

    # ── 4. Evaluation ────────────────────────────────────────────────
    res_ext = evaluate_model(model_ext, test_images, test_labels, "Feature Extraction")
    res_ft = evaluate_model(model_ft, test_images, test_labels, "Fine-Tuned")

    plot_confusion_matrix(res_ext["y_true"], res_ext["y_pred"], "CM — Feature Extraction", figs / "cm_extract.png")
    plot_confusion_matrix(res_ft["y_true"], res_ft["y_pred"], "CM — Fine-Tuned", figs / "cm_finetune.png")

    plot_roc_comparison(
        {"Feature Extraction": res_ext, "Fine-Tuned": res_ft},
        save_path=figs / "roc_comparison.png",
    )

    plot_prediction_grid(model_ft, test_images, test_labels, "Fine-Tuned", save_path=figs / "predictions_grid.png")

    # ── 5. Grad-CAM ──────────────────────────────────────────────────
    for i in range(3):
        plot_grad_cam(model_ft, test_images[i], test_labels[i], "Fine-Tuned", save_path=figs / f"gradcam_{i}.png")

    print("\n✓ Training complete. Figures saved to:", figs)


if __name__ == "__main__":
    main()
