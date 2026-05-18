"""waste_classifier — Transfer learning for binary waste classification."""

from waste_classifier.config import DATA_CFG, MODEL_CFG, PATHS, TRAIN_CFG
from waste_classifier.data import build_datasets, download_dataset, load_test_images
from waste_classifier.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_grad_cam,
    plot_prediction_grid,
    plot_roc_comparison,
    plot_training_curves,
)
from waste_classifier.inference import predict
from waste_classifier.model import build_model, print_trainable_summary
from waste_classifier.train import load_model, train_phase

__all__ = [
    "build_datasets",
    "build_model",
    "DATA_CFG",
    "download_dataset",
    "evaluate_model",
    "load_model",
    "load_test_images",
    "MODEL_CFG",
    "PATHS",
    "plot_confusion_matrix",
    "plot_grad_cam",
    "plot_prediction_grid",
    "plot_roc_comparison",
    "plot_training_curves",
    "predict",
    "print_trainable_summary",
    "TRAIN_CFG",
    "train_phase",
]
