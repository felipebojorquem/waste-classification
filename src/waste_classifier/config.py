"""Centralized configuration for the waste classification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """All project paths relative to project root."""

    root: Path = _ROOT
    data_dir: Path = field(default_factory=lambda: _ROOT / "data" / "o-vs-r-split")
    train_dir: Path = field(default_factory=lambda: _ROOT / "data" / "o-vs-r-split" / "train")
    test_dir: Path = field(default_factory=lambda: _ROOT / "data" / "o-vs-r-split" / "test")
    models_dir: Path = field(default_factory=lambda: _ROOT / "models")
    outputs_dir: Path = field(default_factory=lambda: _ROOT / "outputs")
    figures_dir: Path = field(default_factory=lambda: _ROOT / "outputs" / "figures")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing parameters."""

    img_size: tuple[int, int] = (150, 150)
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42
    class_names: tuple[str, str] = ("O", "R")
    class_labels: dict[int, str] = field(
        default_factory=lambda: {0: "Organic", 1: "Recyclable"}
    )
    dataset_url: str = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
    )


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-3

    # Feature extraction phase
    extract_lr: float = 1e-4
    extract_checkpoint: str = "extract_features_vgg16.keras"

    # Fine-tuning phase
    finetune_lr: float = 1e-5
    finetune_checkpoint: str = "fine_tuned_vgg16.keras"
    unfreeze_from: str = "block5_conv1"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters."""

    base_model: str = "VGG16"
    input_shape: tuple[int, int, int] = (150, 150, 3)
    head_units: tuple[int, ...] = (256,)
    dropout_rate: float = 0.4


# ── Singleton instances ──────────────────────────────────────────────
PATHS = Paths()
DATA_CFG = DataConfig()
TRAIN_CFG = TrainConfig()
MODEL_CFG = ModelConfig()
