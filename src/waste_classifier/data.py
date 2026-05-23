"""Dataset downloading, preprocessing, and tf.data pipeline construction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from waste_classifier.config import DATA_CFG, PATHS


# ── Download & extract ───────────────────────────────────────────────

def download_dataset(url: str = DATA_CFG.dataset_url, dest: Path = PATHS.root / "data") -> Path:
    """Download and extract the dataset if not already present.

    Returns the path to the extracted directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extract_dir = dest / "o-vs-r-split"

    if extract_dir.exists():
        print(f"✓ Dataset already exists at {extract_dir}")
        return extract_dir

    zip_path = dest / "dataset.zip"
    print(f"Downloading dataset → {zip_path}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    print(f"✓ Dataset ready at {extract_dir}")
    return extract_dir


# ── Augmentation layers ──────────────────────────────────────────────

def build_augmentation() -> tf.keras.Sequential:
    """Augmentation pipeline — only active during training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


# ── tf.data pipelines ────────────────────────────────────────────────

def build_datasets(
    train_dir: Path | str = PATHS.train_dir,
    test_dir: Path | str = PATHS.test_dir,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train, validation, and test tf.data.Dataset pipelines.

    - Training: augmentation + rescaling + cache + prefetch.
    - Validation / Test: rescaling + cache + prefetch.
    """
    train_dir, test_dir = str(train_dir), str(test_dir)
    autotune = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=DATA_CFG.val_split,
        subset="training",
        seed=DATA_CFG.seed,
        image_size=DATA_CFG.img_size,
        batch_size=DATA_CFG.batch_size,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=DATA_CFG.val_split,
        subset="validation",
        seed=DATA_CFG.seed,
        image_size=DATA_CFG.img_size,
        batch_size=DATA_CFG.batch_size,
        label_mode="binary",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=DATA_CFG.img_size,
        batch_size=DATA_CFG.batch_size,
        label_mode="binary",
        shuffle=False,
    )

    rescaling = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = build_augmentation()

    # Augmentation goes *after* cache so each epoch sees fresh random transforms.
    train_ds = (
        train_ds
        .map(lambda x, y: (rescaling(x), y), num_parallel_calls=autotune)
        .cache()
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    val_ds = (
        val_ds
        .map(lambda x, y: (rescaling(x), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(autotune)
    )

    test_ds = (
        test_ds
        .map(lambda x, y: (rescaling(x), y), num_parallel_calls=autotune)
        .cache()
        .prefetch(autotune)
    )

    return train_ds, val_ds, test_ds


def load_test_images(
    test_dir: Path | str = PATHS.test_dir,
    n_per_class: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load raw test images as numpy arrays for visualization and evaluation.

    Returns:
        images: (N, H, W, 3) float32 in [0, 1].
        labels: (N,) int array (0 = Organic, 1 = Recyclable).
        filenames: list of file paths.
    """
    test_dir = Path(test_dir)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    def _list(subdir: str) -> list[Path]:
        return sorted(
            p for p in (test_dir / subdir).iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        )[:n_per_class]

    files_o = _list("O")
    files_r = _list("R")
    all_files = files_o + files_r

    images = np.array([
        tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(str(f), target_size=DATA_CFG.img_size)
        )
        for f in all_files
    ])
    images = images.astype("float32") / 255.0

    labels = np.array([0] * len(files_o) + [1] * len(files_r))
    filenames = [str(f) for f in all_files]

    return images, labels, filenames
