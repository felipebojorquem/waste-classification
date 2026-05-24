"""Training pipeline: feature extraction → fine-tuning, with callbacks."""

from __future__ import annotations

import tensorflow as tf

from waste_classifier.config import PATHS, TRAIN_CFG


def get_callbacks(checkpoint_name: str, patience: int = TRAIN_CFG.patience) -> list[tf.keras.callbacks.Callback]:
    """Standard callback stack: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau."""
    ckpt_path = PATHS.models_dir / checkpoint_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            min_delta=TRAIN_CFG.min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience - 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_cls: type = tf.keras.optimizers.Adam,
) -> None:
    """Compile with binary crossentropy + AUC metric."""
    model.compile(
        optimizer=optimizer_cls(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    learning_rate: float,
    checkpoint_name: str,
    epochs: int = TRAIN_CFG.epochs,
    phase_name: str = "Training",
) -> tf.keras.callbacks.History:
    """Run a single training phase (feature extraction OR fine-tuning).

    Returns the History object for metric analysis.
    """
    print(f"\n{'═' * 60}")
    print(f"  {phase_name}")
    print(f"{'═' * 60}")

    compile_model(model, learning_rate)
    callbacks = get_callbacks(checkpoint_name)

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
        verbose=1,
    )

    return history


def load_model(checkpoint_name: str) -> tf.keras.Model:
    """Load a saved model from the models directory."""
    path = PATHS.models_dir / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return tf.keras.models.load_model(str(path))
