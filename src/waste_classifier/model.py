"""Model construction: VGG16 base + custom classification head.

VGG16 (Visual Geometry Group, 2014) is used intentionally as a well-understood
baseline for transfer learning.  For production, consider EfficientNetV2 or
ConvNeXt which offer better accuracy / FLOPs trade-offs.  The modular design
here makes swapping the backbone a one-line change.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from waste_classifier.config import MODEL_CFG


def _build_head(base_output: tf.Tensor) -> tf.Tensor:
    """Classification head: GAP → Dense blocks → sigmoid output."""
    x = GlobalAveragePooling2D()(base_output)
    for units in MODEL_CFG.head_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(MODEL_CFG.dropout_rate)(x)
    return Dense(1, activation="sigmoid", dtype="float32")(x)


def build_model(trainable_base: bool = False, unfreeze_from: str | None = None) -> tf.keras.Model:
    """Build the VGG16-based binary classifier.

    Args:
        trainable_base: If False, all base layers are frozen (feature extraction).
        unfreeze_from: Layer name from which to unfreeze (inclusive).
            Only used when ``trainable_base=True``.

    Returns:
        Compiled-ready ``tf.keras.Model``.
    """
    base = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=MODEL_CFG.input_shape,
    )

    # Freeze strategy
    if not trainable_base:
        base.trainable = False
    else:
        _selective_unfreeze(base, unfreeze_from or "block5_conv1")

    inputs = tf.keras.Input(shape=MODEL_CFG.input_shape)
    x = base(inputs, training=False)  # BN in inference mode (ImageNet stats)
    outputs = _build_head(x)

    return tf.keras.Model(inputs, outputs, name="waste_classifier")


def _selective_unfreeze(base: tf.keras.Model, from_layer: str) -> None:
    """Unfreeze layers from ``from_layer`` onwards, keeping BN frozen."""
    unfreeze = False
    for layer in base.layers:
        if layer.name == from_layer:
            unfreeze = True
        layer.trainable = unfreeze
        # BatchNormalization must stay frozen — its running stats are from ImageNet
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def print_trainable_summary(model: tf.keras.Model) -> None:
    """Print a concise trainable / frozen parameter count."""
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    total = trainable + frozen
    print(f"Total params:     {total:>12,}")
    print(f"Trainable:        {trainable:>12,}  ({trainable / total:.1%})")
    print(f"Frozen:           {frozen:>12,}  ({frozen / total:.1%})")
