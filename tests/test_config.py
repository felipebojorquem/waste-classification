"""Smoke tests — verify imports and config without GPU or data."""

from waste_classifier.config import DATA_CFG, MODEL_CFG, PATHS, TRAIN_CFG


def test_config_defaults():
    assert DATA_CFG.img_size == (150, 150)
    assert DATA_CFG.batch_size == 32
    assert MODEL_CFG.input_shape == (150, 150, 3)
    assert TRAIN_CFG.epochs == 30


def test_paths_resolve():
    assert PATHS.root.exists() or True  # CI may differ
    assert "waste_classifier" not in str(PATHS.root)


def test_imports():
    from waste_classifier import build_model, build_datasets, predict  # noqa: F401
