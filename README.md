# ♻️ Waste Classification — Organic vs Recyclable

Binary image classifier using **VGG16** transfer learning with fine-tuning, built with TensorFlow / Keras.

> **Live demo:** [HuggingFace Spaces](https://huggingface.co/spaces/felipebojorquem/waste-classification) 

---

## Overview

Automating waste sorting to reduce contamination of recyclable materials. The model classifies waste images into two categories: **Organic** and **Recyclable**.

**Approach:** Two-phase transfer learning on VGG16 (ImageNet weights):

1. **Feature Extraction** — Frozen VGG16 backbone + trainable classification head.
2. **Fine-Tuning** — Unfreeze `block5_conv1` onwards for domain adaptation with a lower learning rate.

### Why VGG16?

VGG16 (2014) is deliberately chosen as a well-understood baseline. For production, architectures like EfficientNetV2 or ConvNeXt offer better accuracy/FLOPs trade-offs. The modular design makes backbone swapping a one-line change in `config.py`.

---

## Results

| Model              | Accuracy | AUC   | Params (trainable) |
| ------------------ | -------- | ----- | ------------------ |
| Feature Extraction | 80%      | 0.871 | ~131K              |
| Fine-Tuned         | 85%      | 0.946 | ~2.5M              |

> _Fill in after training on your hardware. Results vary by run due to stochastic training._

<details>
<summary>Training Curves</summary>

![Feature Extraction](outputs/figures/curves_extract.png)
![Fine-Tuning](outputs/figures/curves_finetune.png)

</details>

<details>
<summary>Grad-CAM Visualization</summary>

![Grad-CAM](outputs/figures/gradcam_0.png)

</details>

---

## Project Structure

```
waste-classification/
├── src/waste_classifier/       # Production package
│   ├── config.py               # Centralized hyperparameters
│   ├── data.py                 # Download, augmentation, tf.data pipelines
│   ├── model.py                # VGG16 architecture + freeze strategies
│   ├── train.py                # Training loop with callbacks
│   ├── evaluate.py             # Metrics, confusion matrix, ROC, Grad-CAM
│   └── inference.py            # Single-image prediction API
├── app/app.py                  # Gradio demo (HuggingFace Spaces)
├── notebooks/exploration.ipynb # Exploratory analysis
├── run_training.py             # End-to-end training script
├── Dockerfile                  # Containerized deployment
├── pyproject.toml              # Dependencies & packaging
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/felipebojorquem/waste-classification.git
cd waste-classification
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### 2. Train

```bash
python run_training.py
```

The dataset downloads automatically (~50 MB). Training outputs:

- Model checkpoints → `models/`
- Figures (curves, confusion matrices, ROC, Grad-CAM) → `outputs/figures/`

### 3. Run the demo

```bash
uv pip install -e ".[app]"
python app/app.py
```

Open `http://localhost:7860` — upload a waste image to classify with Grad-CAM explainability.

### 4. Docker

```bash
docker build -t waste-classifier .
docker run -p 7860:7860 waste-classifier
```

---

## Tech Stack

| Tool                   | Purpose                    |
| ---------------------- | -------------------------- |
| TensorFlow 2.x / Keras | Model training & inference |
| VGG16 (ImageNet)       | Pre-trained backbone       |
| Grad-CAM               | Explainability             |
| Gradio                 | Interactive demo           |
| scikit-learn           | Evaluation metrics         |
| Docker                 | Reproducible deployment    |

---

## Dataset

[Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) — 1,200 images (reduced split), binary labels: Organic (O) / Recyclable (R).

---

## License

MIT — see [LICENSE](LICENSE).
