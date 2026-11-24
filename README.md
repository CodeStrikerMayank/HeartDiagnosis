# Heart Disease Neural Network Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-from--scratch-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, multi-layer artificial neural network implemented **from scratch using pure NumPy** (no high-level deep learning frameworks like TensorFlow or PyTorch) to predict the presence of heart disease based on clinical patient indicators.

---

## 📌 Project Overview

This project implements a 6-layer Deep Feedforward Neural Network to diagnose heart disease from clinical health metrics. The entire forward pass, analytical backpropagation with the chain rule, activation functions, loss computation, weight initialization, and mini-batch gradient descent optimizer are custom-coded in Python and NumPy.

### Highlights & Performance
- **From-Scratch Implementation**: Pure vectorised mathematical operations via NumPy.
- **High Diagnostic Accuracy**:
  - **Train Accuracy**: 100.00%
  - **Validation Accuracy**: 98.05%
  - **Test Accuracy**: **96.10%** (Test Loss: 0.1880)
- **He (Kaiming) Weight Initialization**: Configured specifically for LeakyReLU activations to prevent vanishing/exploding gradients.
- **Robust Feature Standardization**: Z-score normalization fitted on the training distribution to ensure numerical stability and fast convergence.

---

## 🧠 Network Architecture

The architecture consists of an input layer, 5 hidden layers with LeakyReLU activations, and a single-neuron Sigmoid output layer for binary classification:

```
Input (13 Features)
       │
       ▼
[Dense (13 → 32) + LeakyReLU (α=0.01)]
       │
       ▼
[Dense (32 → 16) + LeakyReLU (α=0.01)]
       │
       ▼
[Dense (16 → 8)  + LeakyReLU (α=0.01)]
       │
       ▼
[Dense (8  → 4)  + LeakyReLU (α=0.01)]
       │
       ▼
[Dense (4  → 2)  + LeakyReLU (α=0.01)]
       │
       ▼
[Dense (2  → 1)  + Sigmoid Activation]
       │
       ▼
Output: P(Heart Disease = 1)
```

### Mathematical Formulation
1. **Activation Functions**:
   - **LeakyReLU**: $f(z) = \max(0.01z, z)$
   - **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ (with clipping for numerical stability)
2. **Loss Function**:
   - **Binary Cross-Entropy (BCE)**:
     $$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\right]$$
3. **Weight Initialization**:
   - Weights: $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$
   - Biases: $b = \mathbf{0}$
4. **Optimization**:
   - Mini-Batch Gradient Descent (Batch size = 16, Learning rate $\alpha = 0.01$, Epochs = 300).

---

## 📊 Dataset Features

The model trains on `heart.csv` containing 1,025 patient records with 13 clinical biomarkers:

| Feature | Description |
| :--- | :--- |
| `age` | Age of patient (years) |
| `sex` | Sex (1 = male; 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (in mm Hg on admission to the hospital) |
| `chol` | Serum cholesterol in mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg` | Resting electrocardiographic results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | The slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) |
| **`target`** | **Diagnosis (1 = heart disease detected, 0 = normal)** |

---

## 📁 Repository Structure

```
Heart_Neural_Network/
├── src/
│   └── heart_neural_network/
│       └── __init__.py       # Core neural network, data extractor & CLI entrypoint
├── heart.csv                 # Patient dataset
├── pyproject.toml            # Project dependencies & build configuration
├── uv.lock                   # Deterministic lockfile
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites
- Python >= 3.10
- Recommended: [`uv`](https://github.com/astral-sh/uv) (fast Python package manager) or standard `pip`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CodeStrikerMayank/HeartDiagnosis.git
   cd HeartDiagnosis
   ```

2. **Sync dependencies using `uv`**:
   ```bash
   uv sync
   ```
   *(Or using pip: `pip install -e .`)*

---

## 💻 Running the Model

### Option 1: Using the CLI command
```bash
uv run heart-neural-network
```

### Option 2: Running via Python
```bash
uv run python -m heart_neural_network
```

### Expected Output
```text
File found: heart.csv

Starting training: 300 epochs, batch size 16, learning rate 0.01...
Epoch  50/300 | Train Loss: 0.1341, Train Acc: 96.79% | Val Loss: 0.1918, Val Acc: 94.16%
Epoch 100/300 | Train Loss: 0.0172, Train Acc: 99.72% | Val Loss: 0.1076, Val Acc: 96.75%
Epoch 150/300 | Train Loss: 0.0032, Train Acc: 100.00% | Val Loss: 0.0817, Val Acc: 98.05%
Epoch 200/300 | Train Loss: 0.0019, Train Acc: 100.00% | Val Loss: 0.0883, Val Acc: 98.05%
Epoch 250/300 | Train Loss: 0.0014, Train Acc: 100.00% | Val Loss: 0.0922, Val Acc: 98.05%
Epoch 300/300 | Train Loss: 0.0012, Train Acc: 100.00% | Val Loss: 0.0961, Val Acc: 98.05%

Training Complete!
Test Loss: 0.1880 | Test Accuracy: 96.10%
```

---

## 🧑‍💻 Python API Usage

You can also import and use the neural network in your own Python scripts:

```python
from heart_neural_network import Neural_activation_process

# Initialize and train model
model = Neural_activation_process(auto_train=True)

# Evaluate on test set
test_loss, test_acc = model.evaluate(model.x_test, model.y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predict on new normalized patient samples
predictions = model.predict(model.x_test[:5])
print("Predictions:", predictions.flatten())
```

---

## 👤 Author

- **CoreShadowMayank** - [GitHub Profile](https://github.com/CodeStrikerMayank)
