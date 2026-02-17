# 🔋 Project 2 — Feature-Based RUL Prediction (MLP)

A machine learning pipeline for Remaining Useful Life (RUL) prediction using engineered cycle-level features and a fully connected neural network.

This module focuses on interpretable feature-based degradation modeling before moving to sequence-based deep learning approaches.

---

## 🎯 Objective

Given battery cycle data (`.npz` files), the pipeline:

- Computes engineered degradation features per cycle
- Defines End-of-Life (EOL) based on SOH threshold
- Converts cycle index to normalized RUL target
- Performs battery-level cross-validation
- Selects optimal feature groups via ablation + search
- Trains a final regression model

---

## 🧠 Model

- Multi-Layer Perceptron (64 → 32 → 16)
- ReLU activation
- L2 regularization
- MSE loss (regression)
- Median-best-epoch selection from GroupKFold CV

---

## 📊 Validation Strategy

- GroupKFold split (group = battery ID)
- Prevents cycle-level leakage
- Feature group ablation + search
- Median optimal epoch from CV used for final training

---

## 🔬 Feature Engineering

Features are extracted per cycle using:

- Voltage statistics
- dV/dSOC
- dQ/dV
- dT/dV (if temperature available)
- SOH-based degradation indicators

Feature groups are evaluated through systematic ablation and subset search.

---

## 🚀 Training

```bash
python project_2_rul_feature_mlp/train.py
```

Dataset configuration is controlled via:

```python
DATASET_KEY = "NASA"
```

---

## 📁 Outputs

```
artifacts_3/<DATASET>/
├── model (Keras)
├── scaler
├── selected feature indices
├── ablation results (CSV)
├── search results (CSV)
├── Excel summary
└── training report.txt
```

---

## 📈 Target Definition

For cycle `c`:

```
RUL_ratio = (EOL - c) / EOL
```

This normalizes RUL between 0 and 1 for stable regression training.

---

## 🛠 Tech Stack

Python · TensorFlow/Keras · NumPy · SciPy · Scikit-learn · Pandas

---

## 📂 Structure

```
project_2_rul_feature_mlp/
├── train.py
├── dataset.py
├── model.py
├── features.py
├── preprocessing.py
├── cv_search.py
└── artifacts.py
```

---

## 📌 Positioning in Pipeline

Battery Data → Feature Engineering → Feature Selection → MLP Regression → RUL Prediction

This module provides an interpretable baseline before sequence-based LSTM modeling (Project 3).
