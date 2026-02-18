# 🔋 Project 3 — Sequence-based RUL Prediction (LSTM)

An end-to-end deep learning pipeline for Remaining Useful Life (RUL) prediction using per-cycle sequences and an LSTM regression model.

This module extends the feature-based baseline (Project 2) by learning temporal/shape representations directly from cycle-level curves.

---

## 🎯 Objective

Given preprocessed battery `.npz` files, the pipeline:

- Builds per-cycle sequences with multiple channels (e.g., voltage, dQ/dV, dV/dSOC, dT/dV)
- Defines End-of-Life (EOL) using an SOH threshold
- Trains an LSTM model to predict normalized RUL ratio
- Evaluates using battery-level cross-validation (GroupKFold)
- Trains a final model using the median best epoch from CV
- Saves model + scaler + metadata for reproducible inference

---

## 📊 Validation Strategy (Leakage-Free)

- **GroupKFold** split (group = battery file name)
- EarlyStopping + ReduceLROnPlateau during CV
- Final epochs = median best epoch across folds

Outputs:
- `results/<RUN_ID>/cv_folds.csv`
- `results/<RUN_ID>/cv_summary.json`

---

## 🔬 Input Representation

For each cycle `c`, the model consumes a sequence:

- Shape: `(W, F)`
  - `W`: number of points per cycle
  - `F`: number of channels

Default channels (configurable):
- `voltage`
- `dqdv`
- `dv_dsoc`
- `dtdv` 

---

## ▶️ Training

Select dataset in `train.py`:

```python
DATASET_KEY = "NASA"  # MIT / OXFORD / Lab-Li-LCO / Lab-Li-NMC / Lab-Li-EVE*
```

Ensure dataset root is set in `dataset_configs.py`.

Run:

```bash
python project_3_rul_lstm/train.py
```

### Saved Artifacts

```
project_3_rul_lstm/
├── results/<RUN_ID>/
│   ├── cv_folds.csv
│   └── cv_summary.json
└── saved_models/<RUN_ID>/
    ├── rul_lstm.keras
    ├── scaler.joblib
    ├── meta.joblib
    └── meta.json
```

`meta.json` contains:
- dataset key
- EOL threshold
- feature list
- final epochs
- training metrics

---

## 🔎 Evaluate on a New Battery

```bash
python project_3_rul_lstm/evaluate.py \
  --model_dir "project_3_rul_lstm/saved_models/<RUN_ID>" \
  --battery "PATH_TO_NEW_BATTERY.npz"
```

Outputs:
- Ratio-level metrics (R², MAE, RMSE)
- Cycle-level metrics
- `holdout_true_pred.png` (True vs Predicted RUL ratio)

---

## 📦 Data Preprocessing

All datasets used here were preprocessed into a standardized `.npz` format prior to modeling, including:

- Signal cleaning and alignment
- SOH computation and EOL detection
- Derivative feature construction (dV/dSOC, dQ/dV, dT/dV)
- Smoothing and noise handling
- Cycle-level consistency checks

Preprocessing scripts are intentionally excluded to keep the repository focused on modeling and evaluation.


---

## 📌 Positioning in Pipeline

Battery Data → Sequence Builder → LSTM Regression → RUL Prediction

Complements:
- Project 1: Dataset/Brand Classification
- Project 2: Feature-based MLP Baseline

## 📊 Evaluation Results

### OXFORD Dataset

![OXFORD Holdout](outputs/holdout_true_pred_oxford.png)

**CV Summary:** `outputs/cv_summary_oxford.json`

---

### Lab-Li-EVE Dataset

![EVE Holdout](outputs/holdout_true_pred_eve.png)

**CV Summary:** `outputs/cv_summary_eve.json`
