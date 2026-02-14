# Battery RUL Prediction with LSTM (Thesis Code)

This repository contains a reproducible pipeline for Remaining Useful Life (RUL) prediction of Li-ion batteries
using a sequence model (LSTM) trained on per-cycle features.

## Features
- End-to-end pipeline: dataset building → GroupKFold CV (battery-wise split) → final training → model export
- Configurable EOL threshold (e.g., 0.80 / 0.85)
- Saves:
  - CV fold metrics (`results/.../cv_folds.csv`)
  - CV summary (`results/.../cv_summary.json`)
  - Final trained model (`saved_models/.../rul_lstm.keras`)
  - Scaler + metadata (`scaler.joblib`, `meta.joblib`, `meta.json`)
  - Plots (`cv_epochs.png`, `holdout_true_pred.png`)

## Installation
```bash
pip install -r requirements.txt
