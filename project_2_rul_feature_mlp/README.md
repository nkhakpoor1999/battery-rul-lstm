# 🔋 Project 2 — Feature-Based RUL Prediction (MLP)

A machine learning pipeline for Remaining Useful Life (RUL) prediction using engineered cycle-level features and a fully connected neural network.

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
DATASET_KEY = "NASA" # MIT / OXFORD / Lab-Li-LCO / Lab-Li-NMC / Lab-Li-EVE*
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


### 📊 Example Results

#### Oxford Dataset

![Oxford True vs Pred](examples/true_vs_pred_rul_ratio-OXFORD.png)  
![Oxford Ablation](examples/ablation_r2_mean_bar-OXFORD.png)

📄 Full Report: [report-OXFORD.txt](examples/report-OXFORD.txt)

---

#### Lab-Li-EVE Dataset

![Lab True vs Pred](examples/true_vs_pred_rul_ratio-Lab-Li-EVE.png)  
![Lab Ablation](examples/ablation_r2_mean_bar-Lab-Li-EVE.png)

📄 Full Report: [report-Lab-Li-EVE.txt](examples/report-Lab-Li-EVE.txt)
