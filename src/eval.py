from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .data import load_battery_npz, compute_eol

def evaluate_on_battery(
    model,
    scaler,
    battery_path: Path,
    eol_threshold: float,
    clip_ratio_pred: bool,
    features: list,
    savgol_window: int = 21,
    savgol_poly: int = 5
) -> Dict[str, Any]:

    d = load_battery_npz(battery_path, savgol_window=savgol_window, savgol_poly=savgol_poly)
    V = d["voltage"]
    soh = d["soh"]
    eol_idx = compute_eol(soh, threshold=eol_threshold)

    if d["dtdv"] is None:
        d["dtdv"] = np.zeros_like(V)

    W = V.shape[1]
    F = len(features)

    y_true_ratio, y_pred_ratio = [], []

    for c in range(eol_idx + 1):
        seq = np.stack([d[k][c] for k in features], axis=-1)  # (W, F)
        seq_sc = scaler.transform(seq.reshape(-1, F)).reshape(1, W, F)
        pred = float(model.predict(seq_sc, verbose=0).reshape(-1)[0])
        if clip_ratio_pred:
            pred = float(np.clip(pred, 0.0, 1.0))

        true = (eol_idx - c) / eol_idx
        y_true_ratio.append(true)
        y_pred_ratio.append(pred)

    y_true_ratio = np.asarray(y_true_ratio, dtype=float)
    y_pred_ratio = np.asarray(y_pred_ratio, dtype=float)

    r2_r = float(r2_score(y_true_ratio, y_pred_ratio))
    mae_r = float(mean_absolute_error(y_true_ratio, y_pred_ratio))
    rmse_r = float(np.sqrt(mean_squared_error(y_true_ratio, y_pred_ratio)))

    y_true_cycles = y_true_ratio * eol_idx
    y_pred_cycles = y_pred_ratio * eol_idx

    r2_c = float(r2_score(y_true_cycles, y_pred_cycles))
    mae_c = float(mean_absolute_error(y_true_cycles, y_pred_cycles))
    rmse_c = float(np.sqrt(mean_squared_error(y_true_cycles, y_pred_cycles)))

    return {
        "battery": battery_path.name,
        "eol_cycles": int(eol_idx),
        "ratio": {"R2": r2_r, "MAE": mae_r, "RMSE": rmse_r},
        "cycles": {"R2": r2_c, "MAE": mae_c, "RMSE": rmse_c},
        "y_true_ratio": y_true_ratio,
        "y_pred_ratio": y_pred_ratio,
    }

def plot_true_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.xlabel("Cycle index (0..EOL)")
    plt.ylabel("RUL ratio")
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Folder containing rul_lstm.keras, scaler.joblib, meta.joblib")
    parser.add_argument("--battery", type=str, required=True, help="Path to a .npz battery file to evaluate")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    battery_path = Path(args.battery)

    import tensorflow as tf
    model = tf.keras.models.load_model(model_dir / "rul_lstm.keras")
    scaler = joblib.load(model_dir / "scaler.joblib")
    meta = joblib.load(model_dir / "meta.joblib")

    res = evaluate_on_battery(
        model=model,
        scaler=scaler,
        battery_path=battery_path,
        eol_threshold=float(meta["eol_threshold"]),
        clip_ratio_pred=True,
        features=meta["features"],
    )

    print("\n=== Holdout Evaluation ===")
    print("Battery:", res["battery"])
    print("EOL cycles:", res["eol_cycles"])
    print("RATIO :", res["ratio"])
    print("CYCLES:", res["cycles"])

    plot_true_pred(res["y_true_ratio"], res["y_pred_ratio"], model_dir / "holdout_true_pred.png",
                   title=f"Holdout: {res['battery']}")

if __name__ == "__main__":
    main()
