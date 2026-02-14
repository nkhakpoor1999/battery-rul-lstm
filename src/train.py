from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .utils import set_seed, ensure_dir, save_json, now_tag
from .data import build_rul_dataset_time
from .models import build_rul_model_time


def groupkfold_train_eval(
    data_path: Path,
    eol_threshold: float,
    n_splits: int,
    epochs: int,
    batch_size: int,
    verbose: int,
    seed: int,
    savgol_window: int,
    savgol_poly: int,
    l2_reg: float,
    early_stop_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_lr: float,
    features: list,
) -> Tuple[Dict[str, Any], pd.DataFrame]:

    set_seed(seed)

    X, y, groups = build_rul_dataset_time(
        folder=data_path,
        eol_threshold=eol_threshold,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        features=features
    )
    N, W, F = X.shape

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
        X_va_sc = scaler.transform(X_va.reshape(-1, F)).reshape(X_va.shape)

        model = build_rul_model_time(W, F, l2_reg=l2_reg)

        early_stop = EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor,
                                      patience=reduce_lr_patience, min_lr=min_lr)

        hist = model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_va_sc, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        ran_epochs = len(hist.history["loss"])

        y_pred = model.predict(X_va_sc, verbose=0).reshape(-1)
        y_true = y_va.reshape(-1)

        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        rows.append({
            "fold": fold,
            "epochs_ran": ran_epochs,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "train_batteries": ", ".join(sorted(set(groups[tr_idx]))),
            "val_batteries": ", ".join(sorted(set(groups[va_idx]))),
        })

    df = pd.DataFrame(rows)

    summary = {
        "n_splits": int(n_splits),
        "n_samples": int(N),
        "W": int(W),
        "F": int(F),
        "R2_mean": float(df["R2"].mean()),
        "R2_std": float(df["R2"].std(ddof=1)) if n_splits > 1 else 0.0,
        "MAE_mean": float(df["MAE"].mean()),
        "MAE_std": float(df["MAE"].std(ddof=1)) if n_splits > 1 else 0.0,
        "RMSE_mean": float(df["RMSE"].mean()),
        "RMSE_std": float(df["RMSE"].std(ddof=1)) if n_splits > 1 else 0.0,
        "median_best_epoch": int(np.median(df["epochs_ran"].values)),
    }

    return summary, df


def train_final_model(
    data_path: Path,
    eol_threshold: float,
    final_epochs: int,
    batch_size: int,
    verbose: int,
    seed: int,
    savgol_window: int,
    savgol_poly: int,
    l2_reg: float,
    features: list,
) -> Dict[str, Any]:

    set_seed(seed)

    X, y, groups = build_rul_dataset_time(
        folder=data_path,
        eol_threshold=eol_threshold,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        features=features
    )
    N, W, F = X.shape

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X.reshape(-1, F)).reshape(X.shape)

    model = build_rul_model_time(W, F, l2_reg=l2_reg)

    hist = model.fit(X_sc, y, epochs=int(final_epochs), batch_size=batch_size, verbose=verbose)

    y_pred = model.predict(X_sc, verbose=0).reshape(-1)
    y_true = y.reshape(-1)

    metrics = {
        "train_R2": float(r2_score(y_true, y_pred)),
        "train_MAE": float(mean_absolute_error(y_true, y_pred)),
        "train_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "epochs_ran": int(len(hist.history["loss"])),
        "n_samples": int(N),
        "n_batteries": int(len(set(groups))),
        "W": int(W),
        "F": int(F),
    }

    return {
        "model": model,
        "scaler": scaler,
        "history": hist.history,
        "metrics": metrics,
        "W": W,
        "F": F,
        "batteries": sorted(set(groups)),
    }


def plot_cv_epochs(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    plt.plot(df["fold"], df["epochs_ran"], marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Epochs ran")
    plt.title("Early stopping epochs per fold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Base data directory containing dataset folders")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name (e.g., NASA, MIT)")
    parser.add_argument("--eol", type=float, required=True, help="EOL threshold (e.g., 0.8 or 0.85)")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--savgol_window", type=int, default=21)
    parser.add_argument("--savgol_poly", type=int, default=5)
    parser.add_argument("--l2_reg", type=float, default=1e-4)

    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument("--rlr_patience", type=int, default=10)
    parser.add_argument("--rlr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--features", type=str, default="voltage,dqdv,dv_dsoc,dtdv")

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="saved_models")

    args = parser.parse_args()

    features = [x.strip() for x in args.features.split(",") if x.strip()]

    data_path = Path(args.data_dir) / args.dataset
    run_id = f"{args.dataset}_eol{args.eol}_{now_tag()}"

    results_dir = ensure_dir(Path(args.results_dir) / run_id)
    models_dir  = ensure_dir(Path(args.models_dir) / run_id)

    # 1) CV
    summary, fold_df = groupkfold_train_eval(
        data_path=data_path,
        eol_threshold=args.eol,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        seed=args.seed,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        l2_reg=args.l2_reg,
        early_stop_patience=args.early_patience,
        reduce_lr_patience=args.rlr_patience,
        reduce_lr_factor=args.rlr_factor,
        min_lr=args.min_lr,
        features=features
    )

    fold_df.to_csv(results_dir / "cv_folds.csv", index=False)
    save_json(summary, results_dir / "cv_summary.json")
    plot_cv_epochs(fold_df, results_dir / "cv_epochs.png")

    final_epochs = summary["median_best_epoch"]

    # 2) Final train on all data
    final = train_final_model(
        data_path=data_path,
        eol_threshold=args.eol,
        final_epochs=final_epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        seed=args.seed,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        l2_reg=args.l2_reg,
        features=features
    )

    # Save model + scaler + meta
    final["model"].save(models_dir / "rul_lstm.keras")
    joblib.dump(final["scaler"], models_dir / "scaler.joblib")
    meta = {
        "dataset": args.dataset,
        "data_path": str(data_path),
        "eol_threshold": args.eol,
        "features": features,
        "W": final["W"],
        "F": final["F"],
        "final_epochs": final_epochs,
        "seed": args.seed,
        "train_metrics": final["metrics"],
        "batteries": final["batteries"],
    }
    joblib.dump(meta, models_dir / "meta.joblib")
    save_json(meta, models_dir / "meta.json")

    print("\n=== CV Summary ===")
    print(summary)
    print("\n=== Final Train Metrics ===")
    print(final["metrics"])
    print(f"\nSaved to:\n  results: {results_dir}\n  models : {models_dir}\n")


if __name__ == "__main__":
    main()
