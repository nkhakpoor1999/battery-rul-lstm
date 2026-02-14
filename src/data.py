from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import savgol_filter


def _pad_last(x: np.ndarray) -> np.ndarray:
    """Pad last column to restore original length after diff along axis=1."""
    return np.hstack([x, x[:, -1:]])


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """Fill NaNs/infs in 1D array by linear interpolation; fallback to zeros."""
    m = np.isfinite(x)
    if m.any():
        x = x.copy()
        x[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), x[m])
        return x
    return np.zeros_like(x)


def _interp_rows(X: np.ndarray) -> np.ndarray:
    """Apply _interp_nan_1d to each row of a 2D array."""
    return np.vstack([_interp_nan_1d(row) for row in X])


def _savgol_rows(X: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Apply Savitzky–Golay filter row-wise (only if enough points)."""
    if X.shape[1] < window:
        return X
    w = window if window % 2 == 1 else window - 1
    if w < 3:
        return X
    return np.vstack([savgol_filter(row, w, poly) for row in X])


def dvdsoc(voltage: np.ndarray, soc: np.ndarray, eps: float = 1e-9, clip: float = 50.0) -> np.ndarray:
    dV = np.diff(voltage, axis=1)
    dS = np.diff(soc, axis=1)
    out = np.divide(dV, dS, out=np.full_like(dV, np.nan), where=np.abs(dS) > eps)
    out = _pad_last(out)
    out = _interp_rows(out)
    return np.clip(out, -clip, clip)


def compute_eol(soh: np.ndarray, threshold: float = 0.8) -> int:
    idx = np.flatnonzero(soh <= threshold)
    return int(idx[0]) if idx.size else int(len(soh) - 1)


def load_battery_npz(file_path: Path, savgol_window: int = 21, savgol_poly: int = 5) -> Dict[str, np.ndarray]:
    d = np.load(file_path, allow_pickle=True)

    V   = np.asarray(d["voltage"], dtype=float)
    Q   = np.asarray(d["max_capacity"], dtype=float)
    SOC = np.asarray(d["soc"], dtype=float)
    dQdV = np.asarray(d["dqdv"], dtype=float)
    T = np.asarray(d["temperature"], dtype=float) if "temperature" in d.files else None

    dVdS = dvdsoc(V, SOC)

    dTdV = None
    if T is not None:
        clip = 50.0
        dTdV = np.diff(T, axis=1) / np.diff(V, axis=1)
        dTdV = np.where(np.isfinite(dTdV), dTdV, np.nan)
        dTdV = _pad_last(dTdV)
        dTdV = _interp_rows(dTdV)
        dTdV = _savgol_rows(dTdV, savgol_window, savgol_poly)
        dTdV = -np.clip(dTdV, -clip, clip)

    SOH = Q / float(np.nanmax(Q))

    return {
        "voltage": V,
        "max_capacity": Q,
        "soh": SOH,
        "dqdv": dQdV,
        "dv_dsoc": dVdS,
        "dtdv": dTdV,  # can be None
    }


def build_rul_dataset_time(
    folder: Path,
    eol_threshold: float = 0.8,
    savgol_window: int = 21,
    savgol_poly: int = 5,
    features: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = Path(folder)
    files = sorted(folder.glob("*.npz"))
    feats = features or ["voltage", "dqdv", "dv_dsoc", "dtdv"]

    X_list, y_list, g_list = [], [], []

    for f in files:
        d = load_battery_npz(f, savgol_window=savgol_window, savgol_poly=savgol_poly)
        V = d["voltage"]
        eol = compute_eol(d["soh"], threshold=eol_threshold)
        if eol <= 0:
            continue

        if d["dtdv"] is None:
            d["dtdv"] = np.zeros_like(V)

        for c in range(eol + 1):
            X_list.append(np.stack([d[k][c] for k in feats], axis=-1))
            y_list.append((eol - c) / eol)
            g_list.append(f.name)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float).reshape(-1, 1)
    groups = np.asarray(g_list, dtype=object)
    return X, y, groups
