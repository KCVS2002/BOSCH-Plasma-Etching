"""Regression metrics + fold-aggregation helpers used by every experiment."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R² / MAPE(%) for a single fold."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = mean_squared_error(y_true, y_pred)
    # MAPE: protect against zero targets (none expected here, but defensive).
    safe = np.where(np.abs(y_true) < 1e-12, 1e-12, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe)) * 100.0)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": mape,
    }


def aggregate_folds(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """mean ± std over a list of per-fold metric dicts (numeric keys only)."""
    if not per_fold:
        return {}
    keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.asarray([m[k] for m in per_fold], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out
