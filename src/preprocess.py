"""
Preprocessing + feature engineering for smart-meter data.

Steps (each one is something you should be able to explain in an interview):
 1. Missing values  -> linear interpolation along time, remaining gaps -> 0
 2. Outliers        -> "three-sigma rule": cap values above mean + 2*std per customer
                       (sensor glitches would otherwise dominate the scale)
 3. Normalisation   -> per-customer min-max scaling to [0, 1]
                       (so the model learns the SHAPE of usage, not the size of the house)
 4. Reshape         -> 1D daily series  -> 2D (weeks x 7 days) "image" for the CNN branch
 5. Features        -> hand-crafted statistics that capture theft behaviour
"""
import numpy as np
import pandas as pd


def load_dataset(path: str):
    """Works for both the real SGCC `data.csv` and our synthetic file."""
    df = pd.read_csv(path)
    y = df["FLAG"].astype(int).values
    ids = df["CONS_NO"].astype(str).values
    X = df.drop(columns=["CONS_NO", "FLAG"])
    # Real SGCC columns are dates but NOT in order -> sort them
    try:
        X.columns = pd.to_datetime(X.columns)
        X = X.reindex(sorted(X.columns), axis=1)
    except Exception:
        pass
    return X.values.astype("float32"), y, ids


def clean(X: np.ndarray) -> np.ndarray:
    # 1. interpolate missing values per customer (row)
    X = pd.DataFrame(X).interpolate(axis=1, limit_direction="both").fillna(0).values
    # 2. cap outliers per customer
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    X = np.minimum(X, mean + 2 * std)
    return X.astype("float32")


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Statistics computed on CLEAN but UNSCALED kWh values."""
    eps = 1e-6
    n_days = X.shape[1]
    half = n_days // 2
    first, second = X[:, :half].mean(1), X[:, half:].mean(1)
    weekly = X[:, : (n_days // 7) * 7].reshape(len(X), -1, 7).sum(2)
    diffs = np.diff(X, axis=1)
    t = np.arange(n_days)
    slope = np.polyfit(t, X.T, 1)[0]                         # long-term trend
    feats = np.column_stack([
        X.mean(1), X.std(1), X.max(1), np.median(X, 1),
        X.std(1) / (X.mean(1) + eps),                        # coefficient of variation
        (X < 0.05 * (X.mean(1, keepdims=True) + eps)).mean(1),  # share of ~zero days
        second / (first + eps),                              # drop after theft starts
        weekly.std(1) / (weekly.mean(1) + eps),              # weekly irregularity
        np.abs(diffs).mean(1) / (X.mean(1) + eps),           # day-to-day jumpiness
        slope / (X.mean(1) + eps),
        (np.abs(diffs) < 1e-3).mean(1),                      # flat-line days (constant readings)
    ])
    return np.nan_to_num(feats).astype("float32")


def minmax_rows(X: np.ndarray) -> np.ndarray:
    mn = X.min(1, keepdims=True)
    mx = X.max(1, keepdims=True)
    return ((X - mn) / (mx - mn + 1e-6)).astype("float32")


def to_weeks(X: np.ndarray) -> np.ndarray:
    """(customers, days) -> (customers, weeks, 7, 1) for Conv2D."""
    n_weeks = X.shape[1] // 7
    return X[:, : n_weeks * 7].reshape(len(X), n_weeks, 7, 1)


def build_inputs(X_raw: np.ndarray):
    """Full pipeline: returns (daily_scaled, weekly_image, stats_features)."""
    X = clean(X_raw)
    feats = engineer_features(X)
    Xs = minmax_rows(X)
    return Xs, to_weeks(Xs), feats
