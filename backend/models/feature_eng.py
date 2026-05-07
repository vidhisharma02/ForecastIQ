"""
models/feature_eng.py
Builds all required features:
  - Lag features  (adaptive: only lags ≤ len(series)//2)
  - Rolling mean & std (windows: 4, 8, 13, 26 weeks)
  - Calendar: week_of_year, month, quarter, year
  - Holiday flag (US federal holidays)
  - Trend index
Strict time-series train/val split with no data leakage.
"""
import numpy as np
import pandas as pd

# ── US federal holiday Mondays (representative week starts) ──────────
_US_HOLIDAYS = {
    "2019-01-07","2019-05-27","2019-07-08","2019-09-02",
    "2019-11-11","2019-11-25","2019-12-23",
    "2020-01-06","2020-05-25","2020-07-06","2020-09-07",
    "2020-11-09","2020-11-23","2020-12-28",
    "2021-01-04","2021-05-31","2021-07-05","2021-09-06",
    "2021-11-08","2021-11-22","2021-12-27",
    "2022-01-03","2022-05-30","2022-07-04","2022-09-05",
    "2022-11-07","2022-11-21","2022-12-26",
    "2023-01-02","2023-05-29","2023-07-03","2023-09-04",
    "2023-11-06","2023-11-20","2023-12-25",
    "2024-01-01","2024-05-27","2024-07-01","2024-09-02",
    "2024-11-04","2024-11-25","2024-12-23",
}

# All candidate lags — only those that fit the series length are used
_ALL_LAGS = [1, 2, 4, 8, 13, 26, 52]
_ALL_WINDOWS = [4, 8, 13, 26]


def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Given a weekly pd.Series (DatetimeIndex → float),
    return a feature DataFrame including the target column 'y'.

    FIX: Previously used a fixed lag list including lag_52, which caused
    dropna() to wipe all rows for series shorter than ~60 weeks.
    Now lags and rolling windows are capped to half the series length so
    build_features() always returns usable rows for any realistic series.
    """
    df = pd.DataFrame({"y": series.values}, index=pd.DatetimeIndex(series.index))

    n = len(df)
    # Only use lags and windows that leave enough rows after dropna
    max_lag = max(1, n // 2)
    active_lags    = [l for l in _ALL_LAGS    if l <= max_lag]
    active_windows = [w for w in _ALL_WINDOWS if w <= max_lag]

    # ── Calendar features ────────────────────────────────────────────
    iso = df.index.isocalendar()
    df["week_of_year"] = iso.week.astype(int)
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["year"]         = df.index.year
    df["trend"]        = np.arange(len(df), dtype=float)
    df["is_holiday"]   = (
        df.index.strftime("%Y-%m-%d").isin(_US_HOLIDAYS).astype(int)
    )

    # ── Lag features (shift target; no future leakage) ───────────────
    for lag in active_lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # ── Rolling mean & std (shifted by 1 so no current-row leakage) ──
    shifted = df["y"].shift(1)
    for w in active_windows:
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"]  = shifted.rolling(w).std()

    return df.dropna()


def train_val_split(df: pd.DataFrame, val_weeks: int = 8):
    """
    Strictly chronological split. Validation = last val_weeks rows.
    No shuffling. No overlap.
    """
    n = len(df)
    if n < val_weeks + 15:
        raise ValueError(
            f"Need at least {val_weeks + 15} clean feature rows; got {n}. "
            "Upload more historical data."
        )
    return df.iloc[: n - val_weeks].copy(), df.iloc[n - val_weeks :].copy()


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-100 scale)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, np.abs(a - p) / denom)
    return float(np.mean(ratio) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float))))