"""
models/forecaster.py
Implements and compares four forecasting algorithms:
  1. SARIMA  (statsmodels)
  2. Prophet (Meta / Facebook)
  3. XGBoost (gradient boosting with lag features)
  4. LSTM    (TensorFlow / Keras deep learning)

run_all_models() trains all four, validates on hold-out,
auto-selects the best by SMAPE, and returns structured results.
"""
import warnings
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from models.feature_eng import build_features, train_val_split, smape, mae

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

HORIZON = 8   # default forecast weeks


# ── NaN/Inf sanitiser ────────────────────────────────────────────────
def _safe_float(v, fallback=None):
    """Convert to float, replacing NaN/Inf with fallback (None → JSON null)."""
    try:
        f = float(v)
        return fallback if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return fallback


def _safe_list(arr, fallback=0.0):
    """Convert array to list, replacing NaN/Inf with fallback."""
    return [
        fallback if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else round(float(v), 2)
        for v in arr
    ]


# ═══════════════════════════════════════════════════════════
# 1. SARIMA
# ═══════════════════════════════════════════════════════════

def _sarima_fit(series: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    for order, seas in [
        ((1, 1, 1), (1, 1, 0, 52)),
        ((1, 1, 1), (0, 0, 0,  0)),
        ((2, 1, 1), (0, 0, 0,  0)),
    ]:
        try:
            m = SARIMAX(
                series, order=order,
                seasonal_order=seas,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return m.fit(disp=False, maxiter=300)
        except Exception:
            continue
    raise RuntimeError("SARIMA: all configurations failed.")


def predict_sarima(series: pd.Series, horizon: int = HORIZON) -> Tuple[np.ndarray, dict]:
    feat = build_features(series)
    tr, va = train_val_split(feat, horizon)
    train_series = series[series.index <= tr.index[-1]]

    res_val  = _sarima_fit(train_series)
    val_pred = np.nan_to_num(np.array(res_val.forecast(len(va))), nan=0.0, posinf=0.0)
    metrics  = {"smape": smape(va["y"].values, val_pred),
                 "mae":   mae(va["y"].values, val_pred)}

    res_full = _sarima_fit(series)
    forecast = np.nan_to_num(np.array(res_full.forecast(horizon)), nan=0.0, posinf=0.0)
    return np.clip(forecast, 0, None), metrics


# ═══════════════════════════════════════════════════════════
# 2. Prophet
# ═══════════════════════════════════════════════════════════

def _prophet_build(series: pd.Series):
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    dfp = series.reset_index()
    dfp.columns = ["ds", "y"]
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
    )
    try:
        m.add_country_holidays(country_name="US")
    except Exception:
        pass
    m.fit(dfp)
    return m


def predict_prophet(series: pd.Series, horizon: int = HORIZON) -> Tuple[np.ndarray, dict]:
    feat = build_features(series)
    _, va = train_val_split(feat, horizon)
    split_date = va.index[0]

    train_s = series[series.index < split_date]
    m_val   = _prophet_build(train_s)

    # FIX: Use a fixed 7-day frequency future dataframe so Prophet always
    # generates exactly `horizon` future rows, regardless of the irregular
    # spacing in the source data. Previously freq="W" (calendar weeks) could
    # yield fewer rows than horizon when the series timestamps land mid-week,
    # causing a shape mismatch (8,) vs (5,) in the smape() call.
    last_train_date = pd.Timestamp(train_s.index[-1])
    future_dates_val = pd.DataFrame({
        "ds": [last_train_date + pd.DateOffset(weeks=i + 1) for i in range(horizon)]
    })
    # Include training dates so Prophet can anchor its trend correctly
    train_df = pd.DataFrame({"ds": train_s.index.to_list()})
    f_val   = pd.concat([train_df, future_dates_val], ignore_index=True)
    fc_val  = m_val.predict(f_val)

    # Grab exactly the last `horizon` rows (the future rows we appended)
    val_pred = np.nan_to_num(
        fc_val.tail(horizon)["yhat"].values,
        nan=0.0, posinf=0.0,
    )
    # Guard: if still short (edge case), pad with last known value
    if len(val_pred) < len(va["y"]):
        pad = len(va["y"]) - len(val_pred)
        val_pred = np.pad(val_pred, (0, pad), mode="edge")

    metrics = {"smape": smape(va["y"].values, val_pred),
               "mae":   mae(va["y"].values, val_pred)}

    # Retrain on full series and forecast
    m_full  = _prophet_build(series)
    last_full_date = pd.Timestamp(series.index[-1])
    future_dates_full = pd.DataFrame({
        "ds": [last_full_date + pd.DateOffset(weeks=i + 1) for i in range(horizon)]
    })
    full_df = pd.DataFrame({"ds": series.index.to_list()})
    f_full  = pd.concat([full_df, future_dates_full], ignore_index=True)
    fc_full = m_full.predict(f_full)
    forecast = np.nan_to_num(fc_full.tail(horizon)["yhat"].values, nan=0.0, posinf=0.0)
    return np.clip(forecast, 0, None), metrics


# ═══════════════════════════════════════════════════════════
# 3. XGBoost  (walk-forward prediction)
# ═══════════════════════════════════════════════════════════

def predict_xgboost(series: pd.Series, horizon: int = HORIZON) -> Tuple[np.ndarray, dict]:
    import xgboost as xgb

    feat  = build_features(series)
    FCOLS = [c for c in feat.columns if c != "y"]
    tr, va = train_val_split(feat, horizon)

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.04,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(tr[FCOLS], tr["y"],
              eval_set=[(va[FCOLS], va["y"])], verbose=False)

    val_pred = np.nan_to_num(model.predict(va[FCOLS]), nan=0.0, posinf=0.0)
    metrics  = {"smape": smape(va["y"].values, val_pred),
                 "mae":   mae(va["y"].values, val_pred)}

    model.fit(feat[FCOLS], feat["y"])

    extended = series.copy()
    forecasts: list = []
    for _ in range(horizon):
        tmp = build_features(extended)
        if tmp.empty:
            forecasts.append(float(extended.iloc[-1]))
            continue
        row  = tmp.iloc[[-1]][FCOLS]
        # Handle case where walk-forward series has new lag columns not in training
        row  = row.reindex(columns=FCOLS, fill_value=0.0)
        pred = float(np.nan_to_num(model.predict(row)[0], nan=0.0, posinf=0.0))
        pred = max(pred, 0.0)
        forecasts.append(pred)
        new_date = extended.index[-1] + pd.DateOffset(weeks=1)
        extended = pd.concat([extended, pd.Series([pred], index=[new_date])])

    return np.array(forecasts), metrics


# ═══════════════════════════════════════════════════════════
# 4. LSTM
# ═══════════════════════════════════════════════════════════

def predict_lstm(series: pd.Series, horizon: int = HORIZON) -> Tuple[np.ndarray, dict]:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler

    tf.get_logger().setLevel("ERROR")
    SEQ_LEN = min(26, max(8, len(series) // 4))

    vals   = series.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    def _make_seqs(data, sl):
        X, y = [], []
        for i in range(sl, len(data)):
            X.append(data[i - sl: i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    tr_scaled  = scaled[:-horizon]
    val_ctx    = scaled[-(horizon + SEQ_LEN):]
    X_tr, y_tr = _make_seqs(tr_scaled, SEQ_LEN)
    X_va, y_va = _make_seqs(val_ctx,   SEQ_LEN)
    X_tr = X_tr.reshape(-1, SEQ_LEN, 1)
    X_va = X_va.reshape(-1, SEQ_LEN, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    model.fit(X_tr, y_tr, epochs=60, batch_size=16, verbose=0,
              validation_data=(X_va, y_va))

    val_pred_s = model.predict(X_va, verbose=0)
    val_pred   = np.nan_to_num(
        scaler.inverse_transform(val_pred_s).flatten(), nan=0.0, posinf=0.0
    )
    y_va_act = scaler.inverse_transform(y_va.reshape(-1, 1)).flatten()
    metrics  = {"smape": smape(y_va_act, val_pred),
                 "mae":   mae(y_va_act, val_pred)}

    buf = scaled[-SEQ_LEN:].flatten().tolist()
    fc_s: list = []
    for _ in range(horizon):
        seq = np.array(buf[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        p   = float(np.nan_to_num(model.predict(seq, verbose=0)[0, 0], nan=0.0, posinf=0.0))
        fc_s.append(p)
        buf.append(p)

    forecast = np.nan_to_num(
        scaler.inverse_transform(np.array(fc_s).reshape(-1, 1)).flatten(),
        nan=0.0, posinf=0.0,
    )
    return np.clip(forecast, 0, None), metrics


# ═══════════════════════════════════════════════════════════
# Auto-selector
# ═══════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "SARIMA":  predict_sarima,
    "Prophet": predict_prophet,
    "XGBoost": predict_xgboost,
    "LSTM":    predict_lstm,
}


def run_all_models(series: pd.Series, horizon: int = HORIZON) -> dict:
    """
    Train all 4 models on `series`, validate on last `horizon` weeks,
    auto-select best by lowest validation SMAPE.
    Returns a structured dict with per-model metrics and forecasts,
    guaranteed to be NaN/Inf-free (safe for JSON serialisation).
    """
    min_rows = horizon * 2 + 30
    if len(series) < min_rows:
        raise ValueError(
            f"Need ≥ {min_rows} weekly data points; got {len(series)}. "
            "Upload a longer history."
        )

    results: dict = {}
    for name, fn in MODEL_REGISTRY.items():
        try:
            logger.info("Training %s …", name)
            fc, metrics = fn(series, horizon)

            # ── Sanitise every numeric value before storing ──────────
            val_smape = _safe_float(metrics.get("smape"), fallback=9999.0)
            val_mae   = _safe_float(metrics.get("mae"),   fallback=9999.0)
            forecast  = _safe_list(np.clip(fc, 0, None))

            results[name] = {
                "forecast":  forecast,
                "val_smape": round(val_smape, 2),
                "val_mae":   round(val_mae,   0),
                "status":    "success",
            }
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)
            results[name] = {
                "status":    "failed",
                "error":     str(exc),
                "val_smape": 9999.0,
                "val_mae":   9999.0,
                "forecast":  [],
            }

    successful = {k: v for k, v in results.items() if v["status"] == "success"}
    if not successful:
        raise RuntimeError("All 4 models failed to train. Check data quality.")

    best = min(successful, key=lambda k: successful[k]["val_smape"])
    last = series.index[-1]

    forecast_dates = [
        (last + pd.DateOffset(weeks=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    history_values = _safe_list(series.values)

    return {
        "best_model":     best,
        "forecast_dates": forecast_dates,
        "models":         results,
        "history": {
            "dates":  [d.strftime("%Y-%m-%d") for d in series.index],
            "values": history_values,
        },
    }