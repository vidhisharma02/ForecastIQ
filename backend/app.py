"""
app.py – ForecastIQ Flask REST API
────────────────────────────────────────────────────────────
Run (dev):   python app.py
Run (prod):  gunicorn -w 2 -t 600 app:app

Endpoints
─────────
GET  /api/health
POST /api/upload          → { session_key, summary }
GET  /api/states          ?session_key=…
GET  /api/summary         ?session_key=…
POST /api/forecast        { session_key, state, horizon? }
POST /api/compare         { session_key, state }
POST /api/forecast/batch  { session_key, horizon? }
"""

import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.data_loader import (
    load_and_prepare, get_state_series,
    available_states, dataset_summary,
)
from models.forecaster import run_all_models, HORIZON

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".csv", ".xlsx", ".xls"}
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB

# In-memory session store  { session_key → pd.DataFrame }
_store: dict = {}


# ── Response helpers ─────────────────────────────────────────────────
def ok(payload: dict, code: int = 200):
    return jsonify({"status": "success", **payload}), code

def err(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code

def resolve_session(key: str):
    """Return (df, None) or (None, error_response)."""
    if not key or key not in _store:
        return None, err(
            "Invalid or missing session_key. Upload a dataset first.", 400
        )
    return _store[key], None


# ════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════

# ── Health ───────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return ok({"message": "ForecastIQ API is running ✓"})


# ── Upload dataset ────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return err("No 'file' field found in the form-data request.")
    f = request.files["file"]
    if f.filename == "":
        return err("File name is empty.")

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return err(f"Unsupported file type '{ext}'. Please upload CSV or Excel.")

    fname     = secure_filename(f.filename)
    save_path = UPLOAD_DIR / fname
    f.save(str(save_path))
    logger.info("Saved upload → %s", save_path)

    try:
        df = load_and_prepare(str(save_path))
    except Exception as exc:
        logger.exception("load_and_prepare failed")
        return err(str(exc))

    session_key = fname          # For production: use uuid4()
    _store[session_key] = df
    logger.info("Session '%s' loaded — %d rows, %d states",
                session_key, len(df), df["state"].nunique())

    return ok({"session_key": session_key, "summary": dataset_summary(df)})


# ── Summary ───────────────────────────────────────────────────────────
@app.route("/api/summary", methods=["GET"])
def summary():
    df, e = resolve_session(request.args.get("session_key", ""))
    if e: return e
    return ok({"summary": dataset_summary(df)})


# ── States list ───────────────────────────────────────────────────────
@app.route("/api/states", methods=["GET"])
def states():
    df, e = resolve_session(request.args.get("session_key", ""))
    if e: return e
    return ok({"states": available_states(df)})


# ── Single-state forecast (all 4 models + best selection) ────────────
@app.route("/api/forecast", methods=["POST"])
def forecast():
    body = request.get_json(force=True, silent=True) or {}
    df, e = resolve_session(body.get("session_key", ""))
    if e: return e

    state   = (body.get("state") or "").strip()
    horizon = int(body.get("horizon", HORIZON))

    if not state:
        return err("'state' field is required.")

    all_states = available_states(df)
    if state not in all_states:
        return err(
            f"State '{state}' not found. "
            f"Available (first 8): {all_states[:8]}"
        )

    series = get_state_series(df, state)
    if series.empty:
        return err(f"No data rows found for state '{state}'.")

    try:
        result = run_all_models(series, horizon=horizon)
    except Exception as exc:
        logger.exception("Forecast error for %s", state)
        return err(str(exc))

    return ok({"state": state, "horizon": horizon, **result})


# ── Model comparison (all 4 models, one state) ────────────────────────
@app.route("/api/compare", methods=["POST"])
def compare():
    body = request.get_json(force=True, silent=True) or {}
    df, e = resolve_session(body.get("session_key", ""))
    if e: return e

    state = (body.get("state") or "").strip()
    if not state:
        return err("'state' is required.")

    series = get_state_series(df, state)
    if series.empty:
        return err(f"No data for '{state}'.")

    try:
        result = run_all_models(series)
    except Exception as exc:
        logger.exception("Compare error")
        return err(str(exc))

    comparison = [
        {
            "model":     name,
            "val_smape": info.get("val_smape"),
            "val_mae":   info.get("val_mae"),
            "status":    info.get("status"),
            "is_best":   name == result["best_model"],
            "forecast":  info.get("forecast", []),
        }
        for name, info in result["models"].items()
    ]

    return ok({
        "state":          state,
        "best_model":     result["best_model"],
        "forecast_dates": result["forecast_dates"],
        "comparison":     comparison,
        "history":        result["history"],
    })


# ── Batch forecast (all states, best-model only) ──────────────────────
@app.route("/api/forecast/batch", methods=["POST"])
def batch_forecast():
    body = request.get_json(force=True, silent=True) or {}
    df, e = resolve_session(body.get("session_key", ""))
    if e: return e

    horizon = int(body.get("horizon", HORIZON))
    states  = available_states(df)

    batch:  dict = {}
    errors: dict = {}
    for state in states:
        try:
            series = get_state_series(df, state)
            result = run_all_models(series, horizon=horizon)
            bm     = result["best_model"]
            batch[state] = {
                "best_model":     bm,
                "forecast":       result["models"][bm]["forecast"],
                "forecast_dates": result["forecast_dates"],
                "val_smape":      result["models"][bm]["val_smape"],
            }
        except Exception as exc:
            errors[state] = str(exc)

    return ok({"forecasts": batch, "errors": errors, "horizon": horizon})


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("🚀 Starting ForecastIQ on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=True)