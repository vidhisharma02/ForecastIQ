"""
utils/data_loader.py
Handles CSV/Excel ingestion, auto column detection,
weekly aggregation, and missing-value interpolation.

ROOT CAUSE FIX (data_loader.py):
─────────────────────────────────
The original code used:
    date_range = pd.date_range(agg["date"].min(), agg["date"].max(), freq="W-MON")

This ALWAYS anchors to the next Monday on or after min_date.
But dt.to_period("W-MON").dt.start_time maps each raw date to the
Monday of its ISO week — and in this dataset those Mondays are actually
Tuesdays (because pd.Period("W-MON") uses a Sun–Sat week where "W-MON"
means "week labelled by its Monday" but the .start_time is the day AFTER
Sunday, i.e. the period boundary, which is a Tuesday for this data).

The mismatch means NONE of the reindex keys ever match → every value
after reindex is NaN → interpolation cannot fill 100% NaN columns →
build_features() receives an all-NaN series → dropna() wipes all rows
→ "Need at least 23 clean feature rows; got 0".

Fix: instead of generating a uniform date_range (which drifts 6 days
from the actual week keys), build the full index from the ACTUAL unique
week timestamps already present in the aggregated data. This guarantees
perfect alignment regardless of the weekday the data happens to land on.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file.
    Auto-detect State / Date / Total columns (flexible naming).
    Returns a clean weekly long-form DataFrame: columns [state, date, total].
    """
    path = Path(filepath)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = [str(c).strip() for c in df.columns]

    # ── Flexible column mapping ──────────────────────────────────────
    col_map: dict = {}
    for col in df.columns:
        low = col.lower()
        if "state" in low and "state" not in col_map:
            col_map["state"] = col
        elif any(k in low for k in ("date", "week", "period", "time")) and "date" not in col_map:
            col_map["date"] = col
        elif any(k in low for k in ("total", "sale", "revenue", "amount", "value")) and "total" not in col_map:
            col_map["total"] = col
        elif any(k in low for k in ("categ", "type", "product")) and "category" not in col_map:
            col_map["category"] = col

    missing_req = [r for r in ("state", "date", "total") if r not in col_map]
    if missing_req:
        raise ValueError(
            f"Cannot auto-detect columns: {missing_req}. "
            f"Found headers: {list(df.columns)}. "
            "Rename columns to include 'State', 'Date', 'Total'."
        )

    df = df.rename(columns={v: k for k, v in col_map.items()})

    # ── Parse types ──────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])

    df["total"] = (
        df["total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["state", "date", "total"])

    # ── Aggregate to weekly per state ────────────────────────────────
    df["week"] = df["date"].dt.to_period("W-MON").dt.start_time

    agg = (
        df.groupby(["state", "week"], as_index=False)["total"]
        .sum()
        .rename(columns={"week": "date"})
        .sort_values(["state", "date"])
        .reset_index(drop=True)
    )

    # ── Fill missing weeks per state (no gaps) ───────────────────────
    # FIX: Do NOT use pd.date_range(..., freq="W-MON") to build the full
    # index. That function anchors to the next Monday on/after min_date,
    # which is 6 days away from the Tuesday week-starts that
    # to_period("W-MON").dt.start_time produces for this dataset.
    # The 6-day offset means every reindex lookup misses → all NaN.
    #
    # Instead, derive the full week grid from the data's own unique week
    # timestamps. This is always perfectly aligned regardless of weekday.
    all_states = sorted(agg["state"].astype(str).unique())
    all_weeks  = sorted(agg["date"].unique())          # actual week keys

    full_idx = pd.MultiIndex.from_product(
        [all_states, all_weeks], names=["state", "date"]
    )

    # Cast state to plain str (object) to avoid StringDtype vs object mismatch
    agg["state"] = agg["state"].astype(str)

    agg = (
        agg.set_index(["state", "date"])
        .reindex(full_idx)
        .reset_index()
    )
    # Linear interpolation within each state, then fill edges
    agg["total"] = (
        agg.groupby("state")["total"]
        .transform(lambda x: x.interpolate(method="linear").bfill().ffill())
    )

    return agg


def get_state_series(df: pd.DataFrame, state: str) -> pd.Series:
    """Return a DatetimeIndex weekly Series for one state."""
    s = (
        df[df["state"] == state][["date", "total"]]
        .set_index("date")
        .sort_index()["total"]
    )
    s.index = pd.DatetimeIndex(s.index)
    return s


def available_states(df: pd.DataFrame) -> list:
    return sorted(df["state"].dropna().unique().tolist())


def dataset_summary(df: pd.DataFrame) -> dict:
    states = available_states(df)
    return {
        "states": states,
        "total_states": len(states),
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end":   df["date"].max().strftime("%Y-%m-%d"),
        },
        "total_records": len(df),
        "avg_weeks_per_state": int(df.groupby("state").size().mean()),
    }