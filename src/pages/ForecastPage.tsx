// src/pages/ForecastPage.tsx
import { useState, useEffect, useCallback } from "react";
import type { Session } from "../App";
import { api } from "../api/client";
import type { CompareResult, CompareModel } from "../api/client";

interface Props {
  session:      Session;
  initialState: string;
  onBack:       () => void;
}

const MODEL_COLORS: Record<string, string> = {
  SARIMA:  "#38bdf8",
  Prophet: "#fb923c",
  XGBoost: "#4ade80",
  LSTM:    "#c084fc",
};

const LOADING_STEPS = [
  "Fitting SARIMA model…",
  "Training Prophet…",
  "Boosting XGBoost…",
  "Running LSTM…",
  "Selecting best model…",
];

export default function ForecastPage({ session, initialState, onBack }: Props) {
  const [activeState, setActiveState] = useState<string>(initialState);
  const [result,      setResult]      = useState<CompareResult | null>(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState<string | null>(null);
  const [stepIdx,     setStepIdx]     = useState(0);

  // Cycle through loading step messages
  useEffect(() => {
    if (!loading) return;
    setStepIdx(0);
    const iv = setInterval(() => setStepIdx((i) => (i + 1) % LOADING_STEPS.length), 1400);
    return () => clearInterval(iv);
  }, [loading]);

  const runForecast = useCallback(async (state: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.compare(session.key, state);
      setResult(res);
    } catch (e: any) {
      setError(e.message ?? "Forecast failed");
    } finally {
      setLoading(false);
    }
  }, [session.key]);

  useEffect(() => { runForecast(activeState); }, [activeState, runForecast]);

  const handleStateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setActiveState(e.target.value);
  };

  // ── Best model helpers ──────────────────────────────────────────
  const bestModel: CompareModel | undefined =
    result?.comparison.find((m) => m.is_best);

  // ── Mini bar chart data (normalised to max) ─────────────────────
  const maxForecast = (forecast: number[]) =>
    forecast.length ? Math.max(...forecast) : 1;

  const barHeights = (forecast: number[], color: string) => {
    const mx = maxForecast(forecast);
    return forecast.map((v, i) => (
      <div
        key={i}
        className="mc-bar"
        style={{
          height: `${Math.round((v / mx) * 32) + 4}px`,
          background: color,
          opacity: 0.7,
        }}
      />
    ));
  };

  // ── Inline SVG sparkline for the chart section ──────────────────
  const Sparkline = ({
    history,
    forecast,
    dates,
    color,
  }: {
    history:  { dates: string[]; values: number[] };
    forecast: number[];
    dates:    string[];
    color:    string;
  }) => {
    const histVals = history.values;
    const allVals  = [...histVals, ...forecast];
    const mn = Math.min(...allVals);
    const mx = Math.max(...allVals);
    const range = mx - mn || 1;

    const W = 800;
    const H = 180;
    const PAD = { t: 10, r: 20, b: 30, l: 48 };
    const plotW = W - PAD.l - PAD.r;
    const plotH = H - PAD.t - PAD.b;
    const total = allVals.length;

    const x = (i: number) => PAD.l + (i / (total - 1)) * plotW;
    const y = (v: number) => PAD.t + plotH - ((v - mn) / range) * plotH;

    const histPath = histVals
      .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join(" ");

    const fcPath = forecast
      .map((v, i) => {
        const xi = histVals.length - 1 + i;
        return `${i === 0 ? "M" : "L"}${x(xi).toFixed(1)},${y(v).toFixed(1)}`;
      })
      .join(" ");

    // Area fill for forecast
    const fcAreaPts = forecast.map((v, i) => {
      const xi = histVals.length - 1 + i;
      return `${x(xi).toFixed(1)},${y(v).toFixed(1)}`;
    });
    const fcArea = fcAreaPts.length
      ? `M${fcAreaPts[0]} L${fcAreaPts.join(" L")} L${x(histVals.length - 1 + forecast.length - 1).toFixed(1)},${(PAD.t + plotH).toFixed(1)} L${x(histVals.length - 1).toFixed(1)},${(PAD.t + plotH).toFixed(1)} Z`
      : "";

    // Y axis labels
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => mn + t * range);

    // X axis: show ~6 date labels
    const labelIdxs = Array.from({ length: 6 }, (_, i) =>
      Math.round((i / 5) * (total - 1))
    );

    const allDates = [...history.dates, ...dates];

    const fmt = (d: string) => {
      try {
        const dt = new Date(d);
        return `${dt.toLocaleString("default", { month: "short" })} '${String(dt.getFullYear()).slice(2)}`;
      } catch { return d; }
    };

    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
        {/* Grid lines */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD.l} y1={y(t).toFixed(1)}
              x2={W - PAD.r} y2={y(t).toFixed(1)}
              stroke="#1e2d3d" strokeWidth="1"
            />
            <text
              x={PAD.l - 6} y={y(t).toFixed(1)}
              fill="#3a5068" fontSize="9" textAnchor="end" dominantBaseline="middle"
            >
              {t >= 1000 ? `${(t / 1000).toFixed(1)}k` : t.toFixed(0)}
            </text>
          </g>
        ))}

        {/* Forecast area */}
        {fcArea && (
          <path d={fcArea} fill={color} fillOpacity="0.08" />
        )}

        {/* History/forecast divider */}
        {histVals.length > 0 && (
          <line
            x1={x(histVals.length - 1).toFixed(1)} y1={PAD.t}
            x2={x(histVals.length - 1).toFixed(1)} y2={PAD.t + plotH}
            stroke="#2a3f55" strokeWidth="1" strokeDasharray="4,3"
          />
        )}

        {/* History line */}
        <path d={histPath} fill="none" stroke="#3a5068" strokeWidth="1.5" />

        {/* Forecast line */}
        <path d={fcPath} fill="none" stroke={color} strokeWidth="2" />

        {/* Forecast dots */}
        {forecast.map((v, i) => {
          const xi = histVals.length - 1 + i;
          return (
            <circle
              key={i}
              cx={x(xi).toFixed(1)} cy={y(v).toFixed(1)}
              r="3" fill={color} fillOpacity="0.9"
            />
          );
        })}

        {/* X axis labels */}
        {labelIdxs.map((idx) => (
          <text
            key={idx}
            x={x(idx).toFixed(1)}
            y={H - 6}
            fill="#3a5068" fontSize="8.5" textAnchor="middle"
          >
            {allDates[idx] ? fmt(allDates[idx]) : ""}
          </text>
        ))}

        {/* Legend: History / Forecast */}
        <g transform={`translate(${PAD.l + 4}, ${PAD.t + 4})`}>
          <rect width="12" height="2" y="3" fill="#3a5068" rx="1" />
          <text x="16" y="8" fill="#5a7490" fontSize="8.5">History</text>
          <rect width="12" height="2" y="3" x="60" fill={color} rx="1" />
          <text x="76" y="8" fill="#5a7490" fontSize="8.5">Forecast</text>
        </g>
      </svg>
    );
  };

  // ── Render ───────────────────────────────────────────────────────
  return (
    <div className="forecast-page">
      {/* Header */}
      <div className="forecast-header">
        <div className="forecast-title-wrap">
          <p className="forecast-eyebrow">Forecast Analysis</p>
          <h2 className="forecast-state-name">{activeState}</h2>
        </div>
        <button className="forecast-back" onClick={onBack}>
          ← Back
        </button>
      </div>

      {/* State switcher */}
      <div className="state-selector-bar">
        <span className="state-selector-label">Switch state:</span>
        <select
          className="state-selector"
          value={activeState}
          onChange={handleStateChange}
        >
          {session.summary.states.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Loading */}
      {loading && (
        <div className="forecast-loading">
          <div className="fc-spinner" />
          <div className="fc-spinner-text">
            <p style={{ marginBottom: "0.75rem", color: "var(--text-dim)" }}>
              Training all 4 models for <strong style={{ color: "#fff" }}>{activeState}</strong>
            </p>
            <ul className="fc-spinner-steps">
              {LOADING_STEPS.map((step, i) => (
                <li key={step} className={`fc-step ${i === stepIdx ? "active" : ""}`}>
                  {i < stepIdx ? "✓ " : i === stepIdx ? "▶ " : "  "}
                  {step}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="forecast-error">
          <strong>⚠ Error:</strong> {error}
          <br />
          <button
            style={{
              marginTop: "1rem",
              background: "none",
              border: "1px solid var(--red)",
              borderRadius: "var(--radius)",
              color: "var(--red)",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: "0.78rem",
              padding: "5px 14px",
            }}
            onClick={() => runForecast(activeState)}
          >
            Retry
          </button>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Best model banner */}
          {bestModel && (
            <div className="best-model-banner">
              <div>
                <p className="bm-label">Best model selected</p>
                <p className="bm-name">{result.best_model}</p>
              </div>
              <span className="bm-badge">Auto-selected</span>
              <div className="bm-metrics">
                {bestModel.val_smape != null && (
                  <div className="bm-metric">
                    <span className="bm-metric-val">{bestModel.val_smape.toFixed(1)}%</span>
                    <span className="bm-metric-label">sMAPE</span>
                  </div>
                )}
                {bestModel.val_mae != null && (
                  <div className="bm-metric">
                    <span className="bm-metric-val">{bestModel.val_mae.toFixed(0)}</span>
                    <span className="bm-metric-label">MAE</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Sparkline chart for best model */}
          {bestModel && bestModel.forecast.length > 0 && (
            <div className="chart-section">
              <h3 className="chart-section-title">
                {result.best_model} — History + 8-Week Forecast
              </h3>
              <Sparkline
                history={result.history}
                forecast={bestModel.forecast}
                dates={result.forecast_dates}
                color={MODEL_COLORS[result.best_model] ?? "#00d4ff"}
              />
            </div>
          )}

          {/* Model comparison cards */}
          <div className="dash-states-header" style={{ marginBottom: "1rem" }}>
            <h3 className="dash-section-title">All Models</h3>
            <span className="dash-count">4 trained</span>
          </div>

          <div className="models-grid">
            {result.comparison.map((m) => {
              const color = MODEL_COLORS[m.model] ?? "#00d4ff";
              return (
                <div
                  key={m.model}
                  className={`model-card ${m.is_best ? "is-best" : ""}`}
                  style={{ "--model-color": color } as React.CSSProperties}
                >
                  <div className="mc-header">
                    <span className="mc-name">{m.model}</span>
                    {m.is_best && <span className="mc-best-tag">Best</span>}
                    {m.status === "failed" && <span className="mc-failed-tag">Failed</span>}
                  </div>

                  {m.status !== "failed" ? (
                    <>
                      <div className="mc-metrics">
                        <div className="mc-metric">
                          <span className="mc-metric-val">
                            {m.val_smape != null ? `${m.val_smape.toFixed(1)}%` : "—"}
                          </span>
                          <span className="mc-metric-label">sMAPE</span>
                        </div>
                        <div className="mc-metric">
                          <span className="mc-metric-val">
                            {m.val_mae != null ? m.val_mae.toFixed(0) : "—"}
                          </span>
                          <span className="mc-metric-label">MAE</span>
                        </div>
                      </div>
                      {m.forecast.length > 0 && (
                        <div
                          className="mc-forecast-preview"
                          style={{ alignItems: "flex-end", height: "36px" }}
                        >
                          {barHeights(m.forecast, color)}
                        </div>
                      )}
                    </>
                  ) : (
                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
                      Model training failed
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Forecast table */}
          {bestModel && bestModel.forecast.length > 0 && (
            <>
              <div className="dash-states-header" style={{ marginBottom: "1rem" }}>
                <h3 className="dash-section-title">
                  {result.best_model} — Weekly Forecast
                </h3>
              </div>
              <div className="forecast-table-wrap">
                <table className="forecast-table">
                  <thead>
                    <tr>
                      <th>Week</th>
                      <th>Date</th>
                      <th>Forecast</th>
                      <th>Δ vs prev</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bestModel.forecast.map((val, i) => {
                      const prev  = i === 0 ? null : bestModel.forecast[i - 1];
                      const delta = prev != null ? val - prev : null;
                      const pct   = delta != null && prev ? (delta / prev) * 100 : null;
                      return (
                        <tr key={i}>
                          <td className="ft-week">Week {i + 1}</td>
                          <td className="ft-week">{result.forecast_dates[i] ?? "—"}</td>
                          <td className="ft-val">{val.toFixed(0)}</td>
                          <td>
                            {pct != null ? (
                              <span
                                className={`ft-delta ${
                                  pct > 0.5 ? "up" : pct < -0.5 ? "down" : "flat"
                                }`}
                              >
                                {pct > 0 ? "+" : ""}
                                {pct.toFixed(1)}%
                              </span>
                            ) : (
                              <span className="ft-delta flat">—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}