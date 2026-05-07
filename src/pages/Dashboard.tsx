// src/pages/Dashboard.tsx
import type { Session } from "../App";

interface Props {
  session: Session;
  onForecast: (state: string) => void;
}

const MODEL_COLORS: Record<string, string> = {
  SARIMA:  "#38bdf8",
  Prophet: "#fb923c",
  XGBoost: "#4ade80",
  LSTM:    "#c084fc",
};

export default function Dashboard({ session, onForecast }: Props) {
  const { summary } = session;

  const statCards = [
    { label: "Total States",     value: summary.total_states,                   unit: ""   },
    { label: "Weekly Records",   value: summary.total_records.toLocaleString(), unit: ""   },
    { label: "Avg Weeks/State",  value: summary.avg_weeks_per_state,            unit: "wk" },
    { label: "Date Start",       value: summary.date_range.start,               unit: ""   },
    { label: "Date End",         value: summary.date_range.end,                 unit: ""   },
    { label: "Forecast Horizon", value: 8,                                       unit: "wk" },
  ];

  return (
    <div className="dashboard">
      {/* ── Header ── */}
      <div className="dash-header">
        <h2 className="dash-title">Dataset Overview</h2>
        <p className="dash-sub">Select any state to run the full 4-model forecast</p>
      </div>

      {/* ── Stat cards ── */}
      <div className="stat-grid">
        {statCards.map((c) => (
          <div key={c.label} className="stat-card">
            <span className="stat-val">
              {c.value}
              <span className="stat-unit">{c.unit}</span>
            </span>
            <span className="stat-label">{c.label}</span>
          </div>
        ))}
      </div>

      {/* ── Model legend ── */}
      <div className="model-legend">
        <span className="legend-title">Models that will be trained:</span>
        {Object.entries(MODEL_COLORS).map(([name, color]) => (
          <span key={name} className="legend-chip" style={{ borderColor: color, color }}>
            {name}
          </span>
        ))}
      </div>

      {/* ── States grid ── */}
      <div className="dash-states-header">
        <h3 className="dash-section-title">States</h3>
        <span className="dash-count">{summary.states.length} available</span>
      </div>

      <div className="states-grid">
        {summary.states.map((state, i) => (
          <button
            key={state}
            className="state-card"
            style={{ animationDelay: `${i * 18}ms` }}
            onClick={() => onForecast(state)}
          >
            <span className="sc-name">{state}</span>
            <span className="sc-action">Forecast →</span>
          </button>
        ))}
      </div>
    </div>
  );
}