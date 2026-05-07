// src/App.tsx
import { useState } from "react";
import UploadPage from "./pages/UploadPages";
import Dashboard from "./pages/Dashboard";
import ForecastPage from "./pages/ForecastPage";
import "./index.css";

export type View = "upload" | "dashboard" | "forecast";

export interface Session {
  key: string;
  summary: {
    states: string[];
    total_states: number;
    date_range: {
      start: string;
      end: string;
    };
    total_records: number;
    avg_weeks_per_state: number;
  };
}

export default function App() {
  const [view, setView] = useState<View>("upload");
  const [session, setSession] = useState<Session | null>(null);
  const [activeState, setActiveState] = useState<string>("");

  const onUploaded = (s: Session) => {
    setSession(s);
    setView("dashboard");
  };

  const onForecast = (state: string) => {
    setActiveState(state);
    setView("forecast");
  };

  return (
    <div className="app-shell">
      {/* NAVBAR */}
      <header className="navbar">
        <button
          className="brand"
          onClick={() => session && setView("dashboard")}
        >
          <span className="brand-glyph">◈</span>
          <span className="brand-name">ForecastIQ</span>
          <span className="brand-tag">Sales Intelligence</span>
        </button>

        <nav className="nav-items">
          {session && (
            <>
              <button
                className={`nav-pill ${
                  view === "dashboard" ? "nav-active" : ""
                }`}
                onClick={() => setView("dashboard")}
              >
                Dashboard
              </button>

              <button
                className={`nav-pill ${
                  view === "forecast" ? "nav-active" : ""
                }`}
                onClick={() => activeState && setView("forecast")}
                disabled={!activeState}
              >
                Forecast
              </button>
            </>
          )}

          <button
            className="nav-pill nav-upload"
            onClick={() => setView("upload")}
          >
            {session ? "↑ New Dataset" : "Upload"}
          </button>
        </nav>
      </header>

      {/* STATUS BAR */}
      {session && (
        <div className="status-strip">
          <span className="s-dot" />

          <span className="s-text">
            {session.summary.total_states} states
            <span className="s-sep"> · </span>
            {session.summary.date_range.start} →{" "}
            {session.summary.date_range.end}
            <span className="s-sep"> · </span>
            {session.summary.total_records.toLocaleString()} records
          </span>

          <span className="s-file">{session.key}</span>
        </div>
      )}

      {/* PAGE */}
      <main className="page-area">
        {view === "upload" && (
          <UploadPage onUploaded={onUploaded} />
        )}

        {view === "dashboard" && session && (
          <Dashboard
            session={session}
            onForecast={onForecast}
          />
        )}

        {view === "forecast" && session && (
          <ForecastPage
            session={session}
            initialState={activeState}
            onBack={() => setView("dashboard")}
          />
        )}
      </main>
    </div>
  );
}