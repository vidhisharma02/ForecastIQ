// src/api/client.ts
// Fully typed API client for ForecastIQ backend

const BASE = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:5000";

// ── Shared types ─────────────────────────────────────────────────────
export interface DatasetSummary {
  states: string[];
  total_states: number;
  date_range: { start: string; end: string };
  total_records: number;
  avg_weeks_per_state: number;
}

export interface ModelResult {
  forecast: number[];
  val_smape: number;
  val_mae: number;
  status: "success" | "failed";
  error?: string;
}

export interface ForecastResult {
  state: string;
  horizon: number;
  best_model: string;
  forecast_dates: string[];
  models: Record<string, ModelResult>;
  history: { dates: string[]; values: number[] };
}

export interface CompareModel {
  model: string;
  val_smape: number | null;
  val_mae: number | null;
  status: string;
  is_best: boolean;
  forecast: number[];
}

export interface CompareResult {
  state: string;
  best_model: string;
  forecast_dates: string[];
  comparison: CompareModel[];
  history: { dates: string[]; values: number[] };
}

// ── Core fetch ──────────────────────────────────────────────────────
async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res  = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    ...init,
  });
  const json = await res.json();
  if (!res.ok || json.status === "error") {
    throw new Error(json.message ?? `HTTP ${res.status}`);
  }
  return json as T;
}

// ── API surface ─────────────────────────────────────────────────────
export const api = {
  /** Ping the backend */
  health: () =>
    apiFetch<{ message: string }>("/api/health"),

  /** Upload a CSV / Excel file; returns session_key + summary */
  upload: async (file: File): Promise<{ session_key: string; summary: DatasetSummary }> => {
    const form = new FormData();
    form.append("file", file);
    const res  = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
    const json = await res.json();
    if (!res.ok || json.status === "error") throw new Error(json.message ?? "Upload failed");
    return json;
  },

  /** List all state names for a session */
  states: (sessionKey: string) =>
    apiFetch<{ states: string[] }>(`/api/states?session_key=${sessionKey}`),

  /** Full dataset summary */
  summary: (sessionKey: string) =>
    apiFetch<{ summary: DatasetSummary }>(`/api/summary?session_key=${sessionKey}`),

  /**
   * Run all 4 models for one state, auto-select best.
   * Returns per-model forecasts + history.
   */
  forecast: (sessionKey: string, state: string, horizon = 8) =>
    apiFetch<ForecastResult>("/api/forecast", {
      method: "POST",
      body: JSON.stringify({ session_key: sessionKey, state, horizon }),
    }),

  /**
   * Compare all 4 models side-by-side for one state.
   */
  compare: (sessionKey: string, state: string) =>
    apiFetch<CompareResult>("/api/compare", {
      method: "POST",
      body: JSON.stringify({ session_key: sessionKey, state }),
    }),
};