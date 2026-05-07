// src/pages/UploadPage.tsx
import { useState, useCallback } from "react";
import type { DragEvent, ChangeEvent } from "react";
import { api } from "../api/client";
import type { Session } from "../App";

interface Props {
  onUploaded: (s: Session) => void;
}

export default function UploadPage({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setLoading(true);
    setProgress("Uploading file…");
    try {
      const res = await api.upload(file);
      setProgress("Processing dataset…");
      onUploaded({ key: res.session_key, summary: res.summary });
    } catch (e: any) {
      setError(e.message ?? "Upload failed");
    } finally {
      setLoading(false);
      setProgress("");
    }
  }, [onUploaded]);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="upload-page">
      <div className="upload-hero">
        <h1 className="upload-title">Time Series<br/>Forecasting System</h1>
        <p className="upload-sub">
          Upload your state sales CSV/Excel. We train SARIMA · Prophet · XGBoost · LSTM
          and auto-select the best model to forecast the next 8 weeks.
        </p>
      </div>

      <div
        className={`drop-zone ${dragging ? "dz-over" : ""} ${loading ? "dz-loading" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".csv,.xlsx,.xls"
          style={{ display: "none" }}
          onChange={onFileInput}
        />

        {loading ? (
          <div className="dz-spinner-wrap">
            <div className="dz-spinner" />
            <p className="dz-progress">{progress}</p>
          </div>
        ) : (
          <>
            <div className="dz-icon">⬆</div>
            <p className="dz-label">
              {dragging ? "Drop it!" : "Drag & drop your file here"}
            </p>
            <p className="dz-hint">CSV · XLSX · XLS &nbsp;|&nbsp; Max 100 MB</p>
            <button className="dz-btn">Browse Files</button>
          </>
        )}
      </div>

      {error && (
        <div className="upload-error">
          <span className="err-icon">⚠</span> {error}
        </div>
      )}

      <div className="upload-schema">
        <p className="schema-title">Expected columns in your file</p>
        <div className="schema-cols">
          {["State", "Date", "Total", "Category (optional)"].map((col) => (
            <span key={col} className="schema-tag">{col}</span>
          ))}
        </div>
        <p className="schema-note">
          Column names are detected automatically — exact spelling not required.
        </p>
      </div>
    </div>
  );
}