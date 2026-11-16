"""
WaferGuard AI — premium semiconductor inspection dashboard.

Run:
    pip install streamlit plotly pandas numpy scikit-learn requests python-dotenv reportlab
    streamlit run app.py

Model modes:
- If models/wafer_defect_classifier_v9 exists, the app loads the validated local
  MLflow Extra Trees model and performs inference without Databricks API calls.
- If the local model is absent but Databricks credentials are configured, the
  app uses the Databricks Serving endpoint.
- Otherwise, a clearly labelled synthetic Random Forest DEMO model is used.

Databricks endpoint contract expected by this app:
    {"dataframe_records": [{"mean": ..., ..., "segment_8_range": ...}]}

Expected response columns:
    predicted_class, predicted_label, abnormal_probability, prediction_confidence
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import mlflow.sklearn

from utils.data_processor import FEATURE_COLUMNS, engineer_features
from utils.database import (
    init_database, save_feedback, save_inspection, database_configured,
    create_user, authenticate_user, fetch_recent_inspections,
)

load_dotenv()
# Page configuration and design tokens
st.set_page_config(
    page_title="WaferGuard AI | Semiconductor Quality Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAVY = "#0B1220"
NAVY_2 = "#101D33"
BLUE = "#3478F6"
CYAN = "#36BFFA"
GREEN = "#20C997"
RED = "#F05260"
AMBER = "#F5B942"
TEXT = "#EAF1FA"
MUTED = "#9AAAC0"
CARD = "rgba(19, 33, 54, 0.88)"
BORDER = "rgba(148, 170, 201, 0.18)"

# Optional background image: set WAFER_BG_IMAGE to a local image path.
# Image is rendered as a low-opacity background layer; cards remain opaque.
BG_PATH = os.getenv("WAFER_BG_IMAGE", "assets/wafer_background.jpg")
bg_css = ""
if Path(BG_PATH).is_file():
    import base64
    raw = Path(BG_PATH).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = Path(BG_PATH).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    bg_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
          linear-gradient(rgba(5, 12, 25, .84), rgba(5, 12, 25, .91)),
          url("data:{mime};base64,{encoded}") center / cover no-repeat;
        opacity: .15;
        filter: blur(2px);
    }}
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {{
      --wg-bg: {NAVY}; --wg-card: {CARD}; --wg-border: {BORDER};
      --wg-blue: {BLUE}; --wg-text: {TEXT}; --wg-muted: {MUTED};
    }}
    html, body, [class*="css"], .stApp {{
      font-family: 'Inter', sans-serif !important;
    }}
    .stApp {{
      background:
        radial-gradient(ellipse at 12% 0%, rgba(52,120,246,.16), transparent 32%),
        radial-gradient(ellipse at 92% 18%, rgba(54,191,250,.09), transparent 28%),
        linear-gradient(145deg, #0B1220 0%, #0D192B 52%, #08111F 100%);
      color: var(--wg-text);
    }}
    {bg_css}
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stMainBlockContainer"] {{
      position: relative; z-index: 1;
      max-width: 1600px; padding: 1.4rem 2.2rem 3rem;
    }}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
      display: none !important;
    }}
    h1,h2,h3,h4,p,label,span,div {{ letter-spacing: -.01em; }}
    h1,h2,h3,h4 {{ color: #F4F8FF !important; }}
    p, label, .stMarkdown {{ color: #D7E2F0; }}
    .topbar {{
      display:flex; align-items:center; justify-content:space-between;
      gap:1rem; padding: 15px 20px; margin-bottom: 22px;
      border:1px solid var(--wg-border); border-radius:16px;
      background:rgba(15,28,47,.88); backdrop-filter: blur(16px);
      box-shadow:0 12px 34px rgba(0,0,0,.18);
    }}
    .brand {{ display:flex; align-items:center; gap:12px; }}
    .brand-mark {{
      width:42px;height:42px;border-radius:12px;display:grid;place-items:center;
      background:linear-gradient(145deg,#3478F6,#36BFFA);
      color:white;font-size:21px;font-weight:800;
      box-shadow:0 6px 20px rgba(52,120,246,.32);
    }}
    .brand-name {{font-weight:800;font-size:17px;color:#F7FAFF;letter-spacing:.08em;}}
    .brand-sub {{font-size:11px;color:#91A4BE;margin-top:2px;}}
    .status-wrap {{display:flex;gap:9px;align-items:center;flex-wrap:wrap;}}
    .status-pill {{
      display:inline-flex;align-items:center;gap:8px;padding:8px 12px;
      border:1px solid var(--wg-border);border-radius:999px;
      background:rgba(255,255,255,.045);font-size:11px;font-weight:700;
      color:#DCE8F7;white-space:nowrap;
    }}
    .live-dot {{width:8px;height:8px;border-radius:50%;background:#20C997;
      box-shadow:0 0 0 4px rgba(32,201,151,.13);}}
    .demo-dot {{background:#F5B942;box-shadow:0 0 0 4px rgba(245,185,66,.13);}}
    .hero {{padding: 7px 2px 20px;}}
    .eyebrow {{font-size:10px;font-weight:800;letter-spacing:.18em;color:#75B6FF;text-transform:uppercase;}}
    .hero-title {{font-size:clamp(27px,3vw,40px);font-weight:800;color:#F5F8FE;line-height:1.16;margin:8px 0;}}
    .hero-copy {{font-size:14px;color:#AABBD0;max-width:800px;line-height:1.65;}}
    .section-heading {{font-size:17px;font-weight:750;color:#F2F6FC;margin:10px 0 13px;}}
    .section-caption {{font-size:12px;color:#91A4BE;margin-top:-8px;margin-bottom:14px;}}
    .kpi {{
      min-height:112px;padding:17px 18px;border-radius:17px;
      border:1px solid var(--wg-border);background:linear-gradient(145deg,rgba(25,43,68,.96),rgba(15,28,47,.92));
      box-shadow:0 10px 30px rgba(0,0,0,.15);
    }}
    .kpi-label {{font-size:11px;color:#9FB0C6;font-weight:650;}}
    .kpi-value {{font-size:29px;font-weight:800;color:#F5F8FE;margin-top:8px;}}
    .kpi-note {{font-size:10px;color:#8296B1;margin-top:3px;}}
    .panel {{
      border:1px solid var(--wg-border);border-radius:20px;
      background:linear-gradient(145deg,rgba(25,41,64,.96),rgba(14,27,45,.96));
      padding:20px;box-shadow:0 16px 42px rgba(0,0,0,.18);
      min-height:100%;
    }}
    .panel-title {{font-size:17px;font-weight:750;color:#F5F8FE;margin-bottom:4px;}}
    .panel-sub {{font-size:12px;color:#91A4BE;margin-bottom:16px;}}
    .result-card {{
      border-radius:18px;padding:21px;border:1px solid;
      margin:10px 0 15px;
    }}
    .result-good {{background:linear-gradient(120deg,rgba(14,92,73,.38),rgba(14,48,54,.62));border-color:rgba(32,201,151,.38);}}
    .result-bad {{background:linear-gradient(120deg,rgba(117,35,49,.42),rgba(58,25,42,.62));border-color:rgba(240,82,96,.42);}}
    .result-label {{font-size:27px;font-weight:850;letter-spacing:.02em;margin:4px 0;}}
    .result-good .result-label {{color:#5BE0B1;}}
    .result-bad .result-label {{color:#FF7C87;}}
    .result-sub {{font-size:12px;color:#D4DFEC;line-height:1.6;}}
    .risk-chip {{display:inline-block;border-radius:999px;padding:5px 10px;font-size:10px;font-weight:800;letter-spacing:.08em;}}
    .risk-low {{background:rgba(32,201,151,.16);color:#5BE0B1;}}
    .risk-medium {{background:rgba(245,185,66,.16);color:#FFD06A;}}
    .risk-high {{background:rgba(240,82,96,.16);color:#FF8790;}}
    .stButton>button,.stDownloadButton>button {{
      border-radius:11px !important;min-height:43px !important;
      font-weight:700 !important;border:1px solid rgba(117,163,232,.35) !important;
      transition:all .18s ease !important;
    }}
    .stButton>button[kind="primary"] {{
      background:linear-gradient(110deg,#2868E8,#368BFF) !important;color:white !important;
      box-shadow:0 8px 22px rgba(40,104,232,.24) !important;
    }}
    .stButton>button:not([kind="primary"]),.stDownloadButton>button {{
      background:rgba(255,255,255,.055) !important;color:#DCE8F7 !important;
    }}
    .stButton>button:hover,.stDownloadButton>button:hover {{
      border-color:#63A7FF !important;transform:translateY(-1px);
    }}
    [data-testid="stFileUploader"] {{
      border:1px dashed rgba(112,158,220,.48);border-radius:13px;
      background:rgba(5,14,27,.32);padding:12px;
    }}
    [data-testid="stFileUploader"] * {{color:#DCE8F7 !important;}}
    textarea,input {{background:#101F34 !important;color:#F1F6FC !important;}}
    [data-baseweb="tab-list"] {{gap:8px;border-bottom:1px solid var(--wg-border);}}
    [data-baseweb="tab"] {{background:transparent;border-radius:10px 10px 0 0;padding:10px 16px;}}
    [aria-selected="true"] {{color:#75B6FF !important;}}
    [data-testid="stDataFrame"] {{border:1px solid var(--wg-border);border-radius:12px;overflow:hidden;}}
    .footer {{text-align:center;color:#7186A2;font-size:10px;padding:25px 0 4px;letter-spacing:.04em;}}
    .pipeline-track {{
      display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;
      margin:14px 0 20px;
    }}
    .pipeline-step {{
      min-height:91px;padding:13px 14px;border-radius:13px;
      border:1px solid rgba(148,170,201,.16);background:rgba(255,255,255,.035);
    }}
    .pipeline-step.done {{border-color:rgba(32,201,151,.45);background:rgba(32,201,151,.09);}}
    .pipeline-step.active {{border-color:rgba(54,143,255,.72);background:rgba(52,120,246,.16);box-shadow:0 0 0 1px rgba(52,120,246,.12) inset;}}
    .pipeline-number {{font-size:10px;font-weight:800;letter-spacing:.08em;color:#7EA0C8;}}
    .pipeline-step.done .pipeline-number {{color:#5BE0B1;}}
    .pipeline-step.active .pipeline-number {{color:#7DBAFF;}}
    .pipeline-title {{font-size:11px;font-weight:800;color:#EAF2FC;margin-top:9px;}}
    .pipeline-detail {{font-size:10px;color:#91A4BE;margin-top:4px;line-height:1.4;}}
    @media(max-width:850px) {{.pipeline-track{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
    @media(max-width:480px) {{.pipeline-track{{grid-template-columns:1fr;}}}}

    .small-muted {{font-size:11px;color:#8FA2BC;}}
    div[data-testid="stAlert"] {{border-radius:12px;}}
    </style>
    """,
    unsafe_allow_html=True,
)
# Model loading: validated local MLflow model first; Databricks remains optional.
LOCAL_MODEL_PATH = Path(
    os.getenv("WAFERGUARD_LOCAL_MODEL_PATH", "models/wafer_defect_classifier_v9")
)


def databricks_configured() -> bool:
    """Return true when Databricks credentials and endpoint are configured."""
    endpoint = os.getenv("DATABRICKS_ENDPOINT") or os.getenv("DATABRICKS_SERVING_ENDPOINT")
    return bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN") and endpoint)


@st.cache_resource(show_spinner="Loading WaferGuard local MLflow model…")
def load_local_model(model_path: str):
    """Load the validated Extra Trees model downloaded from Databricks MLflow."""
    model = mlflow.sklearn.load_model(model_path)
    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Local model expects {getattr(model, 'n_features_in_', 'unknown')} features; "
            f"the application supplies {len(FEATURE_COLUMNS)}."
        )
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError(f"Unexpected local model classes: {getattr(model, 'classes_', None)}")
    return model


@st.cache_data(ttl=20, show_spinner=False)
def check_databricks_connection() -> tuple[bool, str]:
    """Check Databricks only when its serving endpoint is explicitly selected."""
    if not databricks_configured():
        return False, "Databricks credentials or endpoint not configured"
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "https://" + host
    endpoint = os.getenv("DATABRICKS_ENDPOINT") or os.getenv("DATABRICKS_SERVING_ENDPOINT")
    try:
        response = requests.get(
            f"{host}/api/2.0/serving-endpoints/{endpoint}",
            headers={"Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN', '')}"},
            timeout=(5, 10),
        )
        response.raise_for_status()
        state = response.json().get("state", {})
        ready = str(state.get("ready", "")).upper() == "READY"
        return ready, str(state.get("ready", "Unknown"))
    except requests.RequestException as exc:
        return False, f"Connection check failed: {exc.__class__.__name__}"
    except (ValueError, AttributeError):
        return False, "Unexpected endpoint status response"


def call_databricks_batch(feature_records: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Send one batch through the shared, validated Databricks client."""
    from utils.databricks_client import predict_wafer_in_batches

    predictions = predict_wafer_in_batches(feature_records)
    return [
        {
            **prediction,
            "confidence": prediction["prediction_confidence"],
            "source": "Databricks Model Serving",
            "features": feature_records[index],
        }
        for index, prediction in enumerate(predictions)
    ]


def call_databricks(values: np.ndarray) -> dict[str, Any]:
    """Compatibility wrapper for a single wafer."""
    return call_databricks_batch([engineer_features(values)])[0]


@st.cache_resource(show_spinner="Preparing local demonstration model…")
def build_demo_model() -> tuple[RandomForestClassifier, dict[str, float]]:
    """
    Train a synthetic demonstration RF. This is NOT trained on the UCR Wafer data.
    It exists only as a clearly labelled fallback when no real model is available.
    """
    rng = np.random.default_rng(2026)
    n = 1800
    X = rng.normal(0, 0.18, size=(n, 152))
    y = rng.binomial(1, 0.22, size=n)
    for i in np.where(y == 1)[0]:
        center = int(rng.integers(85, 116))
        width = int(rng.integers(3, 11))
        amplitude = float(rng.uniform(0.45, 1.05))
        lo, hi = max(0, center-width), min(152, center+width)
        X[i, lo:hi] += amplitude
        if lo > 0:
            X[i, lo-2:lo] += amplitude * 0.15
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_val, p)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, pred, labels=[0, 1]).tolist(),
    }
    return model, metrics


def load_model() -> dict[str, Any]:
    """Prefer the locally validated Version 9 model; retain Databricks as fallback."""
    if LOCAL_MODEL_PATH.exists():
        model = load_local_model(str(LOCAL_MODEL_PATH.resolve()))
        return {
            "mode": "local_mlflow",
            "model": model,
            "name": "Local MLflow Extra Trees · Version 9",
        }
    if databricks_configured():
        return {
            "mode": "databricks",
            "name": os.getenv("DATABRICKS_ENDPOINT") or os.getenv("DATABRICKS_SERVING_ENDPOINT"),
        }
    model, metrics = build_demo_model()
    return {"mode": "demo_rf", "model": model, "metrics": metrics, "name": "Synthetic RF Demo"}


MODEL_INFO = load_model()
IS_LOCAL_MODEL = MODEL_INFO["mode"] == "local_mlflow"
IS_DATABRICKS_MODEL = MODEL_INFO["mode"] == "databricks"
IS_REAL_MODEL = IS_LOCAL_MODEL or IS_DATABRICKS_MODEL
MODEL_LABEL = (
    "Local Extra Trees · MLflow Version 9" if IS_LOCAL_MODEL
    else "Databricks Model Serving · Extra Trees v10" if IS_DATABRICKS_MODEL
    else "Synthetic Random Forest · DEMO ONLY"
)
# Synthetic sample wafer creation and prediction helpers
def make_sample(kind: str, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 152)
    values = 0.05 * np.sin(2 * np.pi * x * 3) + rng.normal(0, 0.025, 152)
    if kind == "abnormal":
        values[88:111] += np.hanning(23) * 0.85
        values[113:118] -= np.linspace(0.12, 0.02, 5)
    return values.astype(float)


def predict_values(values: np.ndarray) -> dict[str, Any]:
    """Predict one wafer using the selected real model or labelled demo fallback."""
    if IS_DATABRICKS_MODEL:
        return call_databricks(values)

    if IS_LOCAL_MODEL:
        feature_record = engineer_features(values)
        feature_vector = np.asarray(
            [feature_record[column] for column in FEATURE_COLUMNS],
            dtype=float,
        ).reshape(1, -1)
        model = MODEL_INFO["model"]
        probability = float(model.predict_proba(feature_vector)[0, 1])
        source = "Local MLflow Model · Extra Trees v9"
        features = feature_record
    else:
        model = MODEL_INFO["model"]
        probability = float(model.predict_proba(np.asarray(values).reshape(1, -1))[0, 1])
        source = "Synthetic Random Forest DEMO"
        features = None

    pred = int(probability >= 0.5)
    return {
        "predicted_class": pred,
        "predicted_label": "Abnormal" if pred else "Normal",
        "abnormal_probability": probability,
        "confidence": max(probability, 1 - probability),
        "source": source,
        "features": features,
    }


def risk_level(probability: float) -> tuple[str, str, str]:
    if probability >= 0.75:
        return "HIGH", "risk-high", "Hold the wafer for quality review. Do not release automatically."
    if probability >= 0.40:
        return "MEDIUM", "risk-medium", "Route for secondary inspection before release."
    return "LOW", "risk-low", "Continue through the normal quality-control workflow."


def parse_values_text(text: str) -> np.ndarray:
    cleaned = text.replace("\n", ",").replace("\t", ",").replace(";", ",")
    vals = [v.strip() for v in cleaned.split(",") if v.strip()]
    try:
        arr = np.asarray([float(v) for v in vals], dtype=float)
    except ValueError as exc:
        raise ValueError("Paste comma-separated numeric sensor readings only.") from exc
    if len(arr) != 152:
        raise ValueError(f"Expected 152 readings; found {len(arr)}.")
    if not np.isfinite(arr).all():
        raise ValueError("All 152 readings must be finite numbers.")
    return arr


def parse_sensor_upload(uploaded) -> list[tuple[str, np.ndarray, int | None]]:
    """
    Read a Wafer .ts, CSV or Excel file.
    Each observation must contain 152 measurements; an optional final label is retained.
    """
    filename = uploaded.name
    suffix = Path(filename).suffix.lower()
    records: list[tuple[str, np.ndarray, int | None]] = []

    if suffix == ".ts":
        uploaded.seek(0)
        raw_text = uploaded.getvalue().decode("utf-8", errors="replace")
        for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("@") or line.startswith("%"):
                continue
            if line.lower() in {"timestamps", "classlabel"}:
                continue
            label = None
            if ":" in line:
                signal_text, label_text = line.rsplit(":", 1)
                try:
                    label_value = int(float(label_text.strip()))
                    label = 1 if label_value == -1 else 0 if label_value == 1 else None
                except ValueError:
                    signal_text = line
            else:
                signal_text = line
            signal_text = signal_text.strip().strip(",")
            try:
                values = np.asarray([float(v.strip()) for v in signal_text.split(",") if v.strip()], dtype=float)
            except ValueError:
                # Skip non-data metadata/header lines.
                continue
            if values.size == 0:
                continue
            if len(values) != 152:
                raise ValueError(f"Line {line_no}: expected 152 sensor values, found {len(values)}.")
            if not np.isfinite(values).all():
                raise ValueError(f"Line {line_no}: sensor values contain NaN or infinity.")
            records.append((f"Observation {len(records)+1:04d}", values, label))
    else:
        uploaded.seek(0)
        if suffix == ".csv":
            frame = pd.read_csv(uploaded, header=None)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(uploaded, header=None)
        else:
            raise ValueError("Supported formats are .ts, .csv, .xlsx and .xls.")

        if frame.empty:
            raise ValueError("The uploaded file contains no observations.")

        # A single column of 152 rows is treated as one wafer sequence.
        if frame.shape == (152, 1):
            frame = frame.T

        for row_idx, row in frame.iterrows():
            cells = list(row)
            # Strip an optional trailing class label or ID when the row has 153 values.
            label = None
            if len(cells) == 153:
                try:
                    label_value = int(float(cells[-1]))
                    label = 1 if label_value == -1 else 0 if label_value == 1 else None
                    cells = cells[:152]
                except (TypeError, ValueError):
                    cells = cells[-152:]
            elif len(cells) > 152:
                cells = cells[-152:]
            try:
                values = np.asarray([float(v) for v in cells], dtype=float)
            except (TypeError, ValueError):
                # Skip a header row if it is not numeric.
                if row_idx == 0:
                    continue
                raise ValueError(f"Row {row_idx+1}: sensor columns must be numeric.")
            if len(values) != 152:
                raise ValueError(f"Row {row_idx+1}: expected 152 sensor values, found {len(values)}.")
            if not np.isfinite(values).all():
                raise ValueError(f"Row {row_idx+1}: sensor values contain NaN or infinity.")
            records.append((f"Observation {len(records)+1:04d}", values, label))

    if not records:
        raise ValueError("No valid 152-reading observations were found in the uploaded file.")
    if len(records) > 10000:
        raise ValueError("A maximum of 10,000 observations can be processed in one upload.")
    return records


def fig_layout(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,32,.25)",
        font=dict(family="Inter, sans-serif", color="#DCE7F5", size=11),
        margin=dict(l=18, r=18, t=55 if title else 20, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#14243A", bordercolor="#466080", font_color="#F5F8FC"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,175,205,.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,175,205,.12)", zeroline=False)
    return fig


def create_pdf_report(result: dict[str, Any], values: np.ndarray, wafer_name: str) -> bytes:
    """Create a compact PDF report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="WGTitle", parent=styles["Title"], textColor=colors.HexColor("#12345A")))
    story = [
        Paragraph("WAFERGUARD AI — INSPECTION REPORT", styles["WGTitle"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 8*mm),
    ]
    p = result["abnormal_probability"]
    label = result["predicted_label"]
    risk, _, action = risk_level(p)
    rows = [
        ["Inspection field", "Result"],
        ["Wafer / file", wafer_name],
        ["Classification", label.upper()],
        ["Abnormal probability", f"{p*100:.2f}%"],
        ["Prediction confidence", f"{result['confidence']*100:.2f}%"],
        ["Risk level", risk],
        ["Model source", result["source"]],
        ["Sensor measurements", "152"],
        ["Recommended action", action],
    ]
    table = Table(rows, colWidths=[52*mm, 112*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173B66")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#F1F5FA")),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#CFD9E5")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.extend([table, Spacer(1, 7*mm),
                  Paragraph("Sensor summary", styles["Heading2"]),
                  Paragraph(
                      f"Mean: {np.mean(values):.5f} &nbsp; | &nbsp; "
                      f"Std. dev.: {np.std(values):.5f} &nbsp; | &nbsp; "
                      f"Min: {np.min(values):.5f} &nbsp; | &nbsp; "
                      f"Max: {np.max(values):.5f}",
                      styles["Normal"]),
                  Spacer(1, 5*mm),
                  Paragraph("This report is a decision-support output. Follow the site's approved quality procedures before disposition.", styles["Italic"])])
    doc.build(story)
    return buffer.getvalue()
# Session state
defaults = {
    "current_values": None,
    "current_result": None,
    "current_name": "No wafer selected",
    "history": [],
    "batch_results": None,
    "uploaded_records": None,
        "last_prediction_time": None,
    "pipeline_completed": False,
    "feedback_submitted": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def record_prediction(name: str, values: np.ndarray, result: dict[str, Any]) -> None:
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wafer": name,
        "result": result["predicted_label"],
        "confidence": result["confidence"],
        "abnormal_probability": result["abnormal_probability"],
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:100]
    st.session_state.current_values = np.asarray(values, dtype=float)
    st.session_state.current_result = result
    st.session_state.current_name = name
    st.session_state.last_prediction_time = entry["time"]
    if database_configured():
        try:
            save_inspection(name, result)
        except Exception as exc:
            st.session_state.database_warning = str(exc)


def render_pipeline(placeholder, active_step: int, completed: bool = False) -> None:
    """Render the upload → Databricks → output-generation → result workflow."""
    labels = [
        ("01", "DATA UPLOADED", "File received and validated"),
        ("02", "LOCAL MODEL INFERENCE" if IS_LOCAL_MODEL else "DATABRICKS PROCESSING",
         "34-feature engineering and local model prediction" if IS_LOCAL_MODEL else "Feature engineering and model inference"),
        ("03", "GENERATING OUTPUT", "Aggregating predictions and confidence"),
        ("04", "FINAL OUTPUT", "Inspection results are ready"),
    ]
    cards = []
    for idx, (number, title, detail) in enumerate(labels, start=1):
        if completed or idx < active_step:
            state, symbol = "done", "✓"
        elif idx == active_step:
            state, symbol = "active", "●"
        else:
            state, symbol = "waiting", "○"
        cards.append(
            f'<div class="pipeline-step {state}">'
            f'<div class="pipeline-number">{symbol} {number}</div>'
            f'<div class="pipeline-title">{title}</div>'
            f'<div class="pipeline-detail">{detail}</div></div>'
        )
    placeholder.markdown(
        '<div class="pipeline-track">' + "".join(cards) + '</div>',
        unsafe_allow_html=True,
    )


def run_uploaded_inspection(
    records: list[tuple[str, np.ndarray, int | None]],
    filename: str,
    pipeline_slot,
) -> None:
    """Inspect uploaded wafers in batches; do not save partial output as complete."""
    if not records:
        raise ValueError("No wafer observations were supplied.")
    batch_size = int(os.getenv("WAFERGUARD_BATCH_SIZE", "25"))
    if not 1 <= batch_size <= 100:
        raise ValueError("WAFERGUARD_BATCH_SIZE must be between 1 and 100.")

    output_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing uploaded observations…")
    st.session_state.pipeline_completed = False
    render_pipeline(pipeline_slot, 1)
    try:
        render_pipeline(pipeline_slot, 2)
        total = len(records)
        for batch_start in range(0, total, batch_size):
            group = records[batch_start:batch_start + batch_size]
            if IS_DATABRICKS_MODEL:
                features = [engineer_features(values) for _, values, _ in group]
                predictions = call_databricks_batch(features)
            else:
                predictions = [predict_values(values) for _, values, _ in group]
            if len(predictions) != len(group):
                raise RuntimeError(
                    f"Expected {len(group)} predictions for batch starting at "
                    f"{batch_start + 1}, received {len(predictions)}."
                )
            for (name, values, actual_label), result in zip(group, predictions):
                all_results.append(result)
                output_rows.append({
                    "Observation": name,
                    "Result": result["predicted_label"],
                    "Predicted class": result["predicted_label"],
                    "Predicted class code": int(result["predicted_class"]),
                    "Abnormal probability (%)": round(float(result["abnormal_probability"]) * 100, 2),
                    "Confidence (%)": round(float(result["confidence"]) * 100, 2),
                    "Actual class": ("Abnormal" if actual_label == 1 else "Normal") if actual_label is not None else "Not supplied",
                    "Correct": (int(result["predicted_class"]) == int(actual_label)) if actual_label is not None else None,
                    "Risk": risk_level(float(result["abnormal_probability"]))[0],
                })
            processed = min(batch_start + len(group), total)
            progress.progress(processed / total, text=f"{'Local model inference' if IS_LOCAL_MODEL else 'Databricks inference'}: {processed:,} / {total:,} observations")

        render_pipeline(pipeline_slot, 3)
        results_df = pd.DataFrame(output_rows)
        if len(results_df) != len(records):
            raise RuntimeError(f"Output integrity check failed: {len(results_df)} results for {len(records)} wafers.")

        # Commit history/results only after complete successful inference.
        for (name, values, _), result in zip(records, all_results):
            record_prediction(f"{filename} · {name}", values, result)
        st.session_state.batch_results = results_df
        st.session_state.current_values = np.asarray(records[0][1], dtype=float)
        st.session_state.current_result = all_results[0]
        st.session_state.current_name = f"{filename} · {records[0][0]}"
        st.session_state.uploaded_records = records
        labelled = results_df["Correct"].notna()
        st.session_state.latest_batch_summary = {
            "total": len(results_df),
            "normal": int((results_df["Predicted class"] == "Normal").sum()),
            "abnormal": int((results_df["Predicted class"] == "Abnormal").sum()),
            "labelled": int(labelled.sum()),
            "accuracy": float(results_df.loc[labelled, "Correct"].mean()) if labelled.any() else None,
        }
        st.session_state.pipeline_completed = True
        render_pipeline(pipeline_slot, 4, completed=True)
        progress.progress(1.0, text=f"Completed {total:,} / {total:,} observations")
        time.sleep(0.15)
        progress.empty()
    except Exception:
        progress.empty()
        raise
# Top navigation and hero
try:
    init_database()
    st.session_state.database_warning = None
    if database_configured() and not st.session_state.history:
        st.session_state.history = fetch_recent_inspections(100)
except Exception as database_error:
    st.session_state.database_warning = str(database_error)
if IS_LOCAL_MODEL:
    db_ready, db_status_detail = True, "Local MLflow model loaded and feature schema verified"
    status_text = "LOCAL MODEL ONLINE"
    db_text = "LOCAL INFERENCE · NO API CALL"
    dot_class = "live-dot"
elif IS_DATABRICKS_MODEL:
    db_ready, db_status_detail = check_databricks_connection()
    status_text = "MODEL ONLINE" if db_ready else "MODEL OFFLINE"
    db_text = "DATABRICKS CONNECTED" if db_ready else "DATABRICKS DISCONNECTED"
    dot_class = "live-dot" if db_ready else "live-dot demo-dot"
else:
    db_ready, db_status_detail = False, "No validated local model or configured endpoint"
    status_text = "DEMO MODEL"
    db_text = "DATABRICKS NOT CONFIGURED"
    dot_class = "live-dot demo-dot"
st.markdown(
    f"""
    <div class="topbar" title="{db_status_detail}">
      <div class="brand">
        <div class="brand-mark">W</div>
        <div><div class="brand-name">WAFER AI</div>
        <div class="brand-sub">SEMICONDUCTOR QUALITY INTELLIGENCE</div></div>
      </div>
      <div class="status-wrap">
        <div class="status-pill"><span class="{dot_class}"></span>{status_text}</div>
        <div class="status-pill">◈ {db_text}</div>
      </div>
    </div>
    <div class="hero">
      <div class="eyebrow">INTELLIGENT MANUFACTURING · QUALITY CONTROL</div>
      <div class="hero-title">WaferGuard AI</div>
      <div class="hero-copy">AI-powered wafer process classification and sensor intelligence.
      Inspect a 152-point sensor sequence, review risk signals, and export an auditable inspection report.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if IS_DATABRICKS_MODEL and not db_ready:
    st.error(f"Databricks is configured but the serving endpoint is not ready. Status: {db_status_detail}.")
elif IS_LOCAL_MODEL:
    st.success("LOCAL INFERENCE MODE — Using the validated Databricks-trained Extra Trees Version 9 model from the project models directory. Predictions run locally.")
elif not IS_REAL_MODEL:
    st.warning(
        "DEMO MODE — A synthetic Random Forest is running locally. Its metrics and predictions are illustrative only; "
        "they are not your Databricks model or validated semiconductor quality performance."
    )
# KPI row
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    total_tested = len(hist_df)
    normal_pct = (hist_df["result"].eq("Normal").mean() * 100)
    abnormal_count = int(hist_df["result"].eq("Abnormal").sum())
else:
    total_tested, normal_pct, abnormal_count = 0, 0.0, 0

if IS_REAL_MODEL:
    # The deployed endpoint does not expose validation metrics in inference responses.
    accuracy_display = "—"
    accuracy_note = "Endpoint does not publish evaluation metrics"
else:
    accuracy_display = f"{MODEL_INFO['metrics']['accuracy']*100:.1f}%"
    accuracy_note = "Synthetic demo validation · not production"

kpis = st.columns(4)
kpi_data = [
    ("TOTAL WAFERS TESTED", f"{total_tested:,}", "This app session"),
    ("NORMAL WAFERS", f"{normal_pct:.1f}%", "Share of session inspections"),
    ("ABNORMAL DETECTED", f"{abnormal_count:,}", "Session predictions"),
    ("MODEL METRICS", accuracy_display,
     (f"Recall {MODEL_INFO['metrics']['recall']*100:.1f}% · AUC {MODEL_INFO['metrics']['auc']*100:.1f}%"
      if not IS_REAL_MODEL else accuracy_note)),
]
for col, (label, value, note) in zip(kpis, kpi_data):
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
# Tabs
tab_live, tab_perf, tab_batch, tab_history, tab_ops = st.tabs(
    ["◉  Live Prediction", "▥  Model Performance", "▤  Batch Upload", "◷  History", "⚙  Operations"]
)

with tab_live:
    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Input sensor data</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Upload a Wafer .ts, CSV or Excel dataset. Each observation contains 152 sensor readings.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload wafer dataset",
            type=["ts", "csv", "xlsx", "xls"],
            key="single_upload",
            help="Upload Wafer .ts, CSV or Excel data. Each observation must contain 152 sensor measurements.",
        )
        if uploaded is not None:
            try:
                parsed_records = parse_sensor_upload(uploaded)
                st.session_state.pipeline_completed = False
                st.session_state.uploaded_records = parsed_records
                st.session_state.current_name = uploaded.name
                st.success(f"Dataset uploaded and validated · {len(parsed_records):,} observations · 152 readings each")
                preview_values = np.vstack([r[1] for r in parsed_records[:5]])
                st.caption(f"Preview: {preview_values.shape[0]} observations × 152 sensor measurements")
            except Exception as exc:
                st.session_state.uploaded_records = None
                st.error(f"Dataset validation failed: {exc}")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Load sample · Normal", use_container_width=True):
                st.session_state.current_values = make_sample("normal", 10)
                st.session_state.current_name = "Sample Normal Wafer"
                st.session_state.uploaded_records = [("Sample Normal Wafer", st.session_state.current_values, None)]
                st.rerun()
        with b2:
            if st.button("Load sample · Abnormal", use_container_width=True):
                st.session_state.current_values = make_sample("abnormal", 20)
                st.session_state.current_name = "Sample Abnormal Wafer"
                st.session_state.uploaded_records = [("Sample Abnormal Wafer", st.session_state.current_values, None)]
                st.rerun()

        pipeline_slot = st.empty()
        if st.session_state.uploaded_records:
            render_pipeline(pipeline_slot, 4 if st.session_state.pipeline_completed else 1, completed=st.session_state.pipeline_completed)
            st.caption("Workflow: dataset uploaded → Databricks feature engineering and inference → output generation → final inspection results.")
        else:
            pipeline_slot.info("Upload a dataset to begin the inspection workflow.")

        if st.button("✦  Run AI inspection", type="primary", use_container_width=True):
            try:
                if not st.session_state.uploaded_records:
                    raise ValueError("Upload a Wafer .ts, CSV or Excel dataset, or load a sample wafer first.")
                if IS_DATABRICKS_MODEL and not db_ready:
                    raise ConnectionError(
                        "Databricks credentials are configured, but the serving endpoint is not READY. "
                        f"Status: {db_status_detail}. Check the endpoint before running inference."
                    )
                run_uploaded_inspection(
                    st.session_state.uploaded_records,
                    st.session_state.current_name,
                    pipeline_slot,
                )
                st.success("Inspection completed. Final output is ready below.")
                st.rerun()
            except Exception as exc:
                st.error(f"Inspection could not be completed: {exc}")

        if st.button("Clear upload and results", use_container_width=True):
            st.session_state.current_values = None
            st.session_state.current_result = None
            st.session_state.current_name = "No wafer selected"
            st.session_state.uploaded_records = None
            st.session_state.batch_results = None
            st.rerun()

        if st.session_state.current_values is not None:
            st.markdown(
                f'<div class="small-muted">Loaded: <b>{st.session_state.current_name}</b> · '
                f'{len(st.session_state.current_values)} sensor values</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Inspection decision</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Classification, probability, risk level and recommended next action.</div>', unsafe_allow_html=True)

        result = st.session_state.current_result
        values = st.session_state.current_values
        if result is None:
            st.markdown(
                '<div class="result-card" style="background:rgba(255,255,255,.035);border-color:rgba(148,170,201,.2)">'
                '<div class="result-label" style="color:#AFC2D9">AWAITING INSPECTION</div>'
                '<div class="result-sub">Load a sensor sequence and select Predict wafer to view the model output.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="small-muted">No prediction has been generated in this session.</div>', unsafe_allow_html=True)
        else:
            prob = result["abnormal_probability"]
            conf = result["confidence"]
            abnormal = result["predicted_class"] == 1
            risk, risk_css, action = risk_level(prob)
            result_class = "result-bad" if abnormal else "result-good"
            result_title = "ABNORMAL  ·  REVIEW REQUIRED" if abnormal else "NORMAL  ·  WITHIN MODEL THRESHOLD"
            result_sub = "Potential abnormal process pattern detected. Hold for quality review." if abnormal else "No abnormal pattern detected by the current model. Follow standard release controls."
            st.markdown(
                f'<div class="result-card {result_class}">'
                f'<div class="small-muted">MODEL CLASSIFICATION</div>'
                f'<div class="result-label">{result_title}</div>'
                f'<div class="result-sub">{result_sub}</div>'
                f'<div style="display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap">'
                f'<span class="risk-chip {risk_css}">RISK: {risk}</span>'
                f'<span class="small-muted">Prediction confidence: {conf*100:.1f}%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Abnormal probability", f"{prob*100:.2f}%")
            m2.metric("Prediction confidence", f"{conf*100:.2f}%")
            st.progress(float(prob), text="Abnormal probability")
            st.markdown("**Recommended action**")
            st.write(action)
            st.caption(f"Model source: {result['source']}")

            pdf_bytes = create_pdf_report(result, values, st.session_state.current_name)
            st.download_button(
                "↓  Download inspection PDF",
                data=pdf_bytes,
                file_name=f"WaferGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.batch_results is not None:
        st.markdown('<div class="section-heading">Final inspection output</div>', unsafe_allow_html=True)
        final_df = st.session_state.batch_results
        sm = getattr(st.session_state, "latest_batch_summary", None)
        if sm:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Observations processed", sm["total"])
            f2.metric("Predicted normal", sm["normal"])
            f3.metric("Predicted abnormal", sm["abnormal"])
            f4.metric("Label-based accuracy", f"{sm['accuracy']*100:.1f}%" if sm["accuracy"] is not None else "Not available")
        st.dataframe(final_df, use_container_width=True, hide_index=True)
        st.download_button(
            "↓  Download final inspection output",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="WaferGuard_Final_Inspection_Output.csv",
            mime="text/csv",
            key="live_output_download",
        )

    # Charts for the currently loaded/predicted sensor sequence.
    if st.session_state.current_values is not None:
        values = np.asarray(st.session_state.current_values, dtype=float)
        result = st.session_state.current_result
        st.markdown('<div class="section-heading">Sensor signal intelligence</div>', unsafe_allow_html=True)
        xidx = np.arange(1, 153)
        fig = go.Figure()
        baseline = make_sample("normal", 10)
        fig.add_trace(go.Scatter(x=xidx, y=baseline, mode="lines", name="Reference profile",
                                 line=dict(color=GREEN, width=2, dash="dot"), opacity=.85))
        current_color = RED if result and result["predicted_class"] == 1 else BLUE
        fig.add_trace(go.Scatter(x=xidx, y=values, mode="lines", name="Current wafer",
                                 line=dict(color=current_color, width=2.5)))
        fig.add_vrect(x0=85, x1=115, fillcolor="rgba(245,185,66,.12)",
                      line=dict(color=AMBER, width=1, dash="dash"), annotation_text="Review window 85–115",
                      annotation_position="top left", annotation_font_color="#FFD06A")
        fig.update_layout(xaxis_title="Sensor sequence index (1–152)", yaxis_title="Sensor value")
        fig = fig_layout(fig, 370, "Sensor sequence · current wafer vs illustrative reference")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True})

        # Importance chart: real endpoint does not expose feature importances.
        if MODEL_INFO["mode"] == "demo_rf":
            importances = MODEL_INFO["model"].feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            imp = pd.DataFrame({"Sensor time-step": [f"Sensor {i+1:03d}" for i in top_idx],
                                "Importance": importances[top_idx]})
            imp_fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Sensor time-step",
                             orientation="h", color="Importance", color_continuous_scale=["#2456A6", "#36BFFA"])
            imp_fig.update_layout(coloraxis_showscale=False, xaxis_title="Relative feature importance", yaxis_title="")
            st.plotly_chart(fig_layout(imp_fig, 340, "Demo Random Forest · top 10 sensor time-steps"),
                            use_container_width=True, config={"displaylogo": False, "responsive": True})
            st.caption("Feature importance is from the synthetic demo Random Forest, not from the Databricks Extra Trees endpoint.")
        elif IS_DATABRICKS_MODEL:
            st.info("The deployed Databricks endpoint returns class and probability, but does not expose per-reading feature importance. No time-step importance is fabricated.")
        else:
            st.info("The local Extra Trees model uses 34 engineered features. Per-reading time-step importance is not inferred or fabricated.")

with tab_perf:
    st.markdown('<div class="section-heading">Model performance & validation</div>', unsafe_allow_html=True)
    if IS_REAL_MODEL:
        if IS_LOCAL_MODEL:
            st.info(
                "Local inference uses the Databricks-trained Extra Trees Version 9 model. "
                "Validated on the original 6,164-observation test set: accuracy 98.83%, "
                "abnormal precision 94.45%, recall 94.74%, F1 94.59%, ROC AUC 99.91%. "
                "Local and Databricks Serving predictions matched exactly on all 6,164 observations."
            )
            model_source_display = str(LOCAL_MODEL_PATH)
        else:
            st.info(
                "Live Databricks endpoint inference is connected. The endpoint does not return "
                "evaluation metrics; use the held-out training evaluation for model performance."
            )
            model_source_display = (
                os.getenv("DATABRICKS_ENDPOINT")
                or os.getenv("DATABRICKS_SERVING_ENDPOINT")
                or "Configured"
            )
        perf_cols = st.columns(3)
        for col, label, val in zip(
            perf_cols,
            ["Model family", "Model source", "Input schema"],
            ["Extra Trees", model_source_display, "34 engineered features"],
        ):
            with col:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value" style="font-size:18px">{val}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        metrics = MODEL_INFO["metrics"]
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 score","f1"),("ROC AUC","auc")]):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{metrics[key]*100:.1f}%</div><div class="kpi-note">Synthetic demo validation</div></div>', unsafe_allow_html=True)
        st.warning("These metrics describe a synthetic demonstration dataset only. They must not be quoted as Wafer dataset test performance.")
        cm = np.array([[1, 1], [1, 1]])
        cm_demo = np.asarray(metrics["confusion_matrix"], dtype=int)
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_demo,
            x=["Predicted Normal", "Predicted Abnormal"],
            y=["Actual Normal", "Actual Abnormal"],
            text=cm_demo,
            texttemplate="%{text}",
            colorscale=[[0, "#14243A"], [0.5, "#2468A8"], [1, "#36BFFA"]],
            showscale=True,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        ))
        cm_fig.update_layout(xaxis_title="Predicted class", yaxis_title="Actual class")
        st.plotly_chart(fig_layout(cm_fig, 340, "Synthetic demo validation · confusion matrix"),
                        use_container_width=True, config={"displaylogo": False, "responsive": True})
        st.caption("This confusion matrix is from the synthetic demo validation split, not from the UCR Wafer dataset.")

    # Session prediction distribution donut
    if st.session_state.history:
        hist_counts = pd.Series([h["result"] for h in st.session_state.history]).value_counts()
        donut = go.Figure(go.Pie(
            labels=hist_counts.index.tolist(), values=hist_counts.values.tolist(),
            hole=.68, marker=dict(colors=[GREEN if x == "Normal" else RED for x in hist_counts.index]),
            textinfo="label+percent", hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        st.plotly_chart(fig_layout(donut, 320, "Session inspection distribution"),
                        use_container_width=True, config={"displaylogo": False, "responsive": True})
    else:
        st.caption("The inspection distribution donut will populate after you run predictions.")

    # Probability gauge for latest result
    if st.session_state.current_result:
        prob = st.session_state.current_result["abnormal_probability"] * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob, number={"suffix":"%","font":{"color":TEXT}},
            title={"text":"Latest abnormal probability","font":{"color":TEXT}},
            gauge={"axis":{"range":[0,100],"tickcolor":MUTED},
                   "bar":{"color":RED if prob >= 50 else GREEN},
                   "bgcolor":"rgba(255,255,255,.04)",
                   "steps":[{"range":[0,35],"color":"rgba(32,201,151,.20)"},
                            {"range":[35,70],"color":"rgba(245,185,66,.20)"},
                            {"range":[70,100],"color":"rgba(240,82,96,.20)"}],
                   "threshold":{"line":{"color":"#FFFFFF","width":3},"thickness":.8,"value":50}},
        ))
        st.plotly_chart(fig_layout(gauge, 300), use_container_width=True, config={"displaylogo": False, "responsive": True})

with tab_batch:
    st.markdown('<div class="section-heading">Batch inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Upload a .ts, CSV or Excel file. Each observation must contain 152 sensor readings.</div>', unsafe_allow_html=True)
    batch_file = st.file_uploader("Batch dataset", type=["ts", "csv", "xlsx", "xls"], key="batch_upload")
    if batch_file is not None and st.button("Run batch inspection", type="primary"):
        try:
            parsed = parse_sensor_upload(batch_file)
            batch_size = int(os.getenv("WAFERGUARD_BATCH_SIZE", "25"))
            if not 1 <= batch_size <= 100:
                raise ValueError("WAFERGUARD_BATCH_SIZE must be between 1 and 100.")
            rows = []
            progress = st.progress(0, text="Preparing batch…")
            total = len(parsed)
            for batch_start in range(0, total, batch_size):
                group = parsed[batch_start:batch_start + batch_size]
                if IS_DATABRICKS_MODEL:
                    features = [engineer_features(values) for _, values, _ in group]
                    predictions = call_databricks_batch(features)
                else:
                    predictions = [predict_values(values) for _, values, _ in group]
                if len(predictions) != len(group):
                    raise RuntimeError("Databricks returned an unexpected number of predictions.")
                for (name, values, actual_label), pred in zip(group, predictions):
                    record_prediction(f"{batch_file.name} · {name}", values, pred)
                    rows.append({
                        "Wafer": name,
                        "Result": pred["predicted_label"],
                        "Predicted class code": int(pred["predicted_class"]),
                        "Confidence": round(float(pred["confidence"]) * 100, 2),
                        "Abnormal probability": round(float(pred["abnormal_probability"]) * 100, 2),
                        "Actual class": ("Abnormal" if actual_label == 1 else "Normal") if actual_label is not None else "Not supplied",
                        "Correct": (int(pred["predicted_class"]) == int(actual_label)) if actual_label is not None else None,
                        "Risk": risk_level(float(pred["abnormal_probability"]))[0],
                    })
                processed = min(batch_start + len(group), total)
                progress.progress(processed / total, text=f"Inspecting {processed:,} / {total:,} wafers")
            batch_df = pd.DataFrame(rows)
            if len(batch_df) != total:
                raise RuntimeError(f"Batch output integrity check failed: expected {total}, got {len(batch_df)}.")
            st.session_state.batch_results = batch_df
            progress.progress(1.0, text=f"Completed {total:,} / {total:,} wafers")
            time.sleep(0.15)
            progress.empty()
            st.success(f"Completed inspection for {total:,} wafers.")
        except Exception as exc:
            st.error(f"Batch inspection failed. No complete result was saved: {exc}")

    if st.session_state.batch_results is not None:
        batch_df = st.session_state.batch_results.copy()
        result_column = next(
            (column for column in ("Result", "Predicted class", "Predicted Class") if column in batch_df.columns),
            None,
        )
        if result_column is None:
            st.error("Saved results have no recognised prediction column. Clear results and run again.")
        else:
            b1, b2, b3 = st.columns(3)
            b1.metric("Batch size", len(batch_df))
            b2.metric("Predicted abnormal", int((batch_df[result_column].astype(str).str.lower() == "abnormal").sum()))
            b3.metric("Predicted normal", int((batch_df[result_column].astype(str).str.lower() == "normal").sum()))
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button(
                "↓  Download batch CSV report",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="WaferGuard_Batch_Report.csv",
                mime="text/csv",
            )
            st.caption("Batch predictions are not ground-truth evaluation results unless actual labels are independently provided.")

with tab_history:
    st.markdown('<div class="section-heading">Inspection history</div>', unsafe_allow_html=True)
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history[:10])
        history_df = history_df.rename(columns={
            "time":"Time", "wafer":"Wafer / file", "result":"Result",
            "confidence":"Confidence (0–1)", "abnormal_probability":"Abnormal probability (0–1)"
        })
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "↓  Download session history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="WaferGuard_Inspection_History.csv",
            mime="text/csv",
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No inspections recorded yet. Run a live prediction to populate this table.")
# Footer
if IS_LOCAL_MODEL:
    footer_model = "Local MLflow Extra Trees v9 · 34 engineered features"
elif IS_DATABRICKS_MODEL:
    footer_model = "Databricks Serving · Extra Trees · 34 engineered features"
else:
    footer_model = "Synthetic Random Forest DEMO · 100 trees · 152 raw inputs"
st.markdown(
    f'<div class="footer">WAFERGUARD AI &nbsp;·&nbsp; {footer_model}<br>'
    'Decision-support software. Final wafer disposition must follow approved manufacturing quality procedures.</div>',
    unsafe_allow_html=True,
)


with tab_ops:
    st.markdown('<div class="section-heading">Operations and feedback</div>', unsafe_allow_html=True)
    if database_configured():
        st.success("PostgreSQL is configured. Inspection records are persisted when predictions complete.")
    else:
        st.info("PostgreSQL is not configured. The app is using session-only inspection history.")
    if st.session_state.get("database_warning"):
        st.warning(f"Database notice: {st.session_state.database_warning}")

    st.markdown("#### User account management")
    if not database_configured():
        st.caption("Configure DATABASE_URL in .env and install the database dependency to enable account management.")
    else:
        account_tab, login_tab = st.tabs(["Create account", "Sign in"])
        with account_tab:
            with st.form("create_user_form"):
                new_username = st.text_input("Username", key="new_username")
                new_password = st.text_input("Password (minimum 12 characters)", type="password", key="new_password")
                create_submitted = st.form_submit_button("Create account")
            if create_submitted:
                try:
                    create_user(new_username, new_password)
                    st.success("Account created. You can now sign in.")
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")
        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Sign in")
            if login_submitted:
                try:
                    if authenticate_user(login_username, login_password):
                        st.session_state.authenticated_user = login_username.strip().lower()
                        st.success("Signed in successfully.")
                    else:
                        st.error("Username or password is incorrect.")
                except Exception as exc:
                    st.error(f"Sign-in failed: {exc}")
        if st.session_state.get("authenticated_user"):
            st.caption(f"Signed in as {st.session_state.authenticated_user}")
            if st.button("Sign out"):
                st.session_state.authenticated_user = None
                st.rerun()

    st.markdown("#### Product feedback")
    with st.form("feedback_form"):
        feedback_category = st.selectbox("Feedback category", ["Dashboard", "Prediction", "Batch processing", "Report export", "Other"])
        feedback_rating = st.slider("Overall rating", min_value=1, max_value=5, value=4)
        feedback_message = st.text_area("Comments", placeholder="Describe what worked well or what should improve.")
        feedback_submitted = st.form_submit_button("Submit feedback")
    if feedback_submitted:
        if not database_configured():
            st.warning("Configure PostgreSQL to save feedback. Your feedback has not been stored.")
        else:
            try:
                save_feedback(feedback_category, feedback_rating, feedback_message, st.session_state.get("authenticated_user"))
                st.success("Feedback saved successfully.")
            except Exception as exc:
                st.error(f"Feedback was not saved: {exc}")
