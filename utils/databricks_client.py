"""Reliable batched client for WaferGuard AI Databricks Model Serving."""
from __future__ import annotations

import math
import os
from typing import Any

import requests
from dotenv import load_dotenv
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip().rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
ENDPOINT_NAME = (
    os.getenv("DATABRICKS_ENDPOINT")
    or os.getenv("DATABRICKS_SERVING_ENDPOINT")
    or "waferguard-defect-detection"
).strip()

GLOBAL_FEATURE_COLUMNS = [
    "mean", "stddev", "min_value", "max_value", "range_value",
    "rms", "mean_abs", "first_value", "last_value", "end_to_end_change",
]
SEGMENT_FEATURE_COLUMNS = [
    f"segment_{segment}_{stat}"
    for segment in range(1, 9)
    for stat in ("mean", "stddev", "range")
]
FEATURE_COLUMNS = GLOBAL_FEATURE_COLUMNS + SEGMENT_FEATURE_COLUMNS
EXPECTED_PREDICTION_KEYS = {
    "predicted_class", "predicted_label",
    "abnormal_probability", "prediction_confidence",
}
BATCH_SIZE = int(os.getenv("WAFERGUARD_BATCH_SIZE", "25"))
CONNECT_TIMEOUT = float(os.getenv("DATABRICKS_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = float(os.getenv("DATABRICKS_READ_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("DATABRICKS_MAX_RETRIES", "3"))


def _build_session() -> Session:
    """Create a reusable HTTP session with bounded transient-error retries."""
    retry = Retry(
        total=MAX_RETRIES, connect=MAX_RETRIES, read=MAX_RETRIES, status=MAX_RETRIES,
        backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}), respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def _normalise_record(record: dict[str, Any], index: int) -> dict[str, float]:
    if not isinstance(record, dict):
        raise ValueError(f"Feature record {index} must be a dictionary.")
    missing = [column for column in FEATURE_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Record {index} is missing features: {missing}")
    ordered = {}
    for column in FEATURE_COLUMNS:
        try:
            value = float(record[column])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Record {index}, feature {column!r} is not numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Record {index}, feature {column!r} is not finite.")
        ordered[column] = value
    return ordered


def _normalise_prediction(prediction: Any, index: int) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        raise RuntimeError(f"Prediction {index} is not an object: {prediction!r}")
    missing = EXPECTED_PREDICTION_KEYS - set(prediction)
    if missing:
        raise RuntimeError(f"Prediction {index} is missing output fields: {sorted(missing)}")
    try:
        predicted_class = int(prediction["predicted_class"])
        probability = float(prediction["abnormal_probability"])
        confidence = float(prediction["prediction_confidence"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Prediction {index} contains invalid numeric output.") from exc
    if predicted_class not in (0, 1):
        raise RuntimeError(f"Prediction {index} has invalid class code {predicted_class}.")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Prediction {index} has invalid abnormal probability.")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError(f"Prediction {index} has invalid confidence.")
    expected_class = 1 if probability >= 0.5 else 0
    if predicted_class != expected_class:
        raise RuntimeError(
            f"Prediction {index} is inconsistent: class={predicted_class}, "
            f"abnormal_probability={probability:.6f}, threshold=0.50."
        )
    expected_label = "Abnormal" if predicted_class == 1 else "Normal"
    label = str(prediction["predicted_label"]).strip()
    if label.casefold() != expected_label.casefold():
        raise RuntimeError(f"Prediction {index} label {label!r} conflicts with class {predicted_class}.")
    return {
        "predicted_class": predicted_class,
        "predicted_label": expected_label,
        "abnormal_probability": probability,
        "prediction_confidence": confidence,
    }


def predict_wafer(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Submit up to configured batch size and return validated model predictions."""
    if not DATABRICKS_HOST:
        raise ValueError("DATABRICKS_HOST is missing from .env.")
    if not DATABRICKS_TOKEN:
        raise ValueError("DATABRICKS_TOKEN is missing from .env.")
    if not ENDPOINT_NAME:
        raise ValueError("DATABRICKS_ENDPOINT is missing from .env.")
    if not isinstance(features, list) or not features:
        raise ValueError("Provide at least one wafer feature record.")
    if len(features) > BATCH_SIZE:
        raise ValueError(f"Batch has {len(features)} rows; configured maximum is {BATCH_SIZE}.")
    records = [_normalise_record(row, i + 1) for i, row in enumerate(features)]
    host = DATABRICKS_HOST
    if not host.startswith(("http://", "https://")):
        host = "https://" + host
    url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    try:
        response = _SESSION.post(
            url, headers=headers, json={"dataframe_records": records},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), verify=True,
        )
        if not response.ok:
            raise RuntimeError(
                f"Databricks endpoint returned HTTP {response.status_code}: {response.text[:2000]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Databricks returned invalid JSON: {exc}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Databricks connection failed after retries: {exc}") from exc

    predictions = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(predictions, list):
        raise RuntimeError(f"Unexpected Databricks response: {str(payload)[:1200]}")
    if len(predictions) != len(features):
        raise RuntimeError(f"Expected {len(features)} predictions, received {len(predictions)}.")
    return [_normalise_prediction(item, i + 1) for i, item in enumerate(predictions)]


def predict_wafer_in_batches(features: list[dict[str, Any]], batch_size: int | None = None) -> list[dict[str, Any]]:
    """Split feature rows into bounded requests and preserve original row order."""
    if not features:
        raise ValueError("Provide at least one wafer feature record.")
    size = batch_size or BATCH_SIZE
    if not 1 <= size <= BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {BATCH_SIZE}.")
    combined = []
    for start in range(0, len(features), size):
        combined.extend(predict_wafer(features[start:start + size]))
    if len(combined) != len(features):
        raise RuntimeError("Final prediction count does not match the input row count.")
    return combined
