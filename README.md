# WaferGuard AI

WaferGuard AI is a Streamlit application for semiconductor wafer sensor-sequence classification. It accepts UCR/UEA-style `.ts` files and tabular CSV/Excel uploads, engineers the same 34-feature schema used by the Databricks model, and submits validated records to Databricks Model Serving.

## Current capabilities

- Dark, responsive Streamlit dashboard with Live Prediction, Model Performance, Batch Upload, History, and Operations tabs.
- Databricks Model Serving client with bounded retries, batched requests, strict input/output validation, and response-count checks.
- Shared 34-feature engineering module: 10 global statistics and 3 statistics for each of 8 consecutive 19-reading segments.
- UCR labels are mapped as `1 = Normal` and `-1 = Abnormal`; model target encoding is `0 = Normal` and `1 = Abnormal`.
- CSV/XLSX/XLS/TS ingestion in the shared data processor, including files that already contain all 34 model features.
- Optional PostgreSQL persistence for inspections, feedback, and application users. Passwords are stored as salted PBKDF2 hashes.
- CSV and PDF report exports from the dashboard.
- Endpoint verification script and unit tests.
- Clearly labelled synthetic demo mode when Databricks credentials are absent. Demo predictions and metrics are not evidence of model performance.

## Requirements

- Python 3.10 or newer.
- Databricks Model Serving endpoint and a valid personal access token for live predictions.
- PostgreSQL 14+ is optional; the app remains usable with session-only history when it is not configured.

## Local setup (Windows PowerShell)

```powershell
cd path\to\WaferGuard_AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholders with your own Databricks workspace host, a newly generated token, and the serving endpoint name. Do not share `.env`, upload it to chat, or commit it to Git. The `.gitignore` file excludes it.

For live inference, set:

```dotenv
DATABRICKS_HOST=https://<your-workspace-host>
DATABRICKS_TOKEN=<your-new-private-token>
DATABRICKS_ENDPOINT=waferguard-defect-detection
WAFERGUARD_BATCH_SIZE=25
```

If Databricks credentials are not configured, the app runs in clearly marked synthetic DEMO mode. Do not interpret demo predictions as real wafer classifications.

## Optional PostgreSQL setup

Create an empty PostgreSQL database, then add a connection string to `.env`:

```dotenv
DATABASE_URL=postgresql://username:password@localhost:5432/waferguard
```

The app creates these tables on startup:

- `wg_users`: application usernames and salted password hashes.
- `wg_inspections`: prediction result, probability, confidence, model source, and timestamp.
- `wg_feedback`: category, rating, message, and timestamp.

The Operations tab provides account creation/sign-in and feedback submission. Database persistence is optional; when no `DATABASE_URL` is set, inspection history remains in the current Streamlit session.

## Run the dashboard

```powershell
streamlit run app.py
```

## Run local tests

```powershell
pytest
python -m py_compile app.py utils\*.py scripts\*.py
```

The tests do not contact Databricks or PostgreSQL. They validate feature calculations, supported data parsing, feature schema integrity, prediction validation, and password hashing.

## Verify the real Databricks endpoint

Place one or more labelled `.ts`, `.csv`, `.xlsx`, or `.xls` datasets in the project root, or pass explicit paths/directories:

```powershell
python scripts\verify_waferguard_endpoint.py
python scripts\verify_waferguard_endpoint.py .\datasets\Wafer_TRAIN.ts .\datasets\Wafer_TEST.ts
```

The verification script writes `waferguard_endpoint_verification.csv` with per-observation results. Run this only after configuring `.env`. It sends requests to your live endpoint and may incur Databricks serving costs. The script's measured results, not the dashboard's synthetic demo metrics, should be used to assess endpoint behaviour.

## Project layout

```text
WaferGuard_AI/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
├── utils/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── databricks_client.py
│   └── database.py
├── tests/
│   ├── test_data_processor.py
│   ├── test_databricks_client.py
│   └── test_database.py
├── scripts/
│   ├── test_endpoint.py
│   └── verify_waferguard_endpoint.py
├── assets/
└── reports/
```

## Important model/data notes

Each UCR Wafer observation contains 152 sequential measurements. The source metadata does not provide timestamps, measurement units, or physical sensor names, so the dashboard does not describe those 152 values as seconds, minutes, or named process stages. Model output is decision support and does not replace approved manufacturing quality procedures.

The Databricks endpoint is expected to accept `dataframe_records` containing the 34 engineered features and return a `predictions` list with `predicted_class`, `predicted_label`, `abnormal_probability`, and `prediction_confidence` for each row. If the deployed wrapper uses a different response contract, update the client only after inspecting the actual endpoint response.
