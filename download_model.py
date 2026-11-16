
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("DATABRICKS_HOST"):
    raise ValueError("DATABRICKS_HOST missing in .env")

if not os.getenv("DATABRICKS_TOKEN"):
    raise ValueError("DATABRICKS_TOKEN missing in .env")

import mlflow
from mlflow.artifacts import download_artifacts

# Databricks configuration
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Original trained classifier
MODEL_NAME = (
    "wafer_guard_ai.gold_wafer.wafer_defect_classifier"
)
MODEL_VERSION = "9"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# Download destination inside project root
DESTINATION = Path("models/wafer_defect_classifier_v9")
DESTINATION.mkdir(parents=True, exist_ok=True)

print("Downloading WaferGuard AI trained model...")
print(f"Model: {MODEL_NAME}")
print(f"Version: {MODEL_VERSION}")

downloaded_path = download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path=str(DESTINATION.resolve())
)

print("\nDownload completed!")
print("Location:", downloaded_path)

# Verify local model loading
model = mlflow.sklearn.load_model(downloaded_path)

print("Model type:", type(model).__name__)
print("Input features:", model.n_features_in_)

assert model.n_features_in_ == 34

print("SUCCESS: Original trained model downloaded and verified.")
