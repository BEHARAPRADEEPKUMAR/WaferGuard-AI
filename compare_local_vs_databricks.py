
"""Compare local MLflow predictions with Databricks Model Serving."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_processor import process_uploaded_file
from utils.databricks_client import (
    FEATURE_COLUMNS,
    predict_wafer_in_batches,
)


def main():
    """Compare predictions for the original 6,164 wafer observations."""

    dataset_path = Path("Wafer_TEST.ts")
    local_results_path = Path("local_model_test_predictions.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not local_results_path.exists():
        raise FileNotFoundError(local_results_path)

    # Load original dataset and extract the same 34 features.
    print("Loading original wafer test dataset...")
    df = process_uploaded_file(
        dataset_path.read_bytes(),
        dataset_path.name,
    )

    X_test = df[FEATURE_COLUMNS]

    # Load previously generated local predictions.
    print("Loading local model predictions...")
    local = pd.read_csv(local_results_path)

    if len(local) != len(X_test):
        raise ValueError("Local prediction count does not match dataset.")

    # Request predictions from Databricks Serving.
    print("\nCalling Databricks Model Serving...")
    print("Observations:", len(X_test))
    print("Batch size: 25")

    endpoint_result = predict_wafer_in_batches(
        X_test.to_dict(orient="records"),
        batch_size=25,
    )

    # Normalize the endpoint response.
    if isinstance(endpoint_result, pd.DataFrame):
        endpoint = endpoint_result.copy()
    elif isinstance(endpoint_result, dict):
        endpoint = pd.DataFrame(endpoint_result["predictions"])
    else:
        endpoint = pd.DataFrame(endpoint_result)

    required_columns = [
        "predicted_class",
        "abnormal_probability",
    ]

    missing = [
        column for column in required_columns
        if column not in endpoint.columns
    ]

    if missing:
        raise ValueError(
            f"Databricks response is missing columns: {missing}. "
            f"Received: {endpoint.columns.tolist()}"
        )

    if len(endpoint) != len(local):
        raise ValueError(
            f"Endpoint returned {len(endpoint)} predictions; "
            f"expected {len(local)}."
        )

    # Compare predictions in their original observation order.
    local_class = pd.to_numeric(
        local["predicted_class"], errors="raise"
    ).astype(int)

    endpoint_class = pd.to_numeric(
        endpoint["predicted_class"], errors="raise"
    ).astype(int)

    local_probability = pd.to_numeric(
        local["abnormal_probability"], errors="raise"
    )

    endpoint_probability = pd.to_numeric(
        endpoint["abnormal_probability"], errors="raise"
    )

    probability_difference = (
        local_probability - endpoint_probability
    ).abs()

    class_match = local_class == endpoint_class

    comparison = pd.DataFrame({
        "observation_id": local["observation_id"],
        "actual_class": local["target"],
        "local_predicted_class": local_class,
        "databricks_predicted_class": endpoint_class,
        "local_abnormal_probability": local_probability,
        "databricks_abnormal_probability": endpoint_probability,
        "probability_absolute_difference": probability_difference,
        "class_match": class_match,
    })

    print("\n========== MODEL PARITY RESULTS ==========")
    print("Total observations:", len(comparison))
    print("Matching classes:", int(class_match.sum()))
    print("Different classes:", int((~class_match).sum()))
    print(
        "Class agreement:",
        f"{class_match.mean() * 100:.6f}%",
    )
    print(
        "Maximum probability difference:",
        f"{probability_difference.max():.10f}",
    )
    print(
        "Mean probability difference:",
        f"{probability_difference.mean():.10f}",
    )

    comparison.to_csv(
        "local_vs_databricks_comparison.csv",
        index=False,
    )

    print("\nSaved: local_vs_databricks_comparison.csv")
    print("\nFirst 10 comparison rows:")
    print(comparison.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
