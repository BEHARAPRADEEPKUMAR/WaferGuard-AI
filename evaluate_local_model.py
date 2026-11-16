
"""Evaluate the downloaded WaferGuard AI model on the original test dataset."""

from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.data_processor import process_uploaded_file
from utils.databricks_client import FEATURE_COLUMNS


def main():
    """Run local model evaluation without contacting Databricks."""

    dataset_path = Path("Wafer_TEST.ts")
    model_path = "models/wafer_defect_classifier_v9"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("Loading original test dataset...")
    df = process_uploaded_file(dataset_path.read_bytes(), dataset_path.name)

    X_test = df[FEATURE_COLUMNS]
    y_test = df["target"].astype(int)

    print("Loading downloaded Version 9 model...")
    model = mlflow.sklearn.load_model(model_path)

    if X_test.shape[1] != model.n_features_in_:
        raise ValueError("Model and dataset feature counts do not match.")

    print("\nRunning local predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n========== LOCAL MODEL EVALUATION ==========")
    print(f"Dataset observations: {len(df)}")
    print(f"Input features: {X_test.shape[1]}")
    print(f"Actual Normal: {(y_test == 0).sum()}")
    print(f"Actual Abnormal: {(y_test == 1).sum()}")
    print(f"Predicted Normal: {(y_pred == 0).sum()}")
    print(f"Predicted Abnormal: {(y_pred == 1).sum()}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        digits=4,
        zero_division=0,
    ))

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 6))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 6))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 6))
    print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 6))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 6))
    print("PR AUC:", round(average_precision_score(y_test, y_prob), 6))

    results = df[["observation_id", "class_label", "target"]].copy()
    results["predicted_class"] = y_pred
    results["abnormal_probability"] = y_prob
    results["prediction_correct"] = y_pred == y_test

    output_path = Path("local_model_test_predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"\nPrediction results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
