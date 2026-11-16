import math

import pytest

from utils.databricks_client import FEATURE_COLUMNS, _normalise_prediction, _normalise_record


def test_feature_record_is_ordered_and_finite():
    record = {column: index / 10 for index, column in enumerate(reversed(FEATURE_COLUMNS))}
    normalized = _normalise_record(record, 1)
    assert list(normalized) == FEATURE_COLUMNS
    assert len(normalized) == 34


def test_feature_record_rejects_missing_field():
    record = {column: 0.0 for column in FEATURE_COLUMNS[:-1]}
    with pytest.raises(ValueError, match="missing features"):
        _normalise_record(record, 1)


def test_prediction_validation_accepts_consistent_output():
    result = _normalise_prediction({
        "predicted_class": 1,
        "predicted_label": "Abnormal",
        "abnormal_probability": 0.91,
        "prediction_confidence": 0.91,
    }, 1)
    assert result["predicted_label"] == "Abnormal"


@pytest.mark.parametrize("prediction", [
    {"predicted_class": 1, "predicted_label": "Normal", "abnormal_probability": 0.9, "prediction_confidence": 0.9},
    {"predicted_class": 2, "predicted_label": "Abnormal", "abnormal_probability": 0.9, "prediction_confidence": 0.9},
    {"predicted_class": 1, "predicted_label": "Abnormal", "abnormal_probability": math.nan, "prediction_confidence": 0.9},
])
def test_prediction_validation_rejects_inconsistent_output(prediction):
    with pytest.raises(RuntimeError):
        _normalise_prediction(prediction, 1)
