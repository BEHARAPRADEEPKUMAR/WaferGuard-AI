import numpy as np
import pandas as pd
import pytest

from utils.data_processor import FEATURE_COLUMNS, engineer_features, process_uploaded_file


def test_feature_engineering_returns_exact_34_features():
    values = np.arange(152, dtype=float)
    features = engineer_features(values)
    assert list(features) == FEATURE_COLUMNS
    assert len(features) == 34
    assert features["mean"] == pytest.approx(np.mean(values))
    assert features["stddev"] == pytest.approx(np.std(values, ddof=0))
    assert features["segment_8_range"] == pytest.approx(18.0)


def test_feature_engineering_rejects_wrong_length():
    with pytest.raises(ValueError, match="152"):
        engineer_features(np.arange(151, dtype=float))


def test_feature_engineering_rejects_non_finite_values():
    values = np.zeros(152)
    values[5] = np.nan
    with pytest.raises(ValueError, match="infinity"):
        engineer_features(values)


def test_ts_file_label_mapping_and_feature_count():
    values = ",".join(str(float(x)) for x in range(152))
    content = f"@problemName Wafer\n@univariate true\n{values}:-1\n{values}:1\n".encode()
    result = process_uploaded_file(content, "sample.ts")
    assert len(result) == 2
    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == 0
    assert len([c for c in result.columns if c in FEATURE_COLUMNS]) == 34


def test_pre_engineered_csv_is_accepted():
    frame = pd.DataFrame([{column: 0.1 for column in FEATURE_COLUMNS}])
    result = process_uploaded_file(frame.to_csv(index=False).encode(), "features.csv")
    assert len(result) == 1
    assert all(column in result.columns for column in FEATURE_COLUMNS)
