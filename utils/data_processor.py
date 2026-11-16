"""WaferGuard AI data ingestion and 34-feature engineering.

Accepted inputs:
- UCR/UEA-style .ts files with 152 comma-separated measurements and optional
  trailing class label (-1 = Abnormal, 1 = Normal).
- CSV/XLSX/XLS files containing 152 raw measurement columns, optionally with
  a class_label/class/label column.
- CSV/XLSX/XLS files already containing all 34 engineered feature columns.

The 34 features are calculated to match Notebook 03:
10 global statistics plus mean, population standard deviation and range for
each of eight consecutive 19-measurement segments.
"""

from io import BytesIO

import numpy as np
import pandas as pd

EXPECTED_MEASUREMENTS = 152
SEGMENT_COUNT = 8
SEGMENT_SIZE = 19

GLOBAL_FEATURE_COLUMNS = [
    "mean",
    "stddev",
    "min_value",
    "max_value",
    "range_value",
    "rms",
    "mean_abs",
    "first_value",
    "last_value",
    "end_to_end_change",
]

SEGMENT_FEATURE_COLUMNS = []
for _segment_number in range(1, SEGMENT_COUNT + 1):
    SEGMENT_FEATURE_COLUMNS.extend(
        [
            f"segment_{_segment_number}_mean",
            f"segment_{_segment_number}_stddev",
            f"segment_{_segment_number}_range",
        ]
    )

FEATURE_COLUMNS = GLOBAL_FEATURE_COLUMNS + SEGMENT_FEATURE_COLUMNS
LABEL_NAMES = {"class_label", "class", "label", "target", "classlabel"}
ID_NAMES = {"observation_id", "id", "wafer_id"}


def _normalize_column(value):
    """Normalize a column heading for reliable matching."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_original_label(value):
    """Normalize source labels: 1 means Normal and -1 means Abnormal."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value in {"1", "1.0"}:
        return "1"
    if value in {"-1", "-1.0"}:
        return "-1"
    raise ValueError(
        f"Invalid source class label {value!r}; expected 1 (Normal) or -1 (Abnormal)."
    )


def _feature_row(values, observation_id, label=None):
    """Calculate the 34 model features for one 152-measurement observation."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or values.size != EXPECTED_MEASUREMENTS:
        raise ValueError(
            f"Observation {observation_id}: expected exactly "
            f"{EXPECTED_MEASUREMENTS} measurements, found {values.size}."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Observation {observation_id}: measurements contain null, NaN or infinity."
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    first_value = float(values[0])
    last_value = float(values[-1])

    row = {
        "observation_id": observation_id,
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values, ddof=0)),
        "min_value": minimum,
        "max_value": maximum,
        "range_value": maximum - minimum,
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mean_abs": float(np.mean(np.abs(values))),
        "first_value": first_value,
        "last_value": last_value,
        "end_to_end_change": last_value - first_value,
    }

    # Notebook 03 uses eight consecutive slices of 19 measurements each.
    for segment_index in range(SEGMENT_COUNT):
        start = segment_index * SEGMENT_SIZE
        stop = start + SEGMENT_SIZE
        segment = values[start:stop]
        segment_number = segment_index + 1

        if segment.size != SEGMENT_SIZE:
            raise AssertionError(
                f"Segment {segment_number} must contain {SEGMENT_SIZE} values."
            )

        row[f"segment_{segment_number}_mean"] = float(np.mean(segment))
        row[f"segment_{segment_number}_stddev"] = float(
            np.std(segment, ddof=0)
        )
        row[f"segment_{segment_number}_range"] = float(
            np.max(segment) - np.min(segment)
        )

    if label is not None:
        normalized_label = _normalize_original_label(label)
        row["class_label"] = normalized_label
        row["target"] = 1 if normalized_label == "-1" else 0

    return row


def parse_ts_file(file_bytes, contains_labels=True):
    """Parse a UCR/UEA-style TS file into validated measurement records."""
    try:
        content = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The TS file must contain valid UTF-8 text.") from exc

    records = []
    errors = []

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue

        try:
            if contains_labels:
                if ":" not in line:
                    raise ValueError("Missing ':' label separator.")
                measurement_text, label_text = line.rsplit(":", 1)
                label = _normalize_original_label(label_text)
            else:
                measurement_text = line
                label = None

            values = [float(item.strip()) for item in measurement_text.split(",")]
            if len(values) != EXPECTED_MEASUREMENTS:
                raise ValueError(
                    f"Expected {EXPECTED_MEASUREMENTS} measurements, found {len(values)}."
                )
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise ValueError("Measurements contain NaN or infinity.")

            records.append(
                {
                    "observation_id": len(records) + 1,
                    "measurements": np.asarray(values, dtype=np.float64),
                    "class_label": label,
                }
            )
        except (ValueError, TypeError) as exc:
            errors.append(f"Line {line_number}: {exc}")

    if errors:
        raise ValueError(
            f"Found {len(errors)} invalid TS rows:\n" + "\n".join(errors[:10])
        )
    if not records:
        raise ValueError("No valid observations found in the TS file.")
    return records


def extract_features(records):
    """Convert parsed TS records into a DataFrame containing all 34 features."""
    rows = [
        _feature_row(
            record["measurements"],
            record.get("observation_id", index + 1),
            record.get("class_label"),
        )
        for index, record in enumerate(records)
    ]
    return pd.DataFrame(rows)


def _assign_headerless_columns(df):
    """Assign measurement headings to a headerless 152/153-column table."""
    if df.shape[1] == EXPECTED_MEASUREMENTS:
        df.columns = [f"measurement_{i + 1}" for i in range(EXPECTED_MEASUREMENTS)]
    elif df.shape[1] == EXPECTED_MEASUREMENTS + 1:
        df.columns = (
            [f"measurement_{i + 1}" for i in range(EXPECTED_MEASUREMENTS)]
            + ["class_label"]
        )
    return df


def _is_recognized_header(columns):
    """Return True only when column names describe a supported data schema."""
    normalized = [_normalize_column(column) for column in columns]
    if all(column in normalized for column in FEATURE_COLUMNS):
        return True
    if any(column in LABEL_NAMES for column in normalized):
        return True

    candidates = [
        column for column in normalized
        if column not in ID_NAMES and column != "split_name"
    ]
    if len(candidates) != EXPECTED_MEASUREMENTS:
        return False

    # Support explicit measurement/sensor column names, but do not mistake
    # numeric first-row values from a headerless file for column headings.
    return all(
        column.startswith(("measurement_", "sensor_", "feature_"))
        for column in candidates
    )


def _labels_and_targets(df, label_column):
    """Read source labels and map them to model targets."""
    if label_column is None:
        return None, None

    raw_values = df[label_column]
    normalized_name = _normalize_column(label_column)

    # A column explicitly named target may already use model encoding:
    # 0 = Normal, 1 = Abnormal. Other label columns use source encoding -1/1.
    if normalized_name == "target":
        numeric = pd.to_numeric(raw_values, errors="coerce")
        if numeric.isna().any() or not set(numeric.astype(int).unique()).issubset({0, 1}):
            raise ValueError("A target column must contain only 0 (Normal) or 1 (Abnormal).")
        targets = numeric.astype(int)
        labels = targets.map({0: "1", 1: "-1"})
        return labels, targets

    labels = raw_values.apply(_normalize_original_label)
    targets = labels.map({"1": 0, "-1": 1}).astype(int)
    return labels, targets


def _records_from_dataframe(df):
    """Convert a CSV/Excel DataFrame into 34 model features."""
    if df.empty:
        raise ValueError("The uploaded spreadsheet is empty.")

    df = df.copy()
    df.columns = [_normalize_column(column) for column in df.columns]

    if len(set(df.columns)) != len(df.columns):
        raise ValueError("The file contains duplicate column names after normalization.")

    label_column = next((c for c in df.columns if c in LABEL_NAMES), None)
    id_column = next((c for c in df.columns if c in ID_NAMES), None)

    # Input format A: all 34 pre-engineered model features.
    if all(column in df.columns for column in FEATURE_COLUMNS):
        selected = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if selected.isna().any().any():
            raise ValueError("The 34 feature columns contain missing or non-numeric values.")
        values = selected.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("The 34 feature columns contain infinity.")

        output = selected.copy()
        if id_column:
            output.insert(0, "observation_id", df[id_column].astype(str).values)
        else:
            output.insert(0, "observation_id", range(1, len(output) + 1))

        labels, targets = _labels_and_targets(df, label_column)
        if labels is not None:
            output["class_label"] = labels.values
            output["target"] = targets.values
        return output

    # Input format B: raw 152 sensor measurements.
    excluded = set(ID_NAMES) | {"split_name"}
    if label_column:
        excluded.add(label_column)
    measurement_columns = [c for c in df.columns if c not in excluded]

    if len(measurement_columns) != EXPECTED_MEASUREMENTS:
        raise ValueError(
            "Unsupported dataset. Upload either 152 raw sensor measurement "
            "columns or all 34 engineered feature columns. "
            f"Found {len(measurement_columns)} candidate measurement columns."
        )

    numeric = df[measurement_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Raw sensor measurements contain missing or non-numeric values.")

    values_array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values_array).all():
        raise ValueError("Raw sensor measurements contain infinity.")

    labels, _ = _labels_and_targets(df, label_column)
    rows = []
    for index, values in enumerate(values_array):
        observation_id = (
            df.iloc[index][id_column]
            if id_column
            else index + 1
        )
        label = labels.iloc[index] if labels is not None else None
        rows.append(_feature_row(values, observation_id, label))

    return pd.DataFrame(rows)


def process_uploaded_file(file_bytes, filename):
    """Read TS, CSV, XLSX or XLS files and return all 34 model features."""
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "ts":
        # Determine labelled/unlabelled format from the data rather than
        # silently treating malformed labelled data as unlabelled.
        try:
            records = parse_ts_file(file_bytes, contains_labels=True)
        except ValueError as labelled_error:
            try:
                records = parse_ts_file(file_bytes, contains_labels=False)
            except ValueError:
                raise labelled_error
        return extract_features(records)

    if extension == "csv":
        try:
            df = pd.read_csv(BytesIO(file_bytes))
            if not _is_recognized_header(df.columns):
                df = pd.read_csv(BytesIO(file_bytes), header=None)
                df = _assign_headerless_columns(df)
            return _records_from_dataframe(df)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Unable to read CSV file: {exc}") from exc

    if extension in {"xlsx", "xls"}:
        try:
            df = pd.read_excel(BytesIO(file_bytes))
            if not _is_recognized_header(df.columns):
                df = pd.read_excel(BytesIO(file_bytes), header=None)
                df = _assign_headerless_columns(df)
            return _records_from_dataframe(df)
        except ImportError as exc:
            raise ValueError(
                "Excel support requires openpyxl (.xlsx) or xlrd (.xls)."
            ) from exc
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Unable to read Excel file: {exc}") from exc

    raise ValueError("Unsupported file format. Upload TS, CSV, XLSX or XLS.")


def engineer_features(values, observation_id=1):
    """Return the canonical ordered 34-feature mapping for one wafer."""
    row = _feature_row(values, observation_id)
    return {column: float(row[column]) for column in FEATURE_COLUMNS}
