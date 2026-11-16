import mlflow.sklearn
import pandas as pd
from pathlib import Path

from utils.data_processor import process_uploaded_file, FEATURE_COLUMNS

df = process_uploaded_file(
    Path("Wafer_TEST.ts").read_bytes(),
    "Wafer_TEST.ts"
)

model = mlflow.sklearn.load_model(
    "models/wafer_defect_classifier_v9"
)

X = df[FEATURE_COLUMNS]
y = df["target"].astype(int)
p = model.predict_proba(X)[:, 1]

missed = df.loc[
    (y == 1) & (p < 0.5)
].copy()

missed["abnormal_probability"] = p[(y == 1) & (p < 0.5)]

columns = ["observation_id", "abnormal_probability"]

for segment in range(1, 9):
    columns.extend([
        f"segment_{segment}_stddev",
        f"segment_{segment}_range"
    ])

print(missed[columns].sort_values(
    "abnormal_probability"
).round(4).to_string(index=False))
