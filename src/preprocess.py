"""Load, validate, and split the UCI Cleveland Heart Disease data."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Download the official UCI file when absent and return a clean schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    data = pd.read_csv(path, na_values=["?", "NaN"])
    expected = FEATURES + ["num"]
    if list(data.columns) != expected:
        raise ValueError(f"Unexpected dataset columns: {list(data.columns)}")
    data[TARGET] = (data.pop("num") > 0).astype(int)
    return data


def split_data(data: pd.DataFrame):
    """Create a deterministic stratified 80/20 holdout split."""
    X = data[FEATURES]
    y = data[TARGET]
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )


if __name__ == "__main__":
    frame = load_data()
    train_X, test_X, train_y, test_y = split_data(frame)
    print(f"Rows: {len(frame)} | features: {len(FEATURES)}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print(f"Train/test rows: {len(train_X)}/{len(test_X)}")
