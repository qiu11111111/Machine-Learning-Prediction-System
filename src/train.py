"""Train three leakage-safe classification pipelines."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from preprocess import RANDOM_STATE, load_data, split_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def transformer() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])


def models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def main() -> None:
    X_train, _, y_train, _ = split_data(load_data())
    MODEL_DIR.mkdir(exist_ok=True)
    for name, estimator in models().items():
        pipeline = Pipeline([("preprocess", transformer()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, path)
        print(f"Saved {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
