"""Evaluate saved models and create publication-style figures."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from preprocess import FEATURES, load_data, split_data

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models"
sns.set_theme(style="whitegrid", context="notebook")


def save_eda(data: pd.DataFrame) -> None:
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=180)
    plt.close()

    data[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(
        figsize=(12, 8), bins=20, color="#2878B5", edgecolor="white"
    )
    plt.suptitle("Continuous Feature Distributions")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_distributions.png", dpi=180)
    plt.close()

    ax = sns.countplot(data=data, x="target", hue="target", legend=False)
    ax.set(title="Binary Target Distribution", xlabel="Heart disease", ylabel="Patients")
    ax.set_xticks([0, 1], ["Absent", "Present"])
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    save_eda(data)
    _, X_test, _, y_test = split_data(data)
    records, roc_rows = [], []
    for model_path in sorted(MODELS.glob("*.joblib")):
        model = joblib.load(model_path)
        label = model_path.stem.replace("_", " ").title()
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        records.append({
            "model": label,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        })
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_rows.append((label, fpr, tpr))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
        plt.title(f"Confusion Matrix — {label}")
        plt.tight_layout()
        plt.savefig(FIGURES / f"confusion_matrix_{model_path.stem}.png", dpi=180)
        plt.close()

    metrics = pd.DataFrame(records).sort_values("roc_auc", ascending=False)
    metrics.to_csv(ROOT / "results" / "metrics.csv", index=False)
    for label, fpr, tpr in roc_rows:
        auc = metrics.loc[metrics.model == label, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "roc_curve_comparison.png", dpi=180)
    plt.close()

    best_name = metrics.iloc[0]["model"]
    best_path = MODELS / f"{best_name.lower().replace(' ', '_')}.joblib"
    best = joblib.load(best_path)
    names = best.named_steps["preprocess"].get_feature_names_out()
    estimator = best.named_steps["model"]
    values = (estimator.feature_importances_ if hasattr(estimator, "feature_importances_")
              else abs(estimator.coef_[0]))
    importance = pd.Series(values, index=names).nlargest(15).sort_values()
    importance.plot.barh(figsize=(9, 6), color="#E07A5F")
    plt.title(f"Top Feature Importances — {best_name}")
    plt.xlabel("Importance magnitude"); plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180); plt.close()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
