from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, OUTPUT_DIR, get_company
from .dataset import make_supervised_dataset


def build_model(model_name: str) -> Pipeline:
    if model_name == "logistic":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("scale", StandardScaler(with_mean=False)),
                ("model", classifier),
            ]
        )
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("model", classifier)])
    if model_name == "baseline":
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    raise ValueError("model_name must be one of: logistic, random_forest, baseline")


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    split_index = int(len(X) * (1 - test_size))
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    n_splits = min(5, max(2, len(X) // 80))
    scores = cross_validate(
        model,
        X,
        y,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring=["accuracy", "precision", "recall", "f1"],
        error_score="raise",
    )
    return {metric.replace("test_", ""): float(values.mean()) for metric, values in scores.items() if metric.startswith("test_")}


def train(company: str, feature_mode: str, model_name: str, include_oil: bool) -> dict[str, object]:
    config = get_company(company)
    X, y, vectorizer = make_supervised_dataset(config, feature_mode=feature_mode, include_oil=include_oil)
    if len(X) < 30:
        raise ValueError(f"Not enough matched rows to train: {len(X)}")

    baseline = build_model("baseline")
    model = build_model(model_name)
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    cv_scores = evaluate_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        "company": company,
        "feature_mode": feature_mode,
        "model": model_name,
        "include_oil": include_oil,
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "baseline_f1": float(f1_score(y_test, baseline_pred, zero_division=0)),
        "cv": cv_scores,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "config_key": company,
        "feature_mode": feature_mode,
        "include_oil": include_oil,
    }
    joblib.dump(artifact, MODEL_DIR / f"{company}_{feature_mode}_{model_name}.joblib")
    (OUTPUT_DIR / f"{company}_{feature_mode}_{model_name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stock movement prediction model.")
    parser.add_argument("--company", choices=["koreanair", "lgchem"], required=True)
    parser.add_argument("--feature-mode", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--no-oil", action="store_true", help="Do not include oil news features.")
    args = parser.parse_args()

    report = train(args.company, args.feature_mode, args.model, include_oil=not args.no_oil)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
