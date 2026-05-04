from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.stock_news_prediction.config import MODEL_DIR, OUTPUT_DIR, get_company
from src.stock_news_prediction.dataset import load_targets, make_supervised_dataset
from src.stock_news_prediction.features import add_sentiment_features, clean_text
from .news_service import get_latest_news


MODEL_FILES = {
    "koreanair": MODEL_DIR / "koreanair_tfidf_logistic.joblib",
    "lgchem": MODEL_DIR / "lgchem_tfidf_logistic.joblib",
}

REPORT_FILES = {
    "koreanair": OUTPUT_DIR / "koreanair_tfidf_logistic_report.json",
    "lgchem": OUTPUT_DIR / "lgchem_tfidf_logistic_report.json",
}


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_summary() -> dict[str, object]:
    return {
        key: {
            "model_exists": MODEL_FILES[key].exists(),
            "report": _load_json(REPORT_FILES[key]),
        }
        for key in MODEL_FILES
    }


def predict_recent(company_key: str, limit: int = 20) -> dict[str, object]:
    if company_key not in MODEL_FILES:
        raise ValueError("company_key must be koreanair or lgchem")

    model_path = MODEL_FILES[company_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact is missing: {model_path}")

    artifact = joblib.load(model_path)
    config = get_company(company_key)
    X, y, _ = make_supervised_dataset(config, feature_mode="tfidf", include_oil=True)
    pred = artifact["model"].predict(X)

    frame = pd.DataFrame({"actual": y.astype(int), "predicted": pred.astype(int)})
    recent = frame.tail(limit).reset_index(drop=True)
    return {
        "company": company_key,
        "display_name": config.display_name,
        "total_rows": int(len(frame)),
        "recent": recent.to_dict(orient="records"),
        "report": _load_json(REPORT_FILES[company_key]),
    }


def _latest_news_text(news_payload: dict[str, object], group_key: str) -> str:
    groups = news_payload.get("groups", {})
    group = groups.get(group_key, {}) if isinstance(groups, dict) else {}
    items = group.get("items", []) if isinstance(group, dict) else []
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("title", "")))
        parts.append(str(item.get("summary", "")))
    return clean_text(" ".join(parts))


def _build_latest_feature_frame(company_key: str, news_payload: dict[str, object]) -> tuple[pd.DataFrame, str]:
    config = get_company(company_key)
    artifact = joblib.load(MODEL_FILES[company_key])
    vectorizer = artifact.get("vectorizer")
    if vectorizer is None:
        raise ValueError("Saved artifact does not contain a TF-IDF vectorizer.")

    company_text = _latest_news_text(news_payload, company_key)
    oil_text = _latest_news_text(news_payload, "oil")
    combined_text = clean_text(f"{company_text} {oil_text}")

    targets = load_targets(config)
    latest_close = float(targets["close"].dropna().iloc[-1])
    base = pd.DataFrame({"close": [latest_close], "news_text": [combined_text]})
    base = add_sentiment_features(base, text_col="news_text")

    matrix = vectorizer.transform(base["news_text"])
    tfidf_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf = pd.DataFrame(matrix.toarray(), columns=tfidf_names)
    numeric = base[["close", "sentiment_score", "positive_hits", "negative_hits"]].reset_index(drop=True)
    features = pd.concat([numeric, tfidf.reset_index(drop=True)], axis=1).fillna(0)

    model = artifact["model"]
    expected_columns = getattr(model.named_steps.get("impute"), "feature_names_in_", None)
    if expected_columns is not None:
        features = features.reindex(columns=list(expected_columns), fill_value=0)
    return features, combined_text


def predict_tomorrow_from_latest_news(limit: int = 5) -> dict[str, object]:
    news_payload = get_latest_news(limit=limit)
    predictions: dict[str, object] = {}

    for company_key in ("koreanair", "lgchem"):
        config = get_company(company_key)
        model_path = MODEL_FILES[company_key]
        if not model_path.exists():
            predictions[company_key] = {
                "display_name": config.display_name,
                "error": f"Model artifact is missing: {model_path}",
            }
            continue

        try:
            artifact = joblib.load(model_path)
            features, combined_text = _build_latest_feature_frame(company_key, news_payload)
            model = artifact["model"]
            predicted = int(model.predict(features)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features)[0][predicted])

            predictions[company_key] = {
                "display_name": config.display_name,
                "predicted": predicted,
                "direction": "상승" if predicted == 1 else "하락",
                "confidence": probability,
                "input_news_chars": len(combined_text),
                "latest_close": float(features["close"].iloc[0]) if "close" in features else None,
            }
        except Exception as exc:  # noqa: BLE001 - API should return per-company failures.
            predictions[company_key] = {
                "display_name": config.display_name,
                "error": str(exc),
            }

    return {
        "updated_at": news_payload["updated_at"],
        "news": news_payload,
        "predictions": predictions,
    }
