from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CompanyConfig
from .features import add_sentiment_features, build_tfidf_features


DATE_CANDIDATES = ("predict_date", "일자", "날짜", "date")


def _find_date_col(columns: Iterable[str]) -> str:
    for candidate in DATE_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a date column. Expected one of: {DATE_CANDIDATES}")


def normalize_date(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"\s+", "", regex=True)
    return pd.to_datetime(cleaned, errors="coerce").dt.date


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_targets(config: CompanyConfig) -> pd.DataFrame:
    df = pd.read_csv(config.target_csv, encoding="utf-8-sig")
    date_col = _find_date_col(df.columns)
    return (
        df.assign(date=normalize_date(df[date_col]))
        .dropna(subset=["date"])
        [["date", config.close_col, config.target_col]]
        .rename(columns={config.close_col: "close", config.target_col: "target"})
        .assign(
            close=lambda x: pd.to_numeric(x["close"].astype(str).str.replace(",", ""), errors="coerce"),
            target=lambda x: x["target"].astype(int),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_news(config: CompanyConfig, include_oil: bool = True) -> pd.DataFrame:
    source = config.oil_news_xlsx if include_oil and config.oil_news_xlsx else config.news_xlsx
    df = read_table(source)
    date_col = _find_date_col(df.columns)
    df = df.assign(date=normalize_date(df[date_col])).dropna(subset=["date"])

    text_cols = [col for col in (config.company_news_col, config.oil_news_col) if col in df.columns]
    if not text_cols:
        object_cols = [col for col in df.columns if df[col].dtype == "object" and col != date_col]
        text_cols = object_cols[:2]

    if not text_cols:
        raise ValueError(f"No text columns found in {source}")

    df["news_text"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return (
        df.groupby("date", as_index=False)["news_text"]
        .agg(lambda values: " ".join(v for v in values if v))
        .sort_values("date")
        .reset_index(drop=True)
    )


def parse_embedding(value: object) -> list[float] | None:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if not isinstance(value, str) or not value.startswith("["):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [float(x) for x in parsed]


def load_existing_embeddings(config: CompanyConfig, include_oil: bool = True) -> pd.DataFrame | None:
    source = config.oil_news_xlsx if include_oil and config.oil_news_xlsx else config.news_xlsx
    df = read_table(source)
    date_col = _find_date_col(df.columns)
    embedding_cols = [col for col in df.columns if "embedding" in col.lower() or "임베딩" in col]
    if not embedding_cols:
        return None

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        feature_parts = []
        for col in embedding_cols:
            values = parse_embedding(row[col])
            if values:
                feature_parts.extend(values)
        if feature_parts:
            rows.append({"date": normalize_date(pd.Series([row[date_col]])).iloc[0], "embedding": feature_parts})

    if not rows:
        return None
    return pd.DataFrame(rows).dropna(subset=["date"])


def make_supervised_dataset(
    config: CompanyConfig,
    feature_mode: str = "tfidf",
    include_oil: bool = True,
) -> tuple[pd.DataFrame, pd.Series, object | None]:
    targets = load_targets(config)

    if feature_mode == "embedding":
        embeddings = load_existing_embeddings(config, include_oil=include_oil)
        if embeddings is None:
            raise ValueError("No existing embedding columns were found. Use --feature-mode tfidf first.")
        merged = targets.merge(embeddings, on="date", how="inner")
        feature_df = pd.DataFrame(merged["embedding"].tolist()).add_prefix("emb_")
        return feature_df, merged["target"], None

    news = load_news(config, include_oil=include_oil)
    merged = targets.merge(news, on="date", how="inner")
    merged = add_sentiment_features(merged, text_col="news_text")
    text_features, vectorizer = build_tfidf_features(merged["news_text"])
    numeric = merged[["close", "sentiment_score", "positive_hits", "negative_hits"]].fillna(0).reset_index(drop=True)
    features = pd.concat([numeric, text_features.reset_index(drop=True)], axis=1)
    features = features.fillna(0)
    return features, merged["target"].reset_index(drop=True), vectorizer
