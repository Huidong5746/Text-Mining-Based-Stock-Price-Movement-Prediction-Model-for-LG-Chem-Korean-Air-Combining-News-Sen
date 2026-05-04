from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


POSITIVE_WORDS = {
    "상승",
    "개선",
    "성장",
    "호조",
    "흑자",
    "수익",
    "확대",
    "증가",
    "강세",
    "회복",
    "기대",
    "수주",
    "실적",
}

NEGATIVE_WORDS = {
    "하락",
    "악화",
    "감소",
    "적자",
    "부진",
    "약세",
    "손실",
    "우려",
    "리스크",
    "하향",
    "감산",
    "불확실",
    "비용",
}


def clean_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_keywords(text: str, keywords: set[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


def add_sentiment_features(df: pd.DataFrame, text_col: str = "news_text") -> pd.DataFrame:
    result = df.copy()
    result[text_col] = result[text_col].map(clean_text)
    result["positive_hits"] = result[text_col].map(lambda text: count_keywords(text, POSITIVE_WORDS))
    result["negative_hits"] = result[text_col].map(lambda text: count_keywords(text, NEGATIVE_WORDS))
    result["sentiment_score"] = result["positive_hits"] - result["negative_hits"]
    return result


def build_tfidf_features(texts: pd.Series, max_features: int = 1000) -> tuple[pd.DataFrame, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r"(?u)\b[0-9A-Za-z가-힣]{2,}\b",
    )
    matrix = vectorizer.fit_transform(texts.map(clean_text))
    names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=names), vectorizer
