from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class CompanyConfig:
    key: str
    display_name: str
    keyword: str
    target_csv: Path
    close_col: str
    target_col: str
    news_xlsx: Path
    company_news_col: str
    oil_news_xlsx: Path | None = None
    oil_news_col: str = "oil_news"


COMPANIES: dict[str, CompanyConfig] = {
    "koreanair": CompanyConfig(
        key="koreanair",
        display_name="대한항공",
        keyword="대한항공",
        target_csv=DATA_DIR / "대한항공 주가 타겟.csv",
        close_col="대한종가",
        target_col="대한타겟",
        news_xlsx=DATA_DIR / "대한항공 전처리 (완) 인베딩 (미완).xlsx",
        company_news_col="dh_news",
        oil_news_xlsx=DATA_DIR / "대한항공 + 유가 전처리(완) 인베딩(미완).xlsx",
    ),
    "lgchem": CompanyConfig(
        key="lgchem",
        display_name="LG화학",
        keyword="LG화학",
        target_csv=DATA_DIR / "LG주가타겟.csv",
        close_col="엘지종가",
        target_col="엘지타겟",
        news_xlsx=DATA_DIR / "LG화학 전처리(완) 인베딩(미완).xlsx",
        company_news_col="lg_news",
        oil_news_xlsx=DATA_DIR / "lg화학 + 유가 전처리 (완)  인베딩(미완).xlsx",
    ),
}


def get_company(key: str) -> CompanyConfig:
    try:
        return COMPANIES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPANIES))
        raise ValueError(f"Unknown company '{key}'. Supported values: {supported}") from exc
