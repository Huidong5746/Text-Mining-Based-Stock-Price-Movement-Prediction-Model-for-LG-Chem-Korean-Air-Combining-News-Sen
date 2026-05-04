from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from time import sleep
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import OUTPUT_DIR, get_company


@dataclass
class NewsItem:
    query: str
    title: str
    url: str
    source: str
    collected_date: str


def crawl_naver_news(query: str, start: date, end: date, delay: float = 0.5) -> pd.DataFrame:
    """Collect Naver news search result metadata.

    This intentionally crawls only lightweight search metadata. Full article body
    crawling can be slow and brittle, so keep that as a separate optional step.
    """
    session = requests.Session()
    items: list[NewsItem] = []
    current = start
    while current <= end:
        encoded_query = quote(query)
        date_text = current.strftime("%Y.%m.%d")
        url = (
            "https://search.naver.com/search.naver"
            f"?where=news&query={encoded_query}&pd=3&ds={date_text}&de={date_text}"
        )
        response = session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup.select("a.news_tit"):
            items.append(
                NewsItem(
                    query=query,
                    title=node.get_text(" ", strip=True),
                    url=node.get("href", ""),
                    source="naver_search",
                    collected_date=current.isoformat(),
                )
            )
        sleep(delay)
        current += timedelta(days=1)
    return pd.DataFrame([asdict(item) for item in items])


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl lightweight Naver news metadata.")
    parser.add_argument("--company", choices=["koreanair", "lgchem"], required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    config = get_company(args.company)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    rows = []
    for query in tqdm([config.keyword, "WTI 유가"], desc="queries"):
        rows.append(crawl_naver_news(query, start, end, delay=args.delay))
    result = pd.concat(rows, ignore_index=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.company}_naver_news_{args.start}_{args.end}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
