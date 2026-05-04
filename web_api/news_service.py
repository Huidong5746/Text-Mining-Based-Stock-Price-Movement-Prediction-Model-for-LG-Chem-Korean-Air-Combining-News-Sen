from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


NEWS_QUERIES = {
    "koreanair": {"label": "대한항공", "query": "대한항공 주가"},
    "lgchem": {"label": "LG화학", "query": "LG화학 주가"},
    "oil": {"label": "유가", "query": "WTI 유가 국제유가"},
}


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    summary: str
    published_at: str


def _text(node, selector: str) -> str:
    found = node.select_one(selector)
    return found.get_text(" ", strip=True) if found else ""


def crawl_latest_naver_news(query: str, limit: int = 6) -> list[NewsItem]:
    response = requests.get(
        "https://search.naver.com/search.naver",
        params={"where": "news", "query": query, "sort": "1"},
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    items: list[NewsItem] = []
    for node in soup.select("div.news_area")[:limit]:
        title_node = node.select_one("a.news_tit")
        if not title_node:
            continue
        info_text = " ".join(info.get_text(" ", strip=True) for info in node.select("span.info"))
        items.append(
            NewsItem(
                title=title_node.get_text(" ", strip=True),
                url=title_node.get("href", ""),
                source=_text(node, "a.info.press") or "Naver News",
                summary=_text(node, "a.api_txt_lines.dsc_txt_wrap"),
                published_at=info_text,
            )
        )

    if items:
        return items

    return _fallback_parse_news_links(soup, limit=limit)


def _fallback_parse_news_links(soup: BeautifulSoup, limit: int) -> list[NewsItem]:
    blocked_domains = {
        "www.naver.com",
        "search.shopping.naver.com",
        "dict.naver.com",
        "map.naver.com",
        "terms.naver.com",
        "academic.naver.com",
        "help.naver.com",
        "keep.naver.com",
    }
    blocked_titles = {"NAVER", "네이버뉴스", "Keep에 바로가기", "옵션 가이드"}
    items: list[NewsItem] = []
    seen: set[str] = set()

    for anchor in soup.select('a[href^="http"]'):
        title = anchor.get_text(" ", strip=True)
        url = anchor.get("href", "")
        domain = urlparse(url).netloc
        if (
            not title
            or title in blocked_titles
            or "언론사 선정" in title
            or "보고 싶은 언론사" in title
            or len(title) < 12
            or url in seen
            or domain in blocked_domains
        ):
            continue
        if not any(token in url for token in ("news", "article", "press", "media")):
            continue

        parent_text = anchor.parent.get_text(" ", strip=True) if anchor.parent else ""
        summary = parent_text.replace(title, "").strip()
        if len(summary) > 180:
            summary = summary[:180] + "..."

        items.append(
            NewsItem(
                title=title,
                url=url,
                source=domain or "news",
                summary=summary,
                published_at="최신순",
            )
        )
        seen.add(url)
        if len(items) >= limit:
            break

    return items


def get_latest_news(limit: int = 6) -> dict[str, object]:
    result: dict[str, object] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": {},
    }
    for key, meta in NEWS_QUERIES.items():
        try:
            items = crawl_latest_naver_news(meta["query"], limit=limit)
            error = None
        except requests.RequestException as exc:
            items = []
            error = str(exc)
        result["groups"][key] = {
            "label": meta["label"],
            "query": meta["query"],
            "error": error,
            "items": [asdict(item) for item in items],
        }
    return result
