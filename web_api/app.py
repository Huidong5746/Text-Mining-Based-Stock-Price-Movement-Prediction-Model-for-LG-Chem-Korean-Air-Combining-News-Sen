from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .news_service import get_latest_news
from .prediction_service import predict_recent, predict_tomorrow_from_latest_news, prediction_summary


app = FastAPI(title="Stock News Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/news/latest")
def latest_news(limit: int = Query(default=6, ge=1, le=10)) -> dict[str, object]:
    return get_latest_news(limit=limit)


@app.get("/api/predictions/summary")
def predictions_summary() -> dict[str, object]:
    return prediction_summary()


@app.get("/api/predictions/tomorrow")
def tomorrow_predictions(limit: int = Query(default=5, ge=1, le=10)) -> dict[str, object]:
    return predict_tomorrow_from_latest_news(limit=limit)


@app.get("/api/predictions/{company_key}")
def predictions(company_key: str, limit: int = Query(default=20, ge=1, le=50)) -> dict[str, object]:
    try:
        return predict_recent(company_key, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
