from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .features import add_sentiment_features, clean_text


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_texts(
    texts: list[str],
    model_name: str = "beomi/kcbert-base",
    batch_size: int = 16,
) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    vectors: list[list[float]] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="embedding"):
            batch = [clean_text(text) for text in texts[start : start + batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.extend(pooled.cpu().numpy().astype(np.float32).tolist())
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Create transformer embeddings and simple sentiment features.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file containing a text column.")
    parser.add_argument("--text-col", default="news_text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="beomi/kcbert-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-embedding", action="store_true", help="Only add dictionary sentiment features.")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_excel(input_path) if input_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(input_path)
    df = add_sentiment_features(df, text_col=args.text_col)
    if not args.skip_embedding:
        df["embedding"] = embed_texts(df[args.text_col].fillna("").tolist(), args.model_name, args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
