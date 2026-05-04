from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, get_company
from .dataset import make_supervised_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions with a saved model artifact.")
    parser.add_argument("--artifact", required=True, help="Path under models/ or an absolute path.")
    parser.add_argument("--company", choices=["koreanair", "lgchem"], required=True)
    parser.add_argument("--feature-mode", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--no-oil", action="store_true")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        artifact_path = MODEL_DIR / args.artifact
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.artifact}")

    artifact = joblib.load(artifact_path)
    config = get_company(args.company)
    X, y, _ = make_supervised_dataset(config, feature_mode=args.feature_mode, include_oil=not args.no_oil)
    pred = artifact["model"].predict(X)
    result = pd.DataFrame({"actual": y, "predicted": pred})
    print(result.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
