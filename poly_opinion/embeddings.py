import hashlib
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_article_text(row: pd.Series) -> str:
    title = str(row.get("title", ""))
    text = str(row.get("text", ""))
    return f"Title: {title}\n\nText: {text}"


def _cache_key(df: pd.DataFrame, model_name: str) -> str:
    texts = df.apply(build_article_text, axis=1).tolist()
    payload = json.dumps({"model": model_name, "texts": texts}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def compute_embeddings(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
    cache_dir: str = "cache",
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(df, model_name)
    cache_path = os.path.join(cache_dir, f"{key}.npy")

    if os.path.exists(cache_path):
        print(f"  Loading embeddings from cache ({cache_path})")
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    docs = df.apply(build_article_text, axis=1).tolist()
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    np.save(cache_path, embeddings)
    print(f"  Embeddings cached to {cache_path}")
    return embeddings
