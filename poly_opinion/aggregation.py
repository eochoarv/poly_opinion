from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, rbf_kernel


# --- internal helpers ---

def _filter_journalists(
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    min_articles: int,
) -> list[dict]:
    df = df.copy()
    df["_idx"] = np.arange(len(df))
    result = []
    for journalist, group in df.groupby("journalist"):
        if len(group) < min_articles:
            continue
        result.append({
            "journalist": journalist,
            "indices": group["_idx"].values,
            "n_publications": len(group),
            "first_date": group["date"].min(),
            "last_date": group["date"].max(),
        })
    return result


def _sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 50,
    random_state: int = 42,
) -> float:
    """
    Approximates Earth Mover's Distance in high dimensions by averaging
    1-D Wasserstein distances across random unit-vector projections.
    """
    rng = np.random.RandomState(random_state)
    directions = rng.randn(n_projections, X.shape[1])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    X_proj = X @ directions.T  # (n_X, n_projections)
    Y_proj = Y @ directions.T  # (n_Y, n_projections)
    return float(np.mean([
        wasserstein_distance(X_proj[:, k], Y_proj[:, k])
        for k in range(n_projections)
    ]))


def _mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    gamma=None triggers the median heuristic for automatic bandwidth selection.
    """
    if gamma is None:
        all_pts = np.vstack([X, Y])
        dists = euclidean_distances(all_pts)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0

    XX = rbf_kernel(X, X, gamma).mean()
    YY = rbf_kernel(Y, Y, gamma).mean()
    XY = rbf_kernel(X, Y, gamma).mean()
    return float(np.sqrt(max(XX - 2.0 * XY + YY, 0.0)))


def _pairwise_distances(groups: list[np.ndarray], dist_fn) -> np.ndarray:
    n = len(groups)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(groups[i], groups[j])
            D[i, j] = D[j, i] = d
    return D


# --- public API ---

def aggregate(
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    method: str = "mean",
    min_articles: int = 5,
    wasserstein_projections: int = 50,
    mmd_gamma: Optional[float] = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
        journalist_df        — metadata per journalist
        journalist_embeddings — (N, D) mean-pooled vectors (always computed)
        distance_matrix      — (N, N) pairwise distances via the chosen method
    """
    journalist_data = _filter_journalists(df, article_embeddings, min_articles)
    if not journalist_data:
        raise ValueError(
            f"No journalists survived the min_articles={min_articles} filter. "
            "Lower the threshold or add more data."
        )

    rows, mean_vecs, groups = [], [], []
    for jd in journalist_data:
        vecs = article_embeddings[jd["indices"]]
        mean_v = vecs.mean(axis=0)
        mean_v /= np.linalg.norm(mean_v)
        rows.append({k: v for k, v in jd.items() if k != "indices"})
        mean_vecs.append(mean_v)
        groups.append(vecs)

    journalist_df = pd.DataFrame(rows)
    journalist_embeddings = np.vstack(mean_vecs)

    if method == "mean":
        distance_matrix = cosine_distances(journalist_embeddings)
    elif method == "wasserstein":
        print(f"  Computing sliced Wasserstein distances ({wasserstein_projections} projections)…")
        distance_matrix = _pairwise_distances(
            groups,
            lambda X, Y: _sliced_wasserstein(X, Y, wasserstein_projections),
        )
    elif method == "mmd":
        gamma_label = f"gamma={mmd_gamma}" if mmd_gamma else "median heuristic"
        print(f"  Computing MMD distances ({gamma_label})…")
        distance_matrix = _pairwise_distances(
            groups,
            lambda X, Y: _mmd(X, Y, mmd_gamma),
        )
    else:
        raise ValueError(
            f"Unknown aggregation method {method!r}. Choose: mean | wasserstein | mmd"
        )

    return journalist_df, journalist_embeddings, distance_matrix


def nearest_neighbors(
    journalist_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    journalist_name: str,
    k: int = 10,
) -> pd.DataFrame:
    names = journalist_df["journalist"].tolist()
    if journalist_name not in names:
        raise ValueError(f"{journalist_name!r} not found in journalist list.")
    i = names.index(journalist_name)
    distances = distance_matrix[i]
    idx = np.argsort(distances)[1 : k + 1]
    result = journalist_df.iloc[idx].copy()
    result["distance"] = distances[idx]
    cols = ["journalist", "distance", "cluster_hierarchical", "cluster_hdbscan", "n_publications"]
    return result[[c for c in cols if c in result.columns]]
