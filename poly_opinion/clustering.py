import numpy as np
import hdbscan
from sklearn.cluster import AgglomerativeClustering


def cluster_hdbscan(journalist_embeddings: np.ndarray) -> np.ndarray:
    """Density-based clustering. Label -1 = noise / no cluster."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidean",
    )
    return clusterer.fit_predict(journalist_embeddings)


def cluster_hierarchical(distance_matrix: np.ndarray, n_clusters: int = 6) -> np.ndarray:
    """Force a fixed number of clusters. Useful for exploration."""
    n_clusters = min(n_clusters, len(distance_matrix))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(distance_matrix)
