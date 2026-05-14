import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


def make_umap_projection(
    distance_matrix: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
) -> np.ndarray:
    n = len(distance_matrix)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=min_dist,
        metric="precomputed",
        random_state=42,
    )
    return reducer.fit_transform(distance_matrix)


def plot_journalist_map(
    journalist_df: pd.DataFrame,
    coords: np.ndarray,
    output_path: str = "outputs/journalist_map.png",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=journalist_df["cluster_hierarchical"],
        cmap="tab10",
        s=80,
        alpha=0.85,
    )
    for i, row in journalist_df.iterrows():
        plt.text(coords[i, 0], coords[i, 1], row["journalist"], fontsize=8, alpha=0.8)

    plt.title("Journalist Similarity Map")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Hierarchical cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Map saved to {output_path}")
