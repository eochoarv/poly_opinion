import os

import pandas as pd

from .aggregation import aggregate
from .clustering import cluster_hdbscan, cluster_hierarchical
from .config import Config
from .embeddings import compute_embeddings
from .loader import load_data
from .visualization import make_umap_projection, plot_journalist_map


def run(cfg: Config) -> None:
    os.makedirs(cfg.data.output_dir, exist_ok=True)

    print("Loading data…")
    df = load_data(cfg.data.input_path)
    print(f"  {len(df)} articles from {df['journalist'].nunique()} journalists.")

    print("Computing embeddings…")
    article_embeddings = compute_embeddings(
        df,
        model_name=cfg.embedding.model,
        batch_size=cfg.embedding.batch_size,
        cache_dir=cfg.data.cache_dir,
    )

    print(f"Aggregating journalists (method={cfg.pipeline.aggregation!r})…")
    journalist_df, journalist_embeddings, distance_matrix = aggregate(
        df,
        article_embeddings,
        method=cfg.pipeline.aggregation,
        min_articles=cfg.pipeline.min_articles,
        wasserstein_projections=cfg.pipeline.wasserstein_projections,
    )
    print(f"  {len(journalist_df)} journalists kept after filtering.")

    print("Clustering…")
    journalist_df["cluster_hdbscan"] = cluster_hdbscan(journalist_embeddings)
    journalist_df["cluster_hierarchical"] = cluster_hierarchical(
        distance_matrix,
        n_clusters=cfg.pipeline.n_clusters,
    )

    print("Building UMAP projection…")
    coords = make_umap_projection(
        distance_matrix,
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
    )
    journalist_df["umap_x"] = coords[:, 0]
    journalist_df["umap_y"] = coords[:, 1]

    print("Saving outputs…")
    out = cfg.data.output_dir
    journalist_df.to_csv(f"{out}/journalist_profiles.csv", index=False)
    pd.DataFrame(
        distance_matrix,
        index=journalist_df["journalist"],
        columns=journalist_df["journalist"],
    ).to_csv(f"{out}/journalist_distance_matrix.csv")
    plot_journalist_map(journalist_df, coords, output_path=f"{out}/journalist_map.png")

    print("Done.")
