import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

import umap
import hdbscan


# -----------------------------
# 1. Load and clean data
# -----------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"journalist", "publication_id", "date", "title", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["journalist", "text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 100]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df.reset_index(drop=True)


# -----------------------------
# 2. Embed publications
# -----------------------------

def build_article_text(row: pd.Series) -> str:
    """
    Combine title and body. You can modify this later to include metadata,
    outlet, section, tags, etc.
    """
    title = str(row.get("title", ""))
    text = str(row.get("text", ""))
    return f"Title: {title}\n\nText: {text}"


def compute_embeddings(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
) -> np.ndarray:
    model = SentenceTransformer(model_name)

    docs = df.apply(build_article_text, axis=1).tolist()

    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings)


# -----------------------------
# 3. Aggregate article embeddings
# -----------------------------

def aggregate_by_journalist(
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    min_articles: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Simple version: average all article embeddings for each journalist.

    Later you can replace this with:
    - recency-weighted averages
    - topic-weighted averages
    - distributional distances
    - Wasserstein/MMD between article embeddings
    """
    df = df.copy()
    df["embedding_index"] = np.arange(len(df))

    journalist_rows = []
    journalist_vectors = []

    for journalist, group in df.groupby("journalist"):
        if len(group) < min_articles:
            continue

        idx = group["embedding_index"].values
        vectors = article_embeddings[idx]

        avg_vector = vectors.mean(axis=0)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)

        journalist_rows.append({
            "journalist": journalist,
            "n_publications": len(group),
            "first_date": group["date"].min(),
            "last_date": group["date"].max(),
        })
        journalist_vectors.append(avg_vector)

    journalist_df = pd.DataFrame(journalist_rows)
    journalist_embeddings = np.vstack(journalist_vectors)

    return journalist_df, journalist_embeddings


# -----------------------------
# 4. Compute distances
# -----------------------------

def compute_distance_matrix(journalist_embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine distance:
        0 = very similar
        1 = orthogonal
        2 = opposite direction, though uncommon for normalized text embeddings
    """
    return cosine_distances(journalist_embeddings)


# -----------------------------
# 5. Cluster journalists
# -----------------------------

def cluster_hdbscan(journalist_embeddings: np.ndarray) -> np.ndarray:
    """
    Density-based clustering.
    Label -1 means noise / no clear cluster.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(journalist_embeddings)
    return labels


def cluster_hierarchical(
    distance_matrix: np.ndarray,
    n_clusters: int = 6,
) -> np.ndarray:
    """
    Alternative: force a chosen number of clusters.
    Useful for exploration.
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )

    labels = clustering.fit_predict(distance_matrix)
    return labels


# -----------------------------
# 6. Visualize journalist map
# -----------------------------

def make_umap_projection(journalist_embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    coords = reducer.fit_transform(journalist_embeddings)
    return coords


def plot_journalist_map(
    journalist_df: pd.DataFrame,
    coords: np.ndarray,
    output_path: str = "outputs/journalist_map.png",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=journalist_df["cluster_hdbscan"],
        s=60,
        alpha=0.8,
    )

    for i, row in journalist_df.iterrows():
        plt.text(
            coords[i, 0],
            coords[i, 1],
            row["journalist"],
            fontsize=8,
            alpha=0.75,
        )

    plt.title("Journalist Similarity Map")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# 7. Find nearest journalists
# -----------------------------

def nearest_neighbors(
    journalist_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    journalist_name: str,
    k: int = 10,
) -> pd.DataFrame:
    names = journalist_df["journalist"].tolist()

    if journalist_name not in names:
        raise ValueError(f"{journalist_name} not found.")

    i = names.index(journalist_name)
    distances = distance_matrix[i]

    nearest_idx = np.argsort(distances)[1:k + 1]

    result = journalist_df.iloc[nearest_idx].copy()
    result["distance"] = distances[nearest_idx]

    return result[["journalist", "distance", "cluster_hdbscan", "n_publications"]]


# -----------------------------
# 8. Main pipeline
# -----------------------------

def main():
    input_path = "data/publications.csv"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(input_path)

    print(f"Loaded {len(df)} publications from {df['journalist'].nunique()} journalists.")

    print("Computing article embeddings...")
    article_embeddings = compute_embeddings(df)

    print("Aggregating by journalist...")
    journalist_df, journalist_embeddings = aggregate_by_journalist(
        df,
        article_embeddings,
        min_articles=5,
    )

    print(f"Kept {len(journalist_df)} journalists after filtering.")

    print("Computing distance matrix...")
    distance_matrix = compute_distance_matrix(journalist_embeddings)

    print("Clustering journalists...")
    journalist_df["cluster_hdbscan"] = cluster_hdbscan(journalist_embeddings)
    journalist_df["cluster_hierarchical"] = cluster_hierarchical(
        distance_matrix,
        n_clusters=min(6, len(journalist_df)),
    )

    print("Building UMAP projection...")
    coords = make_umap_projection(journalist_embeddings)
    journalist_df["umap_x"] = coords[:, 0]
    journalist_df["umap_y"] = coords[:, 1]

    print("Saving outputs...")
    journalist_df.to_csv(f"{output_dir}/journalist_profiles.csv", index=False)

    distance_df = pd.DataFrame(
        distance_matrix,
        index=journalist_df["journalist"],
        columns=journalist_df["journalist"],
    )
    distance_df.to_csv(f"{output_dir}/journalist_distance_matrix.csv")

    plot_journalist_map(
        journalist_df,
        coords,
        output_path=f"{output_dir}/journalist_map.png",
    )

    print("Done.")
    print(f"Saved:")
    print(f"- {output_dir}/journalist_profiles.csv")
    print(f"- {output_dir}/journalist_distance_matrix.csv")
    print(f"- {output_dir}/journalist_map.png")


if __name__ == "__main__":
    main()
