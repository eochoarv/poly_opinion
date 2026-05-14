# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
python3 main.py                      # uses config.yaml
python3 main.py --config my.yaml     # override config file
```

The pipeline expects input at `data/publications.csv` and writes results to `outputs/`.
Embeddings are cached in `cache/` (keyed by article content + model name) so re-runs skip encoding.

## Configuration — `config.yaml`

All tunable parameters live here. Key knobs:

| Key | Default | Notes |
|-----|---------|-------|
| `embedding.model` | `all-mpnet-base-v2` | Swap for `all-MiniLM-L6-v2` (faster) or `BAAI/bge-large-en-v1.5` (higher quality) |
| `pipeline.aggregation` | `mean` | `mean` \| `wasserstein` \| `mmd` — see Distance methods below |
| `pipeline.min_articles` | `5` | Journalists with fewer articles are dropped |
| `pipeline.n_clusters` | `6` | For hierarchical clustering; auto-capped at N journalists |
| `pipeline.wasserstein_projections` | `50` | Random projections for sliced Wasserstein |

## Package architecture

```
poly_opinion/          ← importable package
  config.py            ← Config dataclasses + load_config()
  loader.py            ← load_data(): CSV ingestion and validation
  embeddings.py        ← compute_embeddings(): encode articles, cache to disk
  aggregation.py       ← aggregate(): per-journalist distance matrix (all 3 methods)
                          nearest_neighbors(): query utility
  clustering.py        ← cluster_hdbscan(), cluster_hierarchical()
  visualization.py     ← make_umap_projection(), plot_journalist_map()
  pipeline.py          ← run(cfg): orchestrates all stages end-to-end
main.py                ← CLI entrypoint (argparse → load_config → run)
config.yaml            ← runtime configuration
utils.py               ← legacy monolith (superseded, kept for reference)
```

## Distance methods

The `aggregation` setting controls how journalist similarity is measured:

- **`mean`** — cosine distance between mean-pooled article embeddings. Fast, works well when journalists have consistent topic focus. Loses distributional information.
- **`wasserstein`** — sliced Wasserstein distance between each journalist's set of article embeddings. Captures spread and shape of the distribution; better when journalists cover varied topics. O(N² × projections × articles·log·articles).
- **`mmd`** — Maximum Mean Discrepancy with RBF kernel and automatic bandwidth (median heuristic). Similar intent to Wasserstein but kernel-based; more sensitive to the choice of kernel scale.

All three return a symmetric (N × N) distance matrix consumed by UMAP and hierarchical clustering.

## Outputs

| File | Contents |
|------|----------|
| `outputs/journalist_profiles.csv` | One row per journalist: both cluster labels, UMAP coords, article count, date range |
| `outputs/journalist_distance_matrix.csv` | Square pairwise distance matrix (method-dependent) |
| `outputs/journalist_map.png` | 2-D UMAP scatter plot colored by hierarchical cluster |

## Input CSV format

Required columns: `journalist`, `publication_id`, `date`, `title`, `text`.
Rows with `text` shorter than 100 characters or missing `journalist`/`text` are dropped.
