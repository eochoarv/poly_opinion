from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    input_path: str = "data/publications.csv"
    output_dir: str = "outputs"
    cache_dir: str = "cache"


@dataclass
class EmbeddingConfig:
    model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32


@dataclass
class PipelineConfig:
    min_articles: int = 5
    aggregation: str = "mean"
    n_clusters: int = 6
    wasserstein_projections: int = 50


@dataclass
class UMAPConfig:
    n_neighbors: int = 10
    min_dist: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)


def load_config(path: str = "config.yaml") -> Config:
    if not Path(path).exists():
        return Config()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "embedding" in raw:
        cfg.embedding = EmbeddingConfig(**raw["embedding"])
    if "pipeline" in raw:
        cfg.pipeline = PipelineConfig(**raw["pipeline"])
    if "umap" in raw:
        cfg.umap = UMAPConfig(**raw["umap"])
    return cfg
