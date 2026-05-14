import argparse

from poly_opinion.config import load_config
from poly_opinion.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="poly_opinion journalist similarity pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run(load_config(args.config))
