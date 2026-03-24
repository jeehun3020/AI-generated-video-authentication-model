from __future__ import annotations

import argparse

from iseeyou.config import load_config
from iseeyou.data.preprocess import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare face-frame manifests from raw datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_preprocessing(config)


if __name__ == "__main__":
    main()
