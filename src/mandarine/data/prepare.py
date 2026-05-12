"""Prepare deterministic split directories for the modern benchmark."""

from __future__ import annotations

import argparse

from mandarine.common import ensure_dir
from mandarine.config import load_dataset_config
from mandarine.data.splits import create_split_directories, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/modern/dataset/mandarins_cropped.yaml",
        help="Dataset config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.config)
    all_rows = []
    ensure_dir(dataset_config.split_artifact_root)
    for seed in dataset_config.seeds:
        _, rows = create_split_directories(dataset_config, seed)
        all_rows.extend(rows)
    manifest_path = write_manifest(dataset_config.split_artifact_root.parent / "split_manifest.csv", all_rows)
    print(f"Wrote split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
