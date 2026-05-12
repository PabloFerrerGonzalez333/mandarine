"""Dataset splitting helpers."""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from mandarine.common import ensure_dir
from mandarine.config import DatasetConfig


@dataclass(slots=True)
class SplitPaths:
    seed: int
    split_root: Path
    train_good: Path
    val_good: Path
    val_bad: Path
    test_good: Path
    test_bad: Path


def _sorted_images(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    suffixes = {item.lower() for item in extensions}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _copy_many(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)


def create_split_directories(dataset_config: DatasetConfig, seed: int) -> tuple[SplitPaths, list[dict[str, str | int]]]:
    """Create deterministic train/val/test directories for one seed."""

    randomizer = random.Random(seed)
    normal_root = dataset_config.source_root / dataset_config.normal_dir
    abnormal_root = dataset_config.source_root / dataset_config.abnormal_dir

    normal_files = _sorted_images(normal_root, dataset_config.image_extensions)
    abnormal_files = _sorted_images(abnormal_root, dataset_config.image_extensions)

    expected_normals = dataset_config.train_count + dataset_config.val_good_count + dataset_config.test_good_count
    expected_abnormals = dataset_config.val_bad_count + dataset_config.test_bad_count

    if len(normal_files) != expected_normals:
        raise ValueError(f"Expected {expected_normals} normal images, found {len(normal_files)}")
    if len(abnormal_files) != expected_abnormals:
        raise ValueError(f"Expected {expected_abnormals} abnormal images, found {len(abnormal_files)}")

    normal_files = normal_files[:]
    abnormal_files = abnormal_files[:]
    randomizer.shuffle(normal_files)
    randomizer.shuffle(abnormal_files)

    train_good = normal_files[: dataset_config.train_count]
    val_good = normal_files[
        dataset_config.train_count : dataset_config.train_count + dataset_config.val_good_count
    ]
    test_good = normal_files[-dataset_config.test_good_count :]

    val_bad = abnormal_files[: dataset_config.val_bad_count]
    test_bad = abnormal_files[-dataset_config.test_bad_count :]

    split_root = ensure_dir(dataset_config.split_artifact_root / f"seed_{seed}")
    for child in split_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    paths = SplitPaths(
        seed=seed,
        split_root=split_root,
        train_good=split_root / "train" / "good",
        val_good=split_root / "val" / "good",
        val_bad=split_root / "val" / "bad",
        test_good=split_root / "test" / "good",
        test_bad=split_root / "test" / "bad",
    )

    _copy_many(train_good, paths.train_good)
    _copy_many(val_good, paths.val_good)
    _copy_many(val_bad, paths.val_bad)
    _copy_many(test_good, paths.test_good)
    _copy_many(test_bad, paths.test_bad)

    manifest_rows: list[dict[str, str | int]] = []
    split_specs = [
        ("train", "good", train_good, paths.train_good),
        ("val", "good", val_good, paths.val_good),
        ("val", "bad", val_bad, paths.val_bad),
        ("test", "good", test_good, paths.test_good),
        ("test", "bad", test_bad, paths.test_bad),
    ]
    for split_name, label_name, files, target_dir in split_specs:
        for source_path in files:
            manifest_rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "label": label_name,
                    "source_path": str(source_path),
                    "target_path": str(target_dir / source_path.name),
                }
            )

    return paths, manifest_rows


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> Path:
    """Write the split manifest to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "split", "label", "source_path", "target_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
