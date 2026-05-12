"""Typed config loaders for the modern pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import load_yaml, resolve_repo_path


@dataclass(slots=True)
class DatasetConfig:
    name: str
    source_root: Path
    normal_dir: str
    abnormal_dir: str
    image_extensions: tuple[str, ...]
    seeds: tuple[int, ...]
    train_count: int
    val_good_count: int
    test_good_count: int
    val_bad_count: int
    test_bad_count: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    split_artifact_root: Path
    inference_root: Path
    image_size: int
    augmentations: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelConfig:
    name: str
    model_key: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    datamodule: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_config: Path
    model_configs: tuple[Path, ...]
    output_root: Path
    reports_root: Path
    runner: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinalTrainConfig:
    benchmark_config: Path
    seed: int
    output_root: Path
    predictions_root: Path
    reports_root: Path
    trainer: dict[str, Any] = field(default_factory=dict)


def load_dataset_config(path_like: str | Path) -> DatasetConfig:
    """Load dataset config."""

    data = load_yaml(path_like)
    return DatasetConfig(
        name=data["name"],
        source_root=resolve_repo_path(data["source_root"]),
        normal_dir=data["normal_dir"],
        abnormal_dir=data["abnormal_dir"],
        image_extensions=tuple(data.get("image_extensions", [".jpg", ".jpeg", ".png"])),
        seeds=tuple(data["seeds"]),
        train_count=int(data["train_count"]),
        val_good_count=int(data["val_good_count"]),
        test_good_count=int(data["test_good_count"]),
        val_bad_count=int(data["val_bad_count"]),
        test_bad_count=int(data["test_bad_count"]),
        train_batch_size=int(data.get("train_batch_size", 8)),
        eval_batch_size=int(data.get("eval_batch_size", 8)),
        num_workers=int(data.get("num_workers", 0)),
        split_artifact_root=resolve_repo_path(data["split_artifact_root"]),
        inference_root=resolve_repo_path(data["inference_root"]),
        image_size=int(data.get("image_size", 256)),
        augmentations=data.get("augmentations", {}),
    )


def load_model_config(path_like: str | Path) -> ModelConfig:
    """Load model config."""

    data = load_yaml(path_like)
    return ModelConfig(
        name=data["name"],
        model_key=data["model_key"],
        description=data.get("description", ""),
        params=data.get("params", {}),
        datamodule=data.get("datamodule", {}),
        trainer=data.get("trainer", {}),
    )


def load_benchmark_config(path_like: str | Path) -> BenchmarkConfig:
    """Load benchmark config."""

    data = load_yaml(path_like)
    return BenchmarkConfig(
        dataset_config=resolve_repo_path(data["dataset_config"]),
        model_configs=tuple(resolve_repo_path(item) for item in data["model_configs"]),
        output_root=resolve_repo_path(data["output_root"]),
        reports_root=resolve_repo_path(data["reports_root"]),
        runner=data.get("runner", {}),
        trainer=data.get("trainer", {}),
    )


def load_final_train_config(path_like: str | Path) -> FinalTrainConfig:
    """Load final-train config."""

    data = load_yaml(path_like)
    return FinalTrainConfig(
        benchmark_config=resolve_repo_path(data["benchmark_config"]),
        seed=int(data["seed"]),
        output_root=resolve_repo_path(data["output_root"]),
        predictions_root=resolve_repo_path(data["predictions_root"]),
        reports_root=resolve_repo_path(data["reports_root"]),
        trainer=data.get("trainer", {}),
    )
