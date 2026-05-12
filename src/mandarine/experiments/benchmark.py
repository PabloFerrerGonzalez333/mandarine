"""Run the modern benchmark across deterministic seeds and models."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mandarine.common import dump_json, ensure_dir
from mandarine.config import load_benchmark_config, load_dataset_config, load_model_config
from mandarine.data.datamodule import MandarineDataModule
from mandarine.data.splits import create_split_directories, write_manifest
from mandarine.experiments.models import (
    compute_image_aupr,
    create_engine,
    create_model,
    measure_prediction_latency,
    normalise_metric_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modern/benchmark_cpu.yaml", help="Benchmark config YAML.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to run, for example: --models patchcore anomalydino",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of seeds to run, for example: --seeds 13 42",
    )
    return parser.parse_args()


def _merge_trainer_kwargs(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        merged.update(group)
    return merged


def _nan_to_none(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def _select_winner(leaderboard: pd.DataFrame) -> dict[str, Any]:
    ranking = leaderboard.sort_values(
        by=["mean_image_AUROC", "mean_image_AUPR", "mean_latency_ms"],
        ascending=[False, False, True],
    )
    winner = ranking.iloc[0].to_dict()
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in winner.items()}


def main() -> None:
    args = parse_args()
    benchmark_config = load_benchmark_config(args.config)
    dataset_config = load_dataset_config(benchmark_config.dataset_config)
    model_configs = [load_model_config(path) for path in benchmark_config.model_configs]
    if args.models:
        wanted = set(args.models)
        model_configs = [config for config in model_configs if config.name in wanted]
    if not model_configs:
        raise ValueError("No model configurations selected for this benchmark run.")
    seeds = tuple(args.seeds) if args.seeds else dataset_config.seeds

    output_root = ensure_dir(benchmark_config.output_root)
    cache_root = ensure_dir("artifacts/modern/cache")
    ensure_dir(benchmark_config.reports_root)

    split_rows: list[dict[str, str | int]] = []
    run_rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []

    for seed in seeds:
        split_paths, rows = create_split_directories(dataset_config, seed)
        split_rows.extend(rows)

        for model_config in model_configs:
            datamodule = MandarineDataModule(
                dataset_config,
                split_paths,
                train_batch_size=model_config.datamodule.get("train_batch_size"),
                eval_batch_size=model_config.datamodule.get("eval_batch_size"),
            )
            model_output_dir = output_root / f"{model_config.name}_seed_{seed}"
            trainer_kwargs = _merge_trainer_kwargs(benchmark_config.trainer, model_config.trainer)
            model = create_model(model_config, cache_root=cache_root)
            engine = create_engine(model_output_dir, trainer_kwargs)

            status = "ok"
            error_message = None
            metrics: dict[str, Any] = {}
            aupr = float("nan")
            latency_ms = float("nan")

            try:
                engine.fit(model=model, datamodule=datamodule)
                test_metrics = engine.test(model=model, datamodule=datamodule)[0]
                metrics = normalise_metric_dict(test_metrics)
                latency_ms, predictions = measure_prediction_latency(engine, model, datamodule=datamodule)
                aupr = compute_image_aupr(predictions)
            except Exception as exc:  # pragma: no cover - exercised in real runs
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"

            row = {
                "seed": seed,
                "model": model_config.name,
                "status": status,
                "error_message": error_message,
                "image_AUROC": metrics.get("image_AUROC"),
                "image_F1Score": metrics.get("image_F1Score"),
                "image_AUPR": _nan_to_none(aupr),
                "latency_ms": _nan_to_none(latency_ms),
            }
            run_rows.append(row)
            raw_runs.append(
                {
                    "seed": seed,
                    "model": model_config.name,
                    "status": status,
                    "error_message": error_message,
                    "metrics": metrics,
                    "latency_ms": _nan_to_none(latency_ms),
                    "image_aupr": _nan_to_none(aupr),
                    "output_dir": str(model_output_dir),
                }
            )

    split_manifest_path = write_manifest(output_root / "split_manifest.csv", split_rows)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(output_root / "benchmark_runs.csv", index=False)

    success_df = runs_df[runs_df["status"] == "ok"].copy()
    grouped = defaultdict(dict)
    for model_name, model_group in success_df.groupby("model"):
        grouped[model_name]["mean_image_AUROC"] = model_group["image_AUROC"].mean()
        grouped[model_name]["std_image_AUROC"] = model_group["image_AUROC"].std(ddof=0)
        grouped[model_name]["mean_image_AUPR"] = model_group["image_AUPR"].mean()
        grouped[model_name]["std_image_AUPR"] = model_group["image_AUPR"].std(ddof=0)
        grouped[model_name]["mean_image_F1"] = model_group["image_F1Score"].mean()
        grouped[model_name]["std_image_F1"] = model_group["image_F1Score"].std(ddof=0)
        grouped[model_name]["mean_latency_ms"] = model_group["latency_ms"].mean()
        grouped[model_name]["std_latency_ms"] = model_group["latency_ms"].std(ddof=0)
        grouped[model_name]["completed_runs"] = int(model_group.shape[0])

    leaderboard = pd.DataFrame([{"model": model_name, **metrics} for model_name, metrics in grouped.items()])
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(by="mean_image_AUROC", ascending=False)
    leaderboard.to_csv(output_root / "leaderboard.csv", index=False)

    winner = _select_winner(leaderboard) if not leaderboard.empty else {}
    summary = {
        "dataset_config": str(benchmark_config.dataset_config),
        "split_manifest": str(split_manifest_path),
        "winner": winner,
        "runs": raw_runs,
    }
    dump_json(output_root / "metrics_summary.json", summary)

    print(f"Wrote benchmark results to {output_root}")
    if winner:
        print(
            "Winner:",
            winner["model"],
            f"(AUROC={winner['mean_image_AUROC']:.4f}, AUPR={winner['mean_image_AUPR']:.4f})",
        )


if __name__ == "__main__":
    main()
