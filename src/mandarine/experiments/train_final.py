"""Train the winning model on the final split and keep test locked."""

from __future__ import annotations

import argparse

import pandas as pd

from mandarine.common import dump_json, ensure_dir, load_yaml
from mandarine.config import load_benchmark_config, load_dataset_config, load_final_train_config, load_model_config
from mandarine.data.datamodule import MandarineDataModule
from mandarine.data.splits import create_split_directories
from mandarine.experiments.models import (
    compute_image_aupr,
    create_engine,
    create_model,
    measure_prediction_latency,
    normalise_metric_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modern/final_train.yaml", help="Final-training config YAML.")
    return parser.parse_args()


def _select_winner_model_name(benchmark_root):
    summary = load_yaml(benchmark_root / "metrics_summary.json")
    winner = summary.get("winner") or {}
    model_name = winner.get("model")
    if not model_name:
        raise ValueError("Benchmark winner could not be determined. Run the benchmark first.")
    return str(model_name)


def _find_latest_checkpoint(run_root):
    checkpoints = sorted(run_root.rglob("*.ckpt"))
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    args = parse_args()
    final_config = load_final_train_config(args.config)
    benchmark_config = load_benchmark_config(final_config.benchmark_config)
    dataset_config = load_dataset_config(benchmark_config.dataset_config)

    output_root = ensure_dir(final_config.output_root)
    cache_root = ensure_dir("artifacts/modern/cache")
    split_paths, _ = create_split_directories(dataset_config, final_config.seed)
    winner_name = _select_winner_model_name(benchmark_config.output_root)
    model_config_path = next(path for path in benchmark_config.model_configs if path.stem == winner_name)
    model_config = load_model_config(model_config_path)
    datamodule = MandarineDataModule(
        dataset_config,
        split_paths,
        train_batch_size=model_config.datamodule.get("train_batch_size"),
        eval_batch_size=model_config.datamodule.get("eval_batch_size"),
    )

    model = create_model(model_config, cache_root=cache_root)
    trainer_kwargs = dict(benchmark_config.trainer)
    trainer_kwargs.update(model_config.trainer)
    trainer_kwargs.update(final_config.trainer)
    engine = create_engine(output_root, trainer_kwargs)

    engine.fit(model=model, datamodule=datamodule)
    test_metrics = normalise_metric_dict(engine.test(model=model, datamodule=datamodule)[0])
    latency_ms, predictions = measure_prediction_latency(engine, model, datamodule=datamodule)
    image_aupr = compute_image_aupr(predictions)

    predictions_root = ensure_dir(final_config.predictions_root)
    prediction_runs = []
    for image_path in sorted(dataset_config.inference_root.glob("*")):
        if image_path.suffix.lower() not in dataset_config.image_extensions:
            continue
        outputs = engine.predict(model=model, data_path=image_path, return_predictions=True)
        batch = outputs[0] if outputs else None
        pred_score = None
        pred_label = None
        if batch is not None:
            pred_score = getattr(batch, "pred_score", None)
            pred_label = getattr(batch, "pred_label", None)
            if hasattr(pred_score, "detach"):
                pred_score = float(pred_score.detach().cpu().reshape(-1)[0])
            if hasattr(pred_label, "detach"):
                pred_label = bool(pred_label.detach().cpu().reshape(-1)[0])

        prediction_runs.append(
            {
                "image_path": str(image_path),
                "prediction_type": str(type(outputs).__name__),
                "pred_score": pred_score,
                "pred_label": pred_label,
            }
        )

    summary = {
        "winner_model": winner_name,
        "seed": final_config.seed,
        "metrics": {**test_metrics, "image_AUPR": image_aupr, "latency_ms": latency_ms},
        "checkpoint_path": _find_latest_checkpoint(output_root),
        "prediction_samples": prediction_runs,
    }
    dump_json(output_root / "final_model_summary.json", summary)

    pd.DataFrame(prediction_runs).to_csv(predictions_root / "webcam_predictions.csv", index=False)
    print(f"Final model summary written to {output_root / 'final_model_summary.json'}")


if __name__ == "__main__":
    main()
