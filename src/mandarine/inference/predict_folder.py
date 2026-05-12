"""Run folder-level inference with the trained modern model."""

from __future__ import annotations

import argparse

import pandas as pd

from mandarine.common import dump_json, ensure_dir, load_yaml, resolve_repo_path
from mandarine.config import load_benchmark_config, load_final_train_config, load_model_config
from mandarine.experiments.models import create_engine, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modern/final_train.yaml", help="Final-training config YAML.")
    parser.add_argument("--input-dir", default="data/webcam_inference_images", help="Folder with inference images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_config = load_final_train_config(args.config)
    benchmark_config = load_benchmark_config(final_config.benchmark_config)
    final_summary = load_yaml(final_config.output_root / "final_model_summary.json")

    winner_model = final_summary["winner_model"]
    model_config_path = next(path for path in benchmark_config.model_configs if path.stem == winner_model)
    model_config = load_model_config(model_config_path)

    model = create_model(model_config, cache_root=resolve_repo_path("artifacts/modern/cache"))
    engine = create_engine(final_config.output_root, {})
    input_dir = resolve_repo_path(args.input_dir)

    rows = []
    for image_path in sorted(input_dir.glob("*")):
        if not image_path.is_file():
            continue
        outputs = engine.predict(
            model=model,
            ckpt_path=final_summary.get("checkpoint_path"),
            data_path=image_path,
            return_predictions=True,
        )
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

        latest_image = (
            final_config.output_root
            / "Patchcore"
            / "latest"
            / "images"
            / input_dir.name
            / image_path.name
        )
        rows.append(
            {
                "image_path": str(image_path),
                "prediction_type": str(type(outputs).__name__),
                "pred_score": pred_score,
                "pred_label": pred_label,
                "visualization_path": str(latest_image) if latest_image.exists() else None,
            }
        )

    predictions_root = ensure_dir(final_config.predictions_root)
    pd.DataFrame(rows).to_csv(predictions_root / "folder_predictions.csv", index=False)
    dump_json(predictions_root / "folder_predictions.json", {"rows": rows})
    print(f"Wrote predictions for {len(rows)} images to {predictions_root}")


if __name__ == "__main__":
    main()
