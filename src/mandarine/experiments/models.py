"""Model factory and experiment helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO, EfficientAd, Patchcore
from sklearn.metrics import average_precision_score

from mandarine.common import ensure_dir
from mandarine.config import ModelConfig

MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "efficientad": EfficientAd,
    "anomalydino": AnomalyDINO,
}


def create_model(model_config: ModelConfig, cache_root: Path):
    """Instantiate one supported anomalib model."""

    params = dict(model_config.params)
    if model_config.model_key == "efficientad":
        params.setdefault("imagenet_dir", str(cache_root / "imagenette"))
    model_cls = MODEL_REGISTRY[model_config.model_key]
    return model_cls(**params)


def create_engine(default_root_dir: Path, trainer_kwargs: dict[str, Any]) -> Engine:
    """Instantiate a Lightning engine with a stable root dir."""

    ensure_dir(default_root_dir)
    return Engine(default_root_dir=default_root_dir, logger=False, **trainer_kwargs)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def normalise_metric_dict(metrics: dict[str, Any]) -> dict[str, float]:
    """Convert metric tensors into JSON-friendly floats."""

    return {key: _as_float(value) for key, value in metrics.items()}


def flatten_prediction_batches(predictions: list[Any] | None) -> list[Any]:
    """Normalise predict outputs into a flat list of samples/batches."""

    if not predictions:
        return []
    flattened: list[Any] = []
    for item in predictions:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def collect_image_scores(predictions: list[Any] | None) -> tuple[list[float], list[int]]:
    """Extract image-level anomaly scores and labels from predict outputs."""

    scores: list[float] = []
    labels: list[int] = []
    for batch in flatten_prediction_batches(predictions):
        batch_scores = None
        batch_labels = None
        if isinstance(batch, dict):
            batch_scores = batch.get("pred_score")
            if batch_scores is None:
                batch_scores = batch.get("anomaly_score")
            batch_labels = batch.get("gt_label")
            if batch_labels is None:
                batch_labels = batch.get("label")
        else:
            batch_scores = getattr(batch, "pred_score", None)
            if batch_scores is None:
                batch_scores = getattr(batch, "anomaly_score", None)
            batch_labels = getattr(batch, "gt_label", None)
            if batch_labels is None:
                batch_labels = getattr(batch, "label", None)

        if batch_scores is None or batch_labels is None:
            continue

        if hasattr(batch_scores, "detach"):
            batch_scores = batch_scores.detach().cpu().numpy()
        if hasattr(batch_labels, "detach"):
            batch_labels = batch_labels.detach().cpu().numpy()

        scores.extend(np.asarray(batch_scores, dtype=float).reshape(-1).tolist())
        labels.extend(np.asarray(batch_labels, dtype=int).reshape(-1).tolist())

    return scores, labels


def compute_image_aupr(predictions: list[Any] | None) -> float:
    """Compute image-level AUPR from predict outputs."""

    scores, labels = collect_image_scores(predictions)
    if not scores or len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def measure_prediction_latency(engine: Engine, model: Any, *, datamodule: Any) -> tuple[float, list[Any] | None]:
    """Measure average prediction latency per image on the test split."""

    sample_count = len(datamodule.test_data)
    start = time.perf_counter()
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader(), return_predictions=True)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / max(sample_count, 1)) * 1000.0
    return latency_ms, predictions
