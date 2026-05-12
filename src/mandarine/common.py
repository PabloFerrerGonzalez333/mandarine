"""Shared helpers for the modern mandarine pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a repo-relative path into an absolute path."""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory when needed and return it."""

    path = resolve_repo_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def dump_json(path_like: str | Path, payload: Any) -> Path:
    """Persist JSON with stable formatting."""

    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
