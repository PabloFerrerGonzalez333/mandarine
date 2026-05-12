"""Create plots and execute the modern notebooks into HTML."""

from __future__ import annotations

import argparse
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from mandarine.common import ensure_dir, load_yaml, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="artifacts/modern/benchmark", help="Benchmark artifact root.")
    parser.add_argument("--final-root", default="artifacts/modern/final_model", help="Final-model artifact root.")
    return parser.parse_args()


def _render_plot(benchmark_root, reports_root) -> None:
    leaderboard = pd.read_csv(benchmark_root / "leaderboard.csv")
    figure_root = ensure_dir(reports_root / "figures")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(leaderboard["model"], leaderboard["mean_image_AUROC"], color="#d97706")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean image AUROC")
    ax.set_title("Mandarine benchmark leaderboard")
    fig.tight_layout()
    fig.savefig(figure_root / "leaderboard_auroc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(leaderboard["model"], leaderboard["mean_latency_ms"], color="#2563eb")
    ax.set_ylabel("Latency (ms/image)")
    ax.set_title("Inference latency on CPU")
    fig.tight_layout()
    fig.savefig(figure_root / "leaderboard_latency.png", dpi=160)
    plt.close(fig)


def _execute_notebook(notebook_path, output_path) -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        "--ExecutePreprocessor.kernel_name=mandarine-modern",
        "--output-dir",
        str(output_path.parent),
        "--output",
        output_path.stem,
        str(notebook_path),
    ]
    subprocess.run(command, check=True, cwd=str(resolve_repo_path(".")))


def main() -> None:
    args = parse_args()
    benchmark_root = resolve_repo_path(args.benchmark_root)
    final_root = resolve_repo_path(args.final_root)
    reports_root = ensure_dir("docs/html/modern")

    _render_plot(benchmark_root, benchmark_root)
    final_summary = load_yaml(final_root / "final_model_summary.json")

    notebook_paths = [
        resolve_repo_path("notebooks/modern_benchmark_report.ipynb"),
        resolve_repo_path("notebooks/modern_inference_demo.ipynb"),
    ]
    for notebook_path in notebook_paths:
        output_path = reports_root / notebook_path.with_suffix(".html").name
        _execute_notebook(notebook_path, output_path)

    print(f"Reports exported to {reports_root}")
    print(f"Final winner: {final_summary['winner_model']}")


if __name__ == "__main__":
    main()
