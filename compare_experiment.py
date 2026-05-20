#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.training import ExperimentConfig, train_one_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standard QKV and shared-KV attention.")
    parser.add_argument("--data_path", type=str, default="data/synthetic_kv_retrieval.xlsx")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_decode_len", type=int, default=6)
    parser.add_argument("--greedy_eval_every", type=int, default=1)
    return parser.parse_args()


def make_config(args: argparse.Namespace, attention_mode: str) -> ExperimentConfig:
    payload = vars(args).copy()
    payload["attention_mode"] = attention_mode
    return ExperimentConfig(**payload)


def save_summary_csv(summaries: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attention_mode",
        "best_val_exact_match",
        "test_loss",
        "test_token_accuracy",
        "test_exact_match",
        "best_checkpoint",
        "metrics_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def plot_metric(
    standard_metrics: List[Dict[str, float]],
    shared_metrics: List[Dict[str, float]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
) -> None:
    standard_epochs = [int(row["epoch"]) for row in standard_metrics]
    shared_epochs = [int(row["epoch"]) for row in shared_metrics]
    standard_values = [float(row[metric_name]) for row in standard_metrics]
    shared_values = [float(row[metric_name]) for row in shared_metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(standard_epochs, standard_values, marker="o", label="Standard QKV")
    plt.plot(shared_epochs, shared_values, marker="o", label="Shared K/V")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: Standard QKV vs Shared K/V")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    standard_summary, standard_metrics = train_one_experiment(make_config(args, "standard"))
    shared_summary, shared_metrics = train_one_experiment(make_config(args, "shared_kv"))

    output_dir = Path(args.output_dir)
    save_summary_csv(
        summaries=[standard_summary, shared_summary],
        path=output_dir / "comparison_summary.csv",
    )

    plot_metric(
        standard_metrics=standard_metrics,
        shared_metrics=shared_metrics,
        metric_name="val_loss",
        ylabel="Validation Loss",
        output_path=output_dir / "val_loss_comparison.png",
    )
    plot_metric(
        standard_metrics=standard_metrics,
        shared_metrics=shared_metrics,
        metric_name="val_token_accuracy",
        ylabel="Validation Token Accuracy",
        output_path=output_dir / "val_token_accuracy_comparison.png",
    )
    plot_metric(
        standard_metrics=standard_metrics,
        shared_metrics=shared_metrics,
        metric_name="val_exact_match",
        ylabel="Validation Exact Match",
        output_path=output_dir / "val_exact_match_comparison.png",
    )

    print(f"Saved comparison summary: {output_dir / 'comparison_summary.csv'}")
    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
