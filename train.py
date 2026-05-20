#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.training import ExperimentConfig, train_one_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Transformer experiment.")
    parser.add_argument("--data_path", type=str, default="data/synthetic_kv_retrieval.xlsx")
    parser.add_argument("--attention_mode", type=str, choices=["standard", "shared_kv"], default="standard")
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


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(**vars(args))
    train_one_experiment(config)


if __name__ == "__main__":
    main()
