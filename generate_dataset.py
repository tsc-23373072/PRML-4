#!/usr/bin/env python
"""Generate a synthetic key-value retrieval sequence-to-sequence dataset.

The dataset is intentionally designed to probe the distinction between K and V:
- keys are used for matching queries
- values are the content to be returned
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


def make_one_example(
    rng: random.Random,
    num_keys: int,
    num_values: int,
    min_pairs: int,
    max_pairs: int,
    min_queries: int,
    max_queries: int,
) -> Dict[str, object]:
    num_pairs = rng.randint(min_pairs, max_pairs)
    num_queries = rng.randint(min_queries, min(max_queries, num_pairs))

    selected_keys = rng.sample(range(num_keys), k=num_pairs)
    selected_values = [rng.randrange(num_values) for _ in range(num_pairs)]
    key_to_value = {key: val for key, val in zip(selected_keys, selected_values)}

    pair_tokens: List[str] = []
    for key, val in zip(selected_keys, selected_values):
        pair_tokens.extend([f"K{key}", f"V{val}"])

    query_keys = rng.sample(selected_keys, k=num_queries)
    query_tokens = [f"Q{key}" for key in query_keys]
    target_tokens = [f"V{key_to_value[key]}" for key in query_keys]

    src = " ".join(pair_tokens + ["<sep>"] + query_tokens)
    tgt = " ".join(target_tokens)

    return {
        "src": src,
        "tgt": tgt,
        "num_pairs": num_pairs,
        "num_queries": num_queries,
    }


def generate_split(
    split: str,
    size: int,
    rng: random.Random,
    **kwargs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _ in range(size):
        row = make_one_example(rng=rng, **kwargs)
        row["split"] = split
        rows.append(row)

    # Keep columns in a friendly order.
    return [
        {
            "split": row["split"],
            "src": row["src"],
            "tgt": row["tgt"],
            "num_pairs": row["num_pairs"],
            "num_queries": row["num_queries"],
        }
        for row in rows
    ]


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    rng = random.Random(args.seed)
    common = {
        "num_keys": args.num_keys,
        "num_values": args.num_values,
        "min_pairs": args.min_pairs,
        "max_pairs": args.max_pairs,
        "min_queries": args.min_queries,
        "max_queries": args.max_queries,
    }

    rows: List[Dict[str, object]] = []
    rows.extend(generate_split("train", args.train_size, rng, **common))
    rows.extend(generate_split("val", args.val_size, rng, **common))
    rows.extend(generate_split("test", args.test_size, rng, **common))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic KV retrieval dataset.")
    parser.add_argument("--output_xlsx", type=Path, default=Path("data/synthetic_kv_retrieval.xlsx"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/synthetic_kv_retrieval.csv"))
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=800)
    parser.add_argument("--test_size", type=int, default=800)
    parser.add_argument("--num_keys", type=int, default=20)
    parser.add_argument("--num_values", type=int, default=20)
    parser.add_argument("--min_pairs", type=int, default=3)
    parser.add_argument("--max_pairs", type=int, default=6)
    parser.add_argument("--min_queries", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_pairs > args.max_pairs:
        raise ValueError("min_pairs must be <= max_pairs")
    if args.min_queries > args.max_queries:
        raise ValueError("min_queries must be <= max_queries")
    if args.max_queries > args.max_pairs:
        raise ValueError("max_queries must be <= max_pairs")
    if args.num_keys < args.max_pairs:
        raise ValueError("num_keys must be at least max_pairs")

    df = build_dataset(args)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(args.output_xlsx, index=False)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"Saved XLSX dataset: {args.output_xlsx}")
    print(f"Saved CSV dataset:  {args.output_csv}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
