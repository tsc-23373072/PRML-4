#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import load_splits
from src.training import ExperimentConfig, make_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize decoder cross-attention for one sample.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/synthetic_kv_retrieval.xlsx")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="outputs/attention_heatmap.png")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = ExperimentConfig(**checkpoint["config"])
    config.device = args.device

    train_ds, val_ds, test_ds, vocab = load_splits(Path(args.data_path))
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index must be between 0 and {len(dataset)-1}")

    model = make_model(config, vocab).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    item = dataset[args.sample_index]
    src_ids = item["src_ids"].unsqueeze(0).to(device)
    dec_in_ids = item["decoder_input_ids"].unsqueeze(0).to(device)

    with torch.no_grad():
        _, weights = model(
            src_ids=src_ids,
            decoder_input_ids=dec_in_ids,
            src_padding_mask=src_ids.eq(vocab.pad_id),
            tgt_padding_mask=dec_in_ids.eq(vocab.pad_id),
            need_weights=True,
        )

    if not weights["decoder_cross"]:
        raise RuntimeError("No decoder cross-attention weights returned.")

    # Last decoder layer, first sample: [target_len, src_len]
    attn = weights["decoder_cross"][-1][0].detach().cpu().numpy()

    src_labels = vocab.decode(item["src_ids"].tolist(), stop_at_eos=False)
    tgt_labels = vocab.decode(item["decoder_input_ids"].tolist(), stop_at_eos=False)

    plt.figure(figsize=(max(8, 0.55 * len(src_labels)), max(4, 0.55 * len(tgt_labels))))
    plt.imshow(attn, aspect="auto")
    plt.xticks(range(len(src_labels)), src_labels, rotation=45, ha="right")
    plt.yticks(range(len(tgt_labels)), tgt_labels)
    plt.xlabel("Source tokens")
    plt.ylabel("Decoder input tokens")
    plt.title("Decoder Cross-Attention Heatmap")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved attention heatmap: {output_path}")


if __name__ == "__main__":
    main()
