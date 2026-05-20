from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Vocabulary, load_splits, make_collate_fn
from .losses import LabelSmoothingCrossEntropy
from .metrics import exact_match_count, token_accuracy
from .model import TransformerSeq2Seq
from .scheduler import NoamScheduler


@dataclass
class ExperimentConfig:
    data_path: str
    attention_mode: str = "standard"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    epochs: int = 20
    batch_size: int = 64
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 256
    warmup_steps: int = 400
    lr_scale: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 0
    max_decode_len: int = 6
    greedy_eval_every: int = 1

    def validate(self) -> None:
        if self.attention_mode not in {"standard", "shared_kv"}:
            raise ValueError("attention_mode must be 'standard' or 'shared_kv'.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.greedy_eval_every <= 0:
            raise ValueError("greedy_eval_every must be positive.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: ExperimentConfig, vocab: Vocabulary) -> TransformerSeq2Seq:
    return TransformerSeq2Seq(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
        attention_mode=config.attention_mode,
        share_embeddings=True,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _save_metrics_csv(metrics: List[Dict[str, float]], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def _save_checkpoint(
    path: Path,
    model: TransformerSeq2Seq,
    vocab: Vocabulary,
    config: ExperimentConfig,
    epoch: int,
    best_val_exact_match: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab_itos": vocab.itos,
        "vocab_stoi": vocab.stoi,
        "config": asdict(config),
        "epoch": epoch,
        "best_val_exact_match": best_val_exact_match,
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_teacher_forced(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    criterion: LabelSmoothingCrossEntropy,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    correct_tokens = 0
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            src_ids=batch["src_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            src_padding_mask=batch["src_padding_mask"],
            tgt_padding_mask=batch["tgt_padding_mask"],
        )
        loss = criterion(logits, batch["decoder_target_ids"])
        correct, total = token_accuracy(logits, batch["decoder_target_ids"], pad_id=pad_id)

        loss_sum += float(loss.item())
        batch_count += 1
        correct_tokens += correct
        total_tokens += total

    return {
        "loss": loss_sum / max(batch_count, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate_greedy_exact_match(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    eos_id: int,
    pad_id: int,
    max_decode_len: int,
) -> float:
    model.eval()
    matches = 0
    total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        generated = model.greedy_decode(
            src_ids=batch["src_ids"],
            src_padding_mask=batch["src_padding_mask"],
            max_new_tokens=max_decode_len,
        )
        batch_matches, batch_total = exact_match_count(
            generated=generated,
            targets=batch["decoder_target_ids"],
            eos_id=eos_id,
            pad_id=pad_id,
        )
        matches += batch_matches
        total += batch_total

    return matches / max(total, 1)


def train_one_experiment(config: ExperimentConfig) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)

    train_ds, val_ds, test_ds, vocab = load_splits(Path(config.data_path))
    collate_fn = make_collate_fn(vocab.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    model = make_model(config, vocab).to(device)
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.label_smoothing,
        ignore_index=vocab.pad_id,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps,
        scale=config.lr_scale,
    )

    metrics: List[Dict[str, float]] = []
    best_val_exact_match = -1.0
    best_checkpoint_path = Path(config.checkpoint_dir) / f"{config.attention_mode}_best.pt"
    metrics_csv_path = Path(config.output_dir) / f"{config.attention_mode}_metrics.csv"

    print(f"[Info] Device: {device}")
    print(f"[Info] Mode: {config.attention_mode}")
    print(f"[Info] Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"[Info] Vocabulary size: {len(vocab)}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        start_time = time.perf_counter()
        train_loss_sum = 0.0
        train_batches = 0
        train_correct_tokens = 0
        train_total_tokens = 0
        last_lr = 0.0

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(
                src_ids=batch["src_ids"],
                decoder_input_ids=batch["decoder_input_ids"],
                src_padding_mask=batch["src_padding_mask"],
                tgt_padding_mask=batch["tgt_padding_mask"],
            )
            loss = criterion(logits, batch["decoder_target_ids"])
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

            last_lr = scheduler.step()
            optimizer.step()

            correct, total = token_accuracy(logits, batch["decoder_target_ids"], pad_id=vocab.pad_id)
            train_loss_sum += float(loss.item())
            train_batches += 1
            train_correct_tokens += correct
            train_total_tokens += total

        train_loss = train_loss_sum / max(train_batches, 1)
        train_token_acc = train_correct_tokens / max(train_total_tokens, 1)

        val_metrics = evaluate_teacher_forced(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            pad_id=vocab.pad_id,
        )

        if epoch % config.greedy_eval_every == 0 or epoch == config.epochs:
            val_exact_match = evaluate_greedy_exact_match(
                model=model,
                loader=val_loader,
                device=device,
                eos_id=vocab.eos_id,
                pad_id=vocab.pad_id,
                max_decode_len=config.max_decode_len,
            )
        else:
            val_exact_match = float("nan")

        if not np.isnan(val_exact_match) and val_exact_match > best_val_exact_match:
            best_val_exact_match = val_exact_match
            _save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                vocab=vocab,
                config=config,
                epoch=epoch,
                best_val_exact_match=best_val_exact_match,
            )

        elapsed = time.perf_counter() - start_time
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_token_accuracy": train_token_acc,
            "val_loss": val_metrics["loss"],
            "val_token_accuracy": val_metrics["token_accuracy"],
            "val_exact_match": val_exact_match,
            "learning_rate": last_lr,
            "epoch_seconds": elapsed,
        }
        metrics.append(row)
        _save_metrics_csv(metrics, metrics_csv_path)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_token_acc={val_metrics['token_accuracy']:.4f} "
            f"val_exact={val_exact_match:.4f} "
            f"lr={last_lr:.6e} "
            f"time={elapsed:.1f}s"
        )

    # Load the best checkpoint for final test evaluation.
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_teacher_forced(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        pad_id=vocab.pad_id,
    )
    test_exact_match = evaluate_greedy_exact_match(
        model=model,
        loader=test_loader,
        device=device,
        eos_id=vocab.eos_id,
        pad_id=vocab.pad_id,
        max_decode_len=config.max_decode_len,
    )

    summary = {
        "attention_mode": config.attention_mode,
        "best_val_exact_match": float(best_val_exact_match),
        "test_loss": float(test_metrics["loss"]),
        "test_token_accuracy": float(test_metrics["token_accuracy"]),
        "test_exact_match": float(test_exact_match),
        "best_checkpoint": str(best_checkpoint_path),
        "metrics_csv": str(metrics_csv_path),
    }

    summary_path = Path(config.output_dir) / f"{config.attention_mode}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, metrics
