from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]


def _token_sort_key(token: str) -> Tuple[str, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", token)
    if match:
        return match.group(1), int(match.group(2)), token
    return token, -1, token


@dataclass(frozen=True)
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "Vocabulary":
        tokens = set()
        for text in texts:
            tokens.update(str(text).split())

        regular_tokens = sorted(
            [tok for tok in tokens if tok not in SPECIAL_TOKENS],
            key=_token_sort_key,
        )
        itos = SPECIAL_TOKENS + regular_tokens
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.stoi["<sep>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = False) -> List[str]:
        tokens: List[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if stop_at_eos and token == "<eos>":
                break
            if token != "<pad>":
                tokens.append(token)
        return tokens


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Dataset path must end with .xlsx or .csv.")

    required = {"split", "src", "tgt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


class KVRetrievalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        src_tokens = str(row["src"]).split()
        tgt_tokens = str(row["tgt"]).split()

        src_ids = self.vocab.encode(src_tokens + ["<eos>"])
        decoder_input_ids = [self.vocab.bos_id] + self.vocab.encode(tgt_tokens)
        decoder_target_ids = self.vocab.encode(tgt_tokens + ["<eos>"])

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_target_ids, dtype=torch.long),
        }


def pad_1d(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.numel() for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, : seq.numel()] = seq
    return batch


def make_collate_fn(pad_id: int):
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src = pad_1d([item["src_ids"] for item in batch], pad_id)
        dec_in = pad_1d([item["decoder_input_ids"] for item in batch], pad_id)
        dec_tgt = pad_1d([item["decoder_target_ids"] for item in batch], pad_id)

        return {
            "src_ids": src,
            "decoder_input_ids": dec_in,
            "decoder_target_ids": dec_tgt,
            "src_padding_mask": src.eq(pad_id),
            "tgt_padding_mask": dec_in.eq(pad_id),
        }

    return collate


def load_splits(path: Path) -> Tuple[KVRetrievalDataset, KVRetrievalDataset, KVRetrievalDataset, Vocabulary]:
    df = read_table(path)
    vocab = Vocabulary.build(list(df["src"].astype(str)) + list(df["tgt"].astype(str)))

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"].astype(str).str.lower() == split_name]
        if split_df.empty:
            raise ValueError(f"No rows found for split='{split_name}'.")
        splits[split_name] = KVRetrievalDataset(split_df, vocab)

    return splits["train"], splits["val"], splits["test"], vocab
