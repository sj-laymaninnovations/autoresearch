"""
data_loader.py — Streaming, checkpoint-resumable FineWeb-Edu data loader.

Reads parquet shards from `data_dir`, tokenizes the `text` column using
tiktoken cl100k_base, and yields fixed-length (seq_len) chunks as torch
tensors suitable for language model training (input = tokens[:-1],
target = tokens[1:]).

Resume support:
    Progress is tracked in `{out_dir}/data_progress.json`.
    On resume, the loader skips already-processed shards and restores
    the intra-shard row offset.

Usage:
    from data_loader import build_loader
    loader = build_loader(data_dir="sources/fineweb-edu", seq_len=2048,
                          out_dir="checkpoints/config_a", val_fraction=0.005)
    for x, y in loader.train_iter():
        ...
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import pyarrow.parquet as pq
import tiktoken
import torch


# ---------------------------------------------------------------------------
# Tokenizer singleton
# ---------------------------------------------------------------------------

_ENC: Optional[tiktoken.Encoding] = None


def get_tokenizer() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def discover_shards(data_dir: str) -> List[Path]:
    """Return sorted list of .parquet files in data_dir."""
    p = Path(data_dir)
    shards = sorted(p.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No .parquet files found in {data_dir}. "
            "Check that fineweb-edu shards are present."
        )
    return shards


def split_shards(shards: List[Path], val_fraction: float) -> Tuple[List[Path], List[Path]]:
    """Deterministic train/val split by shard index (last N shards = val)."""
    n_val = max(1, int(len(shards) * val_fraction))
    return shards[:-n_val], shards[-n_val:]


# ---------------------------------------------------------------------------
# Progress tracking (resume support)
# ---------------------------------------------------------------------------

class DataProgress:
    """
    Tracks which shards have been consumed and the current row offset.
    Persisted to JSON so training can resume after interruption.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                self._data = {}

    def shard_done(self, shard_name: str) -> bool:
        return self._data.get(shard_name, {}).get("done", False)

    def row_offset(self, shard_name: str) -> int:
        return self._data.get(shard_name, {}).get("row_offset", 0)

    def mark_done(self, shard_name: str):
        self._data.setdefault(shard_name, {})["done"] = True
        self._save()

    def update_offset(self, shard_name: str, offset: int):
        self._data.setdefault(shard_name, {})["row_offset"] = offset
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))


# ---------------------------------------------------------------------------
# Core token stream
# ---------------------------------------------------------------------------

def _stream_tokens(
    shards: List[Path],
    progress: DataProgress,
    split_name: str,
) -> Generator[int, None, None]:
    """
    Yields individual token ids from the given shards.
    Adds an EOS token (tiktoken cl100k_base EOS = 100257 — using special
    token sentinel) between documents so the model learns document boundaries.
    """
    enc = get_tokenizer()
    # cl100k_base doesn't expose a dedicated EOS; use the endoftext token id
    EOS = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    for shard_path in shards:
        shard_name = f"{split_name}:{shard_path.name}"

        if progress.shard_done(shard_name):
            print(f"  [data_loader] skip {shard_path.name} (already done)")
            continue

        start_row = progress.row_offset(shard_name)
        print(f"  [data_loader] reading {shard_path.name} (from row {start_row})")

        pf = pq.ParquetFile(shard_path)
        row_idx = 0

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            texts = rg.column("text").to_pylist()

            for text in texts:
                if row_idx < start_row:
                    row_idx += 1
                    continue

                if not text or not isinstance(text, str):
                    row_idx += 1
                    continue

                ids = enc.encode_ordinary(text)
                for tid in ids:
                    yield tid
                yield EOS

                row_idx += 1

                # Checkpoint offset periodically
                if row_idx % 10_000 == 0:
                    progress.update_offset(shard_name, row_idx)

        progress.mark_done(shard_name)


# ---------------------------------------------------------------------------
# Chunk iterator
# ---------------------------------------------------------------------------

def _chunk_iter(
    token_stream: Generator[int, None, None],
    seq_len: int,
    device: str,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Buffers tokens and yields (x, y) pairs of shape [seq_len].
    x = tokens[:-1], y = tokens[1:]  (standard LM shift).
    """
    buf: List[int] = []
    chunk_size = seq_len + 1  # +1 so we can shift for targets

    for tok in token_stream:
        buf.append(tok)
        if len(buf) >= chunk_size:
            chunk = torch.tensor(buf[:chunk_size], dtype=torch.long, device=device)
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y
            # Slide: keep overlap for continuity
            buf = buf[seq_len:]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class FineWebLoader:
    """
    Checkpoint-resumable streaming loader for FineWeb-Edu parquets.

    Args:
        data_dir:       Path to directory containing *.parquet shards.
        seq_len:        Sequence length for each training example.
        out_dir:        Directory where checkpoints and progress files live.
        val_fraction:   Fraction of shards to hold out as validation.
        device:         "cuda" / "mps" / "cpu"
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        out_dir: str,
        val_fraction: float = 0.005,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.device = device

        shards = discover_shards(data_dir)
        self.train_shards, self.val_shards = split_shards(shards, val_fraction)

        progress_path = Path(out_dir) / "data_progress.json"
        self.progress = DataProgress(progress_path)

        print(f"[FineWebLoader] {len(self.train_shards)} train shards, "
              f"{len(self.val_shards)} val shards  (seq_len={seq_len})")

    def train_iter(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Infinite iterator over training shards (loops over corpus)."""
        while True:
            stream = _stream_tokens(self.train_shards, self.progress, "train")
            yield from _chunk_iter(stream, self.seq_len, self.device)

    def val_iter(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Single-pass iterator over validation shards."""
        stream = _stream_tokens(self.val_shards, self.progress, "val")
        yield from _chunk_iter(stream, self.seq_len, self.device)

    def get_val_batch(
        self, n_batches: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect a fixed val batch for periodic evaluation.
        Returns (x, y) each of shape [n_batches * batch_size, seq_len].
        """
        xs, ys = [], []
        for x, y in self.val_iter():
            xs.append(x)
            ys.append(y)
            if len(xs) >= n_batches * batch_size:
                break

        if not xs:
            raise RuntimeError("Validation shards yielded no data.")

        return (
            torch.stack(xs[: n_batches * batch_size]),
            torch.stack(ys[: n_batches * batch_size]),
        )


def build_loader(
    data_dir: str,
    seq_len: int,
    out_dir: str,
    val_fraction: float = 0.005,
    device: str = "cpu",
) -> FineWebLoader:
    """Convenience factory."""
    return FineWebLoader(
        data_dir=data_dir,
        seq_len=seq_len,
        out_dir=out_dir,
        val_fraction=val_fraction,
        device=device,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../sources/fineweb-edu")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-batches", type=int, default=5)
    args = ap.parse_args()

    print(f"Smoke test: {args.data_dir}")
    loader = build_loader(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        out_dir="/tmp/pdp11_test",
        val_fraction=0.01,
        device="cpu",
    )
    t0 = time.time()
    for i, (x, y) in enumerate(loader.train_iter()):
        if i == 0:
            print(f"  First batch: x.shape={x.shape}, y.shape={y.shape}")
            print(f"  x[:10] = {x[:10].tolist()}")
        if i >= args.n_batches - 1:
            break

    elapsed = time.time() - t0
    print(f"  {args.n_batches} batches in {elapsed:.2f}s")
    print("OK")
