"""
attn11_utf8.py — byte-level (UTF-8) version of the ATTN11 / MathGPT
architecture.

Tokenizer change only — model architecture is unchanged. Each token = one
UTF-8 byte. Vocab = 256 bytes + 4 special tokens = 260. Any language /
character set is encodable without per-script logic.

Token layout:
    0   PAD
    1   BOS
    2   EOS
    3   UNK
    4..259  raw byte values 0..255 (offset by 4)

Why byte-level instead of codepoint-level:
    - codepoint vocab would balloon (>150k Unicode codepoints)
    - bytes give a uniform-cost, lossless, language-agnostic stream
    - the model learns multi-byte UTF-8 sequences as it would any pattern
      (each char is 1-4 contiguous tokens; the model attends across them)

ASCII text is unchanged in shape: 1 char = 1 byte = 1 token. The trained
behavior of the existing div_1d / sub_2d / mul_2d skills is recoverable
by re-training on the same data with this tokenizer (but new checkpoints
are needed — embeddings and lm_head are wider).

Usage:
    python math_lab/attn11_utf8.py \
        --data <train.jsonl> --val-data <val.jsonl> \
        --ckpt-dir <out> --device cuda --epochs 400

The JSONL data format is identical to finetune.py — `{problem, solution_cot,
answer, ...}`. UTF-8 strings work unchanged (they get encoded to bytes
internally).
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Reuse the architecture (MathGPT, blocks, helpers). The model takes
# vocab_size from its config so it works for any tokenizer.
sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    MathGPT, GPTConfig, EMA, cosine_lr, count_params,
    save_checkpoint, load_checkpoint as _load_finetune_checkpoint,
)


# ---------------------------------------------------------------------------
# Tokenizer (byte-level, UTF-8 safe)
# ---------------------------------------------------------------------------

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
BYTE_OFFSET = 4
VOCAB_SIZE = 256 + BYTE_OFFSET  # 260


def byte_encode(text: str) -> list[int]:
    """str → list[int] tokens. Wraps with [BOS, ..., EOS]."""
    out = [BOS_ID]
    for b in text.encode("utf-8"):
        out.append(b + BYTE_OFFSET)
    out.append(EOS_ID)
    return out


def byte_decode(ids: list[int]) -> str:
    """list[int] → str. Stops at the first EOS. Skips other specials."""
    bs = bytearray()
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, BOS_ID, UNK_ID):
            continue
        if BYTE_OFFSET <= i < VOCAB_SIZE:
            bs.append(i - BYTE_OFFSET)
    # 'replace' tolerates partial-token cutoffs at generation time
    return bs.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Prompt template — identical to finetune.py
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "Q: {problem}\nA: "


# ---------------------------------------------------------------------------
# Dataset — mirror of finetune.MathQADataset using byte_encode
# ---------------------------------------------------------------------------

class ByteMathQADataset(Dataset):
    """Tokenizes (problem, cot) pairs with byte_encode and produces fixed-len
    (input_ids, loss_mask) tensors. Loss is computed only on the cot tokens
    (after the prompt prefix), matching finetune.py semantics."""

    def __init__(self, jsonl_path: Path, seq_len: int):
        self.seq_len = seq_len
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                problem = str(obj.get("problem", ""))
                cot = obj.get("solution_cot")
                if cot is None:
                    cot = str(obj.get("answer", ""))
                prompt = PROMPT_TEMPLATE.format(problem=problem)
                full = prompt + cot
                full_ids = byte_encode(full)
                prompt_ids = byte_encode(prompt)
                # prompt_ids includes a trailing EOS; drop it so the cot
                # tokens immediately follow the prompt prefix.
                if prompt_ids and prompt_ids[-1] == EOS_ID:
                    prompt_ids = prompt_ids[:-1]
                if len(full_ids) > seq_len:
                    full_ids = full_ids[:seq_len]
                if len(prompt_ids) > seq_len:
                    prompt_ids = prompt_ids[:seq_len]
                # loss_mask: 1 for cot tokens, 0 for prompt and pad
                loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
                # Pad
                pad_n = seq_len - len(full_ids)
                full_ids = full_ids + [PAD_ID] * pad_n
                loss_mask = loss_mask + [0] * pad_n
                self.samples.append((
                    torch.tensor(full_ids, dtype=torch.long),
                    torch.tensor(loss_mask, dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Generation (byte_encode-aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_bytes(model: MathGPT, prompt: str, max_new: int = 200) -> str:
    """Greedy decode. Identical logic to finetune greedy_generate, but with
    byte_encode/byte_decode."""
    model.eval()
    device = next(model.parameters()).device
    ids = byte_encode(prompt)
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.cfg.seq_len
    for _ in range(max_new):
        ctx = ids_t[:, -seq_len:]
        logits = model(ctx)
        next_id = int(logits[:, -1, :].argmax(-1).item())
        if next_id == EOS_ID:
            break
        ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=device)], dim=1)
    return byte_decode(ids_t[0].tolist())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, label_smoothing,
                    use_amp: bool, dtype):
    model.train()
    total = 0.0
    n = 0
    for ids, mask in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type=device, dtype=dtype):
                model.label_smoothing = label_smoothing
                loss = model(ids, mask)
        else:
            model.label_smoothing = label_smoothing
            loss = model(ids, mask)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.item()) * ids.size(0)
        n += ids.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, label_smoothing) -> float:
    model.eval()
    total = 0.0
    n = 0
    for ids, mask in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        model.label_smoothing = label_smoothing
        loss = model(ids, mask)
        total += float(loss.item()) * ids.size(0)
        n += ids.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None,
                   choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    use_amp = device == "cuda"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,        # ← UTF-8 byte vocab (260)
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MathGPT(cfg).to(device)
    n_params = count_params(model)

    train_ds = ByteMathQADataset(Path(args.data), args.seq_len)
    val_ds = ByteMathQADataset(Path(args.val_data), args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ATTN11 (byte-level, UTF-8) Training")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Vocab  : {VOCAB_SIZE} (256 bytes + 4 specials)")
    print(f"  Data   : {args.data}")
    print(f"  Pairs  : {len(train_ds)} train / {len(val_ds)} val")
    print(f"  Params : {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Config : n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} seq={args.seq_len}")
    print()
    print("-" * 60)
    print(f"  Training {args.epochs} epochs | batch={args.batch}")
    print("-" * 60)

    best_val = float("inf")
    total_steps = max(1, len(train_loader)) * args.epochs
    step = 0
    log_path = ckpt_dir / "train.log"

    for ep in range(args.epochs):
        # Set LR for the first batch of this epoch (cosine over total_steps)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.warmup, total_steps,
                                 args.lr_min, args.lr, restart_period=0)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, scaler, device,
                                      args.label_smoothing, use_amp, dtype)
        step += len(train_loader)

        do_val = (ep + 1) % args.val_every == 0 or ep == args.epochs - 1
        v_str = "skip"
        if do_val:
            vloss = eval_loss(model, val_loader, device, args.label_smoothing)
            v_str = f"{vloss:.4f}"
            if vloss < best_val:
                best_val = vloss
                save_checkpoint(model, opt, ep, step, vloss, ckpt_dir / "best.pt")

        cur_lr = opt.param_groups[0]["lr"]
        msg = (f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}]"
               f" epoch {ep+1:03d}/{args.epochs}"
               f"  train={train_loss:.4f}  val={v_str}"
               f"  lr={cur_lr:.2e}  {time.time()-t0:.1f}s"
               f"{'  * BEST' if (do_val and v_str != 'skip' and float(v_str) <= best_val) else ''}")
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            ckpt_p = ckpt_dir / f"epoch_{ep+1:04d}.pt"
            save_checkpoint(model, opt, ep, step,
                             best_val if v_str == "skip" else float(v_str),
                             ckpt_p)
            msg = f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved checkpoint -> {ckpt_p}"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    print()
    print("=" * 60)
    print(f"  Training complete  |  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints -> {ckpt_dir}")
    print("=" * 60)

    # Quick sample on the first few val pairs (UTF-8 round-trip sanity)
    print()
    print("  Sample generations (last checkpoint):")
    print("-" * 60)
    last_ckpt = max(ckpt_dir.glob("epoch_*.pt"), default=None)
    if last_ckpt is not None:
        m, _ = load_byte_checkpoint(last_ckpt, device)
        for s in val_ds.samples[:3]:
            ids = s[0].tolist()
            # extract the prompt portion (until first non-prompt token via mask)
            mask = s[1].tolist()
            try:
                first_cot = mask.index(1.0)
            except ValueError:
                first_cot = len(ids)
            prompt = byte_decode(ids[:first_cot])
            out = greedy_generate_bytes(m, prompt, max_new=200)
            print(f"  >>> {prompt.strip()}")
            print(f"      {out[len(prompt):].splitlines()[0] if out.startswith(prompt) else out[:80]}")


def load_byte_checkpoint(path: Path, device: str):
    """Load a byte-vocab checkpoint. Wraps finetune.load_checkpoint but
    asserts vocab_size matches — refuses to mix byte and ASCII checkpoints."""
    model, meta = _load_finetune_checkpoint(path, device)
    if model.cfg.vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Checkpoint vocab_size={model.cfg.vocab_size} does not match "
            f"byte vocab ({VOCAB_SIZE}). This checkpoint was trained with "
            f"the ASCII tokenizer (finetune.py) — use the original loader."
        )
    return model, meta


if __name__ == "__main__":
    main()
