"""
finetune.py — Supervised Fine-Tuning on Math Q/A Pairs (CUDA)

Loads training_data.jsonl (or all harder_qa_*.jsonl files), formats each pair
as a supervised learning example, trains a small GPT on GPU with cross-entropy
loss computed ONLY on answer tokens (question tokens are masked out).

Usage:
    python math_lab/finetune.py                          # train on all collected pairs
    python math_lab/finetune.py --data math_lab/results/training_data.jsonl
    python math_lab/finetune.py --epochs 10 --batch 16  # custom hyperparams
    python math_lab/finetune.py --eval-only --ckpt math_lab/results/checkpoints/best.pt
"""

import os
import sys
import json
import math
import time
import random
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent.parent          # autoresearch/
RESULTS    = Path(__file__).parent / "results"
CKPT_DIR   = RESULTS / "checkpoints"
DEFAULT_DATA = RESULTS / "training_data.jsonl"

# ---------------------------------------------------------------------------
# Character-level tokenizer (matches qsparser.asm vocab design)
# ASCII printable chars + 4 special tokens — works without any data download
# ---------------------------------------------------------------------------

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
OFFSET = 4   # ascii_byte + OFFSET = token_id  (matches qsparser.asm)
VOCAB_SIZE = 131  # 4 special + 127 ASCII (0-126)

def char_encode(text: str) -> list[int]:
    ids = [BOS_ID]
    for ch in text:
        b = ord(ch)
        if b < 127:
            ids.append(b + OFFSET)
        else:
            ids.append(UNK_ID)
    ids.append(EOS_ID)
    return ids

def char_decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == EOS_ID:
            break
        if i >= OFFSET:
            out.append(chr(i - OFFSET))
    return "".join(out)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "Q: {problem}\nA: "   # question prefix
ANS_SUFFIX      = "\n"                   # appended after solution

class MathQADataset(Dataset):
    """
    Each sample is a full tokenized (question + answer) sequence.
    loss_mask = 1 only for answer tokens (after "A: ").
    """
    def __init__(self, pairs: list[dict], max_len: int = 512):
        self.samples = []
        short = 0
        for p in pairs:
            problem  = str(p.get("problem",  ""))
            solution = str(p.get("solution", p.get("answer", "")))

            prompt  = PROMPT_TEMPLATE.format(problem=problem)
            full    = prompt + solution + ANS_SUFFIX

            ids = char_encode(full)
            if len(ids) > max_len:
                ids = ids[:max_len]   # truncate

            # Build loss mask: 0 for prompt tokens, 1 for answer tokens
            prompt_ids  = char_encode(prompt)
            prompt_len  = len(prompt_ids)  # includes BOS
            mask = [0] * min(prompt_len, len(ids)) + \
                   [1] * max(0, len(ids) - prompt_len)

            # Pad to max_len
            pad_len = max_len - len(ids)
            ids  = ids  + [PAD_ID] * pad_len
            mask = mask + [0]      * pad_len

            self.samples.append({
                "ids":  torch.tensor(ids,  dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float),
            })

        if short:
            print(f"  Skipped {short} samples (too short)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["ids"], self.samples[idx]["mask"]


def load_pairs(data_path: Path) -> list[dict]:
    """Load JSONL pairs. Fallback: merge all harder_qa_*.jsonl."""
    if data_path.exists():
        pairs = []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return pairs

    # Fallback: all individual files
    files = sorted(RESULTS.glob("harder_qa_*.jsonl"))
    if not files:
        print(f"ERROR: No data found. Run harder_qa.py first.")
        sys.exit(1)

    pairs = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(pairs)} pairs from {len(files)} files (no merged file found).")
    return pairs


# ---------------------------------------------------------------------------
# Tiny GPT model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size:  int = VOCAB_SIZE
    seq_len:     int = 512
    n_layer:     int = 4
    n_head:      int = 4
    n_embd:      int = 128
    dropout:     float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash attention (PyTorch 2.0+) or manual scaled dot-product
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.RMSNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MathGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying (embedding ↔ lm_head)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)          # (B, T, vocab)

        if loss_mask is None:
            return logits

        # Shift: predict next token at each position
        shift_logits = logits[:, :-1, :].contiguous()       # (B, T-1, vocab)
        shift_targets = idx[:, 1:].contiguous()              # (B, T-1)
        shift_mask  = loss_mask[:, 1:].contiguous()          # (B, T-1)

        # Per-token cross-entropy with optional label smoothing
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
            ignore_index=PAD_ID,
            label_smoothing=getattr(self, "label_smoothing", 0.0),
        )                                                     # (B*(T-1),)
        loss_flat = loss_flat * shift_mask.view(-1)

        denom = shift_mask.sum().clamp(min=1)
        return loss_flat.sum() / denom

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 80,
                 temperature: float = 0.3, top_k: int = 40) -> str:
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor(char_encode(prompt)[:-1],   # strip EOS from prompt
                           dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            ids_cond = ids[:, -self.cfg.seq_len:]
            logits = self(ids_cond)[:, -1, :] / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            if next_id.item() == EOS_ID:
                break
            ids = torch.cat([ids, next_id], dim=1)
        return char_decode(ids[0].tolist())


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup: int, total: int, lr_min: float, lr_max: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, step, val_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    model.cfg,
        "epoch":     epoch,
        "step":      step,
        "val_loss":  val_loss,
        "saved_at":  datetime.datetime.now().isoformat(),
    }, path)


def load_checkpoint(path: Path, device: str) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = MathGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MathGPT on Q/A pairs")
    parser.add_argument("--data",      default=str(DEFAULT_DATA))
    parser.add_argument("--epochs",    type=int,   default=20)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--seq-len",   type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--lr-min",    type=float, default=1e-5)
    parser.add_argument("--warmup",    type=int,   default=50,
                        help="LR warmup steps")
    parser.add_argument("--n-layer",   type=int,   default=4)
    parser.add_argument("--n-head",    type=int,   default=4)
    parser.add_argument("--n-embd",    type=int,   default=128)
    parser.add_argument("--weight-decay", type=float, default=1.0,
                        help="AdamW weight decay (Grokking paper: 1.0 for arithmetic generalization)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Cross-entropy label smoothing (0.1 reduces memorization)")
    parser.add_argument("--dropout",   type=float, default=0.15)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data held out for validation")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--ckpt",      default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load checkpoint and run sample generation only")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # -- Setup --------------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype  = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() \
             else torch.float16 if device == "cuda" else torch.float32

    print(f"\n{'='*60}")
    print(f"  MathGPT Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Device : {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"  dtype  : {dtype}")
    print(f"  Data   : {args.data}")

    # -- Load data ----------------------------------------------------------
    pairs = load_pairs(Path(args.data))
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * args.val_split))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    print(f"  Pairs  : {len(train_p)} train / {len(val_p)} val  ({len(pairs)} total)")

    train_ds = MathQADataset(train_p, max_len=args.seq_len)
    val_ds   = MathQADataset(val_p,   max_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=(device=="cuda"))

    # -- Model --------------------------------------------------------------
    cfg = GPTConfig(
        vocab_size = VOCAB_SIZE,
        seq_len    = args.seq_len,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
        dropout    = args.dropout,
    )

    start_epoch = 0
    if args.ckpt and Path(args.ckpt).exists():
        print(f"  Loading checkpoint: {args.ckpt}")
        model, ckpt = load_checkpoint(args.ckpt, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resuming from epoch {start_epoch}")
    else:
        model = MathGPT(cfg).to(device)
    model.label_smoothing = args.label_smoothing

    print(f"  Params : {count_params(model):,}  "
          f"({count_params(model)/1e6:.2f}M)")
    print(f"  Config : n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_embd={cfg.n_embd} seq={cfg.seq_len}")

    # -- Eval-only mode ------------------------------------------------------
    if args.eval_only:
        samples = [
            "Q: What is 48 + 37?\nA: ",
            "Q: What is 125 times 8?\nA: ",
            "Q: If you have 500 dollars and spend 367, how much is left?\nA: ",
        ]
        print(f"\n{'-'*60}")
        print("  Sample generations:")
        print(f"{'-'*60}")
        for prompt in samples:
            response = model.generate(prompt, max_new=60)
            print(f"  {prompt.strip()}")
            print(f"  -> {response[len(prompt):]}")
            print()
        return

    # -- Optimizer ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98),
        weight_decay=args.weight_decay, fused=use_amp,
    )

    import contextlib
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device, dtype=dtype) if use_amp \
                   else contextlib.nullcontext()

    total_train_steps = args.epochs * len(train_loader)
    best_val_loss     = float("inf")
    global_step       = 0

    print(f"\n{'-'*60}")
    print(f"  Training {args.epochs} epochs  |  "
          f"{len(train_loader)} steps/epoch  |  "
          f"batch={args.batch}")
    print(f"{'-'*60}")

    for epoch in range(start_epoch, args.epochs):
        # -- Train ----------------------------------------------------------
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        t0 = time.time()

        for ids, mask in train_loader:
            ids  = ids.to(device,  non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # LR schedule
            current_lr = cosine_lr(global_step, args.warmup,
                                   total_train_steps, args.lr_min, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = model(ids, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

        avg_train = epoch_loss / max(epoch_steps, 1)

        # -- Validate -------------------------------------------------------
        model.eval()
        val_loss  = 0.0
        val_steps = 0
        with torch.no_grad(), autocast_ctx:
            for ids, mask in val_loader:
                ids  = ids.to(device,  non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                val_loss  += model(ids, mask).item()
                val_steps += 1
        avg_val = val_loss / max(val_steps, 1)

        dt = time.time() - t0
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            save_checkpoint(model, optimizer, epoch, global_step,
                            avg_val, CKPT_DIR / "best.pt")

        status = "* BEST" if is_best else ""
        print(f"  epoch {epoch+1:03d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={current_lr:.2e}  {dt:.1f}s  {status}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            path = CKPT_DIR / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(model, optimizer, epoch, global_step, avg_val, path)
            print(f"  Saved checkpoint -> {path}")

    # -- Final checkpoint ---------------------------------------------------
    save_checkpoint(model, optimizer, args.epochs - 1, global_step,
                    best_val_loss, CKPT_DIR / "final.pt")

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoints  -> {CKPT_DIR}")
    print(f"{'='*60}")

    # -- Sample generation --------------------------------------------------
    print(f"\n  Sample generations (best checkpoint):\n{'-'*60}")
    best_model, _ = load_checkpoint(CKPT_DIR / "best.pt", device)
    test_prompts = [
        "Q: What is 48 + 37?\nA: ",
        "Q: What is 125 times 8?\nA: ",
        "Q: If you have 500 dollars and spend 367, how much is left?\nA: ",
    ]
    for prompt in test_prompts:
        response = best_model.generate(prompt, max_new=80)
        answer = response[len(prompt):]
        print(f"  {prompt.strip()}")
        print(f"  -> {answer.strip()}")
        print()


if __name__ == "__main__":
    main()
