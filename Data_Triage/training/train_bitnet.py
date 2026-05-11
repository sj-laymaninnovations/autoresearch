"""
train.py — nanoGPT-style training loop for PDP-11 SLM.

Features:
  - AdamW optimizer with cosine LR decay + linear warmup
  - Gradient clipping (1.0)
  - Mixed precision (bfloat16 on CUDA/CPU; float16 fallback)
  - Logs loss every 100 steps to stdout and logs/train.jsonl
  - Saves checkpoint every 1000 steps (keeps last 3)
  - Checkpoint-resumable (reads latest checkpoint from out_dir)
  - Works on CUDA (3080/3090), MPS (MacBook Pro), and CPU

Usage:
    python train.py --config A              # Config A (gate run)
    python train.py --config A --resume     # Resume from latest checkpoint

DO NOT run Config B/C/D until Config A shows clean decreasing loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

# Local imports
from config import get_config, TrainRun
from data_loader import build_loader
from bitnet import BitGPT


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Model — standard decoder-only transformer with RoPE
# ---------------------------------------------------------------------------

class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with configurable theta."""

    def __init__(self, d_head: int, max_seq: int, theta: float = 500_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = q.shape[2]
        cos = self.cos_cache[:, :, :seq, :]
        sin = self.sin_cache[:, :, :seq, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.d_head  = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model

        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.rope = RoPEEmbedding(self.d_head, cfg.seq_len, cfg.rope_theta)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(
                1, 1, cfg.seq_len, cfg.seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Reshape to [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) / scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.d_model, cfg.d_ff,  bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_ff,  cfg.d_model, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(cfg.vocab_size, cfg.d_model),
            drop  = nn.Dropout(cfg.dropout),
            h     = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f  = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.seq_len, \
            f"Sequence length {T} exceeds max {self.cfg.seq_len}"

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# LR schedule — cosine decay with linear warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, run: TrainRun) -> float:
    t = run.train
    if step < t.warmup_steps:
        return t.lr * step / max(1, t.warmup_steps)
    if step >= t.max_steps:
        return t.lr * t.lr_decay_to

    progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return t.lr * (t.lr_decay_to + (1.0 - t.lr_decay_to) * cosine)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    run: TrainRun,
    out_dir: Path,
    keep: int = 3,
):
    ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "run":       run.to_dict(),
    }, ckpt_path)
    print(f"  [ckpt] saved -> {ckpt_path}")

    # Prune old checkpoints
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"  [ckpt] pruned {old.name}")


def load_latest_checkpoint(
    out_dir: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> int:
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    print(f"  [ckpt] resuming from {latest}")
    state = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return state["step"]


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(
    model: GPT,
    loader,
    n_steps: int,
    ctx,
) -> float:
    model.eval()
    total_loss = 0.0
    for i, (x, y) in enumerate(loader.val_iter()):
        if i >= n_steps:
            break
        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)
        with ctx:
            _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
    model.train()
    return total_loss / max(1, min(n_steps, i + 1))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(run: TrainRun, resume: bool = False):
    t_cfg = run.train
    m_cfg = run.model

    out_dir = Path(t_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "train.jsonl"

    print(f"\n{'='*60}")
    print(f"  PDP-11 SLM — Config {t_cfg.config_letter}")
    print(f"  Device:  {DEVICE}")
    print(f"  d_model: {m_cfg.d_model}  layers: {m_cfg.n_layer}  heads: {m_cfg.n_head}")
    print(f"  seq_len: {m_cfg.seq_len}  dtype: {t_cfg.dtype}")
    print(f"  max_steps: {t_cfg.max_steps:,}  (~{t_cfg.target_tokens/1e6:.0f}M tokens)")
    print(f"{'='*60}\n")

    # Save run config
    run.save(str(out_dir / "run_config.json"))

    # ── Model ──────────────────────────────────────────────────────────────
    model = BitGPT(m_cfg).to(DEVICE)
    print(f"  Parameters: {model.num_params/1e6:.2f}M")

    # ── Mixed precision ────────────────────────────────────────────────────
    ptdtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[t_cfg.dtype]
    use_amp = (DEVICE == "cuda") and (t_cfg.dtype in ("bfloat16", "float16"))
    from contextlib import nullcontext
    ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp \
          else torch.autocast(device_type="cpu",  dtype=ptdtype) if DEVICE == "cpu" \
          else nullcontext()  # MPS: no autocast, use default
    # MPS: manual cast
    if DEVICE == "mps" and ptdtype != torch.float32:
        model = model.to(ptdtype)

    scaler = GradScaler(enabled=(use_amp and ptdtype == torch.float16))

    # ── Optimizer ──────────────────────────────────────────────────────────
    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2 and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": t_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=t_cfg.lr, betas=(t_cfg.beta1, t_cfg.beta2))

    # ── Resume ────────────────────────────────────────────────────────────
    step = 0
    if resume:
        step = load_latest_checkpoint(out_dir, model, optimizer, scaler)
        print(f"  Resumed at step {step:,}")

    # ── Data ──────────────────────────────────────────────────────────────
    loader = build_loader(
        data_dir    = t_cfg.data_dir,
        seq_len     = m_cfg.seq_len,
        out_dir     = t_cfg.out_dir,
        val_fraction= t_cfg.val_fraction,
        device      = DEVICE,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    grad_accum = t_cfg.grad_accum_steps
    accum_loss = 0.0
    t0 = time.time()

    train_data = iter(loader.train_iter())

    print(f"  Starting training from step {step:,} …\n")

    for step in range(step, t_cfg.max_steps):
        # ── LR update ────────────────────────────────────────────────────
        lr = get_lr(step, run)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation micro-steps ─────────────────────────
        for micro_step in range(grad_accum):
            x, y = next(train_data)
            x = x.unsqueeze(0).to(DEVICE)   # [1, seq_len]
            y = y.unsqueeze(0).to(DEVICE)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            loss = loss / grad_accum
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        # ── Optimizer step ────────────────────────────────────────────
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ── Logging ───────────────────────────────────────────────────
        if step % t_cfg.log_every == 0:
            elapsed = time.time() - t0
            tokens_so_far = step * loader.seq_len * t_cfg.batch_size * grad_accum
            record = {
                "step":       step,
                "train_loss": round(accum_loss, 6),
                "lr":         round(lr, 8),
                "tokens_M":   round(tokens_so_far / 1e6, 2),
                "elapsed_s":  round(elapsed, 1),
            }
            print(f"  step {step:6,} | loss {accum_loss:.4f} | lr {lr:.2e} "
                  f"| tok {tokens_so_far/1e6:.1f}M | {elapsed:.0f}s")
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            accum_loss = 0.0

        # ── Validation ────────────────────────────────────────────────
        if step % t_cfg.eval_every == 0 and step > 0:
            val_loss = estimate_val_loss(model, loader, n_steps=20, ctx=ctx)
            print(f"  -- val_loss: {val_loss:.4f} (step {step:,})")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "val_loss": round(val_loss, 6)}) + "\n")

        # ── Checkpoint ────────────────────────────────────────────────
        if step % t_cfg.save_every == 0 and step > 0:
            save_checkpoint(step, model, optimizer, scaler, run, out_dir, t_cfg.keep_checkpoints)

    # Final checkpoint
    save_checkpoint(step, model, optimizer, scaler, run, out_dir, t_cfg.keep_checkpoints)
    print(f"\n  Training complete. {step:,} steps.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PDP-11 SLM training")
    ap.add_argument("--config", default="A", choices=["A", "B", "C", "D"],
                    help="Config letter (A=gate run, B/C/D require A to pass first)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in out_dir")
    ap.add_argument("--device", default=None,
                    help="Override device (cuda/mps/cpu)")
    ap.add_argument("--data-dir", default=None,
                    help="Override train.data_dir from config (e.g. ../sources_prehistoric/fineweb-edu)")
    ap.add_argument("--out-dir", default=None,
                    help="Override train.out_dir / checkpoint directory")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    args = ap.parse_args()

    if args.config != "A":
        print(f"WARNING: Config {args.config} requires Config A to have trained cleanly first.")
        print("  Ensure Config A loss is steadily decreasing before running this.")
        print()

    if args.device:
        DEVICE = args.device

    run = get_config(args.config)
    if args.data_dir:
        run.train.data_dir = args.data_dir
        print(f"  [override] data_dir -> {run.train.data_dir}")
    if args.out_dir:
        run.train.out_dir = args.out_dir
        print(f"  [override] out_dir  -> {run.train.out_dir}")
    if args.max_steps:
        run.train.max_steps = args.max_steps
    if args.warmup_steps:
        run.train.warmup_steps = args.warmup_steps
    train(run, resume=args.resume)
