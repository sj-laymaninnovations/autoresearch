"""
eval.py — Checkpoint evaluation for PDP-11 SLM.

Loads a checkpoint, computes perplexity on the held-out FineWeb-Edu val split,
and generates 5 sample completions at temperature 0.8.

Usage:
    python eval.py --config A
    python eval.py --config A --ckpt checkpoints/config_a/ckpt_0010000.pt
    python eval.py --config A --prompt "The key principle of machine learning is"
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from config import get_config, TrainRun, ModelConfig
from data_loader import build_loader
from train import GPT, DEVICE, get_device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path, model: GPT) -> int:
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    step = state.get("step", 0)
    print(f"  Loaded checkpoint: {ckpt_path.name}  (step {step:,})")
    return step


def find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: GPT,
    loader,
    n_steps: int = 50,
) -> float:
    """
    Compute perplexity on the validation split.
    perplexity = exp(mean_cross_entropy_loss)
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(loader.val_iter()):
        if i >= n_steps:
            break
        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)
        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            count += 1

    if n_steps == 0: return 0.0
    if count == 0:
        raise RuntimeError("No validation data found.")

    mean_loss = total_loss / count
    ppl = math.exp(mean_loss)
    return ppl


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: List[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 200,
) -> List[int]:
    """
    Greedy/sampled autoregressive generation.
    Returns generated token ids (not including prompt).
    """
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = []

    for _ in range(max_new_tokens):
        # Crop to model's seq_len
        ids_cond = ids[:, -model.cfg.seq_len:]
        logits, _ = model(ids_cond)
        logits = logits[:, -1, :]  # last token position

        if temperature == 0.0:
            # Greedy
            next_id = logits.argmax(dim=-1).unsqueeze(1)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        next_tok = next_id.item()
        generated.append(next_tok)
        ids = torch.cat([ids, next_id], dim=1)

        # Stop at EOS (tiktoken endoftext = 100257)
        if next_tok == 100257:
            break

    return generated


# ---------------------------------------------------------------------------
# Sample prompts for generation
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "During the Upper Paleolithic, human populations in Europe created",
    "The cave of Lascaux is famous for its",
    "The Sumerians in Mesopotamia are credited with inventing",
    "The fundamental theorem of calculus states that",
    "Python is a programming language that",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PDP-11 SLM evaluation")
    ap.add_argument("--config",     default="A", choices=["A", "B", "C", "D"])
    ap.add_argument("--ckpt",       default=None,
                    help="Path to specific checkpoint. Defaults to latest in out_dir.")
    ap.add_argument("--data-dir",   default=None,
                    help="Override data_dir from config")
    ap.add_argument("--n-val-steps", type=int, default=50,
                    help="Number of val batches for perplexity estimate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new",    type=int, default=200,
                    help="Max new tokens per generation sample")
    ap.add_argument("--prompt",     default=None,
                    help="Custom prompt (uses DEFAULT_PROMPTS if not set)")
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device

    run = get_config(args.config)
    m_cfg, t_cfg = run.model, run.train

    out_dir = Path(t_cfg.out_dir)

    # ── Find checkpoint ───────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest_checkpoint(out_dir)
        if ckpt_path is None:
            print(f"ERROR: No checkpoint found in {out_dir}")
            print("  Run train.py first.")
            return

    # ── Build model ───────────────────────────────────────────────────────
    print(f"\n  Loading Config {args.config} model…")
    model = GPT(m_cfg).to(DEVICE)
    step = load_checkpoint(ckpt_path, model)
    model.eval()
    print(f"  Parameters: {model.num_params/1e6:.2f}M")

    # ── Data loader (for perplexity) ──────────────────────────────────────
    loader = build_loader(
        data_dir     = args.data_dir if args.data_dir else t_cfg.data_dir,
        seq_len      = m_cfg.seq_len,
        out_dir      = t_cfg.out_dir,
        val_fraction = t_cfg.val_fraction,
        device       = DEVICE,
    )

    # ── Perplexity ────────────────────────────────────────────────────────
    print(f"\n  Computing perplexity ({args.n_val_steps} val batches)…")
    ppl = compute_perplexity(model, loader, n_steps=args.n_val_steps)
    print(f"  Perplexity: {ppl:.2f}  (step {step:,})")

    # ── Generation ────────────────────────────────────────────────────────
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    print(f"\n  Generating {len(prompts)} samples (temp={args.temperature})…")
    print("  " + "-" * 56)

    for i, prompt_text in enumerate(prompts, 1):
        prompt_ids = enc.encode_ordinary(prompt_text)
        gen_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
        )
        gen_text = enc.decode(gen_ids)
        print(f"\n  [{i}] PROMPT: {prompt_text}")
        print(f"       GEN:    {gen_text[:300]}")
        print()

    print("  " + "-" * 56)
    print(f"\n  Evaluation complete (step {step:,}, ppl={ppl:.2f})\n")


if __name__ == "__main__":
    main()
