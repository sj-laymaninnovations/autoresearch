"""
attn11_text.py — causal-LM training on raw text for the ATTN11 architecture.

Mirror of finetune.py / attn11_utf8.py but for free-form text generation
instead of Q/A skill data:
    - Dataset: sliding-window chunks of raw text (no Q/A scaffold)
    - Loss: every token contributes (no prompt mask)
    - Tokenizer: ASCII char-level by default (matches the math checkpoints
      so warm-start from a math model is a direct weight-load)

The architecture (MathGPT) is reused unchanged — only the data shape and
loss-mask semantics differ.

Two experiments we run with this script:
    A. From scratch on TinyShakespeare
    B. Warm-start from a generalized math checkpoint (e.g. div_1d ep_0400)

For (B), pass --init <ckpt_path> --reset-epoch-counter so the cosine LR
schedule restarts cleanly.

Usage:
    python math_lab/attn11_text.py \
        --corpus math_lab/results/corpora/tinyshakespeare.txt \
        --ckpt-dir <out> --device mps --epochs 200
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    MathGPT, GPTConfig, EMA, cosine_lr, count_params,
    save_checkpoint, load_checkpoint,
    char_encode, char_decode, EOS_ID, PAD_ID, BOS_ID, VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Dataset — sliding-window chunks of raw text, full-sequence loss
# ---------------------------------------------------------------------------

class TextChunkDataset(Dataset):
    """Tokenizes a corpus once with char_encode, then carves it into
    overlapping seq_len chunks (stride = seq_len // 2)."""

    def __init__(self, text: str, seq_len: int, stride: int | None = None):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2

        # Tokenize the entire corpus (drop BOS/EOS that char_encode adds —
        # the corpus boundaries don't carry the sentence-pair semantics those
        # tokens were designed for).
        ids = char_encode(text)
        if ids and ids[0] == BOS_ID:
            ids = ids[1:]
        if ids and ids[-1] == EOS_ID:
            ids = ids[:-1]
        self.tokens = torch.tensor(ids, dtype=torch.long)

        # Generate sliding-window starts
        self.starts = list(range(0, max(0, len(self.tokens) - seq_len), self.stride))
        if not self.starts:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.tokens[s : s + self.seq_len]
        # Pad if the very last chunk is short (rare with stride < seq_len)
        if chunk.size(0) < self.seq_len:
            pad = torch.full((self.seq_len - chunk.size(0),), PAD_ID, dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
            mask = torch.cat([
                torch.ones(self.seq_len - pad.size(0), dtype=torch.float32),
                torch.zeros(pad.size(0), dtype=torch.float32),
            ], dim=0)
        else:
            # Causal LM: predict every token. MathGPT.forward shifts internally
            # (the loss is computed on positions 1..T-1 predicting from 0..T-2).
            mask = torch.ones(self.seq_len, dtype=torch.float32)
        return chunk, mask


# ---------------------------------------------------------------------------
# Sampling for vibe-check generations during/after training
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_text(model: MathGPT, prompt: str, max_new: int = 200,
                temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    device = next(model.parameters()).device
    ids = char_encode(prompt)
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.cfg.seq_len
    for _ in range(max_new):
        ctx = ids_t[:, -seq_len:]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        if next_id.item() == EOS_ID:
            break
        ids_t = torch.cat([ids_t, next_id], dim=1)
    return char_decode(ids_t[0].tolist())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, label_smoothing,
                    use_amp, dtype):
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
    p.add_argument("--corpus", required=True, help="path to raw text file")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--init", default=None,
                   help="optional checkpoint to warm-start from (must be ASCII vocab)")
    p.add_argument("--reset-epoch-counter", action="store_true",
                   help="restart LR schedule from epoch 0 even with --init")
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of corpus reserved for val (held out at the END)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None,
                   choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=25,
                   help="emit a generation sample every N epochs")
    p.add_argument("--sample-prompt", default="ROMEO:",
                   help="prompt for the periodic generation sample")
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

    # ---- Corpus split ----
    corpus_path = Path(args.corpus)
    text = corpus_path.read_text(encoding="utf-8")
    n_total = len(text)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_text = text[:n_train]
    val_text = text[n_train:]
    print(f"  Corpus : {corpus_path.name}  |  {n_total:,} chars total")
    print(f"           train={n_train:,} chars  val={n_val:,} chars")

    train_ds = TextChunkDataset(train_text, seq_len=args.seq_len, stride=args.stride)
    val_ds = TextChunkDataset(val_text, seq_len=args.seq_len, stride=args.seq_len)
    print(f"           train chunks={len(train_ds)}  val chunks={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # ---- Model ----
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,    # ASCII (131) — matches math checkpoints
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    if args.init:
        print(f"  Warm-start: loading {args.init}")
        model, meta = load_checkpoint(Path(args.init), device)
        # Sanity check the architecture matches
        if (model.cfg.vocab_size, model.cfg.seq_len, model.cfg.n_layer,
            model.cfg.n_head, model.cfg.n_embd) != (
            cfg.vocab_size, cfg.seq_len, cfg.n_layer, cfg.n_head, cfg.n_embd):
            raise ValueError(
                f"--init checkpoint architecture {(model.cfg)} does not match "
                f"requested {cfg}. Pick a math checkpoint with the same shape "
                f"(6×8×256, vocab=131, seq=256)."
            )
        # Update dropout in the loaded model to match the request (the saved
        # model was probably dropout=0.0 from math training).
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = args.dropout
        start_epoch = 0 if args.reset_epoch_counter else (meta.get("epoch", -1) + 1)
        if start_epoch > 0:
            print(f"  resuming from epoch {start_epoch} (use --reset-epoch-counter to start fresh)")
    else:
        model = MathGPT(cfg).to(device)
        start_epoch = 0

    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.log"

    print()
    print("=" * 60)
    print("  ATTN11 — Causal LM on raw text")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Init   : {'random' if not args.init else args.init}")
    print(f"  Params : {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Config : n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} seq={args.seq_len}")
    print()
    print(f"  Training {args.epochs} epochs (from epoch {start_epoch})  |  batch={args.batch}")
    print("-" * 60)

    best_val = float("inf")
    total_steps = max(1, len(train_loader)) * args.epochs
    step = start_epoch * len(train_loader)

    for ep in range(start_epoch, args.epochs):
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
               f"  ep {ep+1:03d}/{args.epochs}"
               f"  train={train_loss:.4f}  val={v_str}"
               f"  lr={cur_lr:.2e}  {time.time()-t0:.1f}s")
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            ckpt_p = ckpt_dir / f"epoch_{ep+1:04d}.pt"
            save_checkpoint(model, opt, ep, step,
                             best_val if v_str == "skip" else float(v_str),
                             ckpt_p)
            with open(log_path, "a") as f:
                f.write(f"  Saved -> {ckpt_p.name}\n")

        if (ep + 1) % args.sample_every == 0 or ep == args.epochs - 1:
            try:
                gen = sample_text(model, args.sample_prompt, max_new=200,
                                   temperature=0.8, top_k=40)
                snippet = gen.replace("\n", " | ")[:200]
                print(f"    sample: {snippet}")
                with open(log_path, "a") as f:
                    f.write(f"    sample: {snippet}\n")
            except Exception as e:
                print(f"    sample error: {e}")

    print()
    print("=" * 60)
    print(f"  Training complete  |  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints -> {ckpt_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
