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
    """Load JSONL pairs. Fails loudly if the file is missing instead of
    silently globbing unrelated data (that silent fallback has bitten us:
    mistyped paths loaded 30K wrong pairs without error)."""
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        print(f"       refusing silent fallback — specify the correct --data path.")
        sys.exit(1)
    pairs = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not pairs:
        print(f"ERROR: {data_path} exists but contains no valid JSON lines.")
        sys.exit(1)
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

def cosine_lr(step: int, warmup: int, total: int, lr_min: float, lr_max: float,
              restart_period: int = 0) -> float:
    """Cosine LR with optional warm restarts (Loshchilov & Hutter, 2017).
    restart_period=0 means single cosine decay (original behavior)."""
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    post_warmup = step - warmup
    effective_total = total - warmup
    if restart_period > 0:
        # Cosine annealing with warm restarts
        cycle_pos = post_warmup % restart_period
        progress = cycle_pos / restart_period
    else:
        progress = post_warmup / max(effective_total, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].lerp_(v, 1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap EMA weights into the model (call before eval)."""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original weights (call after eval)."""
        model.load_state_dict(self._backup)
        del self._backup


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
# Exact-match probe (runs during training to catch MPS phantom losses)
# ---------------------------------------------------------------------------

import re as _re
_ANSWER_RE = _re.compile(r"####\s*([^\n]+?)\s*$", _re.MULTILINE)
_NUMERIC_RE = _re.compile(r"-?\d+(?:/-?\d+)?(?:\.\d+)?")


def _extract_answer(text):
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUMERIC_RE.findall(text)
    return nums[-1] if nums else None


def _normalize(v):
    s = str(v).strip()
    try:
        fv = float(s)
        if abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return f"{fv:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return s.replace(" ", "").lower()


@torch.no_grad()
def quick_exact_match(model, val_pairs, sample_size=16, max_new=120):
    """Greedy-decode up to `sample_size` val problems, return fraction of
    exact-match correct answers. Used during training to catch numerical
    phantoms — if loss is 'converging' but EM is 0%, weights are corrupt."""
    model.eval()
    device = next(model.parameters()).device
    pairs = val_pairs[:sample_size]
    correct = 0
    for p in pairs:
        prompt = PROMPT_TEMPLATE.format(problem=str(p.get("problem", "")))
        prompt_ids = char_encode(prompt)
        if prompt_ids and prompt_ids[-1] == EOS_ID:
            prompt_ids = prompt_ids[:-1]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for _ in range(max_new):
            ctx = ids[:, -model.cfg.seq_len:]
            logits = model(ctx)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            if nxt.item() == EOS_ID:
                break
            ids = torch.cat([ids, nxt], dim=1)
        generated = char_decode(ids[0].tolist())[len(prompt):]
        pred = _extract_answer(generated)
        expected = p.get("answer", p.get("solution", ""))
        if pred is not None and _normalize(pred) == _normalize(expected):
            correct += 1
    model.train()
    return correct / max(len(pairs), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MathGPT on Q/A pairs")
    parser.add_argument("--data",      default=str(DEFAULT_DATA))
    parser.add_argument("--val-data",  default=None,
                        help="Optional external val JSONL. If set, bypasses --val-split.")
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
    parser.add_argument("--restart-period", type=int, default=0,
                        help="Cosine warm restart period in steps (0=no restarts)")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay rate (0=disabled, 0.999=typical)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (2-4 for CPU/GPU overlap)")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run validation every N epochs (saves time on long runs)")
    parser.add_argument("--em-every", type=int, default=0,
                        help="Run exact-match check every N epochs (0=off). Aborts "
                             "training if EM stays at 0%% for 3 consecutive checks past "
                             "the early-training window — catches MPS phantom losses.")
    parser.add_argument("--em-sample", type=int, default=16,
                        help="Number of val examples to use for in-training EM check")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"],
                        help="Force specific device (default: auto-detect). Use 'cpu' "
                             "when MPS has numerical issues.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for 20-40%% speedup")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="Sleep N seconds every 100 epochs (GPU thermal protection)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Directory to save checkpoints (default: results/checkpoints)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR

    # -- Setup --------------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # MPS has an independent RNG that must be seeded separately for
    # deterministic weight init and dropout patterns. Also clear cached
    # MPS state at startup to avoid carryover across processes.
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(args.seed)
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # Allow forcing a specific device (e.g. "cpu" fallback when MPS is flaky)
    force_device = getattr(args, "device", None)
    if force_device in ("cpu", "cuda", "mps"):
        device = force_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # MPS + fp16 autocast is numerically unstable for our training regime
    # (we saw silent weight corruption producing fake-low losses). Disable
    # autocast on MPS and run in fp32; our models are small enough that the
    # speed cost is acceptable.
    use_amp = device == "cuda"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float32

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"  MathGPT Fine-Tuning")
    print(f"{'='*60}")
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    elif device == "mps":
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device_name = "CPU"
    print(f"  Device : {device}  ({device_name})")
    print(f"  dtype  : {dtype}")
    print(f"  Data   : {args.data}")

    # -- Load data ----------------------------------------------------------
    if args.val_data:
        # External val set — use all of args.data as train, args.val_data as val
        train_p = load_pairs(Path(args.data))
        val_p   = load_pairs(Path(args.val_data))
        random.shuffle(train_p)
        print(f"  Pairs  : {len(train_p)} train / {len(val_p)} val  (external val)")
    else:
        pairs = load_pairs(Path(args.data))
        random.shuffle(pairs)
        n_val   = max(1, int(len(pairs) * args.val_split))
        val_p   = pairs[:n_val]
        train_p = pairs[n_val:]
        print(f"  Pairs  : {len(train_p)} train / {len(val_p)} val  ({len(pairs)} total)")

    train_ds = MathQADataset(train_p, max_len=args.seq_len)
    val_ds   = MathQADataset(val_p,   max_len=args.seq_len)
    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=nw, pin_memory=(device=="cuda"),
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=nw, pin_memory=(device=="cuda"),
                              persistent_workers=(nw > 0))

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

    # -- torch.compile (PyTorch 2.0+ JIT) -----------------------------------
    if args.compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model)

    # -- EMA ----------------------------------------------------------------
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"  EMA    : decay={args.ema_decay}")

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
        weight_decay=args.weight_decay, fused=(device == "cuda"),
    )

    import contextlib
    # GradScaler only supports CUDA; on MPS use autocast without scaling
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
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

    # Periodic exact-match probe history (catches phantom losses on MPS)
    em_history: list[tuple[int, float]] = []

    # Crash/signal safety net: save emergency.pt on SIGTERM/SIGINT or exception.
    # Uses closures so handlers see current epoch/step/val_loss.
    _state = {"epoch": start_epoch, "step": 0, "val": float("inf")}

    def _emergency_save(reason: str):
        try:
            path = ckpt_dir / "emergency.pt"
            save_checkpoint(model, optimizer, _state["epoch"],
                            _state["step"], _state["val"], path)
            print(f"\n  [EMERGENCY SAVE: {reason}] -> {path}", flush=True)
        except Exception as e:
            print(f"\n  [EMERGENCY SAVE FAILED: {e}]", flush=True)

    import signal as _signal
    def _sig_handler(signum, frame):
        _emergency_save(f"signal {signum}")
        sys.exit(0)
    _signal.signal(_signal.SIGTERM, _sig_handler)
    _signal.signal(_signal.SIGINT,  _sig_handler)

    for epoch in range(start_epoch, args.epochs):
        # -- Train ----------------------------------------------------------
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        t0 = time.time()

        for ids, mask in train_loader:
            ids  = ids.to(device,  non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # LR schedule (with optional warm restarts)
            current_lr = cosine_lr(global_step, args.warmup,
                                   total_train_steps, args.lr_min, args.lr,
                                   restart_period=args.restart_period)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = model(ids, mask)

            # Sanity-check loss: must be finite AND within theoretical CE range
            # for our vocab + label smoothing. MPS non-determinism can produce
            # finite-but-numerically-wrong values (e.g. 0.49 when true is 4.87,
            # or -8e28). Retry on any anomaly.
            loss_val = loss.item()
            _V = VOCAB_SIZE
            _s = getattr(model, "label_smoothing", 0.0)
            if _s > 0:
                _pt = 1 - _s + _s / _V
                _po = _s / _V
                import math as _m
                _min_ce = -(_pt * _m.log(_pt) + (_V - 1) * _po * _m.log(_po))
            else:
                _min_ce = 0.0
            _max_ce = math.log(_V) * 5  # wide margin; real CE never exceeds log(V)

            if not torch.isfinite(loss) or loss_val < _min_ce * 0.9 or loss_val > _max_ce:
                print(f"\n  [INSANE LOSS] epoch={epoch} step={global_step} "
                      f"loss={loss_val!r} outside [{_min_ce*0.9:.4f}, {_max_ce:.2f}] "
                      f"— aborting run (MPS numerical corruption).",
                      flush=True)
                sys.exit(2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA weight update
            if ema:
                ema.update(model)

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

        avg_train = epoch_loss / max(epoch_steps, 1)

        # -- Validate (every val_every epochs) ------------------------------
        run_val = ((epoch + 1) % args.val_every == 0) or (epoch + 1 == args.epochs)
        if run_val:
            # Use EMA weights for validation if available
            if ema:
                ema.apply(model)
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
            if ema:
                ema.restore(model)
        else:
            avg_val = best_val_loss  # placeholder for non-val epochs

        dt = time.time() - t0
        # update shared state for signal handlers
        _state["epoch"] = epoch
        _state["step"]  = global_step
        _state["val"]   = avg_val if run_val else _state["val"]
        is_best = run_val and avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            # Save EMA weights as best checkpoint
            if ema:
                ema.apply(model)
            save_checkpoint(model, optimizer, epoch, global_step,
                            avg_val, ckpt_dir / "best.pt")
            if ema:
                ema.restore(model)

        status = "* BEST" if is_best else ""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        val_str = f"val={avg_val:.4f}" if run_val else "val=skip"
        print(f"  [{now}] epoch {epoch+1:03d}/{args.epochs}  "
              f"train={avg_train:.4f}  {val_str}  "
              f"lr={current_lr:.2e}  {dt:.1f}s  {status}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            save_checkpoint(model, optimizer, epoch, global_step, avg_val, path)
            print(f"  [{now}] Saved checkpoint -> {path}")

        # Periodic exact-match probe (catches MPS phantom-loss bugs + tracks real learning)
        if args.em_every > 0 and (epoch + 1) % args.em_every == 0:
            em = quick_exact_match(model, val_p, sample_size=args.em_sample)
            em_history.append((epoch + 1, em, avg_train))
            print(f"  [{now}] exact_match probe: {em*100:.1f}% ({args.em_sample} samples)",
                  flush=True)
            # Save best-EM checkpoint (exact-match is ground truth; val loss lies)
            prev_best_em = max([h[1] for h in em_history[:-1]], default=-1)
            if em > prev_best_em and em > 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                                avg_val, ckpt_dir / "best_em.pt")
                print(f"  [{now}] Saved best_em.pt @ {em*100:.1f}%", flush=True)
            # PHANTOM signal: loss stuck near theoretical floor AND EM=0%.
            # Legitimate training has train >> _min_ce at these checkpoints.
            # Note: _min_ce computed earlier in the training loop.
            phantom_threshold = _min_ce * 1.05  # 5% above theoretical min
            if len(em_history) >= 3:
                recent = em_history[-3:]
                all_zero_em = all(x[1] == 0.0 for x in recent)
                all_near_floor = all(x[2] < phantom_threshold for x in recent)
                if all_zero_em and all_near_floor:
                    print(f"\n  [PHANTOM] EM=0%% AND train_loss ≈ floor ({phantom_threshold:.4f}) "
                          f"for 3 consecutive probes — aborting (numerical corruption).",
                          flush=True)
                    sys.exit(2)

        # Thermal cooldown (prevents GPU overheating on long runs)
        if args.cooldown > 0 and (epoch + 1) % 100 == 0:
            time.sleep(args.cooldown)

    # -- Final checkpoint (use EMA weights if available) --------------------
    if ema:
        ema.apply(model)
    save_checkpoint(model, optimizer, args.epochs - 1, global_step,
                    best_val_loss, ckpt_dir / "final.pt")
    if ema:
        ema.restore(model)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{now}] Training complete")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoints  -> {ckpt_dir}")
    print(f"{'='*60}")

    # -- Sample generation --------------------------------------------------
    print(f"\n  Sample generations (best checkpoint):\n{'-'*60}")
    best_model, _ = load_checkpoint(CKPT_DIR / "best.pt", device)
    test_prompts = [
        # Padded arithmetic format (matches new arith_qa data)
        ("Q: 048 + 037\nA: ", 85, "add"),
        ("Q: 72 - 29\nA: ",   43, "sub"),
        ("Q: 12 * 07\nA: ",   84, "mul"),
        # Traditional format (backward compat)
        ("Q: What is 48 + 37?\nA: ", 85, "add_nl"),
    ]
    for prompt, expected, op in test_prompts:
        response = best_model.generate(prompt, max_new=40)
        raw_answer = response[len(prompt):].split("\n")[0].strip()
        # Attempt to decode reversed answer
        try:
            decoded = int(raw_answer[::-1].lstrip("0") or "0")
        except ValueError:
            decoded = raw_answer
        print(f"  {prompt.strip()}")
        print(f"  -> raw: {raw_answer}  (reversed: {decoded})  expected: {expected}")
        print()


if __name__ == "__main__":
    main()
