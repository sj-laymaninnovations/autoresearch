"""
config.py — Training configuration dataclasses for PDP-11 SLM sweep.

Sweep configs (from ENRICHMENT_PLAN.md):
  Config A: ~7M   params, 140M  Chinchilla tokens  — GATE: must train first
  Config B: ~25M  params, 500M  tokens
  Config C: ~55M  params, 1.1B  tokens
  Config D: ~100M params, 2B    tokens

Architectural constants fixed at training time (cannot change post-train):
  - RoPE theta = 500_000  (confirmed)
  - vocab_size = 100_277   (tiktoken cl100k_base)

DO NOT start B/C/D until Config A produces a clean decreasing loss curve.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Transformer dims
    n_layer:    int   = 6
    n_head:     int   = 4
    d_model:    int   = 128
    d_ff:       int   = 0          # 0 = auto: 4 * d_model
    vocab_size: int   = 100_277    # tiktoken cl100k_base

    # Sequence
    seq_len:    int   = 2048

    # RoPE — LOCKED at 500K per architectural decision; do not change post-train
    rope_theta: float = 500_000.0

    # Dropout
    dropout:    float = 0.0        # 0 for large runs; small runs can use 0.1

    # Bias in Linear layers
    bias:       bool  = False

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, \
            f"d_model {self.d_model} must be divisible by n_head {self.n_head}"
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def approx_params(self) -> int:
        """Back-of-envelope parameter count (no KV cache, no bias)."""
        V, L, D, F = self.vocab_size, self.n_layer, self.d_model, self.d_ff
        embed    = V * D
        attn     = L * (3 * D * D + D * D)   # QKV + out proj
        ffn      = L * (2 * D * F)            # gate + up + down (approx 2-layer)
        lm_head  = V * D                      # weight-tied to embedding usually
        return embed + attn + ffn + lm_head


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Run identity
    name:             str   = "config_a"
    config_letter:    str   = "A"

    # Optimizer
    lr:               float = 3e-4
    weight_decay:     float = 0.1
    beta1:            float = 0.9
    beta2:            float = 0.95
    grad_clip:        float = 1.0

    # Schedule
    warmup_steps:     int   = 5_000
    max_steps:        int   = 50_000      # override per run
    lr_decay_to:      float = 0.1        # fraction of max lr at end of cosine

    # Batch (effective = batch_size * grad_accum_steps)
    batch_size:       int   = 32         # micro-batch (sequences per step)
    grad_accum_steps: int   = 1          # set > 1 if GPU memory limited

    # Token budget (Chinchilla target)
    target_tokens:    int   = 140_000_000

    # Logging / saving
    log_every:        int   = 100
    eval_every:       int   = 500
    save_every:       int   = 1_000
    keep_checkpoints: int   = 3          # keep last N checkpoints

    # Data
    data_dir:         str   = "../sources/fineweb-edu"
    val_fraction:     float = 0.005      # ~0.5% held out as val

    # Precision
    dtype:            str   = "bfloat16" # bfloat16 | float16 | float32

    # Output
    out_dir:          str   = "checkpoints/config_a"

    @property
    def effective_batch_tokens(self) -> int:
        """Tokens consumed per optimizer step."""
        # seq_len comes from model config — caller must supply
        raise NotImplementedError("Use TrainRun.tokens_per_step()")


# ---------------------------------------------------------------------------
# Combined run config
# ---------------------------------------------------------------------------

@dataclass
class TrainRun:
    model:  ModelConfig
    train:  TrainConfig

    def tokens_per_step(self) -> int:
        return self.model.seq_len * self.train.batch_size * self.train.grad_accum_steps

    def steps_for_budget(self) -> int:
        return self.train.target_tokens // self.tokens_per_step()

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "train": asdict(self.train)}

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainRun":
        with open(path) as f:
            d = json.load(f)
        return cls(
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**d["train"]),
        )


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def get_config(letter: str) -> TrainRun:
    """
    Return the canonical TrainRun for the given config letter.

    Config A is the gate. Do not use B/C/D until A trains cleanly.
    """
    letter = letter.upper()

    if letter == "A":
        # ~7M params  |  140M Chinchilla tokens  |  GATE RUN
        model = ModelConfig(
            n_layer    = 6,
            n_head     = 4,
            d_model    = 128,
            seq_len    = 2048,
            rope_theta = 500_000.0,
            dropout    = 0.0,
        )
        train = TrainConfig(
            name             = "config_a",
            config_letter    = "A",
            lr               = 3e-4,
            warmup_steps     = 5_000,
            batch_size       = 32,
            grad_accum_steps = 1,
            target_tokens    = 140_000_000,
            log_every        = 100,
            eval_every       = 500,
            save_every       = 1_000,
            data_dir         = "../sources/fineweb-edu",
            out_dir          = "checkpoints/config_a",
            dtype            = "bfloat16",
        )

    elif letter == "B":
        # ~25M params  |  500M tokens  |  START ONLY AFTER A CONFIRMS CAPACITY LIMIT
        model = ModelConfig(
            n_layer    = 8,
            n_head     = 8,
            d_model    = 256,
            seq_len    = 2048,
            rope_theta = 500_000.0,
            dropout    = 0.0,
        )
        train = TrainConfig(
            name             = "config_b",
            config_letter    = "B",
            lr               = 3e-4,
            warmup_steps     = 5_000,
            batch_size       = 32,
            grad_accum_steps = 4,
            target_tokens    = 500_000_000,
            log_every        = 100,
            eval_every       = 500,
            save_every       = 1_000,
            data_dir         = "../sources/fineweb-edu",
            out_dir          = "checkpoints/config_b",
            dtype            = "bfloat16",
        )

    elif letter == "C":
        # ~55M params  |  1.1B tokens  |  Generalist + coding floor candidate
        model = ModelConfig(
            n_layer    = 10,
            n_head     = 8,
            d_model    = 384,
            seq_len    = 2048,
            rope_theta = 500_000.0,
            dropout    = 0.0,
        )
        train = TrainConfig(
            name             = "config_c",
            config_letter    = "C",
            lr               = 2e-4,
            warmup_steps     = 5_000,
            batch_size       = 32,
            grad_accum_steps = 8,
            target_tokens    = 1_100_000_000,
            log_every        = 100,
            eval_every       = 500,
            save_every       = 1_000,
            data_dir         = "../sources/fineweb-edu",
            out_dir          = "checkpoints/config_c",
            dtype            = "bfloat16",
        )

    elif letter == "D":
        # ~100M params  |  2B tokens  |  Tool use candidate
        # NOTE: requires CORD-19 + arXiv PDFs — do not start until Tier 1 complete
        model = ModelConfig(
            n_layer    = 12,
            n_head     = 8,
            d_model    = 512,
            seq_len    = 65_536,   # 64K context window
            rope_theta = 500_000.0,
            dropout    = 0.0,
        )
        train = TrainConfig(
            name             = "config_d",
            config_letter    = "D",
            lr               = 1e-4,
            warmup_steps     = 5_000,
            batch_size       = 4,           # 64K seqs are large
            grad_accum_steps = 32,
            target_tokens    = 2_000_000_000,
            log_every        = 100,
            eval_every       = 500,
            save_every       = 1_000,
            data_dir         = "../sources/fineweb-edu",
            out_dir          = "checkpoints/config_d",
            dtype            = "bfloat16",
        )

    else:
        raise ValueError(f"Unknown config letter: {letter!r}. Choose A, B, C, or D.")

    run = TrainRun(model=model, train=train)
    # Auto-compute max_steps from token budget
    run.train.max_steps = run.steps_for_budget()
    return run


# ---------------------------------------------------------------------------
# CLI: python config.py A   →  prints config summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    letter = sys.argv[1] if len(sys.argv) > 1 else "A"
    run = get_config(letter)
    m, t = run.model, run.train
    params = m.approx_params()
    tps = run.tokens_per_step()
    steps = run.steps_for_budget()
    print(f"\nConfig {letter}")
    print(f"  Architecture  : {m.n_layer}L × {m.n_head}H × {m.d_model}D  (seq {m.seq_len})")
    print(f"  RoPE theta    : {m.rope_theta:,.0f}")
    print(f"  ~Params       : {params/1e6:.1f}M")
    print(f"  Batch tokens  : {tps:,} / step")
    print(f"  Token budget  : {t.target_tokens/1e6:.0f}M")
    print(f"  ~Steps        : {steps:,}")
    print(f"  LR            : {t.lr}  (warmup {t.warmup_steps:,} steps)")
    print(f"  Out dir       : {t.out_dir}")
    print()
