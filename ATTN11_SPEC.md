# ATTN11 — PDP-11-Inspired Attention Kernel: Implementation Spec

A modern reimplementation of a PDP-11-class machine-learning runtime, ported
to x86-64 (Windows + Linux) and Apple Silicon (training only). Reproduces
the spirit of original PDP-11 MACRO-11 numerical kernels (fixed-point
math, integer quantized weights, byte-pair tokenization) with modern
training tooling layered on top.

This spec covers what is **actually implemented** as of 2026-04-25.

---

## 1. Project goals & size budgets

| Constraint | Limit | Enforcement |
|---|---|---|
| Kernel `.text` section (assembly binary) | **20 KB hard cap** | [`kernel/Makefile`](kernel/Makefile) `SIZE_LIMIT=20480`, checked by `kernel/check_size.py` after every build |
| Model parameters | **5M soft / 50M emergency** | Convention; not auto-enforced |
| PDP-11 era reference (kernel binary) | ~7 KB | Original PDP-11 MACRO-11 was 7 KB; modern x86 kernel given budget for instruction width and ABI overhead |

Motivation: prove a small char-level transformer trained from scratch can
absorb math (arithmetic first, algebra next) under hardware-era-relevant
size constraints, while taking advantage of modern training infrastructure.

---

## 2. Vocabulary & tokenization

**Char-level, 131 tokens.** Defined in [`math_lab/finetune.py`](math_lab/finetune.py)
to match `kernel/qsparser.asm`:

```
PAD_ID = 0      (padding)
BOS_ID = 1      (beginning of sequence)
EOS_ID = 2      (end of sequence)
UNK_ID = 3      (unknown / non-ASCII char)
OFFSET = 4      (token_id = ascii_byte + 4 for ASCII bytes 0–126)

VOCAB_SIZE = 131  (4 special + 127 ASCII)
```

**Encoding rule:** `char_encode(text) → [BOS, ord(c0)+4, ord(c1)+4, …, EOS]`,
with any byte ≥127 replaced by `UNK_ID`.

**Why char-level:**
- No external tokenizer or vocabulary file → reproducible under PDP-11-style
  cold-boot constraints.
- Simple to mirror in the assembly parser (`qsparser.asm`).
- Each digit, operator, and space is one token — natural for arithmetic.

**Trade-off:** sequences are longer than BPE would produce. For the math
domain this is acceptable; `seq_len=128–256` covers all current data.

---

## 3. Model architecture (`MathGPT`)

Defined in [`math_lab/finetune.py`](math_lab/finetune.py) as `MathGPT`. A
standard decoder-only transformer with intentionally-modern conventions:

```
@dataclass GPTConfig:
    vocab_size  = 131       (fixed)
    seq_len     = 128/256   (per-experiment)
    n_layer     = experimental
    n_head      = experimental
    n_embd      = experimental
    dropout     = 0.0–0.15
```

**Block structure (per layer):**
```
x = x + CausalSelfAttention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

| Component | Choice | Notes |
|---|---|---|
| Normalization | **RMSNorm** | matches Llama / Gemma / Qwen convention; cheaper than LayerNorm |
| Attention | Multi-head scaled dot-product | `F.scaled_dot_product_attention(is_causal=True)` (PyTorch flash-attn fallback) |
| Attention bias | none | `c_attn`, `c_proj` are `bias=False` |
| MLP | 2-layer, 4× expansion, **GELU** | `bias=False` throughout |
| Position | **learned absolute** | `nn.Embedding(seq_len, n_embd)` |
| Embedding | tied to lm_head | `lm_head.weight = tok_emb.weight` |
| Init | `normal(0, 0.02)` | for Linear and Embedding |

**Configurations explored (tier-1+2 arithmetic):**

| label | n_layer × n_head × n_embd | params | role |
|---|---|---|---|
| micro | 1×4×32 | ~25K | proof-of-life floor |
| midsize | 2×4×64 | ~120K | early baseline |
| midsize-wide | 4×6×192 | ~1.77M | first multi-digit success (11.9% peak) |
| big | 6×8×128 | ~1.18M | underperformed; head_dim=16 too narrow |
| **mastery** | **6×8×256** | **~4.8M** | current best (~49% peak with CoT+blend init) |

---

## 4. Precision

Three precision regimes — training-time and storage-time are deliberately
separate, mirroring the PDP-11 split between scratch compute and ROM
storage.

### 4.1 Training compute (PyTorch)

Defined in [`math_lab/finetune.py`](math_lab/finetune.py):

| device | dtype | autocast | reason |
|---|---|---|---|
| CUDA | bf16 (fallback fp16) | enabled | standard modern training; bf16 stable, fp16 acceptable |
| **MPS (Apple Silicon)** | **fp32** | **disabled** | **MPS+fp16 produced silent weight corruption** producing fake-low losses on early experiments. Locked to fp32 across the board; speed cost is acceptable at our model sizes. |
| CPU | fp32 | disabled | reliable fallback when MPS phantom-aborts |

### 4.2 Kernel compute format (assembly)

Defined in [`kernel/kernel.h`](kernel/kernel.h):

| name | C type | Q-format | scale | range |
|---|---|---|---|---|
| **q16_t** | `int32_t` | **Q8.16** | 1.0 = 65536 | ±32768.0, resolution ≈ 1/65536 |
| **q8_t** | `int16_t` | **Q8.8** | 1.0 = 256 | ±128.0, resolution = 1/256 |

**Convention:** all kernel arithmetic is Q8.16 (`q16_t`). Q8.8 (`q8_t`) is
storage-only — used for quantized weight matrices on disk and KV cache
entries to halve memory footprint.

### 4.3 Float ↔ integer bridge (`turboquant.asm`)

| direction | function | use |
|---|---|---|
| float32 → Q8.8 | `tq_f32_to_q8` | trained weights → kernel storage |
| Q8.8 → Q8.16 | `tq_q8_to_q16` | storage → compute (with scale shift, lossy if outside range) |
| Q8.8 → Q8.16 (lossless) | `tq_q8_to_q16_exact` | storage → compute (`<< 8`, exact) |
| Q8.16 → Q8.8 | `tq_q16_to_q8` | compute → storage (with rounding/saturation) |

Float→integer paths use SSE2 for throughput. Integer↔integer paths are
scalar (cheap, predictable, fits the size budget).

---

## 5. Kernel components (`kernel/`)

Hard-coded modules totaling ≤ 20 KB `.text`. All Win64 calling convention
(System V shimmed via `io_win64.asm`).

| file | role | PDP-11 lineage |
|---|---|---|
| [`fxmath.asm`](kernel/fxmath.asm) | Q8.16 scalar primitives (`fxmul`, `fxdiv`, `fxabs`, `fxclamp`) | port of FXMATH.MAC |
| [`vecop.asm`](kernel/vecop.asm) | vector ops (`vdot`, `vadd`, `vsub`, `vscl`, `vmax`, `vcpy`, `vclr`, `vsadd`) | port of VECOP.MAC |
| [`matop.asm`](kernel/matop.asm) | matrix-vector ops (`mvmul`, `mvadd`) | port of MATOP.MAC |
| [`actfn.asm`](kernel/actfn.asm) | activation functions | new |
| [`attn_kernel.asm`](kernel/attn_kernel.asm) | attention head computation | new |
| [`qsparser.asm`](kernel/qsparser.asm) | char-level tokenizer matching the Python encoder | new |
| [`qsparser_utf8.asm`](kernel/qsparser_utf8.asm) | optional UTF-8 variant | new |
| [`kvcache.asm`](kernel/kvcache.asm) | KV cache management for inference | new |
| [`turboquant.asm`](kernel/turboquant.asm) | float ↔ Q8.8 quantization | new (no PDP-11 equivalent) |
| [`io_win64.asm`](kernel/io_win64.asm) | platform shim for ABI | new |

**Build:** `make` produces `kernel.dll` (Win) or `kernel.so` (Linux). The
`make size` target verifies the 20 KB cap after every build. Apple Silicon
training does not need the kernel — Python+PyTorch suffices for our scale.

---

## 6. Training pipeline

### 6.1 Data format

JSONL, one record per line. Used by both training and evaluation:

```json
{
  "problem":      "77 * 91",
  "solution":     "7007",
  "answer":       "7007",
  "answer_real":  7007,
  "concept":      "mul_2d",
  "stage":        2,
  "level":        2,
  "solution_cot": "77*91 = 77*(100-9)\n= 7700-693\n= 7007\n#### 7007"
}
```

**Fields:**
- `problem` (required) — input text after the `Q: ` prefix
- `solution` (required) — full target sequence the model is trained to produce
- `answer` — for evaluation; what the EM check compares against (string match after normalization)
- `answer_real` — convenience integer form
- `concept`, `stage`, `level` — taxonomy tags for per-skill analysis
- `solution_cot` (optional) — preferred over `solution` if present (set 2026-04-24); enables Chain-of-Thought training without changing the dataset schema

**Prompt template:**
```
Q: {problem}\nA: {solution}\n
                ^^^^^^^^^^^^
                loss-masked: model only learns to produce solution tokens
```

### 6.2 Training driver

[`math_lab/finetune.py`](math_lab/finetune.py) — single-file training script:

| feature | flag |
|---|---|
| Cosine LR schedule | `--lr` peak, `--lr-min` floor, `--warmup` steps |
| Optional warm-restart | `--restart-period` (Loshchilov & Hutter 2017) |
| Weight decay | `--weight-decay` (default 1.0 — Power et al. 2022 grokking recipe friendly) |
| Label smoothing | `--label-smoothing` (default 0.1) |
| EMA (optional) | `--ema-decay` |
| Best checkpoint (val loss) | automatic `best.pt` |
| Best-EM checkpoint | when `--em-every > 0`, also tracks `best_em.pt` |
| Phantom abort | when `--em-every > 0`: aborts training if 3 consecutive EM probes are 0% AND train_loss is near theoretical floor (catches MPS fp16-style corruption); set `--em-every 0` to disable for grokking experiments |
| Warm-start from checkpoint | `--ckpt path` (loads weights only) |
| Reset epoch counter on warm-start | `--reset-epoch-counter` (added 2026-04-24; see SESSION_LINEAGE.md for the bug story) |
| Embedding init from external | `--init-embeddings file.npy` + `--init-embeddings-labels file.txt` (added 2026-04-24; for transferring digit/operator priors from large LLMs) |

### 6.3 Experiment runner

[`math_lab/autoresearch_canonical.py`](math_lab/autoresearch_canonical.py) —
takes a YAML config, launches training, runs `eval_canonical.py` on the best
checkpoint, appends one row to `results/canonical_experiments.tsv`. Designed
for short feedback loops — every experiment is logged so direct comparison
across runs is trivial.

### 6.4 Skill assembly line (in development)

[`math_lab/skill_runner.py`](math_lab/skill_runner.py) — atomic-skill
training driver per [`math_lab/results/assembly_line_skill_decomp.md`](math_lab/results/assembly_line_skill_decomp.md):

- Single-skill data generated by [`math_lab/datagen/skill_data.py`](math_lab/datagen/skill_data.py)
- Constant-LR regime (Power et al. 2022 grokking config: `lr=1e-3`,
  `weight_decay=1.0`, `dropout=0`, no label smoothing, no phantom abort)
- 4-class classification per (skill, seed): GENERALIZED / PARTIAL /
  MEMORIZED / FAILED based on novel-pair holdout EM
- Persistent registry at `results/skill_registry.tsv`

---

## 7. Inference & evaluation

[`math_lab/eval_canonical.py`](math_lab/eval_canonical.py):
- Loads any saved checkpoint (CPU/MPS/CUDA)
- Greedy-decodes against a val JSONL
- Extracts answer via regex `####\s*(N)` first, falls back to "last numeric
  token" — handles both plain-answer and CoT-format outputs
- Per-stage breakdown for the canonical 5-stage algebra curriculum
- Per-concept breakdown for arithmetic
- Outputs `eval.json` alongside the checkpoint

---

## 8. What's experimental vs. settled

### Settled

- Char-level 131-token vocab + OFFSET=4
- `MathGPT` block topology (RMSNorm + flash-attn-when-available + 4× GELU MLP + tied embedding)
- Q8.16 / Q8.8 fixed-point split for kernel
- 20 KB kernel hard cap
- MPS = fp32-only training
- Constant-LR + weight_decay=1.0 for grokking-style atomic-skill training
- CoT-augmented training data via `solution_cot` field

### Open / per-experiment

- `n_layer`, `n_head`, `n_embd` choices
- LR schedule (cosine vs constant; floor value)
- Embedding init (random vs Gemma/Qwen/OSS-20 PCA-projected vs blended)
- Data composition (single-CoT vs multi-variant CoT vs vanilla)
- Curriculum order (single-skill vs mixed)
- Weight merging across skill checkpoints (Phase 4 of assembly line — not yet attempted)

---

## 9. Notable invariants

- **Validation always uses novel-pair holdout** — operand pairs in val never
  appear in train; commutative twins also held out for symmetric ops. Without
  this, EM measures memorization, not generalization. (Lesson learned the
  hard way mid-project; codified in the assembly-line spec.)
- **Best checkpoint is best-EM where feasible**, not best-val-loss.
  Degenerate-output minima can produce lower val_loss than the actual peak
  capability; trust EM on held-out data.
- **`em_every=0` for grokking experiments** so the phantom-abort detector
  doesn't kill long-run grokking signatures (which look exactly like
  phantom-abort cases for the first thousands of epochs).
- **No auto-promotion across tiers** — operator must reach ≥80% novel-pair
  EM at tier-1 AND tier-2 before tier-3 work begins for that operator (the
  compounding-weakness guard in
  [`assembly_line_skill_decomp.md`](math_lab/results/assembly_line_skill_decomp.md)).

---

## 10. References

- Tokenizer parity: [`kernel/qsparser.asm`](kernel/qsparser.asm) ↔
  [`math_lab/finetune.py`](math_lab/finetune.py) `char_encode`/`char_decode`
- Architecture spec: this file + [`math_lab/finetune.py`](math_lab/finetune.py) `MathGPT` class
- Kernel ABI: [`kernel/kernel.h`](kernel/kernel.h)
- Build + size enforcement: [`kernel/Makefile`](kernel/Makefile),
  [`kernel/check_size.py`](kernel/check_size.py)
- Session research narrative: [`math_lab/SESSION_LINEAGE.md`](math_lab/SESSION_LINEAGE.md)
- Assembly-line research program: [`math_lab/results/assembly_line_skill_decomp.md`](math_lab/results/assembly_line_skill_decomp.md)
