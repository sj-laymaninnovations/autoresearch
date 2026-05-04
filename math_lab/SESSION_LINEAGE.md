# Math model from-scratch — session lineage (2026-04-22 → 2026-04-25)

Multi-day narrative of what was tried, what worked, what didn't, and where we are.
Project goal: train a small language model **from scratch** on math (arithmetic
first, then algebra) under **PDP-11-inspired size constraints** (5M params soft,
50M relaxable, 20 KB hard limit on the assembly kernel binary).

---

## The starting point

- A custom char-level transformer (`MathGPT`) with a 131-token ASCII vocab.
- Custom training stack (`math_lab/finetune.py` + `autoresearch_canonical.py`)
  designed to mirror what a PDP-11-class system might run if it had modern hw.
- A canonical algebra curriculum split into 5 stages (1-step equations →
  exponents/logs/polynomials).
- An earlier "proof-of-life" sweep (1-100 × 1-100 layer × heads) that found:
  - Below ~2×4 (layers × heads), models couldn't even show task-relevant
    structure after 1 epoch.
  - Above ~2×6, width and depth became substitutable.
  - Most provocatively: **delayed mastery** at ~2000 epochs from configs that
    showed no early signal — the user's first encounter with grokking-style
    dynamics.

---

## Day 1 — 2026-04-23: arithmetic exploration

### The setup
- Switched focus to arithmetic from scratch.
- Built canonical-format data with `arith_qa.py` (zero-padded operands,
  reversed-digit answers per Zaremba & Sutskever 2014, Lee et al. 2023).
- Validation set: 143 pairs across tier 1+2 (1-digit and 2-digit operands).

### The runs (15 in 24 hours)

| # | config | data | epochs | peak EM | notes |
|---|---|---|---|---|---|
| 1 | midsize 2×4×64 (120K) | tier1 plain | 3000 | 23.9% headline | ~15% honest — mostly constant-collapse luck |
| 2 | midsize 2×4×64 | tier1+2 rev+`####` | 3000 | 0% | format collapse (model output `####` only) |
| 3 | midsize 2×4×64 | tier1+2 rev | 3000 | 0% | empty-string collapse (model output `\n` only) |
| 4 | midsize 2×4×64 | tier2 fwd-padded | 3000 | 0% | empty-string collapse |
| 5 | bigger 3×4×128 (640K) | tier1+2 rev | ~3hr | 0% | phantom abort killed; partial digit output preserved |
| 6 | 4×6×192 (1.77M) | tier1+2 rev | 3000 | 0% (1.4% peak) | over-trained past peak |
| 7 | 6×8×128 (1.18M) | tier1+2 rev | 3000 | 0% (1.4% peak) | deep+narrow unstable |
| **8** | **4×6×192** | tier1+2 rev | **800** | **11.9% @ ep600** | **🏆 first real multi-digit** |
| 9 | 6×8×128 | tier1+2 rev | 800 | 0.7% | shape wrong for this task |
| 10 | 4×6×192 Phase 2 | tier1+2 rev `lr_min=1e-4` | 1500 | 9.1% | lr floor prevented collapse, but lowered peak |
| 11 | 4×6×192 Phase 3 | tier1-4 rev (3090) | 1500 | 2.2% | tier-3+4 added; tier-1+2 regressed |
| 12 | 6×8×192 Phase 5 (2.7M) | tier1-4 rev | 800 | 0.7% | scale-up collapsed faster than smaller config |

### Day 1 findings

1. **The 23.9% headline on simple tier-1 was misleading.** Per-op breakdown
   showed mostly constant-collapse passes (e.g. sub_1d output `'1'` for
   everything, hit `4-3=1` by luck). Honest tier-1 was ~15%.
2. **Reversed + zero-padded format made things WORSE for the small model.**
   At 120K params, the model collapsed to whitespace-only output rather than
   learning the format-content joint distribution. Three configurations
   (#2, #3, #4) all hit pure empty-output collapse.
3. **Wider/shallower beat deeper/narrower** at the same param budget.
   `4×6×192` (1.77M, 4L 6H wide) beat `6×8×128` (1.18M, 6L 8H narrow) by 17×.
   Likely because head_dim=32 (wider) > head_dim=16 (narrower).
4. **Short training schedule is critical.** Run #8 (800 epochs) hit 11.9%;
   the same architecture run for 3000 epochs (#6) collapsed to 0%. The cosine
   schedule decaying to lr_min=1e-5 over 3000 epochs *unlearns* the peak
   capability that was reached around epoch 400-700.
5. **`val_loss` does NOT track EM** in this regime. The lowest-val-loss
   checkpoint was usually a degenerate "output common tokens" minimum, not the
   peak EM checkpoint. This invalidated default `best.pt` selection — needed
   post-hoc full-checkpoint EM scans to find true peaks.
6. **6×8×192 (Phase 5 scale-up) was WORSE than 4×6×192.** Not capacity-bound
   — adding tier-3+4 data destabilized training; bigger model made the
   instability worse not better. Same data scaled poorly.

---

## Day 2 — 2026-04-24: capacity scale-up, curriculum, embedding transfer, CoT

### Curriculum experiments
- Warm-started 4×6×192 from its tier-1+2 peak checkpoint, continued on
  tier-1-4 data. Got 2.5% → 2.6% on tier-1-4 with the first non-zero tier-3+4
  signal (5/480 = 1%). Far below tier-1+2 init (11.9%), so curriculum
  partially worked but eroded base.
- **Discovered a bug:** `finetune.py` resumed the epoch counter from the
  warm-start checkpoint, so `epochs=800` with init from ep600 actually only
  ran 200 epochs. Patched with `--reset-epoch-counter` flag.
- After fix, "curriculum full" (1400 ep config = real 800 continuation)
  reached tier-1+2 9% peak and tier-1-4 2.6%. Modest, not transformative.

### Tier-1+2 mastery attempt (high-volume data)
- Built `t12master` dataset: exhaustive tier-1 (221 pairs, full space) +
  high-volume tier-2 (2507 train + **320 NOVEL-pair holdout val**, no
  commutative-twin leakage).
- Trained 6×8×256 (~4.8M params, fills 5M budget) on it.
- Result: **first honest tier-2 generalization. 12.5% on novel-pair div_2d**
  (10/80) — real computations, distinct correct quotients across problems.
- Overall only 4.0% — data imbalance (14× more tier-2 than tier-1) starved
  tier-1.

### Embedding transfer from large models (the "what would Qwen do" question)
- Extracted digit embeddings from three open-source LLMs: **Gemma 4 (E4B-it),
  Qwen 3.6 (35B-A3B), gpt-oss-20b**.
- All three independently encode digits as a **number-line manifold**:
  adjacent digits (e.g. 4↔5) cluster more tightly than distant pairs (0↔9).
  Replicated across BPE tokenizers, MoE/dense architectures, and OpenAI/
  Google/Alibaba training data — strong evidence the structure is real and
  convergent, not an artifact of any single model.
- **Math-aware PCA** (fit on ~240 math tokens per model) preserved this
  structure cleanly in 256d. Random-vocab PCA flattened it.
- Used the math-PCA 256d digit/operator vectors as initialization for our
  6×8×256 model.

### Init A/B/C results
| init | overall peak EM | tier-2 novel peak |
|---|---|---|
| Random (baseline) | 14/350 (4.0%) | 14/320 (4.4%) |
| Qwen-only init | 6/350 (1.7%) | 2/320 (0.6%) — **hurt** |
| Three-model **Procrustes-aligned blend** | 24/350 (**6.9%**) | 20/320 (**6.2%**) — modest help |

- Single-model transfer hurt: model-specific quirks acted as noise.
- Procrustes-aligned blend across three models cancelled out idiosyncrasies
  and gave a clean ~70% relative improvement on novel-pair tier-2.

### Chain-of-Thought (the breakthrough)
- Wrote `cot_teacher.py` for LM Studio teacher-LLM CoT generation (planned to
  use OSS-20 from a remote LM Link node).
- Remote node disconnected partway; only 577/2698 succeeded. Local LM Studio
  MLX backend was broken (libpython mismatch from a recent update), so couldn't
  fall back to local OSS-20.
- Pivoted to **rule-based CoT** (`cot_rule.py`): deterministic algorithmic
  decomposition for each operation type:
  - add: place-value (`38+30=68; 68+2=70`)
  - sub: place-value
  - mul: distributive (`82*37 = 82*(30+7) = 82*30 + 82*7 = 2460+574 = 3034`)
  - div: factor verification (`12*24=288; 288/12=24`)
- Trained 6×8×256 on rule-CoT-augmented t12master data.

### CoT result
| ckpt | overall | tier-1 | tier-2 novel |
|---|---|---|---|
| ep50 | 65/350 (18.6%) | 33% | 17.2% |
| **ep150** | **99/350 (28.3%)** ★ | 30% | **90/320 (28.1%)** ★ |
| ep200 | declining | | |
| ep1000 | 0% | | overtrained collapse |

**6× improvement over baseline; 4× improvement over blend-init.**
Same peak-then-collapse pattern as before — peak at ep150, then unlearned by
the cosine schedule going to near-zero LR by ep1000.

### Pre-commit threshold (gambling-fallacy guard)
- The user identified anchoring on an early-promising ~2000-epoch run from
  the prior week and a "one more adjustment away" trap.
- Set explicit threshold before the CoT experiment: ≥25% peak EM = continue
  iterating. <18% = step back and reconsider format/scale/direction.
- CoT result was 28.3%. **Threshold cleared — instinct was correct.**

---

## Day 3 — 2026-04-25: validation + CoT + blend combo

### Currently running: CoT + blend-init combo
- Same 6×8×256 + rule-based CoT data + Procrustes-aligned blend embedding
  init, capped at 400 epochs to avoid the post-peak collapse.
- Tests whether the two interventions (data and embedding priors) stack
  additively. Result expected within 30 min of writing.

---

## Where we are now

### Solid findings
1. **Number-line manifold for digits is universal across LLMs** (Gemma, Qwen,
   OSS-20). The algorithmic geometry is replicable — they all converged.
2. **Algorithm transfer (CoT) > weight transfer (embeddings).** Rule-based
   CoT gave 6×; embedding transfer gave ~70% (modest).
3. **Wider/shallower architecture wins at this param budget for arithmetic.**
4. **Short training is non-negotiable.** Cosine-to-zero LR over many epochs
   destroys the peak. Either short runs (300-500 ep) or higher lr_min, or
   best-by-EM checkpointing instead of best-by-val_loss.
5. **Procrustes-aligned blending across multiple models is more robust than
   single-model transfer** for embedding priors.
6. **`val_loss` is unreliable** in our regime — degenerate output minima
   produce low loss without learning. Always pair with EM probe.

### Best result so far
- **6×8×256 + rule-based CoT, 28.3% EM peak @ ep150, 28.1% on novel-pair
  tier-2 holdout.** First honest 2-digit arithmetic generalization at this
  scale.

### Code/infra deltas committed (today, GitHub)
- `--reset-epoch-counter` flag for warm-start scenarios (was a silent bug:
  `epochs=800` with warm-start from ep600 only ran 200 epochs).
- `--init-embeddings` flag accepting external `.npy` priors with mapping
  to char-vocab IDs.
- `solution_cot` field support in dataset (preferred over `solution`).
- `cot_teacher.py` (LLM-via-LM-Studio CoT generation pipeline).
- `cot_rule.py` (deterministic rule-based CoT generator — fallback that
  ended up being the workhorse).
- Three new training configs (baseline / qwen-init / blend-init).

### Bug fixes / hygiene
- Fixed the `git remote` triple-push-URL config (was pushing to autoresearch
  AND diffusionresearch repos).
- Added `.claude/scheduled_tasks.lock` to `.gitignore`.

### Open questions
1. Does CoT+blend stack? (running now, decision in ~30 min)
2. Does the algorithmic learning generalize beyond tier 1+2? Need to run the
   full 600-problem tier 1-10 quiz on the CoT ep150 checkpoint.
3. Does the same CoT recipe transfer to algebra? Next big experiment after
   arithmetic stabilizes.
4. Can we hold the peak for longer? Try `lr_min=5e-5` or `lr_min=1e-4` at
   shorter total epochs (e.g. 400 with floor).
5. Can we recover the LM Studio teacher pipeline? Local MLX backend is
   broken (python lib mismatch). Either reinstall LM Studio or use OSS-20
   directly via mlx-lm (Python module).

### Next planned arithmetic experiments
- CoT-only with `lr_min=1e-4`, 500 epochs — try to hold peak longer.
- CoT + blend-init combo (running now).
- Comprehensive 600-problem (tier 1-10) quiz on CoT-ep150 checkpoint to see
  out-of-distribution capability.
- If algebra is next: extend `cot_rule.py` to algebra transformations
  (combine-like-terms, distribute, isolate), apply same recipe.

---

## Themes / takeaways for the broader research

1. **Procedural decomposition (CoT) is the highest-leverage small-model
   intervention** for any subject with clean algorithmic structure.
2. **Borrowing structure from big models is real but second-order.** The
   number-line manifold is a beautiful finding that gave us a *frame* for
   thinking about what to inject — but the literal weight transfer was a
   small-magnitude win at small scale. The deeper lesson was:
   *understanding what big models encode tells you what structure to
   bake into your training data,* not what to bake into your weights.
3. **Pre-commit thresholds protect against gambler's-fallacy** in research.
   Setting "we step back if <18%" before running CoT meant we'd have actually
   stopped if it hadn't worked, instead of "one more adjustment"-ing into
   another week.
4. **`val_loss` is a treacherous signal** when degenerate outputs (empty,
   constant, format-only) produce low loss. Always validate with task-level
   metrics (EM) on held-out data, never trust loss alone.
5. **The "one more adjustment" feeling is the trap.** Recognized explicitly
   today; pre-commit thresholds are the antidote.
