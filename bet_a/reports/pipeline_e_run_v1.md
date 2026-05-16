# Pipeline E — Run v1 (QNNPACK aarch* asm, multi-granularity)

**Date:** 2026-05-16
**Wall time:** 23m35s
**Teacher:** `openai/gpt-oss-20b` (local LM Studio, $0 spend)
**Source:** QNNPACK 17 aarch*.S files from NAS via `claudeli` SMB auth (Layman Secrets pattern)
**Extractor:** `bet_a/extract_asm_units.py` — 3 granularities (`comment_block` / `function` / `block`)
**Runner:** `bet_a/pipeline_e.py` — per-granularity prompts

---

## Headline

**163 distilled Q&A pairs from 208 multi-granularity units** out of 17 hand-tuned ARM assembly files. Per Sean's design, the same expert knowledge surfaces at three resolutions (idiom-level comments → function-level architecture → block-level inner loops).

| Granularity | Units extracted | Pairs distilled | Yield |
|---|---|---|---|
| comment_block | 188 | 142 | 76% |
| function | 17 | 20 | 118% (multi-insight) |
| block | 3 | 1 | 33% |
| **Total** | **208** | **163** | **78%** |

---

## Stage funnel

| Stage | Count | Notes |
|---|---|---|
| Units inventoried | 208 | from 17 .S files |
| Preprocessor: too-short comment_block | 26 | < 50 chars after stripping comment leaders |
| Preprocessor: too-long function | 6 | > 600 lines (token-budget guard) |
| Teacher calls | 227 | includes retries |
| Teacher refused | 5 | "body too vague" — legitimate |
| Parse-failed (final, post-retry) | 18 | ~8% of teacher calls |
| Retries used / recovered / failed | 51 / 33 / 18 | retry success rate 65% |
| **Q&A pairs written** | **163** | |
| Format validation failures | 0 | hard fails |
| Pairs with `#### not on own line` warn | 36 (22%) | soft fail, pair still written |
| Pairs with codebase-name leak in question | 1 | "in a library like PyTorch" — general framing, OK |

Median teacher latency 5.3s; p95 13.4s.

---

## Sample pairs (one per granularity)

### comment_block — ARM64 calling convention
> **Q:** What ARM64 calling convention is implied by the function signature of a GEMM kernel that takes several scalar arguments, pointers to matrices, and a pointer to a clamping parameter struct?
>
> **A:** In ARM64, the first eight integer or pointer arguments are passed in registers x0–x7. For a GEMM kernel, the scalar sizes (mr, nr, k) and matrix strides are passed in these registers, while pointers to the input matrices (a, w), output matrix (c), and a struct of clamping parameters are also passed in registers...
> #### ARM64 passes the first eight integer or pointer arguments in registers, using a struct pointer for additional parameters.

### function — FP16 GEMM kernel (8×8 multi-insight)
> **Q:** How does the function efficiently accumulate 8×8 matrix products using ARMv7 NEON FP16 instructions?
>
> **A:** The routine loads eight rows of matrix A into d0–d7 and eight columns of matrix B into q4–q7. For each channel (0–3) it performs a vector multiply-accumulate with the VMLA.F16 instruction... all eight output accumulators (q8–q15) updated in parallel...
> #### The function uses parallel FP16 multiply-accumulate to update all 8×8 outputs in a tight loop

(plus a second pair from the same function on the scale/clamp/quantize sequence — Sean's "1-3 pairs per function for separable insights" rule)

---

## What this run proves

1. **Multi-granularity extraction works** — the same expert source produces 3 distinct training records at different scales.
2. **Per-granularity prompts work** — different rules per granularity produce appropriate output shapes (short instructional for comment_block, multi-insight for function, technique-focused for block).
3. **gpt-oss-20b can handle 12K-character function inputs** in 30s with retry recovery.
4. **The corpus content is technically substantive** — actual ARM mnemonic references (VMLA.F16, VMUL.F16, VMIN.F16, VST1.16, x0-x7 ABI) ground the answers in real ISA knowledge.

## Recurring issues

- **`#### not on own line` (22%):** gpt-oss-20b consistently concatenates the summary line to the last sentence. The v4 Pipeline F prompt had this at 22% on average too; longer Pipeline E prompts didn't make it worse but didn't fix it either. **Probably needs a model with stronger instruction adherence** (claude-sonnet hit 100% on its 3 audit samples) — switching teacher for the final pass is one option, but local-and-free has been the winning constraint.
- **65% retry success rate** (vs Pipeline F v2's 74%) — slightly worse on the longer prompts. Acceptable.

## Combined corpus state

| Source | Pairs |
|---|---|
| Pipeline A (git-history commit-body distillation) | 1,043 |
| **Pipeline E (multi-granularity in-source distillation)** | **+163** |
| **Total** | **1,206** |

---

## Recommended next moves

1. **(M, merge + retrain)** Combine Pipeline A + Pipeline E into one training corpus. Retrain the 5M model on 1,206 pairs to see if the multi-granularity injection moves the needle. **~7 min wall on 3080**, $0.
2. **(X, expand Pipeline E)** Scale to more NAS subdirectories:
   - QNNPACK C/header files (139 files) — opens method/class granularities
   - XNNPACK (TensorFlow Lite mobile asm — likely 100s more files)
   - Other ARM-asm-heavy projects across the 52 orgs
   Likely yields another 1-5K pairs.
3. **(C, switch teacher to claude)** Retry the Pipeline E units that hit the `#### on own line` warning using claude-sonnet via Abacus — proven 100% format compliance on those. Cost: ~$3-5 (spot-check; not full re-run).
4. **(D)** retrieval scaffolding — still the actual Bet A architecture test, deferred.

My read: **(M) first** — quickest signal on whether more data + multi-granularity moves the model. If yes, then (X) at scale. (D) and (C) become tactical follow-ups based on what (M) reveals.

---

*End Pipeline E run v1.*
