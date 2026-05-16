# Bet A — First Training Run (v1)

**Date:** 2026-05-16
**Wall:** 5m6s on RTX 3080
**Model:** 6L × 8H × 256d, char-level (131 vocab), seq_len 1024 → **5.02M params loaded**
**Data:** 936 train / 107 val (90/10 SHA-based split from `qa_records_v1.jsonl`)
**Optimizer:** AdamW, LR 3e-4 → 3e-5 cosine, 500-step warmup, 100 epochs (3,000 steps)
**Driver:** `math_lab/finetune.py` (existing MathGPT lineage)
**Checkpoint:** `bet_a/artifacts/bet_a_v1_6x8_256_char.pt` (60 MB)

---

## Headline

**Training succeeded as a learning curve. Probe failed as a quality test.**

| Outcome | Status |
|---|---|
| Training loop completed | ✅ all 100 epochs, no crash mid-run |
| Val loss trajectory | ✅ 4.13 → 1.997, still descending at epoch 100 |
| Throughput | ✅ ~3s/epoch on 3080 (well under the 200-500 tok/s band CLAUDE.md targets — though that's INFERENCE throughput, separate question) |
| Probe coherence | ❌ **content collapse** — produces grammatical asm-vocab text but cannot condition on the question |
| `####` format anchor | ❌ not emitted in any of 5 probe completions |

The training is honest evidence; the probe failure is **expected at this corpus size and model capacity** and matches the EraGPT v2 finding precisely.

---

## Loss curve

| Epoch | Train | Val | LR | Notes |
|---|---|---|---|---|
| 1 | 4.6184 | 4.1293 | 1.74e-05 | * BEST (random baseline ≈ 4.88 for char-vocab) |
| 5 | 3.1543 | 3.0719 | 8.94e-05 | * BEST |
| 50 | ~2.4 | ~2.4 | 1.5e-04 | mid-warmup descent |
| 91 | 2.0434 | 2.0161 | 3.78e-05 | |
| 100 | **2.0241** | **1.9967** | 3.00e-05 | * BEST (still descending) |

**Generalization gap:** train 2.024 / val 1.997 — val is LOWER than train. Not overfit. Could likely train longer.

---

## Probe results (5 asm-pattern questions)

Full completions in `bet_a/logs/probe_bet_a_v1.log`. Representative output:

> **Q:** When optimizing ARM64 NEON code for performance, what is the difference in benefit between in-order cores like Cortex-A53 and out-of-order cores like Cortex-A72?
>
> **A:** When the CPU instruction a size of a size of a size of a size of a size of a size of a size of the CPU instruction and a simplementation and a size of the CPU instruction and a size of the CPU instruction and a size of the CPU. This approach allows the code the code code be of the CPU implementation and and a simplementation and a simple for a single the CPU is a a simple the…

Vocabulary in the right domain. Grammar fragmentary. **Cannot reproduce specific facts from training data** (e.g. expected reference to 20-25% speedup on A53, the exact data point that appears in 2 training pairs).

All 5 prompts produced similar reductive prose centered on "stack", "size", "instruction", "compiler" — the highest-frequency tokens in the training corpus.

**Hypothesis for the failure:**

1. **Capacity ceiling.** 5M params + char-vocab + free-form prose Q&A is below the capacity threshold for content-conditioned generation. EraGPT v2 saw the same failure mode at Config A (14M) and only crossed the threshold at Config B (113M).
2. **Char-level inefficiency.** Each English word costs 4-12 tokens of model context. The 5M model burns much of its parameter budget learning byte-level English morphology before it can learn structural patterns.
3. **Data too thin / undertrained.** 936 records × 100 epochs is small. Val loss was still descending; 300-500 epochs would likely improve, but probably not break content collapse alone.
4. **Bet A's architecture isn't yet present.** CLAUDE.md's claim is "**model + retrieval scaffolding**". This run trained the model alone, without retrieval. The probe is **not yet a fair test of the thesis** — it's a test of "can a 5M LM solo this." The honest answer is no.

---

## What this run actually proved / disproved

**Proved:**
- The training pipeline works end-to-end (extraction → preprocessor → distillation → training → checkpoint).
- val_loss < random baseline → real learning happened.
- The corpus is well-shaped enough to fit, given enough capacity.

**Disproved:**
- That a 5M model alone, without retrieval, produces useful code commentary at this corpus size.

**Untested (the real Bet A thesis):**
- Whether 5M + retrieval-over-language-specifics meets the 100+ tok/s + useful-output bar.
- Whether throughput holds at 5M on the PDP-11 attention kernel binary (the real CLAUDE.md risk).

---

## Three forward paths

1. **(D, defer-the-test path) Build retrieval scaffolding** — the actual Bet A architecture. The 5M model has captured surface asm vocabulary; pair it with retrieval over ARM ARM + AAPCS64 + (eventual) Track-R syntax tables, and the test becomes whether the COMBINATION produces useful output. This is the design CLAUDE.md called for; we'd be testing the right thing for the first time.

2. **(B, bigger-model path) Step up to Config B (~25M)** — same training data, larger arch. Tests whether capacity is the bottleneck. EraGPT precedent says yes: jumping from 14M to 113M crossed the prose-Q&A threshold. ~25M might be the sweet spot per CLAUDE.md's "throughput cliff at ~25M" note. ~30 min on the 3080.

3. **(L, longer-training path) Train 5M longer** — val was still descending. 100 → 500 epochs at the same arch. Cheap, ~25 min. Will improve val_loss but probably not break content collapse. **Lowest expected ROI.**

My read: **(D) then (B)**.
- (D) is the actual experiment design from CLAUDE.md. Doing (B) before (D) tests the wrong thing.
- (D) starts with building a retrieval index over ARM ARM v8-A reference (we have it via the corpus_plan §3.3 source list), then re-probing the 5M with retrieval-augmented prompts.
- If (D) shows the 5M+retrieval combo works → CLAUDE.md hypothesis validated, scope expansion makes sense.
- If (D) shows it fails → then (B) to test the capacity hypothesis directly.

(L) is essentially yak-shaving; skip unless other paths are blocked.

---

## File manifest

- `bet_a/artifacts/bet_a_v1_6x8_256_char.pt` — best.pt snapshot (60 MB, 5.02M params, val_loss 1.997)
- `bet_a/checkpoints/v1_6x8_256_char/` on Windows — full epoch_*.pt + best.pt + final.pt
- `bet_a/training_data/bet_a_asm_v1_train.jsonl` (936 records)
- `bet_a/training_data/bet_a_asm_v1_val.jsonl` (107 records)
- `bet_a/training_data/bet_a_asm_v1_split.json` (manifest)
- `bet_a/build_training_data.py` (driver)
- `bet_a/logs/bet_a_train_v1.log` (training log)
- `bet_a/logs/probe_bet_a_v1.log` (probe outputs)

---

## Known finetune.py bug

End-of-training "sample generation" block in `math_lab/finetune.py` uses a HARDCODED checkpoint path (`results/checkpoints/best.pt`) instead of the `--ckpt-dir` flag. Causes a `FileNotFoundError` AFTER training completes successfully. Same bug we hit with MathGPT v2 Config-MM on 5/14. **Filed as a follow-up.** Doesn't affect checkpoint quality — training itself completes and best.pt is saved correctly to the `--ckpt-dir`.

---

*End run v1.*
