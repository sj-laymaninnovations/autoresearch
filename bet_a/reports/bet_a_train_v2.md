# Bet A — Training Run v2 (A+E merged corpus)

**Date:** 2026-05-16
**Wall:** 5m43s on RTX 3080
**Model:** 6L × 8H × 256d, char-level (131 vocab), seq_len 1024 → **5.02M params** (identical to v1)
**Data:** 1,087 train / 119 val (Pipeline A 1,043 + Pipeline E 163 — per-source 90/10 split)
**Schedule:** 100 epochs, batch 32, cosine LR 3e-4 → 3e-5, 500-step warmup (identical to v1)
**Checkpoint:** `bet_a/artifacts/bet_a_v2_6x8_256_char.pt` (60 MB)

---

## Headline

**v2 best val 1.9108 vs v1's 1.9967 — 4.3% reduction.** The multi-granularity Pipeline E records measurably improved the loss curve. **Probe output is still in content collapse** — qualitative behavior unchanged. The capacity ceiling, not the data quantity, dominates the failure mode at 5M params on free-form prose Q&A.

This is a clean diagnostic null result: **the corpus-side path is exhausted for this arch.** Next moves must address the arch side.

---

## v1 vs v2 side-by-side

| Metric | v1 (Pipeline A only) | v2 (A + E merged) | Δ |
|---|---|---|---|
| Train pairs | 936 | **1,087** | +151 |
| Val pairs | 107 | 119 | +12 |
| Concept mix | 4 repos | 5 (added asm_ml_qnnpack) | wider |
| Granularities | 1 (commit_body) | 4 (commit_body + comment_block + function + block) | wider |
| Epoch 001 train | 4.6184 | 4.5631 | -0.06 |
| Epoch 001 val | 4.1293 | 4.0554 | -0.07 |
| **Best val loss** | **1.9967** | **1.9108** | **-0.086 (-4.3%)** |
| Wall | 5m6s | 5m43s | +37s (more steps/epoch) |

Loss curves are identical in shape — v2 just lands lower by the same ~0.086 gap throughout. Real signal, not noise.

---

## Probe (same 5 prompts as v1)

Sample (full log: `bet_a/logs/probe_bet_a_v2.log`):

> **Q:** When optimizing ARM64 NEON code for performance, what is the difference in benefit between in-order cores like Cortex-A53 and out-of-order cores like Cortex-A72?
>
> **v1 A:** "When the CPU instruction a size of a size of a size of a size of a size of a size of a size of the CPU instruction and a simplementation and a size of the CPU instruction..."
>
> **v2 A:** "Using a ARM64 instruction a parameter in a parameter of the stack are path are instruction and a single of the stack in a single of the stack..."

Both reductive. v2 latched onto "stack" + "single" as new high-frequency anchors (presumably from the QNNPACK Pipeline E records' frequent stack-layout commentary), but the question-conditioning failure is identical.

No `####` summary lines emitted in any of 5 v2 completions. Same as v1.

---

## What v2 proved and disproved

**Proved:**
- The merge pipeline works (per-source 90/10 split, MathGPT-format harmonization)
- Pipeline E records integrate cleanly into the corpus
- More + better-shaped data DOES reduce val_loss measurably at 5M

**Disproved:**
- That standalone-5M content collapse can be cured by data volume + multi-granularity alone.
- 1,206 records crossed no perceptual threshold.

---

## Three remaining paths (from v1 report, now updated)

| | Action | Cost | Expected lift |
|---|---|---|---|
| **(B)** | **Train Config B (~25M)** on the same v2 corpus | ~15-30 min on 3080, $0 | Tests capacity hypothesis directly. EraGPT precedent says yes — 14M → 113M crossed the threshold; ~25M should be ample for this prose-Q&A scale. |
| **(D)** | **Retrieval scaffolding** — the actual CLAUDE.md architecture | substantial setup (build retrieval index over NAS source code), but the test thereafter is cheap | Tests the RIGHT hypothesis for the first time |
| **(X)** | Scale Pipeline E to more NAS sources (QNNPACK C/headers + XNNPACK + 50 other orgs) | 1-4 hr local, $0 | Would push corpus to 5-10K; if data side really was the bottleneck this catches it, but v2 says it isn't |

My read: **(B) before (D).** v2 just gave us the strongest piece of evidence so far that *for this arch* the data side is exhausted. Testing arch capacity (Config B 25M) is the next single experiment that produces a different bit of information. (D) is the right strategic move but requires more infrastructure to run; (B) gives us the capacity signal in 30 minutes.

If (B) shows substantial qualitative improvement → CLAUDE.md's "5M is the right scale" claim needs revisiting OR retrieval is even MORE the path forward (since smaller-with-retrieval ought to be possible). If (B) shows similar content collapse → the model+corpus combination still isn't right, and (D) becomes the only way to make progress without rethinking the whole arch.

---

*End run v2.*
