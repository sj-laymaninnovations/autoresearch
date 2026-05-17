# Bet A — v3 4-way comparison: capacity vs balance vs corpus

**Date:** 2026-05-16
**Parallel runs:** 3080 (Config B 25M) + 3090 (5M balanced) — ~16 min wall each
**Compute:** Local CUDA, $0 spent
**Question this report answers:** *Can any single-variable change (capacity, balance, corpus) cure the standalone-5M content-collapse failure mode?*

---

## TL;DR

**No single dimension fixes content collapse.** Capacity (5M→25M) only buys ~2% val improvement. Balancing minorities hurts val on natural-distribution val set and introduces n-gram repetition pathology. The corpus side was already shown exhausted in v2. **The failure mode requires both capacity AND retrieval — or a much larger model — to break, per EraGPT precedent.**

---

## All-four side-by-side

| Run | Arch | Data | Best val | Train end | Train↓val gap | Probe verdict |
|---|---|---|---|---|---|---|
| v1 | 5M (6L · 8H · 256) | A only (1043) | 1.9967 | 2.024 | ~0 | content collapse |
| v2 | 5M (same) | A+E (1206) | 1.9108 | 1.93 | ~0 | content collapse (unchanged) |
| **B** | **25M (8L · 8H · 512)** | A+E (1206) | **1.8677** | 1.36 | **−0.58** | content collapse + overfit |
| **v3-bal** | 5M (same as v1/v2) | balanced (3070) | 1.9939 | 1.46 | −0.54 | new pathology (n-gram loops) |

**Diagonal trend:**
- v1 → v2 = +163 records, −0.086 val
- v2 → B  = 5x arch, −0.043 val
- v2 → v3-bal = 2.5x records, **+0.083 val (worse)**

---

## Probe excerpts

Same 5 prompts across all four runs. Sampling the Cortex-A53 vs Cortex-A72 prompt:

| Run | Output (truncated) |
|---|---|
| v1 | "When the CPU instruction a size of a size of a size of a size..." |
| v2 | "Using a ARM64 instruction a parameter in a parameter of the stack..." |
| **B (25M)** | "The kernel uses a specific often ARM64 kernel code multiple ARM64 CPU (ENEON) instructions and are parallel..." |
| **v3-bal** | "Using ARM64 NEON instructions can process multiple multiple multiple extra instructions..." |

Observations:

- **v1 → v2** swapped one set of high-frequency anchors ("CPU instruction", "size") for another ("parameter", "stack"). Same shape.
- **B (25M)** shows the first signs of *topical variation* — "ENEON" coined from "NEON", references to "kernel", "MMU", "page", "filter instruction" in different prompts. Outputs differ across prompts (small win). Still doesn't answer the actual question.
- **v3-balanced** introduces a NEW pathology — **n-gram repetition loops** ("multiple multiple multiple", "the the"). The wider corpus reduced the dominant-anchor effect but the model lacks the capacity to use the wider vocabulary coherently. Worse than v2.

---

## What each variable demonstrated

### Capacity (5M → 25M)
- ✅ Val improves 4.6% (1.911 → 1.868)
- ✅ Outputs are more topically varied
- ❌ Severe overfit (train 1.36 / val 1.94, gap 0.58)
- ❌ Still no `####` summary lines
- ❌ Still doesn't condition on the asked question

Diagnosis: 25M crosses some threshold (token variety up, vocabulary breadth up) but not the *prompt-conditioning* threshold. EraGPT analog: 14M → 113M crossed conditioning; 14M → 25M did not. **Per CLAUDE.md's stated ~25M throughput cliff, going past ~25M sacrifices the inference-time speed claim the whole bet is predicated on.**

### Concept balance (1087 → 3070 with minorities upsampled)
- ❌ Val worsens by 4% (1.911 → 1.994)
- ❌ New repetition pathology
- ❌ Probe still in collapse with new failure shape

Diagnosis: The natural concept imbalance was a STRENGTH for val_loss because Linux ARM64 (the majority class) dominates the val set. Balancing redistributed model attention away from the majority, which is correct in principle but the model lacks the capacity to learn the minority classes well. With 21x upsampling of dav1d records, the model sees them 21 times during training, but at 5M it can't encode 5 distinct asm idiolects.

### Corpus quantity (1043 → 1206)
- ✅ Val improved 4.3% (v1 → v2)
- ❌ No qualitative shift

Diagnosis: From v2 already in `bet_a_train_v2.md`. Data side exhausted for this arch.

---

## What this means for Bet A's actual thesis

CLAUDE.md said: "A 5M-parameter model + retrieval scaffolding can produce useful code generation at 100+ tok/s on consumer hardware."

We've now tested three pieces in isolation:
- 5M alone on 1043 records: content collapse
- 5M alone on 1206 records (v2): content collapse
- 5M alone on 3070 balanced records: content collapse + n-gram pathology
- 25M alone on 1206 records: content collapse with broader vocabulary
- **5M + retrieval: NOT YET TESTED**

**This is the moment to go to path (D) — retrieval scaffolding.** Every other lever short of "much bigger arch than the 25M cliff CLAUDE.md sets" has been pulled.

If 5M + retrieval doesn't work either, then CLAUDE.md's stated arch hypothesis is itself wrong and Bet A needs scope reconsideration. That's the discipline-gate moment.

---

## Recommendation

Move to **(D) retrieval scaffolding** as the next concrete experiment.

Concrete first step for (D) — small enough to ship in a single session:
1. Build a **symbol-and-API retrieval index** over the NAS source-code corpus (Track R per corpus_plan §2.3). For ARM64 alone, this means:
   - QNNPACK function signatures + adjacent docstrings (Pipeline E already harvested some of this)
   - Linux kernel `arch/arm64/include/asm/*.h` header symbols
   - dav1d / FFmpeg / x264 public API function lists
2. **Query-then-prompt** pattern: at inference time, the prompt = "Q: ... \nRELEVANT CONTEXT: \n{retrieved-snippets}\nA:".
3. Probe the v2 or B (25M) checkpoint *with this retrieval-augmented prompt* on the same 5 questions.
4. Compare to standalone probe.

This tests CLAUDE.md's actual claim. Doesn't require retraining. Cheap and informative.

Alternative if you want to skip (D) for now: **stop and re-evaluate the un-staged commitment.** Three days of work, four runs, no qualitative break. The original CLAUDE.md commitment was un-staged with a specific recommended sanity-check (kernel-on-code throughput) that we never actually executed. If retrieval testing is also going to fail, deferring further compute spend until the thesis is reconsidered is the by-the-book Block-5 ("Diagnostic / Recovery") move.

---

## File manifest

- `bet_a/artifacts/bet_a_v2_B_8x8_512_char.pt` — Config B 25M, val 1.868
- `bet_a/artifacts/bet_a_v3_balanced_6x8_256_char.pt` — 5M balanced, val 1.994
- `bet_a/training_data/bet_a_asm_v3_balanced_train.jsonl` — 3070 balanced records
- `bet_a/build_balanced_corpus.py` — balancing driver
- `bet_a/logs/bet_a_train_B_25M.log`, `bet_a_train_v3_balanced.log`, `probe_bet_a_AB.log`

---

*End run v3 4-way report.*
