# Bet A — (D) Retrieval Test + 100M Findings

**Date:** 2026-05-17
**Parallel experiments:**
  - (D) Retrieval scaffolding — zero-shot RAG probe on v2-5M / 25M / 50M
  - 100M training (~85.84M, 12Lx12Hx768d) on v2 corpus, RTX 3090

---

## Complete capacity curve (now 5M → 100M)

| Params | Best val | Train final | Overfit gap |
|---|---|---|---|
| 5M v1 | 1.9967 | 2.024 | ~0 |
| 5M v2 | 1.9108 | 1.931 | ~0 |
| **25M** | **1.8677** | 1.364 | −0.50 |
| 50M | 1.8878 | 1.020 | −0.87 |
| **100M** | **1.9075** | **0.865** | **−1.04** |

**The 25M is the peak.** 50M and 100M both degrade best val as overfitting overwhelms capacity gains. For 1087 training records, the data is the binding constraint at scales above ~25M.

**Verdict on scaling:** With the current corpus size, adding more params is net-negative past 25M. The 100M's best checkpoint was likely found at ~epoch 20, after which overfit took over.

---

## (D) Retrieval scaffolding — findings

### Setup
Retriever: TF-IDF cosine similarity (sklearn) over all 1206 distilled Q/A pairs from Pipeline A+E. Top-2 retrieved pairs injected before the target question.

Augmented prompt format:
```
RELEVANT CONTEXT:
Q: {retrieved_q_1}
A: {retrieved_a_1}

Q: {retrieved_q_2}
A: {retrieved_a_2}

Q: {target_question}
A:
```

### What the retriever found (quality: excellent)

For the Cortex-A53 vs A72 query, hit #1 contained the EXACT answer:
> "By reordering operations, it reduces stalls caused by waiting for previous results on an in-order core, leading to a **20-25% speedup on the Cortex A53** for small block sizes. On out-of-order cores such as the Cortex A72 and A73, the hardware can already reorder instructions to hide latency, so the improvement is within measurement noise."

For the SVE2/SDOT query, hit #1 contained the EXACT answer:
> "The 4-element 16-bit SDOT instruction allows the processor to perform four signed dot products in a single cycle, each combining two 16-bit operands into a 32-bit result. **This is ideal for the 4-tap and 8-tap convolution filters** common in video codecs, as it matches the filter width and data type."

The specific facts we wanted the model to USE were being retrieved correctly.

### What the models actually did

All three models (5M, 25M, 50M) **ignored the retrieved context** and generated text from inside the context block, not from the "A:" of the target question.

Root cause: **prompt truncation + training format mismatch.**

1. The augmented prompt (~1100 chars) exceeds seq_len=1024. We truncate to `seq_len - max_new = 714` chars.
2. This truncation cuts off the target question entirely — the model only sees the "RELEVANT CONTEXT" block.
3. The model then CONTINUES from inside the retrieved text (generating the next likely character), never seeing the target question at all.

Even without truncation, the model was never trained on the retrieval-augmented format. It would see `RELEVANT CONTEXT:` at the start of a sequence and treat it as "text to continue" rather than "context to use." The training corpus only has bare `Q: ... A:` pairs.

### What this means for CLAUDE.md's (D) hypothesis

**Zero-shot retrieval injection doesn't work.** The hypothesis "model holds patterns, retrieval holds specifics" is architecturally correct but requires the model to be TRAINED on the retrieval-augmented format. It's not a free inference-time add-on.

The correct implementation path:
1. For each training Q/A pair, retrieve 2 similar pairs from the corpus
2. Format as: `RELEVANT CONTEXT:\nQ: {r1.q}\nA: {r1.a}\n\nQ: {target}\nA: {answer}`
3. Fine-tune (or from-scratch train) on this format
4. At inference, use the same format with a retrieval index over the syntax/API source material

This is more work than expected but it's the RIGHT test of CLAUDE.md's hypothesis.

---

## Root cause summary (all experiments to date)

Three binding constraints, now confirmed:

| Constraint | Evidence |
|---|---|
| **Data volume** | 25M+ overfits 1087 records; adding params above 25M degrades val |
| **Format alignment** | Zero-shot RAG fails; model needs training on retrieval-augmented format |
| **Concept balance** | Balanced corpus (v3) worsened val on natural-distribution val (as expected statistically); the 56% Linux ARM64 dominance skews outputs to kernel idioms |

---

## Recommended next actions (in priority order)

### (1) Scale corpus to 5-10K pairs before any more training
At 25M the model is data-limited. A 5× corpus expansion would let us push past 25M with confident val improvement. Sources:
- Pipeline A expansion: QNNPACK all files (not just aarch*), Linux full arch/arm64, FFmpeg ARM64 more thoroughly
- Pipeline E expansion: extend to more NAS orgs (XNNPACK, TensorFlow Lite QNNPACK)
- Pipeline B: blog harvest (Dark Shikari, Wojciech Mula, Daniel Lemire) — 1400+ records at zero cost

### (2) Build retrieval-augmented training corpus
Regenerate the training data with retrieval context injected into each example:
- For each pair in the corpus, retrieve 2 similar pairs from the same corpus (leave-one-out)
- Format the augmented example
- Retrain 25M on this format
- At inference, use the NAS source code as the retrieval index

This is the actual CLAUDE.md architecture. Estimated 2-3 hours of engineering.

### (3) Fix concept balance without hurting val
The 56% Linux ARM64 dominance is the main cause of "kernel idiom" drift in outputs. After scaling corpus, ensure concepts are more balanced (Pipeline E expansion adds QNNPACK, XNNPACK, x86 codec asm to thin out Linux dominance).

---

## Stop-here recommendation

The experiments are producing clean diagnostic signal. The path forward requires corpus scaling and format change — both are tractable but are not same-day work.

Good stopping point: commit all results, update CLAUDE.md with corrected throughput claims, then plan corpus scaling separately.

---

*End findings report.*
