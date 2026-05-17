# Bet A — Capacity Curve (5M → 50M)

**Date:** 2026-05-16–17
**Question:** How does output quality improve as model capacity scales from 5M to 50M params on the same v2 corpus?

---

## Five-way loss comparison

| Run | Arch | Params | Best val | Train final | Overfit gap | Wall |
|---|---|---|---|---|---|---|
| v1 | 6L·8H·256d | 5.02M | 1.9967 | 2.024 | ~0 | 5m6s |
| v2 (A+E) | same | 5.02M | 1.9108 | 1.931 | ~0 | 5m43s |
| v3-bal | same | 5.02M | 1.9939 | 1.455 | −0.54 | ~9m |
| B (25M) | 8L·8H·512d | 25.77M | **1.8677** | 1.364 | −0.50 | 16m |
| **50M** | 10L·10H·640d | 49.90M | 1.8878 | 1.020 | **−0.87** | **21m** |

Note: 50M best val (1.888) is slightly *worse* than 25M (1.868). The 50M model's best checkpoint was achieved early in training before overfit took over; by epoch 100, val had climbed to 2.19 while train reached 1.02.

---

## Throughput ladder (RTX 3080, eager decode)

| Arch | Params | tok/s | vs 5M |
|---|---|---|---|
| 5M (6L·8H·256d) | 5.02M | **270** | — |
| 25M (8L·8H·512d) | 25.77M | 182 | −33% |
| **50M (10L·10H·640d)** | **49.90M** | **141** | **−48%** |
| 100M (12L·12H·768d) | 85.84M | 107 | −60% |
| 200M (16L·16H·1024d) | 202.54M | 76 | −72% |

50M at 141 tok/s — well above the 100 tok/s Bet A target.

---

## Probe quality (Cortex-A53 vs A72 prompt, truncated)

| Run | Output start |
|---|---|
| 5M v1 | "When the CPU instruction a size of a size of a size..." |
| 5M v2 | "Using a ARM64 instruction a parameter of the stack..." |
| 25M | "The kernel uses a specific often ARM64 CPU (ENEON) instructions and are parallel..." |
| **50M** | **"In ARM64 CPU NEON processing the processes in pixels in parallel, the code can be accessed..."** |

**Trend:** Each capacity doubling adds topically-relevant vocabulary. 50M shows NEON, SIMD, pixels from Pipeline E QNNPACK influence. Different prompts elicit different domain anchors (Cortex/NEON → pixel processing, SVE2 → SIMD, stack allocation → stack pointer idioms).

**Not crossed:** Question-conditioned coherent response. The rounded-right-shift and partial-register prompts still drift to Linux kernel interrupt/exception territory (overrepresentation in corpus). No `####` summary lines emitted.

---

## Key finding: vocabulary scales with capacity, coherence does not

The probe outputs across the 5 models reveal two separable phenomena:

1. **Domain vocabulary** (which ARM64 concepts the model names) improves smoothly with capacity:
   - 5M: "stack", "size", "single" (generic)
   - 25M: "kernel", "NEON" (garbled as ENEON), "page"
   - 50M: "NEON", "SIMD", "pixels", "parallel" (clean domain terms)

2. **Question conditioning** (whether the output answers the asked question) does NOT improve:
   - All models produce grammatically plausible ARM64 prose that doesn't address the specific question
   - Topical variation across prompts improves (50M gives different answers to NEON vs SVE2 vs stack questions), but correctness doesn't

This confirms the CLAUDE.md thesis *needs* to separate the two:
- Vocabulary/domain → solvable by more corpus + capacity
- Conditioning/reasoning → requires retrieval scaffolding (the model doesn't have the SPECIFIC FACT "Cortex-A53 benefits 20-25% from scheduling optimization")

---

## What's left of the "pure training" path

The corpus (1,206 pairs) is severely data-limited for 50M+ parameters. Doubling from 25M to 50M actually worsened best val (1.867 → 1.888) because the model overfits so completely that no checkpoint survives without generalization loss.

To make pure training work without retrieval at 50M would need an estimated **10-20× more training pairs** (10,000-25,000 records) to fill the model's capacity without overfit. That's 3-4 more Pipeline A/E expansion cycles on the NAS corpus.

---

## Recommendation

The data side and the capacity side have now been explored thoroughly:
- 5M → corpus exhausted at 1.2K records
- 25M → capacity increment marginal; overfit begins
- 50M → capacity increment marginal; overfit worse; vocabulary better

**The clean next experiment is (D) retrieval scaffolding** — test whether the existing trained models (5M or 25M) with ARM64 API/syntax retrieved into the prompt actually answer questions correctly. This doesn't require more training or data; it tests whether the split "model holds patterns / retrieval holds specifics" hypothesis is correct. One session of work for the retrieval index; one probe run; clean diagnostic.

If retrieval works at 5M → CLAUDE.md's hypothesis validated. If it doesn't → we need 100M+ model class.

---

*End capacity curve report.*
