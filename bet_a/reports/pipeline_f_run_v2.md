# Pipeline F — Run v2 (Pipeline A α + β + γ scope)

**Date:** 2026-05-15
**Wall time:** α (4m20s) + β+γ (1h37m30s) = **~1h42m total**
**Teacher:** `openai/gpt-oss-20b` (local LM Studio, $0 spend)
**Prompt:** v4 (`distillation_prompt_v4.md`)
**Preprocessor:** `bet_a/preprocessor.py`
**Sources added since v1.1:**
  - α: LoongArch (both x264 + dav1d) + RISC-V (dav1d) hunks
  - β: **FFmpeg** — libavcodec / libavfilter / libavutil / libswscale / libswresample / libpostproc across aarch64, arm, x86, loongarch, riscv
  - γ: **Linux kernel** — `arch/arm64/{kernel,lib,crypto,mm,kvm}/*.S` (sparse-clone, depth 10000)

---

## Headline

**1,043 distilled Q&A pairs from 1,148 unique commits across 4 source repos.**

This is **14× the v1.1 corpus** (74 pairs → 1,043 pairs) for ~1.7 hours of wall time and **$0 API spend**. All format validation rules held (0 codebase-name leaks, 0 missing `####` lines, 0 inline-`####` from gpt-oss-20b).

---

## Stage funnel (cumulative across α + β + γ)

| Stage | Count | Notes |
|---|---|---|
| Unique commits inventoried | 1,148 | x264 (90) + dav1d (64) + FFmpeg (382) + Linux ARM64 (612) |
| Already-done (resume) | 101 | from v1.1's 74 pairs across 69 commits + α's 32 new from new resume |
| Preprocessor: too-short | 159 | bodies < 80 chars |
| Preprocessor: Type-D | 189 | benchmark-table-only commits the teacher would have hallucinated through |
| Teacher calls | 1,012 | ~5s/call median, 10.5s p95 |
| Teacher refused | 42 | "body too vague" — model declining |
| Parse-failed (final, post-retry) | 36 | ~3.5% of teacher calls — much lower than v1's 20% thanks to retry logic |
| Retries used / recovered / failed | 140 / 104 / 36 | retry success rate 74% |
| **Q&A pairs written** | **1,043** | |
| Pairs per successful commit | 1.07 | |
| Format validation failures | **0** | |

---

## Per-repo yield

| Repo | Hunks harvested | Unique commits | Pairs distilled | Pairs / commit |
|---|---|---|---|---|
| Linux ARM64 | 681 | 612 | **678** | 1.11 |
| FFmpeg | 747 | 382 | 258 | 0.68 |
| x264 | 100 | 90 | 70 | 0.78 |
| dav1d | 83 | 64 | 37 | 0.58 |
| **TOTAL** | **1,611** | **1,148** | **1,043** | 0.91 |

**Linux ARM64 is the dominant single source** — every one of 612 unique commits had a substantive body, and the kernel maintainers (Catalin Marinas, Will Deacon, Mark Rutland) write rationale-rich patch descriptions by convention. Pair-per-commit ratio (1.11) is the highest because many kernel patches contain multiple separable insights.

**dav1d's low ratio (0.58)** is partly because the cross-ISA (RISC-V / LoongArch) additions skewed toward bare-benchmark commits that the preprocessor caught.

---

## What the preprocessor caught (decisive)

Across all 4 repos, the preprocessor heuristic dropped **348 commits** (159 too-short + 189 Type-D) **before any teacher call**.

Without the preprocessor, those 348 commits would have hit the teacher, and based on the audit-v2 finding (both gpt-oss-20b and claude-sonnet-4 hallucinate rationale for bare benchmarks), **most of those would have produced fabricated training examples**. That's exactly the failure mode the architecture was designed to prevent.

Cost savings (counterfactual): if those had been Abacus calls, 348 × ~$2/call = **~$700 of wasted credits + ~700 polluted training examples**.

---

## Parse-fail rate dropped dramatically

| Run | Parse-fail rate (final) | Retry logic |
|---|---|---|
| v1 (initial) | 14/71 = **20%** | none |
| v1.1 (resume + retry) | 0 (after retry) | added |
| v2 α | 1/41 = 2% | retry |
| v2 β+γ | 36/1012 = **3.5%** | retry |

The retry logic recovered 74% of transient parse failures (104 of 140 retries succeeded). The remaining 3.5% are commits where the model fails twice — likely commits with unusual character sequences (escaped quotes, embedded code blocks) that consistently break gpt-oss-20b's JSON serializer.

Worth pursuing in v2.1: a **second retry with a different temperature** (currently 0.3) might recover more. ROI is modest (3.5% of 1,012 ≈ 36 more pairs) but free.

---

## License posture audit (per corpus_plan §8)

| Source | License | Posture confirmed |
|---|---|---|
| x264 | GPL-2.0-or-later | fair-use training ✓ |
| dav1d | BSD-2-Clause | fair-use training (plus permissive) ✓ |
| FFmpeg | LGPL-2.1-or-later (some files GPL) | fair-use training ✓ |
| Linux kernel `arch/arm64` | GPL-2.0 with Linus's syscall exception | fair-use training ✓ |

All four repos are open-source with established conventions for derived-work production. The corpus_plan §8 "fair-use training" posture (you confirmed earlier) covers all four.

---

## What this corpus IS and IS NOT

**IS:**
- Real frontier-engineer commentary, distilled from production assembly work
- Cross-ISA (ARM64 primary, x86, RISC-V, LoongArch secondary) — gives the model the "language-agnostic patterns" CLAUDE.md envisioned
- Validated for format (zero leaks, zero missing `####`)
- A working **training dataset** for a 5M model with retrieval scaffolding

**IS NOT:**
- Hand-audited beyond the v1 10-pair sample. The 969 pairs added since the audit have NOT been individually inspected.
- Cleaned of unintentional duplicates (same idiom phrased differently across repos)
- Balanced — Linux ARM64 represents 65% of the corpus

---

## Recommendations

### Pipeline F v2.1 (small, ~10 min)
Single second-retry on parse-fails using temperature=0.7 to break repeating-token loops. Could lift corpus +36 pairs. Free.

### Quality audit at scale
The v1 audit covered 10 of 74 pairs (~14%). Same ratio on 1,043 = ~140 pairs. Too many for a single eyeball pass. Options:
- (a) Sample 30 pairs across all 4 repos, weighted by yield. Confirm v1 audit findings hold.
- (b) Run an LLM-as-judge pass (gpt-oss-20b can self-evaluate at ~5s/pair, ~90 min total) to flag suspect pairs for Sean to spot-check.

### Pipeline B — blog harvest
Still on the table. With the corpus now at 1,043 it's no longer the only path to scale. **Decision point:** is 1,043 enough for first training pass on a 5M model, or do we want Pipeline B's ~1,400 blog records on top before training?

### First training run (the actual point)
1,043 pairs at ~80 tokens average answer = ~83K answer tokens. With prompt template ("Q: ... A: ..."), maybe 130K tokens total. For a 5M-parameter MathGPT-style model that's a tiny training set — would need to be combined with general-pretrain data or trained as a fine-tune from a checkpoint.

This is the **next concrete experimental milestone** for Bet A.

---

## File manifest

- `bet_a/distilled/qa_records_v1.jsonl` — 1,043 distilled Q&A pairs
- `bet_a/curated/x264_hunks.jsonl` — 100 hunks
- `bet_a/curated/dav1d_hunks.jsonl` — 83 hunks
- `bet_a/curated/ffmpeg_hunks.jsonl` — 747 hunks
- `bet_a/curated/linux_arm64_hunks.jsonl` — 681 hunks
- `bet_a/reports/pipeline_f_run_stats.json` — final stats
- `bet_a/reports/pipeline_f_skipped.jsonl` — 457 cumulative skip entries
- `bet_a/pipeline_f.py` — runner (REPOS dict expanded to 4 repos)
- `bet_a/extract_hunks.py` — reusable hunk extractor

Upstream clones (in `raw/`, not committed — see `bet_a/.gitignore`):
- `raw/x264_src/` (9.8 MB, 3,223 commits)
- `raw/dav1d_src/` (16 MB, 2,792 commits)
- `raw/ffmpeg_src/` (143 MB, depth 8000)
- `raw/linux_src/` (2.1 GB, sparse to arch/arm64, depth 10000)

---

*End run v2.*
