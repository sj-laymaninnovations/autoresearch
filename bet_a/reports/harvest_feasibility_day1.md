# Pipeline A — Feasibility Harvest, Day 1

**Date:** 2026-05-15
**Pipeline:** A (git history harvest of asm + commit messages)
**Scope:** x264 + dav1d, shallow clone (`--depth 5000`)

---

## Summary

Pipeline A is feasible. From two of four planned repos at shallow depth, we get **119 raw hunks**, **109 with substantive commit bodies (91%)**, median 47–89 added lines per hunk. Full-depth clones plus FFmpeg + Linux ARM64 should multiply this 5–10× to ~600–1,200 raw hunks across the four primary code repositories.

**Verdict:** Proceed to Week-1 full Pipeline A run.

---

## What was harvested

```
~/Documents/autoresearch/bet_a/
├── raw/
│   ├── x264_src/      9.4 MB clone (depth 5000)   46 asm files
│   └── dav1d_src/     16 MB clone (depth 5000)    90 asm files
└── curated/
    ├── x264_hunks.jsonl     57 hunks
    └── dav1d_hunks.jsonl    62 hunks
```

**Filter applied:** Only hunks with `n_added ≥ 5` survive (drops trivial whitespace/formatting tweaks).

## Counts

| Metric | x264 | dav1d |
|---|---|---|
| Hunks total | 57 | 62 |
| Substantive commit body (≥40 chars) | 52 (91%) | 57 (91%) |
| ≥3 inline asm comments in added lines | 9 (16%) | 2 (3%) |
| ≥30 added lines | 45 (79%) | 39 (63%) |
| Median added-lines per hunk | 89 | 47 |

## Top samples (longest commit bodies)

**x264:**
1. `c1c9931d` — *"Improve pixel-a.S Performance by Using SVE/SVE2"* — 2,875-char body, 523 added lines on `common/aarch64/pixel-a-sve.S`
2. `4664f5aa` — *"aarch64: Improve scheduling in sad_x3/sad_x4"* — 1,441-char body
3. `dc755eab` — *"aarch64: Use rounded right shifts in dequant"* — 1,398-char body

**dav1d:**
1. `ec5c3052` — *"AArch64: Optimize lane load/store in MC functions"* — 6,993-char body
2. `01558f3f` — *"AArch64: Add HBD subpel filters using 128-bit SVE2"* — 4,977-char body

These are exactly the rationale-rich expert commentary the corpus plan targets.

---

## Key finding — re-route required

**The corpus plan assumed inline asm comments would carry the rationale. They don't.**

Only **9% of hunks** have ≥3 inline `;` or `//` comments in their added lines. But **91% of hunks** have substantive commit body text. The "WHY" lives in `git log` messages, not in `; comments above the next instruction`.

Implications for `bet_a/corpus_plan.md`:
- `commentary_records.jsonl` source primary = **commit message body**, not inline comments
- `code_records.jsonl` = the asm hunks, optionally with `annotation_quality: "low"` for the inline-comment-poor majority
- The distillation prompt (Pipeline F) should explicitly use the commit body as the expert claim and produce a Q&A pair where the question prompts the technique and the answer rephrases the body + summary

## Open issues surfaced

1. **Shallow clone limitation.** depth=5000 captured ~5000 commits per repo. Full history would yield significantly more — needs a `git fetch --unshallow` step before the Week-1 full run. Adds disk: x264 full ≈ 30 MB, dav1d full ≈ 60 MB. Trivial.

2. **Commit body quality varies.** Some are "fix typo" (already filtered by our `≥5 added lines` threshold). Some are 3-line summaries; some are 7K-char essays. The corpus plan's per-source 20% cap means a single 7K-char commit body shouldn't dominate.

3. **No FFmpeg, no Linux ARM64 yet.** Today is feasibility. Week-1 deliverable includes both. FFmpeg's `libavcodec/aarch64/*.S` history is the biggest single yield expected.

4. **No commentary from blog / mailing list yet.** Today's harvest is git-only. Pipeline B (blog) + Pipeline C (ML) come Week-1 onward. The Pipeline A signal alone is enough to commit to the harvest direction.

---

## Recommended next step (waiting for Sean's nod)

Either:
- **(W)** Full Week-1 Pipeline A run on all 4 repos (`git fetch --unshallow` + FFmpeg + Linux ARM64 clone). ~30 min wall, ~150 MB disk, ~600–1,200 hunks expected.
- **(B)** Move to Pipeline B (blog harvest) first — Dark Shikari / Wojciech Mula / Daniel Lemire. Tests a different source format before scaling git harvest.
- **(D)** Sketch the distillation prompt against today's top-10 highest-body-content hunks (no API call yet — just write the prompt and a sample Q&A by hand). Gates the eventual $20 Abacus audit spend.

My read: **(D) before (W) or (B)**. We just learned the corpus shape isn't what the plan assumed. Updating the distillation prompt against real samples — by hand, no API spend — is the cheapest way to de-risk Pipeline F before it consumes credits.

---

*End feasibility report.*
