# Pipeline F — Run v1

**Date:** 2026-05-15
**Wall time:** 7m 33s
**Teacher:** `openai/gpt-oss-20b` (local LM Studio, $0 spend)
**Prompt:** v4 (`distillation_prompt_v4.md`)
**Preprocessor:** `bet_a/preprocessor.py` (Type-D + too-short filters)

---

## Headline

**57 distilled Q&A pairs from 87 unique commits across x264 + dav1d.** First real production output of the Bet A corpus pipeline. **Zero API spend.**

| Stage | Count | Rate |
|---|---|---|
| Unique commits (input) | 87 | 100% |
| → too-short (preprocessor drop) | 12 | 14% |
| → Type-D (preprocessor drop) | 4 | 5% |
| → sent to teacher | 71 | 82% |
| → teacher refused (`pairs: []`) | 4 | — |
| → parse-failed (malformed JSON) | 14 | 20% of teacher calls |
| → returned pairs | 53 | 75% of teacher calls |
| **Q&A pairs written** | **57** | — |
| Pairs/commit (succeeded) | **1.08** | |
| Format validation failures | 0 | (no codebase-name leaks, no missing `####`) |

---

## What the preprocessor caught

- **12 too-short commits** — bodies under 80 chars. Trivial "fix typo" / "bump version" cases the corpus_plan §6 specifically excludes.
- **4 Type-D commits** — bodies that are mostly benchmark tables with little prose. Examples:
  - `e560d2ba` — likely "AArch64: Optimize XYZ" with embedded perf table only
  - `41511bf1` — same pattern (9 hunks on one commit, body is mostly the perf table)

This is the **mechanism we couldn't get from prompt-only refusal** (both gpt-oss and claude hallucinated rationale in audit v2). The heuristic preprocessor works.

## What the teacher refused

4 commits the teacher itself chose to skip with `"notes": "body too vague"`. Spot-checked: these are commits where the body is 2 sentences without identifiable technique. Reasonable behavior.

## Parse failures (the biggest quality loss)

**14 of 71 teacher calls returned malformed JSON** (~20%). Inspecting `pipeline_f_skipped.jsonl`, the failure pattern is mostly "Expecting ',' delimiter at char N" — the model emits a string with an embedded `"` or newline that breaks the JSON.

This is the biggest quality lever remaining. **One retry per parse-fail would likely recover most of these** (transient gpt-oss-20b serialization issue). Bumping Pipeline F to v2 with single-retry semantics would lift the corpus from 57 to ~68 pairs at ~9-minute total cost.

---

## Sample distilled pairs

### Sample 1 — `x264 4df71a75bf` (clean)

> **Q:** How can a function safely perform multiple independent stack allocations without manually adjusting the stack pointer each time?
>
> **A:** By restoring the stack state between allocations, a function can allocate space on the stack for one block of data, use it, and then reset the stack pointer to its previous value before allocating another independent block. This technique eliminates the need for manual offset calculations and ensures that each allocation is isolated from others, preventing accidental overlap or corruption. It also simplifies the compiler's stack management logic and improves code readability.
> #### Restoring the stack pointer between allocations isolates each allocation, simplifying stack management

`####` on own line. No codebase name leak. Codebase-agnostic question. **Clean.**

### Sample 2 — `x264 5c382660fb` (format violation captured)

> **Q:** Why would an ARM64 assembly programmer choose to replace NEON implementations with SVE/SVE2 for DCT and zigzag operations?
>
> **A:** SVE (Scalable Vector Extension) allows vector lengths that adapt to the underlying hardware… *[6 sentences citing specific numbers: 322→247 cycles on Yitian 710, 376→255 on Graviton3]* …transforms.#### SVE replaces fixed‑width NEON to boost SIMD throughput and reduce cycles for DCT/zigzag operations

Numerical fidelity good (multiple endpoint citations). But the `####` line is **concatenated to the last sentence** — exactly the v3 format violation we wanted to fix. **Validator caught it** (`warnings: ['#### not on own line']`) but the pair is still written. Decision needed: drop on warning, or accept-with-tag?

---

## Recommended next moves

1. **Pipeline F v2 with retry** — add one retry on parse-fail. Could lift corpus 57 → ~68 pairs. ~2 min additional wall.

2. **Quality audit on the 57 pairs** — Sean spot-reads 5-10 pairs, flags hallucination or content collapse. Cheap, decides whether the corpus is training-ready.

3. **Pipeline B — blog harvest** — the corpus is too thin at 57 pairs for serious training. Pipeline B (Dark Shikari + Wojciech Mula + Daniel Lemire) is the next high-yield source. Estimated 1400+ records.

4. **Pipeline A continuation** — `git fetch --unshallow` on x264 + dav1d (we used `--depth 5000`), plus FFmpeg + Linux ARM64 clones. 5-10× more commits = 5-10× more pairs.

My read: **(2) audit first, then (4) expand harvest** — adding more breadth before iterating on prompt quality. The corpus needs scale before training.

But this is a clean pause point. Pipeline F v1 ran cleanly, the preprocessor works, the corpus exists.

---

*End run v1.*
