# Distillation Prompt — v2 (locked, ready for audit)

**Date:** 2026-05-15
**Supersedes:** `distillation_prompt_sketch_v1.md`
**Decisions baked in (all five approved by Sean):**

| | Decision | How it's enforced |
|---|---|---|
| (a) | Drop commits with body < 80 chars | **Preprocessor filter** (not prompt). Applied before any teacher call. |
| (b) | Allow 1-3 Q&A pairs per commit when separable | **Prompt instruction** (see below). Cap of 3. |
| (c) | Refuse bare-benchmark commits | **Prompt instruction**: return `null` pair. Preprocessor drops the null. |
| (d) | Normalize measurement tables to ranges | **Prompt instruction**: extract range + endpoints, not verbatim tables. |
| (e) | Strip codebase names from question text | **Prompt instruction**: question must be codebase-agnostic; answer may cite domain context (e.g., "H.264 dequant"). |

---

## The prompt (v2, locked)

```
You are reformulating commit messages from expert assembly programmers
into Q&A training pairs for a small (5M-parameter) language model. The
student model will produce ARM64 assembly with retrieval scaffolding
handling language-specific syntax. The student needs IDIOMS, PATTERNS,
and RATIONALE — not syntax tables.

The commit body below describes one or more optimization techniques.
Produce 1-3 Q&A pairs (one per separable insight) where each pair
satisfies ALL the following:

QUESTION rules:
  - Asks about the underlying principle, idiom, or pattern.
  - Phrased as a question a general ARM64 programmer might ask.
  - Does NOT mention specific codebases (x264, dav1d, FFmpeg, Linux,
    etc.) or specific function/file names. The question must be
    applicable beyond the source commit.

ANSWER rules:
  - Preserves the expert's claim verbatim in substance — DO NOT add
    facts not in the commit body.
  - 3-7 sentences of instructional prose.
  - INCLUDES the expert's measurement numbers as evidence. For
    multi-row benchmark tables, extract the RANGE (e.g.,
    "8-30% speedup across blend/avg/mask functions") plus the
    notable endpoints (worst case, best case). Do NOT dump the
    verbatim table.
  - MAY cite the domain context that ties to the measurements
    (e.g., "in H.264 dequantization kernels", "for AV1 motion
    compensation") because the measurements are tied to it.
  - Ends with a single line `#### <single-sentence summary>`
    capturing the takeaway. No trailing punctuation on the `####`
    line.

REFUSAL rules:
  - If the commit body lacks substantive rationale (e.g., is only
    a benchmark table with no explanatory text, or only repeats the
    subject line), return `{"pairs": []}` and explain in `notes`.
  - If the commit body is too vague to extract a generally-applicable
    idiom, return `{"pairs": []}` and explain in `notes`.

OUTPUT (strict JSON, no preamble, no markdown fences):
  {
    "pairs": [
      {"question": "...", "answer": "..."},
      {"question": "...", "answer": "..."}
    ],
    "notes": "optional brief reason if pairs is empty"
  }

INPUT:
  commit_subject: {commit_subject}
  commit_body: |
    {commit_body}
```

---

## Hand-drafted exemplars

The three from v1 are unchanged and still valid. Two new ones cover the edge cases v1 noted but didn't ground:

### Exemplar 4 — Type E (multi-insight split into TWO pairs)

**Source:** dav1d `ec5c3052` "AArch64: Optimize lane load/store in MC functions"

Same source as v1 Exemplar 3, but now showing the **split** version since
decision (b) is in effect. Two separable insights → two pairs.

```json
{
  "pairs": [
    {
      "question": "What is the performance cost of partial-register lane writes (e.g., MOV V0.S[1], W0) in ARM64 NEON code on out-of-order cores, and how do you break the resulting dependency chain?",
      "answer": "Lane-by-lane writes create a false dependency chain: the destination vector register cannot be renamed because each lane write merges with the prior state of the register. On out-of-order cores (Neoverse V1/V2, Cortex-X3/X1), this serializes operations the OoO engine would otherwise parallelize. The mitigation is to fill the full register first with a non-merging write (MOV V0.16B, V_src.16B, or LDR Q0, [src]) before any lane-write sequence. Subsequent lane writes still merge, but the dependency chain restarts at the full-width write. Measured impact on AV1 motion compensation: 8-30% speedup across blend/avg/mask/w_mask functions in 8bpc, with up to 65% on 16bpc blend variants where the dependency chain was longest.\n#### Avoid partial-register writes in hot NEON code; start with a full-register write before any lane updates"
    },
    {
      "question": "When extracting a single lane from an ARM64 NEON vector to memory, is there a cheaper alternative to ST1 {Vn.S}[0], [dst] for the 0th-lane case?",
      "answer": "Yes — for the 0th-lane extract, use the FP scalar store form (STR S0, [dst]). It produces the same memory result as ST1 {V0.S}[0], [dst] but is a cheaper micro-op because it doesn't carry the lane-index decoding. Use it whenever you know at compile time that only the 0th lane needs to land in memory. This optimization is part of the same patch family that targets partial-register-write dependency chains in motion-compensation kernels.\n#### For 0th-lane extracting stores, prefer FP scalar store (STR S0) over lane-extract (ST1)"
    }
  ],
  "notes": null
}
```

### Exemplar 5 — Type D (bare-benchmark refusal)

**Synthetic example** (no real candidate among today's top hunks — they all had rationale). Constructed to show what the prompt should do when only numbers, no "why."

**Input subject:** *aarch64: Add idct dimension 4 fast path*

**Input body (synthetic):**

```
        Cortex A53   A72   A73   X1
Before: 250          165   170   140
After:  240          158   163   135
```

**Expected output:**

```json
{
  "pairs": [],
  "notes": "Body is only benchmark numbers; no explanatory rationale that would support a generally-applicable Q&A. Skipping."
}
```

### Exemplar 6 — Short-body rejection (preprocessor filter)

**Source:** any commit whose body field is 0-79 chars (e.g., "Fix
typo", "Bump version", "Address review comments").

**Expected behavior:** **Never sent to the teacher.** Filtered at the
preprocessor by decision (a). Recorded in `harvest_stats.json` as
`filtered_short_body`.

---

## Preprocessing pipeline (formalized)

The filter chain before any teacher call:

```
input:   raw_hunks.jsonl (one record per (commit, file_path) hunk)
  |
  v
[1] dedupe by hash(commit_sha + file_path + first_100_chars(hunk))
  |
  v
[2] group by commit_sha
    -> aggregate all file_paths under one commit
    -> body becomes the commit message body (shared across hunks)
  |
  v
[3] drop commits where body_chars < 80      (decision a)
  |
  v
[4] drop commits where subject is in
    {"fix typo", "bump version", "merge", "release N.N.N", "..."}  (heuristic)
  |
  v
[5] dedupe by hash(commit body first 500 chars)  (catch cherry-picks)
  |
  v
output:  commits_ready_for_distillation.jsonl
```

Step 3-5 typically drop 30-50% of raw hunks per my read of the harvest
sample. That leaves ~70-50% of raw hunks proceeding to the teacher.

---

## What's still uncertain

1. **Teacher faithfulness.** The prompt rules (especially "do not add
   facts not in the commit body") are enforced by the prompt. Frontier
   teachers WILL sometimes embellish or generalize. The audit will tell
   us how often. Disagreement signal goes back into prompt v3.

2. **Pair-count distribution.** Decision (b) caps at 3, but I haven't
   measured how many commits actually have multiple separable insights.
   Could be ~10%, could be ~40%. Audit will measure.

3. **Refusal rate.** Decision (c)'s refusal rate is the second uncertain
   number. If teachers refuse ~5% of inputs (close to my read), we lose
   little. If they refuse 30%, the bare-benchmark category is much
   bigger than expected and we should harvest blog/ML for those
   commits' surrounding rationale.

The audit's job is to measure these three numbers, not just to qualify
the prompt.

---

## Audit plan (waiting Sean's go-ahead before spend)

- **Budget:** $5-8 of the $20 Abacus credits (≈25-40% of pool)
- **Sample size:** 12 commit records (small but covers Type A/B/C +
  the long-body and short-rationale variants)
- **Teachers:** gpt-4o, claude-sonnet-4, gemini-2.5-flash (3-way
  round-robin, same input to all three on each commit)
- **Total Q&A pairs expected:** 12 commits × 3 teachers × 1-3 pairs
  per commit = ~36-100 Q&A pairs
- **Output:** `bet_a/reports/distillation_audit_v1.md` with side-by-side
  teacher comparison + my hand-drafted ground-truth + manual flags
  ("matches truth", "adds invented facts", "weak"), plus the three
  uncertainty measurements above.
- **Decision after audit:** prompt v3 OR full distillation OR pivot
  teacher API.

---

*End v2.*
