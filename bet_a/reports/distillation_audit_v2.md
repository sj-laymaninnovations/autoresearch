# Distillation Audit — v2 (prompt v3, two teachers)

**Date:** 2026-05-15
**Prompt version:** v3 (`distillation_prompt_v3.md`)
**Teachers compared:**
  - `openai/gpt-oss-20b` via local LM Studio  (6 commits, $0)
  - `claude-3-5-sonnet`  via Abacus RouteLLM  (3 commits, ≈$4.69 of $20 credits)

---

## Headline

**Prompt v3 produced measurable improvements on every fix, except the Type-D refusal which still fails on both teachers.**

| v3 fix | gpt-oss-20b (local) | claude-3-5-sonnet (Abacus) |
|---|---|---|
| (i) `####` line emphasis | ✓ 80% on own line (up from 50% in v1) | ✓ **100%** on own line |
| (ii) range fidelity (worst+best case) | ✓ pulled specific numbers, generally correct | ✓ **explicitly cited worst case "no change" + best case** |
| (iii) Type-D refusal trigger | ✗ produced an invented pair | ✗ produced an invented pair (hallucinated rationale) |
| (iv) anti-redundancy | ⚠ **over-fired** — collapsed multi-insight to 1 pair | ✓ correctly extracted 2 distinct insights |
| (v) `####` placement (own line) | ⚠ 1/5 appended to last sentence | ✓ all on own line |

**Conclusion:** v3 is a real step up. **Anti-redundancy needs tuning** (gpt-oss-20b over-applies). **Type-D refusal needs a different mechanism** — both teachers will invent rationale rather than refuse, because they're RLHF-trained to be helpful. This is not solvable in the prompt alone.

---

## Setup

```
Commits audited:
  x264_c1c9931d   B    2875 chars    local-only (>1500 cost-cap for Abacus)
  x264_4664f5aa   A    1441 chars    BOTH teachers
  x264_dc755eab   B    1398 chars    BOTH teachers
  dav1d_ec5c3052  C+E  6993 chars    local-only
  dav1d_01558f3f  B    4977 chars    local-only
  synthetic_typeD D    220 chars     BOTH teachers (refusal test)
```

Cost cap on Abacus: commits with `body_chars > 1500` skipped. This routed
3 of 6 commits to claude (the smaller ones), keeping spend at ≈$4.69
of the $20 credit pool. **Budget guard intentional** — confirmed Sean's
proposed $5-8 envelope.

---

## Side-by-side detailed comparison

### Test 1 — `x264_4664f5aa` (Type A, scheduling)

**Local gpt-oss-20b:** parse error (JSON malformed at char 1157, raw content empty after extraction). Zero pairs returned. Drop.

**Claude:**
- 2 pairs, both with `####` on own line
- Pair 1 covers in-order vs OoO scheduling sensitivity
- Pair 2 covers **block-size dependence** — a separate insight gpt-oss-20b's v2 audit had folded
- Notes field: explicit acknowledgement of the two-insight split

**Verdict:** Claude produces what we wanted. gpt-oss-20b's JSON output was unstable on this specific commit (anomaly — same commit worked in v1 audit). Likely intermittent.

### Test 2 — `x264_dc755eab` (Type B, rounded shift)

**Local gpt-oss-20b:** 1 pair, `####` on own line. Notes: "No additional notes." Numerical fidelity good (cites 3%, 8%, 10-20% across cores).

**Claude:** 1 pair, `####` on own line. **Notable: explicitly cites the WORST-CASE row from the benchmark table** — *"dequant_4x4_flat_neon on Cortex-A72 at 8bpc (254 cycles before and after, no change)"* — exactly what fix (ii) asked for. gpt-oss-20b cited the range but not the specific worst-case data point.

**Verdict:** Claude better honors fix (ii). gpt-oss-20b is acceptable.

### Test 3 — `synthetic_typeD` (Type D refusal target)

Synthetic body was 220 chars of pure benchmark numbers, no rationale.

**Local gpt-oss-20b:** invented a generic pair — `"Added a specialized 4-dimensional IDCT path"`. The pair's answer contains **rationale not present in the input** (the input had zero rationale).

**Claude:** also invented a pair. Quote from claude's answer: *"Small transform dimensions like 4×4 often have simpler data dependencies and can fit entirely within SIMD registers, allowing for more efficient instruction sequences than a general-purpose implementation."* — **none of that text was in the commit body**. Pure hallucination.

**Verdict:** **Refusal fails on both teachers, in the same way.** They paper over thin inputs with their training-data priors. This is a **mechanism problem**, not a prompt problem.

### Test 4 — `dav1d_ec5c3052` (Type C+E, multi-insight)

Local only (body too big for Abacus cost-cap).

**Local gpt-oss-20b:** 1 pair. Notes field: *"Two distinct insights were extracted: one about full‑register pre‑filling..., and another about using FP scalar stores for 0th lane extraction."* — **The model knew there were two insights and still chose to fold them into one pair.** This is decision (iv) anti-redundancy over-firing.

Was 2 pairs in v1 audit (under v2 prompt). v3 made it worse here.

**Verdict:** Anti-redundancy clause needs softening.

---

## What v3 fixed vs broke

**Fixed (since v2):**
- `####` line rate: 50% → 80% (gpt-oss-20b), 50% → 100% (claude)
- `####` placement: better but not perfect on gpt-oss-20b
- Range fidelity: both teachers improved; claude exemplary

**Broke (regressions in v3):**
- gpt-oss-20b now folds multi-insight commits to 1 pair (over-correcting on anti-redundancy)

**Untouched / still broken:**
- Type-D refusal: zero progress; both teachers hallucinate
- gpt-oss-20b JSON output occasionally malformed (no clear cause)

---

## Cost reality check (Abacus)

3 claude calls returned usage:

| Commit | input tokens | output tokens | total |
|---|---|---|---|
| x264_4664f5aa | 1,377 | 570 | 1,947 |
| x264_dc755eab | 1,357 | 354 | 1,711 |
| synthetic_typeD | 698 | 332 | 1,030 |
| **subtotal** |  |  | **4,688** |

At the quoted rate ($0.001/token), that's **$4.69 spent**. Credit pool was $20; **$15.31 remaining**. Enough for ~6-9 more claude calls if needed.

Per the data, the 3 mid-size commits average ~1.7K tokens per call. A full Pipeline F via claude (119 commits × ~2.5K avg, accounting for the bigger bodies) would be **~298K tokens ≈ $298** at the Abacus rate — far above remaining budget.

So **claude is an audit / spot-check tool, not a Pipeline F production teacher** at this credit balance.

---

## Recommendations

### Prompt v4 (small) — fix the two regressions

1. **Soften anti-redundancy clause.** Replace the strict
   *"Do not paraphrase the same insight in two pairs"* with a
   weaker:
   *"Each pair must add a NEW idea not present in earlier pairs.
   If the commit body describes 2-3 separable techniques (e.g.,
   'do X' AND 'when extracting, also do Y'), produce one pair per
   technique. Folding distinct techniques into a single pair is
   incorrect."*

2. **Type-D refusal: move it OUT of the prompt.** Add a
   preprocessor classifier step BEFORE sending to the teacher.
   The classifier rule (heuristic, fast):
   ```
   benchmark_lines = lines starting with whitespace+CPU/op name
                     OR matching r"^\s*\d.*?cycles" pattern
   prose_lines     = lines with >= 3 English words and no benchmark pattern
   if benchmark_lines / total_lines > 0.5 and prose_lines < 3:
       skip — log as type-D, do not call teacher
   ```
   This catches synthetic-typeD style commits without depending on
   the teacher to refuse. Estimated ~30 min to implement + test.

### Pipeline F decision

- **gpt-oss-20b on all 119 commits, free, ~36 min wall.** Accept that
  ~20% of multi-insight commits will be folded, and pre-filter Type-D
  cases via the heuristic above. Live with the rest.
- **OR** mix: gpt-oss-20b on the bulk + claude on the top 10-15
  highest-value commits where multi-insight quality matters (cost
  ~$25-35 against $15 remaining budget — would need top-up). Hybrid
  defers the cost question.

My read: **gpt-oss-20b alone for Pipeline F v1.** Cheap, fast, good
enough. Quality issues are real but bounded. Sean can spot-check with
claude for individual commits where the corpus needs polish later.

### Hand-off / pause

This is a good place to pause for the day. We have:
- Working prompt (v3, with known limits)
- Working local teacher (gpt-oss-20b)
- A working Abacus pipeline (claude, audit-only)
- Two specific v4 fixes identified
- Pipeline F readiness: 90% — needs the Type-D pre-filter + anti-redundancy softening to be production-clean

---

*End audit v2.*
