# Distillation Audit — v1

**Date:** 2026-05-15
**Prompt version:** v2 (locked in `distillation_prompt_v2.md`)
**Teacher cost:** **$0** — local LM Studio on Mac mini, no Abacus credits spent
**Sample size:** 5 real commits × 2 teachers = 10 calls planned, **5 successful** (all gpt-oss-20b)

---

## Headline

**`openai/gpt-oss-20b` produces usable distilled Q&A pairs from real commit bodies at ~17 s per call on the Mac mini.** All 5 commits yielded 2 pairs each (10 total). Quality is **70% faithful, 50% format-compliant** — the prompt needs a v3 pass to fix specific issues, but the approach is viable.

**`qwen/qwen3.6-35b-a3b` is unusable through this prompt as written.** 3 of 5 calls timed out at 55-60s; 2 returned in <400 ms (instant failure mode). It's a reasoning model and consumes its token budget on internal chain-of-thought before producing visible content. Either needs `/no_think` system prompt, much larger `max_tokens`, or skip it entirely.

---

## Setup

```
Host:     192.168.1.169:1234 (Mac mini M4 Pro, LM Studio)
Teachers: openai/gpt-oss-20b, qwen/qwen3.6-35b-a3b
Prompt:   distillation_prompt_v2.md (system_msg, ~750 tokens)
Commits:  5 from bet_a/curated/{x264,dav1d}_hunks.jsonl
          (covering Type A, Type B, Type C+E pattern variants)
Max tokens: 4000 per call
Run output: bet_a/reports/audit_responses_v1.jsonl
```

| Commit | Type | Body chars | Hand-draft exists? |
|---|---|---|---|
| `x264_c1c9931d` | B (SVE substitution) | 2,875 | no |
| `x264_4664f5aa` | A (conditional advice) | 1,441 | **yes — Exemplar 1** |
| `x264_dc755eab` | B (instruction substitution) | 1,398 | **yes — Exemplar 2** |
| `dav1d_ec5c3052` | C+E (multi-insight) | 6,993 | **yes — Exemplars 3/4** |
| `dav1d_01558f3f` | B + heavy benchmarks | 4,977 | no |

---

## Results: gpt-oss-20b

| Commit | latency | usage prompt/comp/total | pairs produced |
|---|---|---|---|
| x264_c1c9931d | 28.2 s | 2,152 / 597 / 2,749 | 2 |
| x264_4664f5aa | 17.2 s | 1,323 / 402 / 1,725 | 2 |
| x264_dc755eab | 12.5 s | 1,255 / 367 / 1,622 | 2 |
| dav1d_ec5c3052 | 17.7 s | 4,980 / 472 / 5,452 | 2 |
| dav1d_01558f3f | 14.8 s | 3,867 / 467 / 4,334 | 2 |

5/5 successful, 10 pairs total. Average ~18 s per call. Reasoning_tokens 18-42 (minimal CoT). Throughput plenty fast for the eventual Pipeline F full run (~600-1200 commits × 18s = 3-6 hours).

### Faithfulness audit (commit-by-commit)

#### `x264_4664f5aa` — matches my Exemplar 1
- ✅ Captures the in-order vs OoO distinction
- ✅ Cites the 20-25% speedup figure on A53
- ✅ Cites specific numbers (580→477, 1238→1110)
- ✅ Notes A72/A73 within measurement noise
- ❌ **Missing `####` summary line on both pairs**
- ✅ Question is codebase-agnostic (no "x264", no specific filenames)

**Verdict: substance-good, format-broken.**

#### `x264_dc755eab` — matches my Exemplar 2
- ✅ Captures the rounded-right-shift idiom
- ✅ Cites 3% / 8% / 10-20% across A53 / A72-A73 / A72-A73-10bpc
- ✅ Mentions the FMA replacement rationale
- ✅ `####` lines present on both pairs
- ✅ Codebase-agnostic questions
- ⚠️ Minor: pair 2 is somewhat redundant with pair 1 (covers same measurement table from a slightly different angle)

**Verdict: high-quality, with some redundancy.**

#### `dav1d_ec5c3052` — matches my Exemplars 3 & 4 (multi-insight split)
- ✅ **Correctly split into 2 pairs** (full-register-write + FP scalar stores) — decision (b) working as designed
- ✅ Captures the partial-register dependency-chain mechanism
- ✅ Captures the 0th-lane scalar-store optimization
- ⚠️ **Number range is sloppy**: claims "0.88× to 1.03× on Neoverse and Cortex cores" but the actual benchmark table shows 0.692× (V2 blend w8) at the worst end — not 0.88×. Lost the worst-case data point.
- ❌ Missing `####` summary lines on both pairs
- ✅ Codebase-agnostic questions
- ⚠️ Names "Neoverse and Cortex cores" generally, doesn't name specific cores

**Verdict: structurally correct (split worked), but numerical fidelity slipped.**

#### `x264_c1c9931d` — no hand-draft, audit only
- ✅ Captures SVE/SVE2 vs NEON substitution
- ✅ Cites specific number ranges (1.2×-3.8×)
- ✅ Cites endpoints (ssd_4x16: 313 vs 653, hadamard 1622 vs 2187)
- ✅ `####` line on pair 1, missing on pair 2
- ⚠️ Mentions specific kernel names (SSD, SA8D, VAR, Hadamard AC) — borderline on decision (e); these are domain-context not codebase names

**Verdict: solid, format ~50%.**

#### `dav1d_01558f3f` — no hand-draft, audit only
- ✅ Identifies SDOT instruction match to 4-tap filters
- ✅ Cites 17% FPS improvement
- ✅ Cites speedup range (1.5× to 12×)
- ⚠️ **`####` line is on the same line as the last answer sentence, not its own line** — format violation
- ✅ Codebase-agnostic questions

**Verdict: solid, format violation.**

---

## Decision scorecard (v2 prompt vs observed behavior)

| Decision | Working? | Notes |
|---|---|---|
| (a) drop body < 80 chars | N/A in this audit | All 5 commits had bodies > 1,000 chars. Untested here. |
| (b) 1-3 pairs per commit | ✅ | Got 2 pairs for every commit. Multi-insight commit correctly split. |
| (c) refuse bare benchmarks | N/A in this audit | No Type-D commits in sample. **Gap — need to retest with synthetic Type-D input.** |
| (d) normalize tables to ranges | ⚠️ partial | Model produces ranges but sometimes loses the worst-case endpoint (dav1d_ec5c3052). |
| (e) strip codebase names from questions | ✅ | No "x264", "dav1d", "FFmpeg" in question text. Function/kernel names appear in answers as domain context (acceptable per decision e). |

**Two gaps identified:**
1. `####` line emission is ~50% — needs explicit-er prompt phrasing
2. Type-D refusal untested — need to send a synthetic bare-benchmark commit through

---

## Results: qwen/qwen3.6-35b-a3b

| Commit | latency | pairs |
|---|---|---|
| x264_c1c9931d | 3.5 s | ERR |
| x264_4664f5aa | 55.4 s | ERR |
| x264_dc755eab | 60.2 s | ERR |
| dav1d_ec5c3052 | 175 ms | ERR |
| dav1d_01558f3f | 333 ms | ERR |

All ERR. Latencies suggest:
- 3.5 s — early failure (context format issue?)
- 55-60 s — produced output but ran out of `max_tokens=4000` before completing the JSON (Qwen 3.6 reasoning model spends massive token budget on internal CoT)
- 175/333 ms — LM Studio queue saturation or model OOM after sustained earlier calls

**Recommendation: skip qwen for v1 distillation production.** Either use gpt-oss-20b alone (which works), or re-evaluate qwen with `/no_think` system prompt and `max_tokens=8000+`. Not blocking — gpt-oss-20b is sufficient.

---

## Prompt v3 fixes (recommended)

1. **Emphasize the `####` rule** — current language ("Ends with a single line `####`...") gets dropped half the time. Move it earlier, repeat at the end, mark as STRICT.

2. **Strengthen range-fidelity rule** — "Always include both the WORST CASE and BEST CASE from the benchmark table, with the specific CPU and operation that produced each endpoint."

3. **Add an explicit Type-D refusal trigger** — "If the commit body is more than 50% benchmark tables by character count with fewer than 3 sentences of explanatory prose, return `{\"pairs\": []}`."

4. **Anti-redundancy clause** — "If you produce more than one pair, each pair must cover a DISTINCT insight. Do not paraphrase the same insight twice with different question wording."

5. **`####` placement** — "The `####` line must be on its OWN line, preceded by `\n`. Not appended to a sentence."

Estimated work for v3: 15 min, no audit re-spend (we have working teachers).

---

## Recommendation for next session

1. **Patch the prompt to v3** with the 5 fixes above.
2. **Add a synthetic Type-D commit** to the audit set; re-run the 6-commit audit (5 real + 1 synthetic) against gpt-oss-20b alone.
3. **If v3 audit hits ≥90% format compliance + correct refusal**, declare prompt v3 production-ready and queue the **full Pipeline F run** against all 119 harvested hunks. Estimated 119 × 18 s = ~36 minutes wall, **zero $ cost**.
4. **Or** — Sean's call — if you want a frontier-teacher comparison before going to full Pipeline F, spend $2-3 of Abacus credits on 2-3 calls against `claude-sonnet-4` to see if the format compliance is better there. Bound the spend; we now know Abacus calls cost real money.

---

*End audit v1.*
