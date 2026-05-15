# Distillation Prompt Sketch — v1

**Date:** 2026-05-15
**Purpose:** De-risk Pipeline F (commit-body → Q&A distillation) BEFORE spending Abacus credits. Draft the prompt template by hand against today's top harvested hunks; surface edge cases the prompt will need to handle.

---

## Why this sketch exists

Today's feasibility harvest (`harvest_feasibility_day1.md`) found that
**rationale lives in commit body text**, not in inline asm comments. That
changes the input shape for Pipeline F. The corpus_plan had assumed the
input was "annotated code"; the reality is "commit message body + diff
context."

Before we spend the $20 Abacus credits on a 50-Q&A audit run, we should:
1. Write the prompt against actual commit-body input
2. Hand-produce the kind of Q&A we want the teachers to produce
3. Note edge cases the prompt has to handle
4. THEN spend credits to see if frontier teachers actually match what we hand-produced

---

## The prompt template (v1)

```
You are reformulating commit messages from expert assembly programmers
into Q&A training pairs for a small language model. The student model
will produce ARM64 assembly with retrieval scaffolding handling
language-specific syntax.

The commit body below describes an optimization technique. Produce ONE
Q&A pair where:

1. The QUESTION asks about the underlying principle, idiom, or pattern
   (not the specific commit). Phrase it as a generally-applicable
   question another engineer might ask. Avoid mentioning the specific
   code base, function names, or commit identifiers in the question.

2. The ANSWER preserves the expert's claim verbatim in substance —
   INCLUDING the measurement numbers if present — but rephrases as
   instructional prose. 3-7 sentences. End with a single line
   `#### <single-sentence summary>` capturing the takeaway.

CRITICAL constraints:
  - Do NOT add facts not in the commit body. The commit body is the
    truth source. If it's wrong, the Q&A is wrong.
  - Do NOT drop the measurement numbers. They are the evidence.
  - The `#### ` line must be exactly `#### <prose>` with no trailing
    punctuation.
  - Return strict JSON. No preamble, no markdown fences.

INPUT:
  commit_subject: {commit_subject}
  commit_body: |
    {commit_body}

OUTPUT (strict JSON):
  {"question": "...", "answer": "..."}
```

---

## Hand-drafted exemplars (3)

These are what we WANT the teachers to produce. If a teacher's output
deviates substantively from these on the same input, we have a prompt
problem.

### Exemplar 1 — `x264 4664f5aa` (in-order vs out-of-order scheduling)

**Input subject:** *aarch64: Improve scheduling in sad_x3/sad_x4*

**Input body (excerpt):** "around a 20-25% speedup on Cortex A53 for
the small sizes (much smaller difference for bigger sizes though), while
it doesn't make much of a difference at all (mostly within measurement
noise) for the out-of-order cores (A72 and A73)."

**Hand-drafted Q&A:**

```json
{
  "question": "When optimizing ARM64 NEON code (e.g., sad_x3/sad_x4 sum-of-absolute-differences kernels), how does manual instruction scheduling affect performance differently on in-order cores like Cortex-A53 versus out-of-order cores like Cortex-A72/A73?",
  "answer": "On in-order cores (Cortex-A53), manual instruction scheduling can produce a 20-25% speedup for small block sizes because the core cannot itself reorder dependent operations — exposing instruction-level parallelism is the programmer's job. The benefit shrinks substantially at larger block sizes where loop overhead dominates ILP. On out-of-order cores (Cortex-A72, A73), the same scheduling rewrite produces a difference within measurement noise — the OoO engine already extracts the parallelism the manual schedule was creating. This means hand-scheduling effort is far more valuable when targeting in-order cores; on OoO targets it should be evaluated against the maintenance cost.\n#### Manual scheduling pays off on in-order ARM cores (A53: ~20-25% for small blocks); on OoO cores (A72/A73), the engine absorbs the same gain"
}
```

### Exemplar 2 — `x264 dc755eab` (rounded right shift idiom)

**Input subject:** *aarch64: Use rounded right shifts in dequant*

**Input body (excerpt):** "Don't manually add in the rounding constant
(via a fused multiply-add instruction) when we can just do a plain
rounded right shift. [...] around 3% faster on the Cortex A53, around
8% faster for 8bpc on Cortex A72/A73, and around 10-20% faster for
10bpp on A72/A73."

**Hand-drafted Q&A:**

```json
{
  "question": "In an ARM64 NEON dequantization loop where output requires rounded right-shifting, when should you prefer a single rounded-right-shift instruction (URSHR / SRSHR) over a fused multiply-add followed by a plain right-shift?",
  "answer": "Always prefer the rounded-right-shift instruction when the operation reduces to 'multiply, add rounding constant, shift right.' The dedicated rounded-shift instruction collapses the rounding-constant injection into the shift in a single micro-op, freeing the multiplier and reducing register pressure on the inner loop. Measured impact on H.264 dequant kernels: ~3% faster on Cortex-A53, ~8% faster for 8bpc on Cortex-A72/A73, and 10-20% faster for 10-bit-depth dequant on A72/A73. The 10bpc gain is largest because the shift runs on wider operands, amplifying the saved fused-multiply-add per inner step.\n#### Use rounded-right-shift directly; don't simulate rounding with a manual FMA plus plain shift"
}
```

### Exemplar 3 — `dav1d ec5c3052` (partial-register dependency chains)

**Input subject:** *AArch64: Optimize lane load/store in MC functions*

**Input body (excerpt):** "Partial register writes can create long
dependency chains, which can reduce performance on out-of-order CPUs.
This patch removes most of these kinds of problems in MC functions by
filling the full register before other lane loading instructions. Most
lane extracting stores can also be optimized using FP scalar stores
when the 0th lane would be extracted." + benchmark tables.

**Hand-drafted Q&A:**

```json
{
  "question": "What is the performance cost of partial-register lane writes in ARM64 NEON code on out-of-order cores, and how should you structure lane loads and extracting stores to mitigate it?",
  "answer": "Lane-by-lane writes (MOV V0.S[1], W0) create a false dependency chain: the destination vector register cannot be renamed because each new lane write merges with the prior state. On out-of-order cores (Neoverse V1/V2, Cortex-X3/X1), this serializes operations the OoO engine would otherwise parallelize. The mitigation is to fill the full register first with a non-merging write (MOV V0.16B, V_src.16B or LDR Q0, [src]) before any lane-write sequence; subsequent lane writes still merge but the dependency chain restarts at the full-width write. For extracting stores where only the 0th lane is needed, prefer the FP scalar store form (STR S0, [dst]) over the lane-extract form (ST1 {V0.S}[0], [dst]) — same effect, cheaper micro-op. Measured impact on AV1 motion-compensation kernels: 8-30% speedup across blend/avg/mask/w_mask functions in 8bpc; up to 65% speedup on 16bpc blend variants where the dependency chain was longest.\n#### Avoid partial-register writes in hot NEON code — start with a full-register write before any lane updates, and use FP scalar stores for 0th-lane extracts"
}
```

---

## Pattern catalog (what teachers will see)

Across just today's 5 sampled hunks, three commit-body shapes appear:

### Type A — Conditional advice
**Shape:** "X helps for context A but not for context B"
Example: scheduling helps in-order, indifferent on OoO.
Q&A target: question is "how does X differ between A and B?", answer covers both.

### Type B — Instruction substitution
**Shape:** "Use Y instead of X when condition holds, with measurement"
Example: rounded-right-shift instead of FMA+shift.
Q&A target: question is "when should you prefer Y over X?", answer covers when + why + measurements.

### Type C — Pattern / anti-pattern
**Shape:** "Avoid Z because of mechanism M; do W instead, with measurement"
Example: partial-register writes create false dependency; full-width writes break the chain.
Q&A target: question is "what is the cost of Z and how to mitigate?", answer covers the mechanism + the alternative + measurements.

These three types cover ~80% of what the harvest will yield. The
remaining 20% is two harder categories:

### Type D — Bare benchmark dump (no rationale)
**Shape:** Title is the technique, body is a benchmark table only.
Example: `01558f3f` "Add HBD subpel filters using 128-bit SVE2" — the
body is mostly numbers; the rationale is implicit (SVE2 is faster).
Q&A target: ?? — we need the prompt to either (a) refuse to produce
Q&A, or (b) extract the technique name from the subject and ask "when
is SVE2 worth the binary-size cost?" without inventing rationale not
in the body.

### Type E — Multiple insights in one commit
**Shape:** Body teaches 2+ separable lessons.
Example: `ec5c3052` actually has TWO claims (full-register first AND
FP scalar stores for 0th lane). We folded both into one Q&A above, but
the cleaner approach may be TWO Q&A pairs per such commit.

---

## Open questions for the prompt

1. **Type D handling:** Should the prompt produce Q&A pairs when the body
   is bare benchmarks? My read: refuse with a `null` Q&A and let the
   pipeline drop these. We can revisit if we lose too many records.

2. **Type E handling:** Should the prompt produce ONE Q&A per commit
   (forcing fold) or MULTIPLE when separable lessons are present?
   Recommend: instruct the prompt to produce 1-3 pairs as needed, with
   a strict cap.

3. **Subject vs body weight:** Some commits have great bodies; some have
   great subjects + thin bodies. Current prompt weights body heavily.
   If subject is "Use ldp/stp for paired loads" with empty body, we may
   want to still produce a Q&A — but then we're relying on the teacher's
   prior knowledge, not the expert's. **Decision proposed: drop commits
   with body < 80 chars before sending to the teacher.** Keeps the
   "expert is truth source" property.

4. **Measurement-table normalization:** Many bodies have multi-table
   benchmark dumps (e.g., 7 CPU cores × 10 ops). Verbatim preservation
   would blow up token budget. Recommend: prompt instructs the teacher
   to extract the RANGE ("8-30% across functions") not the table.

5. **Cross-source naming hygiene:** Don't leak `dav1d` or `x264` into
   the question text — keeps the model general-purpose for code that's
   not from these codebases.

---

## Next-step recommendations

1. **Bake decisions 1-5 above into the prompt** (estimated +10 lines).
2. **Hand-draft 3 more exemplars from harder cases** (Type D, Type E,
   and a 100-char body) so we have edge-case ground truth.
3. **Spend $5-8 of Abacus credits** running 10-15 audit Q&A pairs across
   3 teachers (gpt-4o, claude-sonnet-4, gemini-2.5-flash). That's a
   feasibility check, not the full audit — saves ~$12-15 for a v2 pass.
4. **Compare teacher output to hand-drafts.** Disagreements = prompt
   bugs.

Before going further, let me confirm with Sean:
- (a) Drop commits with body < 80 chars: yes / no
- (b) Allow 1-3 Q&A pairs per commit: yes / no
- (c) Refuse to distill bare-benchmark commits: yes / no
- (d) Normalize measurement tables to ranges in the answer: yes / no
- (e) Strip codebase names (`x264`, `dav1d`, `FFmpeg`) from questions: yes / no

---

*End sketch v1.*
