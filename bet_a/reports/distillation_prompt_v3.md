# Distillation Prompt — v3 (post-audit-v1 fixes)

**Date:** 2026-05-15
**Supersedes:** `distillation_prompt_v2.md`
**Audit feeding this version:** `distillation_audit_v1.md`

## What changed since v2

| Fix | Audit-v1 symptom | v3 change |
|---|---|---|
| (i) `####` line emphasis | emitted on ~50% of pairs | Move rule earlier, repeat at end, mark STRICT |
| (ii) Range fidelity | dropped worst-case endpoint (0.692× rounded up to 0.88×) | Require BOTH worst-case AND best-case with the specific CPU/op |
| (iii) Type-D refusal trigger | untested in v1 audit; need concrete criterion | "if >50% of body chars are benchmark tables AND <3 explanatory sentences, refuse" |
| (iv) Anti-redundancy | gpt-oss-20b sometimes paraphrased the same insight in pair 1 + pair 2 | "each pair MUST cover a DISTINCT insight" |
| (v) `####` placement | sometimes appended to last sentence, not on its own line | "on its own line, preceded by \n" |

## The v3 prompt (locked)

```
You are reformulating commit messages from expert assembly programmers
into Q&A training pairs for a small (5M-parameter) language model.

STRICT OUTPUT FORMAT (read this first):
  - Return ONE JSON object, no preamble, no markdown fences.
  - Schema: {"pairs": [{"question":"...","answer":"..."}], "notes":"..."}
  - Each answer MUST end with `\n#### <single-sentence summary>` on its own line.
    Not appended to the last sentence. Not on the same line as prose.
  - The `####` line has no trailing punctuation.

TASK:
The commit body below describes one or more optimization techniques.
Produce 1-3 Q&A pairs, one per separable insight.

QUESTION rules:
  - Asks about the underlying principle, idiom, or pattern.
  - Phrased as a question a general ARM64 programmer might ask.
  - Does NOT mention specific codebases (x264, dav1d, FFmpeg, Linux,
    etc.) or specific function/file names.
  - Codebase-agnostic. Re-usable beyond the source commit.

ANSWER rules:
  - 3-7 sentences of instructional prose, then the `####` line.
  - Preserves the expert's claim verbatim in substance — do NOT add
    facts not in the commit body.
  - When the body contains a benchmark table, ALWAYS include BOTH the
    worst-case and the best-case data point with the specific CPU and
    operation that produced each endpoint. Then summarize the range.
    Do NOT dump the verbatim table.
  - MAY cite the domain context that ties to the measurements
    (e.g., "in H.264 dequantization", "for AV1 motion compensation").

PAIR-DISTINCTNESS RULE:
  - If you produce more than one pair, each must cover a DISTINCT
    insight. Do not paraphrase the same insight in two pairs.
  - If only one insight is present, return ONE pair.

REFUSAL RULES (return {"pairs": [], "notes": "<reason>"} if):
  - The commit body is more than 50% benchmark tables (by character
    count) AND contains fewer than 3 explanatory sentences. Bare
    benchmark dumps are not enough to support a generally-applicable
    Q&A.
  - The commit body is too vague (under 3 sentences) to extract a
    generally-applicable idiom.

REMEMBER: every answer ends with `\n#### <one-line summary>` on its
own line. No exceptions.

INPUT:
  commit_subject: {commit_subject}
  commit_body: |
    {commit_body}
```

## Audit-v2 plan

Same 5 real commits from audit-v1 + 1 synthetic Type-D commit (bare
benchmark only, body is pure numbers). Total 6 commits.

Two teachers, same prompt v3:
- `openai/gpt-oss-20b` via local LM Studio (~$0, ~108 s wall)
- `claude-sonnet-4` via Abacus (bounded $ spend, audit-only)

Cost guard: send the smallest 2 commits to Abacus first, measure actual
usage from the response, then decide whether to send the 3rd. If even
the smallest commit blows the budget, stop at 1 Abacus call and report.

Pass criteria for declaring "v3 ready for Pipeline F":
- `####` line on ≥90% of pairs, on its own line
- Type-D commit returns `{"pairs": []}` with a reasonable note
- No pair-internal redundancy (qualitatively assessed)
- Numerical fidelity: worst-case and best-case both cited in ≥80% of
  benchmark-bearing commits

If criteria not met: prompt v4, no further teacher spend until v4 is in.
