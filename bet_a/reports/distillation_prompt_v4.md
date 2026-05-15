# Distillation Prompt — v4 (small surgery from v3)

**Date:** 2026-05-15
**Supersedes:** `distillation_prompt_v3.md`
**Audit feeding this version:** `distillation_audit_v2.md`

## Changes since v3

| Fix | v3 symptom | v4 change |
|---|---|---|
| Soften anti-redundancy | gpt-oss-20b collapsed multi-insight commits to 1 pair (e.g. dav1d_ec5c3052) | Replace "do not paraphrase" with explicit instruction to produce one pair per separable technique |
| Move Type-D refusal out of prompt | Both teachers (oss + claude) hallucinate rationale for bare-benchmark inputs rather than refusing | Drop the per-prompt Type-D criteria; refusal handled by `bet_a/preprocessor.py` BEFORE the teacher sees the commit |

Everything else from v3 (output format, question codebase-stripping,
range fidelity, `####` line rules) carries over unchanged.

## The v4 prompt (locked)

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
Produce 1-3 Q&A pairs.

PAIR-COUNT RULE (this is the load-bearing rule):
  - Each pair must add a NEW idea not present in earlier pairs.
  - If the commit body describes 2 or 3 separable techniques
    (e.g., "do X" AND "in case Y, also do Z"), produce ONE PAIR
    PER TECHNIQUE. Folding distinct techniques into a single pair is
    INCORRECT and will degrade training quality.
  - If only ONE insight is present, return ONE pair.
  - Do NOT paraphrase the same insight twice with different wording.

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

REFUSAL:
  - If the commit body is too vague (under 3 explanatory sentences) to
    extract a generally-applicable idiom, return:
      {"pairs": [], "notes": "body too vague"}
  - Refusal is rare; most commits have substantive rationale.

REMEMBER: every answer ends with `\n#### <one-line summary>` on its
own line. No exceptions.

INPUT:
  commit_subject: {commit_subject}
  commit_body: |
    {commit_body}
```

## Type-D preprocessor (separate from prompt)

The decision-c "refuse Type-D" requirement now lives in
`bet_a/preprocessor.py` as a heuristic classifier that runs BEFORE
the teacher call:

```
classify(body):
  lines = body.split('\n')
  benchmark_lines = lines matching one of:
    - r'^\s*\S+:\s+\d'         (label colon numbers, e.g. "ssd_4x4: 250")
    - r'^\s*[\d.]+\s+[\d.]+'   (two+ numbers in a row)
    - lines with >= 3 numeric tokens
  prose_lines = lines with >= 3 English words, no leading digit
  ratio = len(benchmark_lines) / max(1, len(lines))
  if ratio > 0.5 and len(prose_lines) < 3:
    return "type_D"
  return "process"
```

This catches the synthetic-typeD case (3 lines, all numeric, 0 prose
lines) without depending on the teacher's discretion.
