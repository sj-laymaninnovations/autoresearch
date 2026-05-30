# pace_mathgpt

Pipeline that turns a mathgpt training jsonl into a PACE curriculum.

## Why

Mathgpt's merged `training_data.jsonl` is a mix of clean and toxic rows:
older generators produce placeholder-CoT traces like `"752 ... 56 = 696"` or
`"Result = 3628800"` regardless of the actual problem. Feeding those into
PACE's Stage 3 or Stage 6 bakes the wrong reasoning into the weights with
much higher gradient signal than the flat format does — the exact failure
mode the canvas warns against ("erroneous chains will corrupt them quickly",
[canvas.md:134](../../../promptengineering/canvas.md)).

This pipeline filters those rows out before PACE rewrites them.

## Run

```
python math_lab/pace_mathgpt/run.py math_lab/results/training_data.jsonl
```

Defaults tuned for mathgpt's sub-100M regime:

| Knob | Default | Why |
|---|---|---|
| `--profile` | `tiny` | Scaffolding ratios 1.5-2.0:1 instead of the canvas's 3-5:1, because tiny models have tight context windows and few attention heads. |
| `--envelope-style` | `lite` | Newline labels (`Reasoning:`, `Answer:`) instead of XML tags. Preserves Layer 0 arithmetic basins by keeping raw numerals adjacent to the gradient signal. |
| `--critique-bank` | `math` | Stage 5 uses arithmetic-error templates (wrong product, off-by-one, missed carry, etc.) instead of prose critiques. |
| `--stages` | `1,2,3,5,6` | Skips Stage 4 (tool use), 7 (RAG), 8 (multi-turn). Mathgpt's job is reasoning, not agentic behaviour. |

## Pipeline steps

1. **verify_solution.py** — Arithmetic auditor. Evaluates each `LHS = RHS`
   step, confirms `####` matches `answer`, rejects placeholder-CoT patterns
   (`...`, `Result =`, arrows). Writes `audit_out/{clean.jsonl,
   rejected.jsonl, audit_stats.json}`. Policy is **REJECT, never
   auto-repair** — auto-fixing could introduce new wrong arithmetic.

2. **adapt.py** — Schema mapper. Converts mathgpt
   `{problem, solution, answer, level}` to PACE
   `{instruction, output, reasoning, thinking, difficulty, metadata}`.
   Writes `adapted_out/adapted_with_reasoning.jsonl`.

3. **PACE compiler** — Runs once per emitted stage. Uses the tuned defaults
   above. Writes `dist/stage_N_curriculum.jsonl` plus phase concatenations
   in canvas ingestion order.

4. **PACE validator** — Schema check, surfaces quality flags
   (below-ratio, missing reasoning, missing thinking) as informational
   warnings. Stages that weren't emitted show up as `[SKIP]`, not `[FAIL]`.

## What the output looks like

Stage 3 (CoT) on `834 - 345`:

```
User:
What is 834 - 345?
---
Reasoning:
834 - 345 = 489

Answer:
489
```

Stage 5 (contrastive critique) on the same problem:

```
User:
What is 834 - 345?
---
Draft:
479

Critique:
- A carry was not propagated in the addition.
- A borrow was not applied in the subtraction, producing the wrong digit.

Revised:
489
```

The draft contains a real wrong number (489 - 10 = 479, simulating a missed
borrow). The critique is bound to the specific flaw type. The revision is
the correct answer. None of these properties held in earlier versions of
PACE — the demo would have produced a draft like `"Great question! 489"`
critiqued randomly for "uses jargon", which trains the model to ignore
critique content.

## Known considerations

* **Below-ratio warnings on Stage 3 simple-arithmetic rows.** Single-line
  reasoning like `"834 - 345 = 489"` can't hit the 2:1 scaffolding ratio
  even at the tiny profile. Two reasonable responses: filter those rows
  for Stage 3 only, or accept them — the model still benefits from seeing
  real arithmetic at all.

* **The combinatorics rows are doubly wrong.** Even the `answer` field is
  wrong: every "arrange N items" question has `answer: 3628800` (= 10!)
  regardless of N. The auditor catches the bad trace (placeholder_cot) and
  drops the row, so PACE never sees them. But the underlying generator
  needs to be fixed if you want combinatorics back in the training set.

* **Stage 6 thinking trace is derived from the same reasoning.** Mathgpt
  data has no separate `thinking` field, so the adapter synthesises one by
  prefixing the reasoning with `"Let me work through this."` This is
  cheap and preserves the real arithmetic, but a future improvement would
  be to distill genuine first-person thinking traces from a teacher model.
