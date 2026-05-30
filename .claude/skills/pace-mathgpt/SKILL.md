---
name: pace-mathgpt
description: Use this skill whenever the user wants to prepare mathgpt training data through PACE, audit training jsonl arithmetic correctness, run the curriculum compiler, integrate PACE outputs with finetune.py, or troubleshoot training-data quality. Triggers on phrases like "PACE", "curriculum", "audit the training data", "prompt corrections", "fix training data", "finetune-ready data", "Stage 3 / Stage 5 / Stage 6 curriculum", "lite envelope", "math critique bank". Also use when the user starts a training run and you suspect the data may contain placeholder-CoT rows or wrong arithmetic.
---

# PACE for mathgpt

PACE (Prompt Archaeology & Curriculum Engine) rewrites raw `{problem, solution, answer}` Q&A into 8 progressive SFT curriculum stages, each targeting a known small-model failure mode. The mathgpt integration lives in [math_lab/pace_mathgpt/](../../../math_lab/pace_mathgpt/) and is self-contained inside the autoresearch repo — no separate `promptengineering` repo required.

## When to invoke this skill

1. **Before any new training run** on `training_data.jsonl` or any merged mathgpt dataset. The auditor finds ~52% toxic rows in current `training_data.jsonl`; training without filtering bakes wrong arithmetic into the weights with high gradient signal.
2. **When the user asks about training data quality, prompt corrections, or curriculum.**
3. **When a training run produces a model that "rushes to the answer" or gets simple arithmetic wrong** — these are the failure modes PACE Stage 3 (CoT) and Stage 6 (latent reasoning) are designed to fix.
4. **When the user adds a new datagen script.** Audit the new generator's output before merging into `training_data.jsonl`.

## What to actually do

### Default workflow (full pipeline)

```bash
python math_lab/pace_mathgpt/run.py math_lab/results/training_data.jsonl
```

Produces (under `math_lab/pace_mathgpt/pipeline_out/`):

| Directory | Contents | Use |
|---|---|---|
| `audit_out/` | `clean.jsonl`, `rejected.jsonl`, `audit_stats.json` | Inspect what was filtered and why |
| `adapted_out/` | `adapted_with_reasoning.jsonl` | Intermediate PACE-shaped rows |
| `dist/` | `stage_{1,2,3,5,6}_curriculum.jsonl`, `phase_{1,2,3,4}_curriculum.jsonl` | Raw PACE curriculum (lite envelope) |
| `finetune_ready/` | `stage_N.jsonl`, `phase_N.jsonl` | **What `finetune.py` consumes directly** |

### Training against the curriculum

`finetune.py` expects `{problem, solution_cot, solution, answer}` rows. The `finetune_ready/` outputs match that schema exactly. Common training calls:

```bash
# Train against a single stage (the most common use):
python math_lab/finetune.py --data math_lab/pace_mathgpt/pipeline_out/finetune_ready/stage_3.jsonl

# Train through the canvas's prescribed phase order:
python math_lab/finetune.py --data math_lab/pace_mathgpt/pipeline_out/finetune_ready/phase_1.jsonl
# ...later resume from a checkpoint on phase_2, etc.
```

The canvas-prescribed phase order ([math_lab/pace_mathgpt/pace_lib/canvas.md:446-457](../../../math_lab/pace_mathgpt/pace_lib/canvas.md)):

1. **Phase 1** (Format & Syntax) — Stages 1, 4 (mathgpt: just Stage 1)
2. **Phase 2** (Constraints & State) — Stages 2, 8 (mathgpt: just Stage 2)
3. **Phase 3** (Reasoning & Grounding) — Stages 3, 6, 7 (mathgpt: 3 and 6)
4. **Phase 4** (Self-Correction) — Stage 5

### Useful one-offs

* **Audit a new dataset without compiling**:
  ```
  python math_lab/pace_mathgpt/verify_solution.py math_lab/results/<new>.jsonl --out-dir /tmp/audit
  ```
* **Preview-mode (first N rows only)**:
  ```
  python math_lab/pace_mathgpt/run.py <input.jsonl> --sample 200
  ```
* **Different scaffolding profile for a larger model**:
  ```
  python math_lab/pace_mathgpt/run.py <input.jsonl> --profile small
  ```
* **Different envelope style (XML for any non-tiny use)**:
  ```
  python math_lab/pace_mathgpt/run.py <input.jsonl> --envelope-style xml
  ```

## Important defaults to keep

These defaults exist because of empirical findings; do not change them without checking with the user:

* **`--profile tiny`** — scaffolding ratios 1.5-2.0:1 instead of the canvas's 3-5:1, because mathgpt is sub-100M params with tight context windows.
* **`--envelope-style lite`** — preserves Layer 0 arithmetic basins ([canvas.md:478-484](../../../math_lab/pace_mathgpt/pace_lib/canvas.md)) by keeping raw numerals adjacent to the gradient signal instead of wrapped in `<final_answer>` tags.
* **`--critique-bank math`** — Stage 5 uses arithmetic-error templates (off-by-one, missed carry, wrong operation, decimal misplaced, etc.) instead of prose critiques.
* **Stages 4, 7, 8 are skipped** — tool use, RAG, multi-turn are not mathgpt's job.
* **Auto-repair is OFF.** The auditor REJECTS dirty rows; it does not attempt to fix them. Auto-fixing risks introducing new wrong arithmetic, which is exactly the failure mode this exists to prevent.

## Known data-quality findings (as of 2026-05-29)

Running the auditor against the current `math_lab/results/training_data.jsonl` (5,087 rows):

* **2,421 clean (47.6%)** — 100% of `cot_*` domains plus partial `arithmetic`
* **2,461 rejected: placeholder_cot** — `algebra`, `combinatorics`, `number_theory`, half of `arithmetic` use traces like `"752 ... 56 = 696"` or `"Result = 3628800"` instead of real arithmetic
* **205 rejected: step_arithmetic_error** — the `"Solve for x"` generator emits `"730 + 418 = 417.5"` style traces with wrong intermediates
* **Combinatorics is doubly broken:** every `"arrange N items"` question has `answer: 3628800` (= 10!) regardless of N. The auditor catches the bad trace, but the underlying generator needs fixing before combinatorics can be re-included.

If the user's training failure modes look like "model rushes to answer", "model gets simple arithmetic wrong", or "val_loss looks good but task accuracy is bad", the most likely cause is that they were training on the toxic 52% — feed PACE output instead.

## Resyncing with upstream PACE

The PACE library lives vendored at [math_lab/pace_mathgpt/pace_lib/](../../../math_lab/pace_mathgpt/pace_lib/). Upstream is at `F:\projects\promptengineering`. To pull upstream changes:

```bash
cp F:/projects/promptengineering/pace_compiler.py math_lab/pace_mathgpt/pace_lib/
cp F:/projects/promptengineering/validate.py math_lab/pace_mathgpt/pace_lib/
cp F:/projects/promptengineering/canvas.md math_lab/pace_mathgpt/pace_lib/
```

The PACE compiler has had substantial correctness fixes (see [project_pace_compiler_fixes.md](~/.claude/projects/F--projects-autoresearch/memory/project_pace_compiler_fixes.md) in user memory). If resyncing from a different upstream, verify those fixes are still in place: `Stage 5 critiques bound to flaw category`, `Stage 7 doesn't synthesise context from answer`, `enforce_answer_last doesn't auto-pad with generic filler`.

## Do NOT

* **Do not feed raw `training_data.jsonl` to `finetune.py` directly** without auditing first. Half of it is toxic. Run the PACE pipeline.
* **Do not silently auto-repair rejected rows.** They are quarantined in `rejected.jsonl` for a reason. Fixing the upstream generator is the right move; auto-patching the rejected rows is not.
* **Do not change the lite-envelope default for tiny models** without measuring. The XML envelope wraps every numeric answer in `<final_answer>` tags, putting 6+ structural tokens between the model's number-line manifold and the gradient signal at the answer position. For mathgpt this matters.
* **Do not skip the validator step.** The `[WARN]` lines surface real data-quality issues (below-ratio rows, missing reasoning) that should inform training decisions.
