# PACE Guide — Reading & Using the Curriculum

This guide is for **humans**. The Claude-facing instructions live in
`.claude/skills/pace-mathgpt/SKILL.md`. Per-script details live in
`math_lab/pace_mathgpt/README.md`. This document explains what PACE *is*,
why it's wired up this way for mathgpt, and what to do with the output.

## What is PACE

PACE rewrites flat `{problem, solution, answer}` Q&A into 8 progressive
SFT curriculum stages. Each stage targets a known small-model failure
mode — premature token emission, sycophantic critique, context neglect,
catastrophic turn forgetting, etc.

The full theory is in `pace_lib/canvas.md`. The short version:

> **Tiny models can't dynamically execute complex prompts at inference
> time, so we must bake those empirical insights directly into the weights
> via curriculum-shaped SFT data.**

## Why this exists for mathgpt specifically

Three reasons:

1. **The current `training_data.jsonl` is 52% toxic.** Older generators
   produce placeholder-CoT traces (e.g. `"752 ... 56 = 696"`) where the
   intermediate "step" is meaningless. Training on these with a flat format
   already costs accuracy; training with PACE's deeper scaffolding would
   amplify the corruption ([canvas.md:134](pace_lib/canvas.md): "erroneous
   chains will corrupt them quickly").

2. **mathgpt's failure modes match the canvas catalogue exactly.** "Rushes
   to answer", "wrong arithmetic with confident reasoning", "val_loss looks
   good but task accuracy doesn't" — these are Stage 3/6 problems. The
   answer-last principle and the thinking-block scaffolding directly target
   them.

3. **Sub-100M params is even further down the small-model spectrum than
   the canvas targets (<3B).** Every PACE failure mode is more severe at
   mathgpt's scale, not less. The defaults are tuned for this.

## What lives where

```
math_lab/pace_mathgpt/
├── verify_solution.py      # arithmetic auditor (REJECT, don't auto-repair)
├── adapt.py                # mathgpt → PACE schema mapper
├── pace_to_finetune.py     # PACE compiled → finetune.py-ready schema
├── run.py                  # one-command orchestrator
├── pace_lib/
│   ├── pace_compiler.py    # vendored PACE compiler (don't edit without resyncing)
│   ├── validate.py         # vendored PACE validator
│   └── canvas.md           # vendored design canvas — read this
├── pipeline_out/           # produced by run.py — committable
│   ├── audit_out/
│   ├── adapted_out/
│   ├── dist/               # raw PACE curriculum
│   └── finetune_ready/     # what finetune.py consumes
├── README.md               # per-script details
└── PACE_GUIDE.md           # this file
```

The PACE library is vendored — autoresearch is fully self-contained.
Upstream lives at `F:\projects\promptengineering`. Resync procedure is in
the skill file.

## End-to-end: from a fresh dataset to a trained model

### 1. Generate or merge the source jsonl

You probably already have one of:
- `math_lab/results/training_data.jsonl` — the merged 5,087-row blob
- A new per-skill `harder_qa_*.jsonl` from `math_lab/datagen/`
- A teacher-distilled CoT file

### 2. Run the pipeline

```bash
python math_lab/pace_mathgpt/run.py math_lab/results/training_data.jsonl
```

Watch for these in the output:

* **Audit stats**: how many rows survived. If <50% clean, the source has
  a generator bug worth fixing (not just a few one-offs).
* **Stage 5 critique diversity**: low diversity (<30%) on huge datasets is
  expected because the math-flaw bank has ~7 categories × ~3 templates
  each. Don't worry unless it dips below 15%.
* **Below-ratio warnings on Stage 3**: single-step arithmetic
  (`834 - 345 = 489`) can't satisfy a 2:1 scaffolding ratio. Either drop
  those rows for Stage 3 specifically, or accept that real arithmetic at
  any length is still better than fake reasoning at perfect ratio.

### 3. Train

`finetune.py` reads `{problem, solution_cot, solution, answer}`. The
`finetune_ready/` outputs match this exactly:

```bash
# Single-stage training (most common for a focused run):
python math_lab/finetune.py \
  --data math_lab/pace_mathgpt/pipeline_out/finetune_ready/stage_3.jsonl

# Phase-by-phase per the canvas curriculum order:
python math_lab/finetune.py \
  --data math_lab/pace_mathgpt/pipeline_out/finetune_ready/phase_1.jsonl
# checkpoint → resume on phase_2.jsonl → phase_3.jsonl → phase_4.jsonl
```

### 4. Evaluate

The audit step does NOT touch held-out validation. Use your existing
novel-pair holdouts ([user_methodology](~/.claude/projects/F--projects-autoresearch/memory/user_methodology.md)
in user memory) — PACE training data was never seen at eval time.

## Reading a PACE row

A Stage 3 row from `dist/stage_3_curriculum.jsonl`:

```json
{
  "prompt": "User:\nWhat is 834 - 345?",
  "response": "Reasoning:\n834 - 345 = 489\n\nAnswer:\n489",
  "stage": 3,
  "description": "Pre-Rationalized CoT (Answer-Last)",
  "difficulty_score": 0.3,
  "envelope_style": "lite"
}
```

* **Envelope is "lite"**: newline labels (`User:`, `Reasoning:`, `Answer:`)
  instead of XML tags. The numeric answer `489` sits adjacent to the
  gradient signal at the end of the sequence, preserving Layer 0 arithmetic
  basins ([canvas.md:478-484](pace_lib/canvas.md)).
* **Answer-last**: the response builds reasoning before emitting the answer.
  Tiny models don't have the latent depth to compute deep arithmetic in one
  forward pass; the auto-regressive reasoning gives them more compute steps.
* **Difficulty 0.3** = level 3 from the source data (`level / 10`).

Same problem, Stage 5 (contrastive critique):

```json
{
  "prompt": "User:\nWhat is 834 - 345?",
  "response": "Draft:\n479\n\nCritique:\n- A carry was not propagated in the addition.\n- A borrow was not applied in the subtraction, producing the wrong digit.\n\nRevised:\n489",
  "stage": 5,
  "flaw_category": "missed_carry_borrow"
}
```

The draft contains a **real wrong number** (489 − 10 = 479, simulating a
missed borrow). The critique **names the specific flaw type** that
produced it. The revision is correct. None of this was true in earlier
PACE — historical drafts were cosmetic ("Great question! 489") and
critiques were random.

## What to do when the audit rejects a lot

If the auditor rejects >40% of a source dataset, the right move depends
on what the rejection reason is:

| Reason | Action |
|---|---|
| `placeholder_cot` | Fix the upstream generator. Examples: replace `"...".join(...)` with the actual operator string in `datagen/<X>.py`. |
| `step_arithmetic_error` | The generator computes one thing and writes another. Fix the trace-emission code. |
| `final_mismatches_answer` | The `answer` field and the `####` line disagree. Usually a bug in answer-computation. |
| `missing_solution` / `missing_problem` | Schema issue. The generator forgot a field. |

**Do NOT auto-repair.** Repairing risks introducing new wrong arithmetic.
Quarantine in `rejected.jsonl` and fix the source.

## Reading the canvas

`pace_lib/canvas.md` is the design doc. The sections most relevant to
mathgpt:

* **Foundational Principle: Answer-Last Token Positioning** (top of file)
  — why every PACE output ends with the answer
* **Epoch 3: Step-by-Step Rationalization** — Stage 3 design
* **Epoch 5: Self-Refinement & Metaprompting** — Stage 5 design
* **Epoch 6: Latent System Reasoning** — Stage 6 design
* **Circuit-Aware Data Design** — Layer 0 arithmetic basins (~line 478)
  and why raw numerals matter for tiny models

You can skip Epochs 4 (tool use), 7 (RAG), 8 (constitutional), 9
(structured output), 10 (multi-turn), 11 (KV-cache) — none of those
stages are emitted for mathgpt.
