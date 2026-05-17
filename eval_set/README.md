# MathGPT Held-Out Evaluation Set — v1

**Project:** Layman Innovations / MathGPT
**Purpose:** Methodologically-clean ground-truth instrument for benchmarking
50M–300M parameter reasoning specialists across all future training runs.

---

## Design Principles

1. **Zero contamination with training corpora.** Problems are drawn exclusively
   from sources that post-date the distillation cutoff of all HuggingFace
   reasoning corpora we use (OpenR1-Math-220k, AM-Thinking-v1-Distilled,
   OpenThoughts3, Polaris-Dataset-53K, NuminaMath, s1K). Accepted sources:
   AIME 2025+, AIME 2026+, Putnam 2025+, USAMO 2025+, and hand-authored
   originals.

2. **Deterministic ground truth only.** Recorded answers come from official
   competition answer keys or from hand-derived solutions verified with SymPy.
   LLM consensus is never used as ground truth.

3. **Verifiable answers.** Every problem has a single, unambiguous numerical
   or symbolic answer that can be checked programmatically (`verify_answer.py`).

4. **Signal across the small-model range.** Difficulty spans easy / medium /
   hard so we detect meaningful capability differences at 50M–300M params,
   not just floor or ceiling effects.

---

## Directory Structure

```
eval_set/
├── problems/               # One JSON file per problem (EVAL-001.json, ...)
├── verification/
│   ├── schema.py           # Pydantic schema + validator
│   ├── verify_answer.py    # Programmatic answer verification (SymPy + string)
│   ├── contamination_check.py   # Streaming fingerprint comparison vs training corpora
│   ├── contamination_report.md  # Output of contamination check
│   └── solutions/          # Worked solutions for hand-authored problems
├── review/
│   ├── generate_review_prompts.py
│   ├── review_prompt_chatgpt.md
│   ├── review_prompt_gemini.md
│   └── review_prompt_claude.md
├── manifest.json           # Running index: counts, distributions, schema version
└── README.md               # This file
```

---

## Sources and Post-Cutoff Rationale

| Source | Why post-cutoff |
|---|---|
| AIME 2025 (Feb 2025) | After knowledge cutoff of all 2024-era distillation runs |
| AIME 2026 (Feb 2026) | Definitively post-cutoff; not in any public dataset as of writing |
| Hand-authored | Original content; zero contamination risk |

**Excluded sources (training-set contamination risk):**
- AIME pre-2025 (extensively in NuminaMath, OpenR1, and others)
- MATH-500 (directly in multiple training corpora)
- GSM8K (ubiquitous in all SFT and distillation datasets)
- AMC 8/10/12 pre-2025 (in NuminaMath)
- Putnam pre-2025 (in various reasoning corpora)

---

## Verification Approach

**Primary (deterministic):**
- `verify_answer.py` uses SymPy for symbolic equivalence checking
- Falls back to normalized string comparison for integers and simple fractions
- Handles common format variations (decimals ↔ fractions, mixed numbers, etc.)

**Secondary (LLM review):**
- `generate_review_prompts.py` produces prompts for ChatGPT / Gemini / Claude
- Reviewer solves independently, then we compare
- Used to flag ambiguous problems, not to set ground truth

**Contamination check:**
- `contamination_check.py` streams first 10K rows from each training corpus
- Compares normalized fingerprints with cosine similarity
- Any match > 0.85 flagged for human review

---

## Adding New Problems

1. Create `problems/EVAL-NNN.json` following the schema in `verification/schema.py`
2. If the answer comes from an official key, set `"verification_method": "known_key"`
3. If hand-authored, add worked solution to `verification/solutions/EVAL-NNN_solution.py`
   and confirm `verify_answer.py` returns PASS
4. Run `contamination_check.py` on the new problem
5. Update `manifest.json` (or re-run the manifest generator)

---

## Running the LLM Review

```bash
# Generate prompts
python review/generate_review_prompts.py

# Then paste review_prompt_chatgpt.md into ChatGPT, etc.
# Collect model answers and compare against our recorded answers.
```

---

## Known Limitations

- Contamination check is a streaming sample (first 10K rows), not exhaustive
- AIME 2026 problems require manual entry of official text + answer key
- Hand-authored problems may inadvertently resemble textbook problems in style

---

## Schema Version

**v1** — 2026-05-17
