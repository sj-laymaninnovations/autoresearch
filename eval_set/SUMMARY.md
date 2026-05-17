# MathGPT Eval Set — Session Summary

**Generated:** 2026-05-17
**Session target:** 30 problems (15 verified + 15 shells); expandable to 50-100

---

## Problem Count

| Category | Total | Verified | Needs Human Entry |
|---|---|---|---|
| Hand-authored originals | 15 | **15** | 0 |
| AIME 2026 I+II | 10 | 0 | 10 |
| AIME 2025 I | 5 | 0 | 5 |
| **TOTAL** | **30** | **15** | **15** |

---

## Verified Problems by Distribution (EVAL-016 → EVAL-030)

| Category | Count | Problems |
|---|---|---|
| Algebra | 4 | EVAL-016 (rect), EVAL-017 (abs val), EVAL-018 (LCM), EVAL-019 (quadratic k), EVAL-020 (quadratic fit) |
| Geometry | 3 | EVAL-021 (inradius), EVAL-022 (hexagon), EVAL-023 (tangent) |
| Number Theory | 3 | EVAL-024 (divisors), EVAL-025 (mod exp), EVAL-026 (gcd count) |
| Combinatorics | 2 | EVAL-027 (digit sum), EVAL-028 (coin flip) |
| Word Problem | 2 | EVAL-029 (fruit purchase), EVAL-030 (work rates) |

**Difficulty:** 5 easy / 7 medium / 3 hard (hand-authored only)

---

## Verification Tooling Status

| Tool | Status | Notes |
|---|---|---|
| `verification/schema.py` | ✅ Working | Pydantic v2; 4 cross-field validators |
| `verification/verify_answer.py` | ✅ Working | 16/16 unit tests pass; SymPy + fraction + float + string |
| `verification/contamination_check.py` | ✅ Working | Streams 4 corpora × 10K rows; Jaccard similarity |
| `review/generate_review_prompts.py` | ✅ Working | Produces 3 model-specific review prompts |

---

## Contamination Check Results

**Result: CLEAN** — 0 flags above 0.70 Jaccard threshold across 4 corpora × 10K sampled rows.

Corpora checked:
- `open-r1/OpenR1-Math-220k`
- `NovaSky-UC-Berkeley/Sky-T1-Data-17K`
- `AI-MO/NuminaMath-CoT`
- `qfq/openai-math-s1k`

> Note: streaming sample only (10K rows each). Full exhaustive check recommended before
> finalizing the eval set for publication.

---

## Problems Requiring Human Action

**EVAL-001 → EVAL-015** (`status: needs_verification`) — AIME 2025 + 2026 shells.

For each:
1. Locate official problem text from AoPS or MAA archives
2. Paste into `"problem_text"` field
3. Enter official answer (integer 0-999) into `"answer"` field
4. Set `"status": "verified"` and `"category"` to the correct value
5. Run `python verification/schema.py problems/EVAL-00N.json` to validate
6. Re-run `verify_answer.py` against the recorded answer

Official sources:
- AIME 2026: https://artofproblemsolving.com/wiki/index.php/2026_AIME
- AIME 2025: https://artofproblemsolving.com/wiki/index.php/2025_AIME

---

## Notable Issues Found During Construction

1. **EVAL-026 answer corrected** — Initially wrote 400 (by mistake); SymPy brute-force
   confirmed 300. Shows value of computational verification over mental math.

2. **EVAL-027 answer corrected** — Stars-and-bars gives 219, not 282 (initial guess).
   Corrected by writing and running the exhaustive loop.

3. **AIME 2025/2026 blocks** — Cannot safely verify official answers from memory;
   must be filled from official sources to avoid poisoning the eval set with wrong answers.

4. **Zero-shot RAG incompatibility** (noted from MathGPT parallel work) — The current
   char-level models were not trained on retrieval-augmented format; injected context
   is echoed rather than used. Eval set is not affected, but this informs training design.

---

## Recommended Next Steps

### Immediate (to reach 30 fully-verified problems)
- Fill EVAL-001 → EVAL-015 from official AIME sources (1-2 hrs)
- Run `python verification/contamination_check.py` on full set after filling
- Run LLM review prompts (`review/review_prompt_*.md`) and compare answers

### Expansion to 50 problems
- Add AIME 2025 II (problems 1, 4, 8, 12, 15) — EVAL-031 → EVAL-035
- Add 5 more hand-authored hard problems:
  - 2 calculus (limits, integrals with clean integer answers)
  - 2 contest-style number theory
  - 1 combinatorics (generating function or inclusion-exclusion)

### Expansion to 100+ problems
- Add USAMO 2025 problems (proof problems — requires extending schema)
- Add AIME 2026 Part II
- Add Putnam 2025 B-1 through B-3 (accessible first-half problems)

### Tooling improvements
- Add `--verify-all` flag to run all problems through `verify_answer.py`
- Integrate with MathGPT training loop: after each epoch, run eval set and log EM
- Add per-problem timeout to eval harness (prevent runaway generation)

---

## File Manifest

```
eval_set/
├── problems/
│   ├── EVAL-001.json ... EVAL-015.json  [needs_verification — AIME skeletons]
│   └── EVAL-016.json ... EVAL-030.json  [verified — hand-authored]
├── verification/
│   ├── schema.py              ✓ Pydantic v2 schema + validator
│   ├── verify_answer.py       ✓ SymPy + numeric + string verification (16/16 tests)
│   ├── contamination_check.py ✓ Streaming Jaccard check (4 corpora)
│   ├── contamination_report.md ✓ Result: CLEAN
│   └── solutions/
│       └── hand_authored_batch1.py  ✓ All 15 solutions verified
├── review/
│   ├── generate_review_prompts.py  ✓
│   ├── review_prompt_chatgpt.md    ✓
│   ├── review_prompt_gemini.md     ✓
│   └── review_prompt_claude.md     ✓
├── manifest.json   ✓
├── README.md       ✓
└── SUMMARY.md      ✓ (this file)
```
