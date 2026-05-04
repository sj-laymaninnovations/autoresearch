"""
pedagogy_tag.py — apply rich pedagogical taxonomy to word-problem records.

Adds the following fields per record (preserves existing fields):
  topic           : top-level subject ("Arithmetic")
  subtopic        : operation family ("Addition", "Subtraction", "Multiplication", "Division")
  skill           : the atomic skill id (mirror of concept stripped)
  sequence        : integer ordering within subtopic, lower = comes first in curriculum
  operator        : +, -, *, /
  operand_width   : [digits_a, digits_b] for the two operands
  nuance          : list of distinguishing variant tags (e.g. ["no_carry"], ["borrow"], ["clean_div"])
  lesson          : human-readable lesson title
  grade_us        : US grade level as string ("1", "2", ...)
  age_range       : age range string ("6-7", ...)
  format          : "word_problem" or "naked"  (these are word problems → "word_problem")

Designed to be applied to either the live wordproblems_overnight_graded.jsonl
file or used at write-time by future runs of distill_wordproblems.py.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

# ---------------------------------------------------------------------------
# Skill catalog. The single source of truth — distill_wordproblems.py and
# pedagogy_tag.py both read from here.
# ---------------------------------------------------------------------------

SKILL_CATALOG = {
    "add_1d": {
        "topic": "Arithmetic", "subtopic": "Addition",
        "operator": "+", "operand_width": [1, 1],
        "sequence": 1,
        "nuance": ["single_digit"],
        "lesson": "Add single-digit numbers",
        "grade_us": "1", "age_range": "6-7",
    },
    "add_2d": {
        "topic": "Arithmetic", "subtopic": "Addition",
        "operator": "+", "operand_width": [2, 2],
        "sequence": 2,
        "nuance": ["two_digit", "with_or_without_carry"],
        "lesson": "Add two-digit numbers (with and without regrouping)",
        "grade_us": "2", "age_range": "7-8",
    },
    "sub_2d_no_borrow": {
        "topic": "Arithmetic", "subtopic": "Subtraction",
        "operator": "-", "operand_width": [2, 2],
        "sequence": 1,
        "nuance": ["two_digit", "no_borrow"],
        "lesson": "Subtract two-digit numbers without regrouping",
        "grade_us": "2", "age_range": "7-8",
    },
    "sub_2d_borrow": {
        "topic": "Arithmetic", "subtopic": "Subtraction",
        "operator": "-", "operand_width": [2, 2],
        "sequence": 2,
        "nuance": ["two_digit", "borrow"],
        "lesson": "Subtract two-digit numbers with regrouping (borrowing)",
        "grade_us": "3", "age_range": "8-9",
    },
    "mul_1d": {
        "topic": "Arithmetic", "subtopic": "Multiplication",
        "operator": "*", "operand_width": [1, 1],
        "sequence": 1,
        "nuance": ["single_digit", "memorized_table"],
        "lesson": "Multiplication tables (single-digit factors)",
        "grade_us": "3", "age_range": "8-9",
    },
    "mul_2d": {
        "topic": "Arithmetic", "subtopic": "Multiplication",
        "operator": "*", "operand_width": [2, 2],
        "sequence": 2,
        "nuance": ["two_digit", "long_multiplication"],
        "lesson": "Multiply two-digit numbers (long multiplication)",
        "grade_us": "4", "age_range": "9-10",
    },
    "div_1d": {
        "topic": "Arithmetic", "subtopic": "Division",
        "operator": "/", "operand_width": [2, 2],   # dividend up to 2 digits, divisor up to 2 digits
        "sequence": 1,
        "nuance": ["clean_division", "single_digit_quotient"],
        "lesson": "Clean division (no remainder, single-digit quotient)",
        "grade_us": "3", "age_range": "8-9",
    },
}

# Skills not yet trained but reserved in the curriculum sequence
RESERVED_FUTURE_SKILLS = {
    "sub_1d":      {"subtopic": "Subtraction",    "sequence": 0, "grade_us": "1"},
    "add_3d":      {"subtopic": "Addition",       "sequence": 3, "grade_us": "3"},
    "sub_3d":      {"subtopic": "Subtraction",    "sequence": 3, "grade_us": "3"},
    "mul_2d_by_1d":{"subtopic": "Multiplication", "sequence": 2, "grade_us": "4"},
    "div_2d":      {"subtopic": "Division",       "sequence": 2, "grade_us": "4"},
    "decimal_add": {"subtopic": "Decimals",       "sequence": 1, "grade_us": "5"},
    "frac_add":    {"subtopic": "Fractions",      "sequence": 1, "grade_us": "5"},
}


def tag_record(rec: dict, format_kind: str = "word_problem") -> dict:
    """Add pedagogical fields to a record. Idempotent (safe to re-tag)."""
    skill_id = rec.get("concept", "").replace("word_problem_", "")
    meta = SKILL_CATALOG.get(skill_id)
    if meta is None:
        return rec
    rec["topic"]         = meta["topic"]
    rec["subtopic"]      = meta["subtopic"]
    rec["skill"]         = skill_id
    rec["sequence"]      = meta["sequence"]
    rec["operator"]      = meta["operator"]
    rec["operand_width"] = list(meta["operand_width"])
    rec["nuance"]        = list(meta["nuance"])
    rec["lesson"]        = meta["lesson"]
    rec["grade_us"]      = meta["grade_us"]
    rec["age_range"]     = meta["age_range"]
    rec["format"]        = format_kind
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--format", default="word_problem",
                    choices=["word_problem", "naked"])
    args = ap.parse_args()

    in_p = Path(args.in_path)
    out_p = Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    n = 0
    with in_p.open() as fin, out_p.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = tag_record(rec, format_kind=args.format)
            fout.write(json.dumps(rec) + "\n")
            n += 1
            key = (rec.get("subtopic", "?"), rec.get("grade_us", "?"))
            counts[key] = counts.get(key, 0) + 1

    print(f"Tagged {n} records → {out_p}")
    print()
    print("Distribution by (subtopic, grade):")
    for (sub, g), c in sorted(counts.items()):
        print(f"  grade {g}  {sub:18s}  {c:5d}")


if __name__ == "__main__":
    main()
