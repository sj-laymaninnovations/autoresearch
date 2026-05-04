"""
build_arith_generalist.py — combine all 7 atomic skill datasets into a single
balanced training set with format augmentation.

Goal: produce a JSONL the generalist model trains on, with:
  - balanced per-skill counts (target N records each, oversample small skills)
  - 4-6 format variants per problem (same CoT, varied surface form) to fix
    the "rigid format" failure mode we diagnosed today

Output schema is identical to per-skill JSONL: {problem, solution_cot,
answer, ...}. The variants live only in the `problem` field.

Run:
    python math_lab/datagen/build_arith_generalist.py \
        --target-per-skill 7000 \
        --out math_lab/results/skills/arith_generalist_train.jsonl

Validation set is held out per-skill (10%) and saved as a separate file
with no augmentation — so val EM measures pure skill, not memorization
of the canonical format.
"""
from __future__ import annotations

import argparse, json, random, re
from collections import Counter
from pathlib import Path
from typing import Iterable

# Map skill_id → (train_jsonl, val_jsonl) for the *latest, post-fix* dataset.
# These are the same files our per-skill chat models were trained on.
SOURCES = {
    "add_1d":            ("add_1d_full_train_seed42_20260427_194420.jsonl",
                          "add_1d_full_val_seed42_20260427_194420.jsonl"),
    "add_2d":            ("add_2d_train_seed42_20260427_194421.jsonl",
                          "add_2d_val_seed42_20260427_194421.jsonl"),
    "sub_2d_no_borrow":  ("sub_2d_no_borrow_train_seed42_20260427_194525.jsonl",
                          "sub_2d_no_borrow_val_seed42_20260427_194525.jsonl"),
    "sub_2d_borrow":     ("sub_2d_borrow_rich_train_seed42_20260426_182042.jsonl",
                          "sub_2d_borrow_rich_val_seed42_20260426_182042.jsonl"),
    "mul_1d":            ("mul_1d_full_train_seed42_20260426_225659.jsonl",
                          "mul_1d_full_val_seed42_20260426_225659.jsonl"),
    "mul_2d":            ("mul_2d_full_train_seed42_20260426_195403.jsonl",
                          "mul_2d_full_val_seed42_20260426_195403.jsonl"),
    "div_1d":            ("div_1d_expanded_train_seed42_20260426_110442.jsonl",
                          "div_1d_expanded_val_seed42_20260426_110442.jsonl"),
}

# ---------------------------------------------------------------------------
# Format-augmentation: turn one problem string into many surface forms
# ---------------------------------------------------------------------------

# Rough operator → English mapping for natural-language variants.
OP_ENGLISH = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def parse_problem(p: str) -> tuple[int, str, int]:
    """`'12 + 7'` -> (12, '+', 7). Returns (None, None, None) on parse fail."""
    m = re.match(r"\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*$", p.strip())
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def variants(problem: str, rng: random.Random) -> list[str]:
    """Return 4-6 surface forms of the same arithmetic expression."""
    a, op, b = parse_problem(problem)
    if a is None:
        return [problem]
    forms = [
        f"{a} {op} {b}",                    # canonical
        f"{a}{op}{b}",                      # no-space compact
        f"{a}  {op}  {b}",                  # extra spaces
        f"{a} {op} {b} = ",                 # with trailing equals
        f"What is {a} {op} {b}?",           # natural-language question
        f"Find: {a} {op} {b}",              # instruction prefix
    ]
    # Include the words version 30% of the time (avoid exploding count)
    if rng.random() < 0.3 and op in OP_ENGLISH:
        forms.append(f"{a} {OP_ENGLISH[op]} {b}")
    return forms


# ---------------------------------------------------------------------------
# Per-skill load + balance + augment
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def balance_oversample(records: list[dict], target: int,
                        rng: random.Random) -> list[dict]:
    """Oversample (with replacement) to reach `target` if records < target;
    downsample if records > target."""
    if len(records) >= target:
        rng.shuffle(records)
        return records[:target]
    out = list(records)
    while len(out) < target:
        out.append(records[rng.randint(0, len(records) - 1)])
    rng.shuffle(out)
    return out


def augment_records(records: list[dict], skill: str,
                     rng: random.Random) -> list[dict]:
    """Replace `problem` with one randomly-chosen surface form per record.
    Each record's CoT is unchanged — the model learns: many forms → one
    canonical CoT path → one answer."""
    out = []
    for r in records:
        forms = variants(r.get("problem", ""), rng)
        new_p = rng.choice(forms)
        new_r = dict(r)
        new_r["problem"] = new_p
        new_r["concept"] = f"arith_{skill}"  # tag with skill for analytics
        out.append(new_r)
    return out


def build(skills_dir: Path, target_per_skill: int, seed: int = 42):
    rng = random.Random(seed)
    train_records = []
    val_records = []
    for skill, (train_name, val_name) in SOURCES.items():
        train_p = skills_dir / train_name
        val_p   = skills_dir / val_name
        if not train_p.exists() or not val_p.exists():
            print(f"  [skip] {skill}: missing files")
            continue
        # Train: balance + augment
        recs = load_jsonl(train_p)
        recs = balance_oversample(recs, target_per_skill, rng)
        recs = augment_records(recs, skill, rng)
        train_records.extend(recs)
        # Val: NO oversampling, NO augmentation — measures pure skill on
        # canonical format (so EM is comparable to per-skill metrics)
        v = load_jsonl(val_p)
        # tag each val record with the skill for per-skill scoring later
        for r in v:
            r["concept"] = f"arith_{skill}"
        val_records.extend(v)
        print(f"  {skill:18s}  train={len(recs):>6d}  val={len(v):>4d}")
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-dir",
                    default="math_lab/results/skills",
                    type=Path)
    ap.add_argument("--target-per-skill", type=int, default=7000)
    ap.add_argument("--out-train",
                    default="math_lab/results/skills/arith_generalist_train.jsonl",
                    type=Path)
    ap.add_argument("--out-val",
                    default="math_lab/results/skills/arith_generalist_val.jsonl",
                    type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"  target-per-skill: {args.target_per_skill}")
    print()
    train, val = build(args.skills_dir, args.target_per_skill, args.seed)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with args.out_val.open("w") as f:
        for r in val: f.write(json.dumps(r) + "\n")

    print()
    print(f"  total train: {len(train):,}  -> {args.out_train}")
    print(f"  total val:   {len(val):,}    -> {args.out_val}")
    print()
    print("  format distribution in train (sample):")
    cnt = Counter()
    for r in train[:10000]:
        p = r["problem"]
        if "What is" in p:    cnt["natural_lang"] += 1
        elif "Find:" in p:    cnt["instruction"] += 1
        elif " = " in p:      cnt["equals"] += 1
        elif "  " in p:       cnt["extra_space"] += 1
        elif " " not in p:    cnt["no_space"] += 1
        else:
            for w in (" plus ", " minus ", " times ", " divided by "):
                if w in p:
                    cnt["english_op"] += 1
                    break
            else:
                cnt["canonical"] += 1
    for k, v in cnt.most_common():
        print(f"    {k:15s}  {v:>5d}")


if __name__ == "__main__":
    main()
