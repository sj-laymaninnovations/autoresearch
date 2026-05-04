"""
build_curriculum_piles.py — bagpile-style curriculum data piles.

Pile 1 — basics:                add_1d + mul_1d (full tables)
Pile 2 — two-digit no regrouping: add_2d-no-carry + sub_2d_no_borrow
Pile 3 — regrouping + division:  add_2d-with-carry + sub_2d_borrow + mul_2d + div_1d
Pile 4 — word problems:           word_problem (distilled)

Format augmentation (4-6 surface forms per problem) is applied to each pile
so the curriculum model is robust across phrasings — the chat-OOD failure
mode we diagnosed.

Each pile gets its own train+val JSONL pair. Val is per-skill canonical
(no augmentation) so EM is comparable to per-skill metrics.

Usage:
    python math_lab/datagen/build_curriculum_piles.py
"""
from __future__ import annotations

import json, random, re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_arith_generalist import (
    SOURCES, load_jsonl, augment_records, parse_problem,
)


SKILLS_DIR = Path("math_lab/results/skills")
OUT_DIR    = SKILLS_DIR

# Pile membership — by skill_id
PILES = {
    1: ["add_1d", "mul_1d"],
    2: ["add_2d_no_carry", "sub_2d_no_borrow"],   # add_2d is split below
    3: ["add_2d_with_carry", "sub_2d_borrow",
        "mul_2d", "div_1d"],
    4: ["word_problem"],
}

# How many epochs each pile gets trained for (for the curriculum_runner)
PILE_EPOCHS = {1: 600, 2: 300, 3: 300, 4: 200}


def split_add_2d(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """add_2d records carry both no-carry and with-carry; split by parity of
    ones-sum."""
    no_c, with_c = [], []
    for r in records:
        a, op, b = parse_problem(r.get("problem", ""))
        if a is None or op != "+":
            with_c.append(r)
            continue
        if (a % 10) + (b % 10) >= 10:
            with_c.append(r)
        else:
            no_c.append(r)
    return no_c, with_c


def load_word_problems():
    """word_problem dataset has distinct path."""
    train = load_jsonl(SKILLS_DIR / "wordproblems_train_seed42.jsonl")
    val   = load_jsonl(SKILLS_DIR / "wordproblems_val_seed42.jsonl")
    return train, val


def main(seed=42):
    rng = random.Random(seed)

    # Load each base skill once
    raw = {}
    for skill, (train_name, val_name) in SOURCES.items():
        t = load_jsonl(SKILLS_DIR / train_name)
        v = load_jsonl(SKILLS_DIR / val_name)
        raw[skill] = {"train": t, "val": v}
        print(f"  loaded {skill:18s}: train={len(t):>6d} val={len(v):>4d}")

    # Split add_2d into carry and no-carry shards
    add_no_c_t, add_with_c_t = split_add_2d(raw["add_2d"]["train"])
    add_no_c_v, add_with_c_v = split_add_2d(raw["add_2d"]["val"])
    raw["add_2d_no_carry"]   = {"train": add_no_c_t,   "val": add_no_c_v}
    raw["add_2d_with_carry"] = {"train": add_with_c_t, "val": add_with_c_v}
    print(f"  split add_2d:  no_carry train={len(add_no_c_t)} val={len(add_no_c_v)}  | "
          f"with_carry train={len(add_with_c_t)} val={len(add_with_c_v)}")

    # word_problem
    wp_train, wp_val = load_word_problems()
    raw["word_problem"] = {"train": wp_train, "val": wp_val}
    print(f"  loaded word_problem: train={len(wp_train):>6d} val={len(wp_val):>4d}")
    print()

    # Build each pile
    for pile_n, members in PILES.items():
        train, val = [], []
        for skill in members:
            if skill not in raw:
                print(f"  [pile {pile_n}] skip {skill}: not loaded")
                continue
            t = raw[skill]["train"]
            v = raw[skill]["val"]
            # Tag every record with pile + skill for analysis
            t_aug = augment_records(t, skill, rng)
            for r in t_aug:
                r["pile"] = pile_n
            for r in v:
                r["pile"]    = pile_n
                r["concept"] = f"arith_{skill}"
            train.extend(t_aug)
            val.extend(v)
            print(f"  [pile {pile_n}] {skill:22s} train={len(t_aug):>6d} val={len(v):>4d}")

        rng.shuffle(train)
        rng.shuffle(val)
        train_p = OUT_DIR / f"curriculum_pile{pile_n}_train.jsonl"
        val_p   = OUT_DIR / f"curriculum_pile{pile_n}_val.jsonl"
        with train_p.open("w") as f:
            for r in train: f.write(json.dumps(r) + "\n")
        with val_p.open("w") as f:
            for r in val: f.write(json.dumps(r) + "\n")
        print(f"  pile {pile_n} totals: train={len(train):>6d}  val={len(val):>4d}  "
              f"→ {train_p.name}")
        print()


if __name__ == "__main__":
    main()
