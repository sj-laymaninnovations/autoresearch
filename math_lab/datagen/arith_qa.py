"""
arith_qa.py — Arithmetic Q/A Generator with Reversed Digits + Padding

Generates focused arithmetic training data optimized for transformer learning:
  - Zero-padded operands and answers (digit alignment)
  - Reversed output digits (LSB-first) for natural carry propagation
  - Balanced operations (+, -, *, //)
  - Commutative augmentation for + and *
  - Curriculum tiers: 1-digit, 2-digit, 3-digit operands

References:
  - Zaremba & Sutskever 2014: reversed output format
  - Power et al. 2022: grokking dynamics
  - Lee et al. 2023: teaching arithmetic to small transformers

Output: math_lab/results/harder_qa_arith_<op>_<tag>_<timestamp>.jsonl
"""

import json
import random
import datetime
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Tier definitions ──────────────────────────────────────────────────────────
# Each tier defines operand range and answer padding width.

TIERS = {
    1: {"lo": 1, "hi": 9,   "op_width": 1, "ans_width": {"add": 2, "sub": 1, "mul": 2, "div": 1}},
    2: {"lo": 10, "hi": 99,  "op_width": 2, "ans_width": {"add": 3, "sub": 2, "mul": 4, "div": 2}},
    3: {"lo": 100, "hi": 999, "op_width": 3, "ans_width": {"add": 4, "sub": 3, "mul": 6, "div": 3}},
}


def pad_num(n: int, width: int) -> str:
    """Zero-pad a non-negative integer to `width` digits."""
    return str(abs(n)).zfill(width)


def reverse_str(s: str) -> str:
    """Reverse a string (for LSB-first output)."""
    return s[::-1]


# ── Problem generators ────────────────────────────────────────────────────────

def gen_add(rng: random.Random, tier: int) -> dict:
    t = TIERS[tier]
    a = rng.randint(t["lo"], t["hi"])
    b = rng.randint(t["lo"], t["hi"])
    ans = a + b
    aw = t["ans_width"]["add"]

    a_s = pad_num(a, t["op_width"])
    b_s = pad_num(b, t["op_width"])
    ans_s = pad_num(ans, aw)
    rev_ans = reverse_str(ans_s)

    problem = f"{a_s} + {b_s}"
    solution = f"{rev_ans}\n#### {rev_ans}"

    return {
        "problem": problem, "solution": solution,
        "answer": ans, "answer_reversed": rev_ans,
        "domain": "arith_add", "level": tier, "source": "arith_qa",
    }


def gen_sub(rng: random.Random, tier: int) -> dict:
    t = TIERS[tier]
    a = rng.randint(t["lo"], t["hi"])
    b = rng.randint(t["lo"], t["hi"])
    if a < b:
        a, b = b, a  # ensure non-negative result
    ans = a - b
    aw = t["ans_width"]["sub"]

    a_s = pad_num(a, t["op_width"])
    b_s = pad_num(b, t["op_width"])
    ans_s = pad_num(ans, aw)
    rev_ans = reverse_str(ans_s)

    problem = f"{a_s} - {b_s}"
    solution = f"{rev_ans}\n#### {rev_ans}"

    return {
        "problem": problem, "solution": solution,
        "answer": ans, "answer_reversed": rev_ans,
        "domain": "arith_sub", "level": tier, "source": "arith_qa",
    }


def gen_mul(rng: random.Random, tier: int) -> dict:
    t = TIERS[tier]
    a = rng.randint(t["lo"], t["hi"])
    b = rng.randint(t["lo"], t["hi"])
    ans = a * b
    aw = t["ans_width"]["mul"]

    a_s = pad_num(a, t["op_width"])
    b_s = pad_num(b, t["op_width"])
    ans_s = pad_num(ans, aw)
    rev_ans = reverse_str(ans_s)

    problem = f"{a_s} * {b_s}"
    solution = f"{rev_ans}\n#### {rev_ans}"

    return {
        "problem": problem, "solution": solution,
        "answer": ans, "answer_reversed": rev_ans,
        "domain": "arith_mul", "level": tier, "source": "arith_qa",
    }


def gen_div(rng: random.Random, tier: int) -> dict:
    """Backward generation: pick quotient and divisor, compute dividend."""
    t = TIERS[tier]
    quotient = rng.randint(t["lo"], t["hi"])
    divisor = rng.randint(max(2, t["lo"]), t["hi"])
    dividend = quotient * divisor
    aw = t["ans_width"]["div"]

    # dividend might exceed tier's op_width — use its actual width
    div_width = max(t["op_width"], len(str(dividend)))
    d_s = pad_num(dividend, div_width)
    b_s = pad_num(divisor, t["op_width"])
    ans_s = pad_num(quotient, aw)
    rev_ans = reverse_str(ans_s)

    problem = f"{d_s} / {b_s}"
    solution = f"{rev_ans}\n#### {rev_ans}"

    return {
        "problem": problem, "solution": solution,
        "answer": quotient, "answer_reversed": rev_ans,
        "domain": "arith_div", "level": tier, "source": "arith_qa",
    }


GENERATORS = {
    "add": gen_add,
    "sub": gen_sub,
    "mul": gen_mul,
    "div": gen_div,
}

# Commutative operations get augmented (a op b) AND (b op a)
COMMUTATIVE = {"add", "mul"}


# ── Main generation logic ─────────────────────────────────────────────────────

def generate_arith_pairs(
    ops: list[str],
    tiers: list[int],
    n_per_combo: int,
    seed: int = 42,
    augment_commutative: bool = True,
) -> list[dict]:
    """
    Generate arithmetic Q/A pairs.

    Args:
        ops: list of operation names ("add", "sub", "mul", "div")
        tiers: list of difficulty tiers (1, 2, 3)
        n_per_combo: pairs per (operation, tier) combo
        seed: random seed
        augment_commutative: if True, also generate b op a for + and *

    Returns:
        list of JSONL-compatible dicts
    """
    rng = random.Random(seed)
    pairs = []
    seen = set()

    for op in ops:
        gen_fn = GENERATORS[op]
        for tier in tiers:
            count = 0
            attempts = 0
            while count < n_per_combo and attempts < n_per_combo * 10:
                attempts += 1
                p = gen_fn(rng, tier)

                # Dedup by problem string
                key = p["problem"]
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(p)
                count += 1

                # Commutative augmentation: swap operands
                if augment_commutative and op in COMMUTATIVE:
                    parts = p["problem"].split(f" {'+' if op == 'add' else '*'} ")
                    if len(parts) == 2 and parts[0] != parts[1]:
                        sym = "+" if op == "add" else "*"
                        comm_problem = f"{parts[1]} {sym} {parts[0]}"
                        if comm_problem not in seen:
                            seen.add(comm_problem)
                            comm = dict(p)
                            comm["problem"] = comm_problem
                            pairs.append(comm)

    rng.shuffle(pairs)
    return pairs


def save_pairs(pairs: list[dict], tag: str = "") -> Path:
    """Save pairs as JSONL to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out_path = RESULTS_DIR / f"harder_qa_arith{suffix}_{ts}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} arithmetic pairs -> {out_path}")
    return out_path


def print_stats(pairs: list[dict]):
    """Print distribution summary."""
    from collections import Counter
    by_domain = Counter(p["domain"] for p in pairs)
    by_tier = Counter(p["level"] for p in pairs)

    print(f"\n  Total pairs: {len(pairs)}")
    print(f"  By operation: {dict(sorted(by_domain.items()))}")
    print(f"  By tier:      {dict(sorted(by_tier.items()))}")

    # Show samples
    print(f"\n  Samples:")
    for p in pairs[:6]:
        print(f"    Q: {p['problem']}  ->  A: {p['solution'].split(chr(10))[0]}"
              f"  (real: {p['answer']})")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate arithmetic Q/A pairs with reversed digits + padding")
    parser.add_argument("--ops", nargs="+", default=["add", "sub", "mul", "div"],
                        choices=["add", "sub", "mul", "div"],
                        help="Which operations to generate")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="Difficulty tiers (1=1-digit, 2=2-digit, 3=3-digit)")
    parser.add_argument("--n", type=int, default=300,
                        help="Pairs per (operation, tier) combo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--no-commutative", action="store_true",
                        help="Disable commutative augmentation for + and *")
    args = parser.parse_args()

    pairs = generate_arith_pairs(
        ops=args.ops,
        tiers=args.tiers,
        n_per_combo=args.n,
        seed=args.seed,
        augment_commutative=not args.no_commutative,
    )

    print_stats(pairs)
    out = save_pairs(pairs, tag=args.run_tag)
