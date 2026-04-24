"""
cot_qa.py -- Multi-step Chain-of-Thought Q/A Generator for Math Training

Generates 2-4 step word problems with explicit CoT solutions.
Each step shows a visible computation so the model learns to chain
arithmetic operations.  All answers are verified with integer arithmetic
using backward generation (pick answer first, derive inputs).

Output: math_lab/results/harder_qa_cot_<category>_<run_tag>_<timestamp>.jsonl
"""

import json
import random
import datetime
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent.parent / "results"

# 256 seq_len - 2 (BOS/EOS) - 8 (wrapper: "Q: " + "\nA: " + "\n")
MAX_CONTENT_LEN = 246


# ── Length validation ─────────────────────────────────────────────────────────

def _check_length(problem: str, solution: str) -> bool:
    """Verify the full training example fits in seq_len=256."""
    return len(problem) + len(solution) <= MAX_CONTENT_LEN


# ── Template functions ────────────────────────────────────────────────────────
# Each takes (rng, level) and returns {"problem", "solution", "answer"}.
# Backward generation ensures all divisions produce exact integers.

# --- Category: sequential (2-3 step add/sub/mul) ---

def store_inventory(rng: random.Random, level: int) -> dict:
    """A shop has X items. Y boxes of Z arrive. W sell. How many left?"""
    boxes = rng.randint(2, 9)
    per_box = rng.randint(2, 15) * level
    received = boxes * per_box
    start = rng.randint(received + 1, received * 3) * level
    sold = rng.randint(1, start + received - 1)
    total = start + received
    answer = total - sold

    problem = (f"A shop has {start} items. {boxes} boxes of "
               f"{per_box} arrive. {sold} sell. How many left?")
    solution = (f"{boxes}*{per_box}={received}\n"
                f"{start}+{received}={total}\n"
                f"{total}-{sold}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


def earnings(rng: random.Random, level: int) -> dict:
    """Sam earns $X/hr for Y hrs, spends $Z. How much left?"""
    rate = rng.randint(5, 25) * level
    hours = rng.randint(2, 8)
    earned = rate * hours
    spent = rng.randint(1, earned - 1)
    answer = earned - spent

    problem = f"Sam earns ${rate}/hr for {hours} hrs, spends ${spent}. How much left?"
    solution = (f"{rate}*{hours}={earned}\n"
                f"{earned}-{spent}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


def combine_split(rng: random.Random, level: int) -> dict:
    """Amy has A and Ben has B coins. Split equally, how many each?"""
    half = rng.randint(3, 50) * level
    total = half * 2
    a = rng.randint(1, total - 1)
    b = total - a

    problem = f"Amy has {a} and Ben has {b} coins. Split equally, how many each?"
    solution = (f"{a}+{b}={total}\n"
                f"{total}/2={half}\n#### {half}")
    return {"problem": problem, "solution": solution, "answer": half}


# --- Category: percentage (2 step) ---

def pct_discount(rng: random.Random, level: int) -> dict:
    """A $X shirt is Y% off. Sale price?"""
    pct = rng.choice([10, 20, 25, 50])
    divisor = 100 // pct
    price = rng.randint(2, 30) * divisor * level
    discount = price * pct // 100
    answer = price - discount

    problem = f"A ${price} shirt is {pct}% off. Sale price?"
    solution = (f"{price}*{pct}/100={discount}\n"
                f"{price}-{discount}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


def tax_calc(rng: random.Random, level: int) -> dict:
    """An item costs $X. Tax is Y%. Total?"""
    pct = rng.choice([5, 10, 20, 25])
    divisor = 100 // pct
    price = rng.randint(2, 30) * divisor * level
    tax = price * pct // 100
    answer = price + tax

    problem = f"An item costs ${price}. Tax is {pct}%. Total?"
    solution = (f"{price}*{pct}/100={tax}\n"
                f"{price}+{tax}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


# --- Category: rate (2 step) ---

def unit_price(rng: random.Random, level: int) -> dict:
    """X pens cost $Y. How much for Z?"""
    unit = rng.randint(2, 20) * level
    count_a = rng.randint(2, 10)
    total_cost = unit * count_a
    count_b = rng.randint(2, 10)
    answer = unit * count_b

    problem = f"{count_a} pens cost ${total_cost}. How much for {count_b}?"
    solution = (f"{total_cost}/{count_a}={unit}\n"
                f"{unit}*{count_b}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


def speed_distance(rng: random.Random, level: int) -> dict:
    """A car goes X mph for Y hrs, then Z mph for W hrs. Total miles?"""
    s1 = rng.randint(10, 60) * level
    h1 = rng.randint(1, 5)
    s2 = rng.randint(10, 60) * level
    h2 = rng.randint(1, 5)
    d1 = s1 * h1
    d2 = s2 * h2
    answer = d1 + d2

    problem = (f"A car goes {s1} mph for {h1} hrs, "
               f"then {s2} mph for {h2} hrs. Total miles?")
    solution = (f"{s1}*{h1}={d1}\n"
                f"{s2}*{h2}={d2}\n"
                f"{d1}+{d2}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


# --- Category: comparison (3 step) ---

def price_compare(rng: random.Random, level: int) -> dict:
    """Store A: X for $Y. Store B: Z for $W. Cheaper per item at A by how much?"""
    up_a = rng.randint(2, 15) * level
    up_b = up_a + rng.randint(1, 10) * level
    count_a = rng.randint(2, 8)
    count_b = rng.randint(2, 8)
    total_a = up_a * count_a
    total_b = up_b * count_b
    answer = up_b - up_a

    problem = (f"Store A: {count_a} for ${total_a}. Store B: "
               f"{count_b} for ${total_b}. Cheaper per item at A by how much?")
    solution = (f"{total_a}/{count_a}={up_a}\n"
                f"{total_b}/{count_b}={up_b}\n"
                f"{up_b}-{up_a}={answer}\n#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer}


# --- Category: multi_step (2-3 step) ---

def profit_calc(rng: random.Random, level: int) -> dict:
    """Buy X at $Y each, sell all for $Z. Profit per item?"""
    count = rng.randint(2, 10)
    cost_each = rng.randint(3, 20) * level
    margin = rng.randint(1, 10) * level
    sell_each = cost_each + margin
    total_cost = cost_each * count
    total_sell = sell_each * count
    profit = total_sell - total_cost
    profit_each = margin  # profit / count = margin * count / count = margin

    problem = (f"Buy {count} at ${cost_each} each, "
               f"sell all for ${total_sell}. Profit per item?")
    solution = (f"{cost_each}*{count}={total_cost}\n"
                f"{total_sell}-{total_cost}={profit}\n"
                f"{profit}/{count}={profit_each}\n#### {profit_each}")
    return {"problem": problem, "solution": solution, "answer": profit_each}


def average_calc(rng: random.Random, level: int) -> dict:
    """Scores: A, B, C. Average?"""
    n = rng.choice([3, 4, 5])
    avg = rng.randint(10, 50) * level
    total = avg * n
    # Build scores that sum to total
    scores = []
    remaining = total
    for i in range(n - 1):
        lo = max(1, avg - 20 * level)
        hi = min(remaining - (n - 1 - i), avg + 20 * level)
        if hi < lo:
            hi = lo
        s = rng.randint(lo, hi)
        scores.append(s)
        remaining -= s
    scores.append(remaining)

    scores_str = ", ".join(str(s) for s in scores)
    sum_str = "+".join(str(s) for s in scores)
    problem = f"Scores: {scores_str}. Average?"
    solution = (f"{sum_str}={total}\n"
                f"{total}/{n}={avg}\n#### {avg}")
    return {"problem": problem, "solution": solution, "answer": avg}


# ── Template registry ─────────────────────────────────────────────────────────

COT_TEMPLATES = {
    "sequential":  [store_inventory, earnings, combine_split],
    "percentage":  [pct_discount, tax_calc],
    "rate":        [unit_price, speed_distance],
    "comparison":  [price_compare],
    "multi_step":  [profit_calc, average_calc],
}

ALL_TEMPLATES = []
for _cat, _fns in COT_TEMPLATES.items():
    for _fn in _fns:
        ALL_TEMPLATES.append((_cat, _fn))


# ── Core generator ────────────────────────────────────────────────────────────

def generate_cot_problem(rng: random.Random, level: int,
                         category: str = None) -> Optional[dict]:
    """Generate a single multi-step CoT problem."""
    if category and category in COT_TEMPLATES:
        fn = rng.choice(COT_TEMPLATES[category])
        cat = category
    else:
        cat, fn = rng.choice(ALL_TEMPLATES)

    result = fn(rng, level)

    if not _check_length(result["problem"], result["solution"]):
        return None

    return {
        "problem":  result["problem"],
        "solution": result["solution"],
        "answer":   result["answer"],
        "domain":   f"cot_{cat}",
        "level":    level,
        "source":   "cot_generated",
    }


def generate_cot_pairs(
    n_pairs: int = 500,
    level: int = 3,
    category: str = None,
    seed: int = None,
) -> list[dict]:
    """Generate n_pairs of multi-step CoT problems."""
    rng = random.Random(seed)
    pairs = []
    max_attempts = n_pairs * 5

    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        result = generate_cot_problem(rng, level, category)
        if result is not None:
            pairs.append(result)

    if len(pairs) < n_pairs:
        print(f"WARNING: Only generated {len(pairs)}/{n_pairs} "
              f"(some exceeded 256-char limit)")

    return pairs


# ── Save (mirrors harder_qa.py exactly) ───────────────────────────────────────

def save_pairs(pairs: list[dict], domain: str, run_tag: str = "") -> Path:
    """Save generated pairs as JSONL to results directory.

    Filename starts with 'harder_qa_' so merge_results.py picks it up.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    out_path = RESULTS_DIR / f"harder_qa_{domain}{suffix}_{ts}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} CoT Q/A pairs -> {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    CATEGORIES = list(COT_TEMPLATES.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="Generate multi-step Chain-of-Thought math Q/A pairs")
    parser.add_argument("--category", default="all",
                        choices=CATEGORIES,
                        help="Problem category (default: all)")
    parser.add_argument("--n",        type=int, default=500,
                        help="Number of pairs to generate")
    parser.add_argument("--level",    type=int, default=3,
                        choices=[3, 4, 5],
                        help="Difficulty level 3-5 (default: 3)")
    parser.add_argument("--run-tag",  default="",
                        help="Optional tag for output filename")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed (omit for different batch each run)")
    args = parser.parse_args()

    cat = None if args.category == "all" else args.category
    domain = f"cot_{args.category}" if cat else "cot_all"

    pairs = generate_cot_pairs(
        n_pairs=args.n,
        level=args.level,
        category=cat,
        seed=args.seed,
    )

    out = save_pairs(pairs, domain, args.run_tag)

    print(f"\nSample (first 3 problems):")
    for p in pairs[:3]:
        full = f"Q: {p['problem']}\nA: {p['solution']}\n"
        print(f"  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  Solution: {p['solution']}")
        print(f"  Chars: {len(full)+2}/256")
        print()
