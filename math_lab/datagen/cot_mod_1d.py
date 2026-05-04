"""
cot_mod_1d.py — Atomic mod_1d skill (single-digit modulo, a % b for a, b ∈ [0..9]).

Universe: 100 pairs (a ∈ [0,9] × b ∈ [0,9]). Excludes b=0 (undefined).
Effective universe: 90 pairs. Answer space: 0..8.

Five rich-CoT variants:
  table       : direct lookup
  divide_then : "5 % 3: 5/3=1 r 2; so 5 % 3 = 2"
  subtract    : "5 % 3: 5-3=2; 2<3 so 2; so 5 % 3 = 2"
  multiples   : "5 % 3: multiples of 3 are 0,3; 5-3=2; so 5 % 3 = 2"
  small_a     : "if a < b, a % b = a"  (fast path)
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs():
    """All a%b for a∈[0,9], b∈[1,9] — skip b=0 (undefined)."""
    return [(a, b, a % b) for a in range(10) for b in range(1, 10)]


def _ex_mod_1d(a, b):
    """Deterministic sibling example, always different from (a, b)."""
    for ea, eb in [(7, 3), (8, 5), (5, 2), (9, 4), (6, 7)]:
        if (ea, eb) != (a, b) and eb != 0:
            return ea, eb, ea % eb
    return 1, 3, 1


def variant_table(a, b, ans):
    ea, eb, eans = _ex_mod_1d(a, b)
    return (f"Example: {ea} % {eb} = {eans}\n"
            f"{a} % {b} = {ans}\n"
            f"#### {ans}")


def variant_divide_then(a, b, ans):
    ea, eb, eans = _ex_mod_1d(a, b)
    eq, er = ea // eb, ea % eb
    if ea < eb:
        ex = f"{ea} % {eb}: {ea} < {eb}, remainder is {ea} = {eans}"
    else:
        ex = f"{ea} % {eb}: {ea}/{eb}={eq} r {er} = {eans}"
    if b == 1:
        return (f"Example: {ex}\n{a} % 1 = 0 (any number mod 1 is 0)\n#### 0")
    if a < b:
        return (f"Example: {ex}\n{a} % {b}: since {a} < {b}, the remainder is {a}\n#### {a}")
    q = a // b
    r = a % b
    return (f"Example: {ex}\n"
            f"{a} % {b}: divide {a} by {b}\n"
            f"{a} / {b} = {q} remainder {r}\n"
            f"#### {r}")


def variant_subtract(a, b, ans):
    ea, eb, eans = _ex_mod_1d(a, b)
    if ea < eb:
        ex = f"{ea} < {eb}, so {ea} % {eb} = {ea}"
    else:
        esteps = []
        cur = ea
        while cur >= eb:
            esteps.append(f"{cur}-{eb}={cur-eb}")
            cur -= eb
        ex = f"{ea} % {eb}: " + ", ".join(esteps) + f" → {eans}"
    if b == 1:
        return (f"Example: {ex}\n{a} % 1 = 0\n#### 0")
    if a < b:
        return (f"Example: {ex}\n{a} < {b}, so {a} % {b} = {a}\n#### {a}")
    steps = []
    cur = a
    while cur >= b:
        steps.append(f"{cur} - {b} = {cur - b}")
        cur -= b
    return (f"Example: {ex}\n"
            f"{a} % {b}: subtract {b} until result < {b}\n"
            + "\n".join(steps)
            + f"\n{cur} < {b}, stop\n"
            + f"#### {cur}")


def variant_multiples(a, b, ans):
    ea, eb, eans = _ex_mod_1d(a, b)
    if ea < eb:
        ex = f"{ea} < {eb} so {ea} % {eb} = {ea}"
    else:
        em = max(m for m in range(0, ea + 1, eb) if m <= ea)
        ex = f"{ea} % {eb}: largest mult of {eb} ≤ {ea} is {em}; {ea}-{em}={eans}"
    if b == 1:
        return (f"Example: {ex}\n{a} % 1 = 0\n#### 0")
    if a < b:
        return (f"Example: {ex}\nmultiples of {b}: 0, {b}, ...\n{a} < {b} so {a} % {b} = {a}\n#### {a}")
    multiples = [b * k for k in range(0, a // b + 2) if b * k <= a + b]
    largest = max(m for m in multiples if m <= a)
    return (f"Example: {ex}\n"
            f"{a} % {b}: largest multiple of {b} that is \u2264 {a} is {largest}\n"
            f"{a} - {largest} = {ans}\n"
            f"#### {ans}")


def variant_small_a(a, b, ans):
    ea, eb, eans = _ex_mod_1d(a, b)
    ex = (f"{ea} % {eb} = {ea} (since {ea} < {eb})" if ea < eb
          else f"{ea} % {eb} = {eans}")
    if a < b:
        return (f"Example: {ex}\n{a} % {b} = {a} (since {a} < {b})\n#### {a}")
    if b == 1:
        return (f"Example: {ex}\n{a} % 1 = 0\n#### 0")
    return (f"Example: {ex}\n{a} % {b} = {ans}\n#### {ans}")


VARIANTS = {
    "table":        variant_table,
    "divide_then":  variant_divide_then,
    "subtract":     variant_subtract,
    "multiples":    variant_multiples,
    "small_a":      variant_small_a,
}


def to_record(a, b, ans, vname, vfn):
    return {
        "problem":      f"{a} % {b}",
        "solution":     str(ans),
        "answer":       str(ans),
        "answer_real":  ans,
        "concept":      "mod_1d",
        "stage": 1, "level": 1,
        "solution_cot": vfn(a, b, ans),
        "cot_source":   f"rule_based_{vname}",
    }


def build(seed=42, holdout_frac=0.0):
    """No-holdout default: full table. Mod is finite — generalize == memorize."""
    pairs = gen_pairs()
    rng = random.Random(seed)
    train_records = []
    for _ in range(3):     # ×3 reps so we get more steps per epoch
        for a, b, ans in pairs:
            for vname, vfn in VARIANTS.items():
                train_records.append(to_record(a, b, ans, vname, vfn))
    rng.shuffle(train_records)
    val_records = [{
        "problem": f"{a} % {b}", "solution": str(ans), "answer": str(ans),
        "answer_real": ans, "concept": "mod_1d", "stage": 1, "level": 1,
    } for a, b, ans in pairs]
    return train_records, val_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="math_lab/results/skills")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    train, val = build(seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out / f"mod_1d_full_train_seed{args.seed}_{ts}.jsonl"
    val_p   = out / f"mod_1d_full_val_seed{args.seed}_{ts}.jsonl"
    with open(train_p, "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(val_p, "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")
    print(f"Train: {len(train)} records (90 pairs × 5 variants × 3 reps)")
    print(f"Val:   {len(val)} pairs")
    print(f"  -> {train_p.name}")
    print(f"  -> {val_p.name}")

    print("\nSample 7 % 3:")
    for vn, vf in VARIANTS.items():
        print(f"  [{vn}]")
        for line in vf(7, 3, 7 % 3).split("\n"):
            print(f"    {line}")


if __name__ == "__main__":
    main()
