"""
skill_data.py — Atomic-skill data generator for the assembly-line program.

Each skill enumerates its full unique input space, holds out 50% as
novel-pair val (commutative twins also excluded), writes train/val JSONL
with consistent format. Used by skill_runner.py.

Usage:
    python -m math_lab.datagen.skill_data --skill add_1d_no_carry --out-dir results/skills/

Skills implemented (Phase 2 minimum):
    digit_echo, digit_pair, digit_compare,
    add_1d_no_carry, add_1d_with_carry,
    sub_1d_no_borrow,
    mul_1d, div_1d,
    add_2d_no_carry, add_2d_with_carry,
    sub_2d_no_borrow, sub_2d_with_borrow,
    mul_2d_by_1d, mul_2d_by_2d, div_2d
"""

import argparse, json, random, datetime
from pathlib import Path
from collections import Counter

# ── Skill generators ─────────────────────────────────────────────────────────
# Each returns list[(problem_str, answer_int, optional_extra_dict)]

def gen_digit_echo():
    return [(str(d), d, {}) for d in range(10)]

def gen_digit_pair():
    return [(f"{a}{b}", int(f"{a}{b}"), {}) for a in range(10) for b in range(10)]

def gen_digit_compare():
    return [(f"{a},{b}", max(a, b), {}) for a in range(10) for b in range(10)]

def gen_add_1d_no_carry():
    return [(f"{a} + {b}", a+b, {}) for a in range(10) for b in range(10) if a+b < 10]

def gen_add_1d_with_carry():
    return [(f"{a} + {b}", a+b, {}) for a in range(10) for b in range(10) if a+b >= 10]

def gen_sub_1d_no_borrow():
    return [(f"{a} - {b}", a-b, {}) for a in range(10) for b in range(10) if a >= b]

def gen_mul_1d():
    return [(f"{a} * {b}", a*b, {}) for a in range(10) for b in range(10)]

def gen_div_1d():
    """Single-digit-quotient division: dividend up to 9*9=81, divisor 2-9, clean.
    Quotient stays 1-digit. Total: 80 unique pairs (was 23 with the narrower
    'both 1-digit' definition). Expanded 2026-04-25 for div training balance."""
    pairs = {}
    for q in range(0, 10):
        for d in range(2, 10):
            a = q * d
            pairs[(a, d)] = q
    return [(f"{a} / {d}", q, {}) for (a, d), q in pairs.items()]

def gen_add_2d_no_carry():
    out = []
    for a in range(10, 100):
        for b in range(10, 100):
            # No carry means each digit pair sums < 10
            if (a%10 + b%10 < 10) and (a//10 + b//10 < 10):
                out.append((f"{a} + {b}", a+b, {}))
    return out

def gen_add_2d_with_carry():
    out = []
    for a in range(10, 100):
        for b in range(10, 100):
            # At least one column carries
            if not ((a%10 + b%10 < 10) and (a//10 + b//10 < 10)):
                out.append((f"{a} + {b}", a+b, {}))
    return out

def gen_sub_2d_no_borrow():
    out = []
    for a in range(10, 100):
        for b in range(10, 100):
            if a < b: continue
            # No borrow: each column digit a >= b
            if (a%10 >= b%10) and (a//10 >= b//10):
                out.append((f"{a} - {b}", a-b, {}))
    return out

def gen_sub_2d_with_borrow():
    out = []
    for a in range(10, 100):
        for b in range(10, 100):
            if a < b: continue
            # At least one column borrows
            if not ((a%10 >= b%10) and (a//10 >= b//10)):
                out.append((f"{a} - {b}", a-b, {}))
    return out

def gen_mul_2d_by_1d():
    return [(f"{a} * {b}", a*b, {}) for a in range(10, 100) for b in range(2, 10)]

def gen_mul_2d_by_2d():
    return [(f"{a} * {b}", a*b, {}) for a in range(10, 100) for b in range(10, 100)]

def gen_div_2d():
    out = []
    for q in range(10, 100):
        for d in range(2, 100):
            ans = q
            dividend = q * d
            if 100 <= dividend <= 9999:
                out.append((f"{dividend} / {d}", q, {}))
    return out

REGISTRY = {
    'digit_echo': (gen_digit_echo, False),
    'digit_pair': (gen_digit_pair, False),
    'digit_compare': (gen_digit_compare, True),  # commutative-like (max)
    'add_1d_no_carry': (gen_add_1d_no_carry, True),
    'add_1d_with_carry': (gen_add_1d_with_carry, True),
    'sub_1d_no_borrow': (gen_sub_1d_no_borrow, False),
    'mul_1d': (gen_mul_1d, True),
    'div_1d': (gen_div_1d, False),
    'add_2d_no_carry': (gen_add_2d_no_carry, True),
    'add_2d_with_carry': (gen_add_2d_with_carry, True),
    'sub_2d_no_borrow': (gen_sub_2d_no_borrow, False),
    'sub_2d_with_borrow': (gen_sub_2d_with_borrow, False),
    'mul_2d_by_1d': (gen_mul_2d_by_1d, False),
    'mul_2d_by_2d': (gen_mul_2d_by_2d, True),
    'div_2d': (gen_div_2d, False),
}


def commutative_key(problem):
    """Canonicalize commutative problem so (a, b) and (b, a) hash the same."""
    parts = problem.split()
    if len(parts) != 3:
        return problem
    a, op, b = parts
    if op in ('+', '*'):
        return f"{op}:" + ",".join(sorted([a, b]))
    return f"{op}:{a},{b}"


def build_skill_data(skill, holdout_frac=0.5, seed=42):
    """Generate train/val for a skill with novel-pair holdout."""
    if skill not in REGISTRY:
        raise ValueError(f"unknown skill: {skill!r}; known: {list(REGISTRY)}")
    gen_fn, is_commutative = REGISTRY[skill]
    pairs = gen_fn()

    # Group commutative twins so they go together in train OR val (no leakage)
    if is_commutative:
        groups = {}
        for problem, ans, extra in pairs:
            k = commutative_key(problem)
            groups.setdefault(k, []).append((problem, ans, extra))
        group_keys = list(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(group_keys)
        n_val_groups = max(1, int(len(group_keys) * holdout_frac))
        val_keys = set(group_keys[:n_val_groups])
        train_pairs, val_pairs = [], []
        for k, plist in groups.items():
            if k in val_keys:
                val_pairs.extend(plist)
            else:
                train_pairs.extend(plist)
    else:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * holdout_frac))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    def to_record(problem, ans, extra):
        return {
            'problem': problem,
            'solution': str(ans),
            'answer': str(ans),
            'answer_real': ans,
            'concept': skill,
            'stage': 0,
            **extra,
        }

    train_records = [to_record(*p) for p in train_pairs]
    val_records = [to_record(*p) for p in val_pairs]
    return train_records, val_records


def save(records, path):
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skill', required=True, choices=list(REGISTRY))
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--holdout-frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train, val = build_skill_data(args.skill, args.holdout_frac, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = out_dir / f'{args.skill}_train_{ts}.jsonl'
    val_path = out_dir / f'{args.skill}_val_{ts}.jsonl'
    save(train, train_path)
    save(val, val_path)

    print(f'Skill: {args.skill}')
    print(f'  Train: {len(train)} pairs -> {train_path.name}')
    print(f'  Val:   {len(val)} novel-pair holdout -> {val_path.name}')
    print(f'  Sample train: {train[:3]}')


if __name__ == '__main__':
    main()
