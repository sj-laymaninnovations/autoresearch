"""
cot_div_expanded.py — Expanded div_1d: same skill (1-digit quotient) but
much larger operand space.

Original: q∈[0,9], d∈[2,9] → 80 unique pairs
Expanded: q∈[0,9], d∈[2,99] → 980 unique pairs (12× more)

Each pair gets 4 rich-CoT variants (factorization/buildup/countdown/inverse),
matching cot_div_rich.py format. Total training records: 980 × 4 = 3920
(after train/val split).

Stays in spirit with "div_1d = single-digit-quotient division". The model
still has to learn the underlying skill, but with much more varied operand
patterns to generalize from.
"""
import argparse
import json
import datetime
import random
from pathlib import Path


def gen_pairs():
    """Returns [(a, d, q), ...] for q∈[0,9], d∈[2,99], a = q*d."""
    pairs = {}
    for q in range(0, 10):
        for d in range(2, 100):
            a = q * d
            pairs[(a, d)] = q
    return [(a, d, q) for (a, d), q in pairs.items()]


def variant_factorization(a, d, q):
    return f"{a} = {d} * {q}\n{a}/{d} = {q}\n#### {q}"


def variant_buildup(a, d, q):
    if q == 0:
        return f"{d} * 0 = 0\n{a}/{d} = 0\n#### 0"
    if q == 1:
        return f"{d} * 1 = {d}\n{a}/{d} = 1\n#### 1"
    multiples = "\n".join(f"{d} * {k} = {d*k}" for k in range(1, q+1))
    return f"{multiples}\n{a}/{d} = {q}\n#### {q}"


def variant_countdown(a, d, q):
    if q == 0:
        return f"{a} - 0*{d} = {a} (none fit)\n{a}/{d} = 0\n#### 0"
    steps = []
    cur = a
    for k in range(1, q+1):
        next_cur = cur - d
        steps.append(f"{cur} - {d} = {next_cur}")
        cur = next_cur
    steps_str = "\n".join(steps)
    return f"{steps_str}\nstep count = {q}\n{a}/{d} = {q}\n#### {q}"


def variant_inverse(a, d, q):
    if q == 0:
        return f"{a}/{d} : how many {d} in {a} ?\n0 * {d} = 0 = {a}\n{a}/{d} = 0\n#### 0"
    return f"{a}/{d} : how many {d} in {a} ?\n{q} * {d} = {a}\n{a}/{d} = {q}\n#### {q}"


VARIANTS = {
    'factorization': variant_factorization,
    'buildup': variant_buildup,
    'countdown': variant_countdown,
    'inverse': variant_inverse,
}


def to_record(a, d, q, variant_name, variant_fn):
    return {
        'problem': f'{a} / {d}',
        'solution': str(q),
        'answer': str(q),
        'answer_real': q,
        'concept': 'div_1d',
        'stage': 1,
        'level': 1,
        'solution_cot': variant_fn(a, d, q),
        'cot_source': f'rule_based_{variant_name}',
    }


def build(holdout_frac=0.2, seed=42):
    pairs = gen_pairs()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * holdout_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_records = []
    for a, d, q in train_pairs:
        for vname, vfn in VARIANTS.items():
            train_records.append(to_record(a, d, q, vname, vfn))

    val_records = []
    for a, d, q in val_pairs:
        val_records.append({
            'problem': f'{a} / {d}',
            'solution': str(q),
            'answer': str(q),
            'answer_real': q,
            'concept': 'div_1d',
            'stage': 1, 'level': 1,
        })
    rng.shuffle(train_records)
    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'div_1d_expanded_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'div_1d_expanded_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    # Sanity samples
    print('\nSample train records (first 3):')
    for r in train[:3]:
        print(f'  Q: {r["problem"]:10s}  variant: {r["cot_source"]}')
        for line in r['solution_cot'].split('\n')[:5]:
            print(f'    {line}')
        print()


if __name__ == '__main__':
    main()
