"""
cot_mul_dense.py — Density follow-up for mul_2d.

Prior cot_mul_rich.py with 1000 pairs × 4 variants = 3200 records hit 15.0%
PARTIAL (peak ep200) on Windows 3090 2026-04-26 — same recipe that took
div_1d to 95.4%. Hypothesis: mul_2d's universe is much larger (8100 pairs vs
div_1d's 980), so 1000 sampled pairs = only 10% coverage vs div_1d's 100%.

This script bumps coverage to ~25% of the universe (2500 unique pairs).
Same 4 variants. Total: 10000 train records.

Operand space and variants are unchanged from cot_mul_rich.py — only the
sampling density changes, isolating the density variable.
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs(n_unique, seed=42):
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n_unique:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if (a, b) in seen: continue
        seen.add((a, b))
        out.append((a, b, a * b))
    return out


def variant_partial_products(a, b, ans):
    b_tens = (b // 10) * 10
    b_ones = b - b_tens
    p1 = a * b_tens
    p2 = a * b_ones
    return (f"{a} * {b} = {a}*({b_tens}+{b_ones})\n"
            f"= {a}*{b_tens} + {a}*{b_ones}\n"
            f"= {p1} + {p2}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_distributive_a(a, b, ans):
    a_tens = (a // 10) * 10
    a_ones = a - a_tens
    p1 = a_tens * b
    p2 = a_ones * b
    return (f"{a} * {b} = ({a_tens}+{a_ones})*{b}\n"
            f"= {a_tens}*{b} + {a_ones}*{b}\n"
            f"= {p1} + {p2}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_area_decomp(a, b, ans):
    a_t = (a // 10) * 10
    a_o = a - a_t
    b_t = (b // 10) * 10
    b_o = b - b_t
    p1 = a_t * b_t
    p2 = a_t * b_o
    p3 = a_o * b_t
    p4 = a_o * b_o
    return (f"{a} * {b} = ({a_t}+{a_o})*({b_t}+{b_o})\n"
            f"= {a_t}*{b_t} + {a_t}*{b_o} + {a_o}*{b_t} + {a_o}*{b_o}\n"
            f"= {p1} + {p2} + {p3} + {p4}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_commutative(a, b, ans):
    if a > b:
        s, l = b, a
    else:
        s, l = a, b
    s_tens = (s // 10) * 10
    s_ones = s - s_tens
    p1 = l * s_tens
    p2 = l * s_ones
    return (f"{a} * {b} = {l} * {s} (commute)\n"
            f"= {l}*({s_tens}+{s_ones})\n"
            f"= {p1} + {p2}\n"
            f"= {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'partial_products': variant_partial_products,
    'distributive_a': variant_distributive_a,
    'area_decomp': variant_area_decomp,
    'commutative': variant_commutative,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} * {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'mul_2d',
        'stage': 2, 'level': 2,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(n_unique=2500, holdout_frac=0.2, seed=42):
    pairs = gen_pairs(n_unique, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * holdout_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_records = []
    for a, b, ans in train_pairs:
        for vname, vfn in VARIANTS.items():
            train_records.append(to_record(a, b, ans, vname, vfn))

    val_records = [{
        'problem': f'{a} * {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'mul_2d',
        'stage': 2, 'level': 2,
    } for a, b, ans in val_pairs]
    rng.shuffle(train_records)
    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-unique', type=int, default=2500)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(n_unique=args.n_unique, holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'mul_2d_dense_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'mul_2d_dense_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'Coverage: {args.n_unique}/8100 = {100*args.n_unique/8100:.1f}% of universe')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')


if __name__ == '__main__':
    main()
