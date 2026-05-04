"""
cot_sub_2d_no_borrow.py — Atomic sub_2d (no borrow case).

Universe: a > b ∈ [10, 99] AND a%10 >= b%10 (no borrow needed).
|universe| = 4500 such pairs.  Answer ∈ [1, 89].

Variants:
  column          : standard column subtraction (no borrow path)
  decompose       : a - b = a - b_t - b_o
  partial_left    : (a_t - b_t)*10 + (a_o - b_o)
  near_round      : a - b = a - (b_round_up) + delta
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs(n_unique, seed=42):
    """No-borrow pairs: a > b AND a%10 >= b%10."""
    rng = random.Random(seed)
    universe = []
    for a in range(10, 100):
        for b in range(10, 100):
            if a > b and (a % 10) >= (b % 10):
                universe.append((a, b, a - b))
    rng.shuffle(universe)
    return universe[:n_unique]


def _ex_sub_no_borrow(a, b):
    # Pick sibling with same ones-digit constraint: ea%10 >= eb%10
    for ea, eb in [(76, 43), (85, 51), (63, 21), (97, 34), (54, 32)]:
        if (ea, eb) != (a, b) and ea > eb and ea % 10 >= eb % 10:
            return ea, eb, ea - eb
    return 78, 45, 33


def variant_column(a, b, ans):
    ea, eb, eans = _ex_sub_no_borrow(a, b)
    ea_t, ea_o = ea // 10, ea % 10
    eb_t, eb_o = eb // 10, eb % 10
    ex = (f"{ea} - {eb}: ones {ea_o}-{eb_o}={ea_o-eb_o}; "
          f"tens {ea_t}-{eb_t}={ea_t-eb_t}; = {eans}")
    a_t, a_o = a // 10, a % 10
    b_t, b_o = b // 10, b % 10
    return (f"Example: {ex}\n"
            f"{a} - {b}:\n"
            f"ones: {a_o} - {b_o} = {a_o - b_o}\n"
            f"tens: {a_t} - {b_t} = {a_t - b_t}\n"
            f"= {a_t - b_t}{a_o - b_o}\n"
            f"#### {ans}")


def variant_decompose(a, b, ans):
    ea, eb, eans = _ex_sub_no_borrow(a, b)
    eb_t = (eb // 10) * 10; eb_o = eb - eb_t; es1 = ea - eb_t
    ex = f"{ea} - {eb} = {ea} - {eb_t} - {eb_o} = {es1} - {eb_o} = {eans}"
    b_t = (b // 10) * 10
    b_o = b - b_t
    step1 = a - b_t
    return (f"Example: {ex}\n"
            f"{a} - {b} = {a} - {b_t} - {b_o}\n"
            f"= {step1} - {b_o}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_partial_left(a, b, ans):
    ea, eb, eans = _ex_sub_no_borrow(a, b)
    ea_t = (ea // 10) * 10; ea_o = ea - ea_t
    eb_t = (eb // 10) * 10; eb_o = eb - eb_t
    ex = (f"{ea} - {eb} = ({ea_t}+{ea_o}) - ({eb_t}+{eb_o})"
          f" = {ea_t-eb_t} + {ea_o-eb_o} = {eans}")
    a_t = (a // 10) * 10
    a_o = a - a_t
    b_t = (b // 10) * 10
    b_o = b - b_t
    return (f"Example: {ex}\n"
            f"{a} - {b} = ({a_t}+{a_o}) - ({b_t}+{b_o})\n"
            f"= {a_t - b_t} + {a_o - b_o}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_near_round(a, b, ans):
    ea, eb, eans = _ex_sub_no_borrow(a, b)
    eb_r = ((eb // 10) + 1) * 10; eos = eb_r - eb; ei = ea - eb_r
    ex = f"{ea} - {eb} = {ea} - {eb_r} + {eos} = {ei} + {eos} = {eans}"
    b_round = ((b // 10) + 1) * 10
    overshoot = b_round - b
    intermediate = a - b_round
    return (f"Example: {ex}\n"
            f"{a} - {b} = {a} - {b_round} + {overshoot}\n"
            f"= {intermediate} + {overshoot}\n"
            f"= {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'column': variant_column,
    'decompose': variant_decompose,
    'partial_left': variant_partial_left,
    'near_round': variant_near_round,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} - {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'sub_2d_no_borrow',
        'stage': 2, 'level': 2,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(n_unique=2000, holdout_frac=0.2, seed=42):
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
        'problem': f'{a} - {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'sub_2d_no_borrow',
        'stage': 2, 'level': 2,
    } for a, b, ans in val_pairs]
    rng.shuffle(train_records)
    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-unique', type=int, default=2000)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(n_unique=args.n_unique, holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'sub_2d_no_borrow_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'sub_2d_no_borrow_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'Coverage: {args.n_unique}/4500 = {100*args.n_unique/4500:.1f}% of universe')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')


if __name__ == '__main__':
    main()
