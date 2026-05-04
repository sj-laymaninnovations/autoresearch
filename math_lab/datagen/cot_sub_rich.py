"""
cot_sub_rich.py — Rich-CoT data for atomic sub_2d_with_borrow.

Counterpart to cot_div_expanded / cot_mul_rich. Same recipe:
  ~1000 unique (a, b) pairs from the borrow-required universe × 4 reasoning
  variants → ~4000 train records, novel-pair holdout.

Borrow universe: a, b ∈ [10, 99] with a > b AND (a%10) < (b%10), so column
subtraction needs to borrow from tens.  |universe| = 1620 such pairs.

Variants:
  column_borrow   : standard column algorithm with explicit borrow step
  decompose       : 53-27 = 53-20-7 = 33-7 = 26
  complement      : 53-27 = 53-30+3 = 23+3 = 26
  count_up        : 53-27: 27+3=30, 30+23=53, gap = 3+23 = 26
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs(n_unique, seed=42):
    """Borrow-required pairs: a > b AND a%10 < b%10."""
    rng = random.Random(seed)
    universe = []
    for a in range(10, 100):
        for b in range(10, 100):
            if a > b and (a % 10) < (b % 10):
                universe.append((a, b, a - b))
    rng.shuffle(universe)
    return universe[:n_unique]


def _ex_sub_borrow(a, b):
    # Sibling always satisfies borrow constraint: ea%10 < eb%10, ea > eb
    for ea, eb in [(52, 37), (71, 48), (63, 28), (84, 57), (43, 19)]:
        if (ea, eb) != (a, b) and ea > eb and ea % 10 < eb % 10:
            return ea, eb, ea - eb
    return 54, 38, 16


def variant_column_borrow(a, b, ans):
    """Standard column subtraction with borrow."""
    ea, eb, eans = _ex_sub_borrow(a, b)
    ea_t, ea_o = ea // 10, ea % 10
    eb_t, eb_o = eb // 10, eb % 10
    eo_new = ea_o + 10 - eb_o
    ex = (f"{ea} - {eb}: ones borrow {ea_o}+10={ea_o+10}, "
          f"{ea_o+10}-{eb_o}={eo_new}; tens {ea_t-1}-{eb_t}={ea_t-1-eb_t}; = {eans}")
    a_t, a_o = a // 10, a % 10
    b_t, b_o = b // 10, b % 10
    new_a_o = a_o + 10
    ones_diff = new_a_o - b_o
    new_a_t = a_t - 1
    tens_diff = new_a_t - b_t
    return (f"Example: {ex}\n"
            f"{a} - {b}:\n"
            f"ones: {a_o} - {b_o} → borrow → {new_a_o} - {b_o} = {ones_diff}\n"
            f"tens: {a_t} → {new_a_t}, {new_a_t} - {b_t} = {tens_diff}\n"
            f"= {tens_diff}{ones_diff}\n"
            f"#### {ans}")


def variant_decompose(a, b, ans):
    """Split b: a - b = a - b_tens - b_ones."""
    ea, eb, eans = _ex_sub_borrow(a, b)
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


def variant_complement(a, b, ans):
    """Round b up to next 10, subtract, add back the overshoot."""
    ea, eb, eans = _ex_sub_borrow(a, b)
    eb_up = ((eb // 10) + 1) * 10; eos = eb_up - eb; ei = ea - eb_up
    ex = f"{ea} - {eb} = {ea} - {eb_up} + {eos} = {ei} + {eos} = {eans}"
    b_t_up = ((b // 10) + 1) * 10
    overshoot = b_t_up - b
    intermediate = a - b_t_up
    return (f"Example: {ex}\n"
            f"{a} - {b} = {a} - {b_t_up} + {overshoot}\n"
            f"= {intermediate} + {overshoot}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_count_up(a, b, ans):
    """Count up from b to a in two hops: b → next_ten → a."""
    ea, eb, eans = _ex_sub_borrow(a, b)
    en10 = ((eb // 10) + 1) * 10; eh1 = en10 - eb; eh2 = ea - en10
    ex = f"{ea} - {eb}: {eb}+{eh1}={en10}, {en10}+{eh2}={ea}, gap={eh1}+{eh2}={eans}"
    next_ten = ((b // 10) + 1) * 10
    hop1 = next_ten - b
    hop2 = a - next_ten
    return (f"Example: {ex}\n"
            f"{a} - {b}: count up from {b} to {a}\n"
            f"{b} + {hop1} = {next_ten}\n"
            f"{next_ten} + {hop2} = {a}\n"
            f"gap = {hop1} + {hop2} = {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'column_borrow': variant_column_borrow,
    'decompose': variant_decompose,
    'complement': variant_complement,
    'count_up': variant_count_up,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} - {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'sub_2d_with_borrow',
        'stage': 2, 'level': 2,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(n_unique=1000, holdout_frac=0.2, seed=42):
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
        'concept': 'sub_2d_with_borrow',
        'stage': 2, 'level': 2,
    } for a, b, ans in val_pairs]
    rng.shuffle(train_records)
    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-unique', type=int, default=1000)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(n_unique=args.n_unique, holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'sub_2d_borrow_rich_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'sub_2d_borrow_rich_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    # Sanity sample
    print('\nSample variants for "53 - 27":')
    for vname, vfn in VARIANTS.items():
        print(f'  [{vname}]')
        for line in vfn(53, 27, 53-27).split('\n'):
            print(f'    {line}')
        print()


if __name__ == '__main__':
    main()
