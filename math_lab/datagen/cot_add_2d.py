"""
cot_add_2d.py — Atomic add_2d (handles BOTH no-carry and with-carry).

Universe: a, b ∈ [10, 99] = 8100 pairs (same as mul_2d). Answer ∈ [20, 198].

Mixed: each sampled pair could be either carry or no-carry (whichever the
operands produce). The carry / no-carry distinction is reflected in the
column variant's CoT but the model trains on both.

Recipe target: 50% coverage = 4000 unique pairs × 4 variants = 16000 records.
With dropout=0.05 and 300 epochs (matching the mul_2d recipe). The simpler
operation should land it in the 90%+ range easily.

Variants:
  column         : column-wise with explicit carry handling when needed
  decompose      : (a_t + b_t)*10 + (a_o + b_o)  (handles overflow inline)
  partial        : a + b_t + b_o
  near_round     : a + b = a + (round_up_b) - delta
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
        out.append((a, b, a + b))
    return out


def _ex_add_2d(a, b):
    for ea, eb in [(35, 47), (23, 58), (41, 39), (67, 24), (52, 31)]:
        if (ea, eb) != (a, b):
            return ea, eb, ea + eb
    return 11, 22, 33


def variant_column(a, b, ans):
    """Standard column algorithm with carry annotation."""
    ea, eb, eans = _ex_add_2d(a, b)
    ea_t, ea_o = ea // 10, ea % 10
    eb_t, eb_o = eb // 10, eb % 10
    eo = ea_o + eb_o
    if eo >= 10:
        ex = (f"{ea} + {eb}: ones {ea_o}+{eb_o}={eo}, carry 1, "
              f"write {eo-10}; tens {ea_t}+{eb_t}+1={ea_t+eb_t+1}; = {eans}")
    else:
        ex = f"{ea} + {eb}: ones {ea_o}+{eb_o}={eo}; tens {ea_t}+{eb_t}={ea_t+eb_t}; = {eans}"
    a_t, a_o = a // 10, a % 10
    b_t, b_o = b // 10, b % 10
    ones_sum = a_o + b_o
    if ones_sum >= 10:
        ones_digit = ones_sum - 10
        carry = 1
        tens_sum = a_t + b_t + carry
        return (f"Example: {ex}\n"
                f"{a} + {b}:\n"
                f"ones: {a_o} + {b_o} = {ones_sum} → carry 1, write {ones_digit}\n"
                f"tens: {a_t} + {b_t} + 1 = {tens_sum}\n"
                f"= {tens_sum}{ones_digit}\n"
                f"#### {ans}")
    else:
        tens_sum = a_t + b_t
        return (f"Example: {ex}\n"
                f"{a} + {b}:\n"
                f"ones: {a_o} + {b_o} = {ones_sum} (no carry)\n"
                f"tens: {a_t} + {b_t} = {tens_sum}\n"
                f"= {tens_sum}{ones_sum}\n"
                f"#### {ans}")


def variant_decompose(a, b, ans):
    """Split both into tens+ones, sum components."""
    ea, eb, eans = _ex_add_2d(a, b)
    ea_t = (ea // 10) * 10; ea_o = ea - ea_t
    eb_t = (eb // 10) * 10; eb_o = eb - eb_t
    ex = (f"{ea} + {eb} = ({ea_t}+{ea_o}) + ({eb_t}+{eb_o})"
          f" = {ea_t+eb_t} + {ea_o+eb_o} = {eans}")
    a_t = (a // 10) * 10
    a_o = a - a_t
    b_t = (b // 10) * 10
    b_o = b - b_t
    tens = a_t + b_t
    ones = a_o + b_o
    return (f"Example: {ex}\n"
            f"{a} + {b} = ({a_t}+{a_o}) + ({b_t}+{b_o})\n"
            f"= {tens} + {ones}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_partial(a, b, ans):
    """Add b in two pieces to a."""
    ea, eb, eans = _ex_add_2d(a, b)
    eb_t = (eb // 10) * 10; eb_o = eb - eb_t; es1 = ea + eb_t
    ex = f"{ea} + {eb} = {ea} + {eb_t} + {eb_o} = {es1} + {eb_o} = {eans}"
    b_t = (b // 10) * 10
    b_o = b - b_t
    step1 = a + b_t
    return (f"Example: {ex}\n"
            f"{a} + {b} = {a} + {b_t} + {b_o}\n"
            f"= {step1} + {b_o}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_near_round(a, b, ans):
    """Round b up to next 10, add, then subtract delta."""
    ea, eb, eans = _ex_add_2d(a, b)
    eb_r = ((eb // 10) + 1) * 10; ed = eb_r - eb; ei = ea + eb_r
    ex = f"{ea} + {eb} = {ea} + {eb_r} - {ed} = {ei} - {ed} = {eans}"
    b_round = ((b // 10) + 1) * 10
    delta = b_round - b
    intermediate = a + b_round
    return (f"Example: {ex}\n"
            f"{a} + {b} = {a} + {b_round} - {delta}\n"
            f"= {intermediate} - {delta}\n"
            f"= {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'column': variant_column,
    'decompose': variant_decompose,
    'partial': variant_partial,
    'near_round': variant_near_round,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} + {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'add_2d',
        'stage': 2, 'level': 2,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(n_unique=4000, holdout_frac=0.2, seed=42):
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
        'problem': f'{a} + {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'add_2d',
        'stage': 2, 'level': 2,
    } for a, b, ans in val_pairs]
    rng.shuffle(train_records)
    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-unique', type=int, default=4000)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(n_unique=args.n_unique, holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'add_2d_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'add_2d_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'Coverage: {args.n_unique}/8100 = {100*args.n_unique/8100:.1f}% of universe')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    print('\nSamples:')
    for prob_a, prob_b in [(53, 27), (82, 39), (11, 22)]:
        print(f'\n{prob_a} + {prob_b} = {prob_a + prob_b}')
        for vname, vfn in VARIANTS.items():
            print(f'  [{vname}]')
            for line in vfn(prob_a, prob_b, prob_a + prob_b).split('\n'):
                print(f'    {line}')


if __name__ == '__main__':
    main()
