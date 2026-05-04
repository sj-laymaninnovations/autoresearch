"""
cot_div_rich.py — Richer CoT formats for division-only training data.

Atomic div_1d failed at 5M with the terse "d*q=a; a/d=q" CoT (3/3 seeds
FAILED). This module generates 4 richer reasoning-path variants per pair,
designed to give the model multiple lenses on the same fact:

  factorization     : "a = d × q ; a / d = q ; #### q"
  buildup           : "d × 1 = d ; d × 2 = 2d ; ... ; d × q = a ; #### q"
  countdown         : "a - d = a-d ; (a-d) - d = a-2d ; ... ; 0 (q steps) ; #### q"
  inverse_question  : "a / d : how many d in a ? q ; q × d = a ; #### q"

Edge: q=0 gets a single direct line per variant since looping isn't useful.

Total per skill: 80 unique pairs × 4 variants = 320 records (vs 80 before).
"""
import argparse, json, datetime, random
from pathlib import Path

# 80 unique div_1d pairs (same as skill_data.py expanded definition):
def gen_div_1d_pairs():
    """Returns [(a, d, q), ...] — quotient stays 1-digit, dividend up to 81."""
    pairs = {}
    for q in range(0, 10):
        for d in range(2, 10):
            a = q * d
            pairs[(a, d)] = q
    return [(a, d, q) for (a, d), q in pairs.items()]


def _ex_div_1d(a, d, q):
    for ea, ed, eq in [(12, 3, 4), (42, 7, 6), (18, 6, 3), (45, 9, 5), (16, 4, 4)]:
        if (ea, ed) != (a, d):
            return ea, ed, eq
    return 8, 2, 4


def variant_factorization(a, d, q):
    ea, ed, eq = _ex_div_1d(a, d, q)
    ex = f"{ea} = {ed} * {eq}; {ea}/{ed} = {eq}"
    return (f"Example: {ex}\n"
            f"{a} = {d} * {q}\n"
            f"{a}/{d} = {q}\n"
            f"#### {q}")


def variant_buildup(a, d, q):
    ea, ed, eq = _ex_div_1d(a, d, q)
    if eq <= 1:
        ex = f"{ed} * {eq} = {ed*eq}; {ea}/{ed} = {eq}"
    else:
        ex = "; ".join(f"{ed}*{k}={ed*k}" for k in range(1, eq + 1)) + f"; {ea}/{ed}={eq}"
    if q == 0:
        return (f"Example: {ex}\n{d} * 0 = 0\n{a}/{d} = 0\n#### 0")
    if q == 1:
        return (f"Example: {ex}\n{d} * 1 = {d}\n{a}/{d} = 1\n#### 1")
    multiples = "\n".join(f"{d} * {k} = {d*k}" for k in range(1, q + 1))
    return (f"Example: {ex}\n"
            f"{multiples}\n"
            f"{a}/{d} = {q}\n"
            f"#### {q}")


def variant_countdown(a, d, q):
    ea, ed, eq = _ex_div_1d(a, d, q)
    if eq == 0:
        ex = f"{ea}/{ed}: none fit, = 0"
    else:
        esteps = "; ".join(f"{ea-ed*k+ed}-{ed}={ea-ed*k}" for k in range(1, min(eq+1, 4)))
        ex = f"{ea}/{ed}: " + esteps + ("..." if eq > 3 else "") + f" ({eq} steps) = {eq}"
    if q == 0:
        return (f"Example: {ex}\n{a} - 0*{d} = {a} (none fit)\n{a}/{d} = 0\n#### 0")
    steps = []
    cur = a
    for k in range(1, q + 1):
        next_cur = cur - d
        steps.append(f"{cur} - {d} = {next_cur}")
        cur = next_cur
    steps_str = "\n".join(steps)
    return (f"Example: {ex}\n"
            f"{steps_str}\n"
            f"step count = {q}\n"
            f"{a}/{d} = {q}\n"
            f"#### {q}")


def variant_inverse(a, d, q):
    ea, ed, eq = _ex_div_1d(a, d, q)
    ex = f"{ea}/{ed}: how many {ed} in {ea}? {eq} * {ed} = {ea}, so = {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{a}/{d} : how many {d} in {a} ?\n"
                f"0 * {d} = 0 = {a}\n"
                f"{a}/{d} = 0\n"
                f"#### 0")
    return (f"Example: {ex}\n"
            f"{a}/{d} : how many {d} in {a} ?\n"
            f"{q} * {d} = {a}\n"
            f"{a}/{d} = {q}\n"
            f"#### {q}")


VARIANTS = {
    'factorization': variant_factorization,
    'buildup': variant_buildup,
    'countdown': variant_countdown,
    'inverse': variant_inverse,
}


def to_record(a, d, q, variant_name, variant_fn, hold_out=False):
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


def build(holdout_frac=0.5, seed=42):
    """Returns (train_records, val_records).
    Hold out 50% of unique (a, d) pairs from train; their CoT variants don't appear.
    Val records have a single 'plain' answer (no CoT) for clean EM evaluation.
    """
    pairs = gen_div_1d_pairs()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * holdout_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_records = []
    for a, d, q in train_pairs:
        for vname, vfn in VARIANTS.items():
            train_records.append(to_record(a, d, q, vname, vfn))

    # Val: single record per pair, plain answer
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
    args = p.parse_args()

    train, val = build(seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'div_1d_richcot_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'div_1d_richcot_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants) -> {train_p.name}')
    print(f'Val:   {len(val)} novel-pair holdout -> {val_p.name}')

    # Show samples
    print('\nSample variants for "12 / 3":')
    for vname, vfn in VARIANTS.items():
        print(f'  [{vname}]')
        for line in vfn(12, 3, 4).split('\n'):
            print(f'    {line}')
        print()


if __name__ == '__main__':
    main()
