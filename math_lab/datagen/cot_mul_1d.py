"""
cot_mul_1d.py — Atomic mul_1d skill (single-digit × single-digit).

Tiny universe: a, b ∈ [0,9] = 100 pairs. Answer ∈ [0,81] (45 distinct values
post-commutativity, but trained on full pairs).

Five rich-CoT variants:
  table       : direct table lookup ("7 × 8 = 56")
  count       : repeated addition ("7+7+7+7+7+7+7+7 = 56")
  doubling    : doubling for even multiplicand ("7 × 8 = 7×4 doubled = 28×2 = 56")
  near_round  : round one to 10 ("7 × 8 = 7×10 - 7×2 = 70 - 14 = 56")
  commute     : commute then table ("7 × 8 = 8 × 7 = 56")
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs():
    return [(a, b, a * b) for a in range(10) for b in range(10)]


def _ex_mul_1d(a, b):
    for ea, eb in [(3, 7), (6, 8), (4, 9), (5, 6), (2, 8)]:
        if (ea, eb) != (a, b):
            return ea, eb, ea * eb
    return 2, 3, 6


def variant_table(a, b, ans):
    ea, eb, eans = _ex_mul_1d(a, b)
    return (f"Example: {ea} * {eb} = {eans}\n"
            f"{a} * {b} = {ans}\n"
            f"#### {ans}")


def variant_count(a, b, ans):
    ea, eb, eans = _ex_mul_1d(a, b)
    if ea == 0 or eb == 0:
        ex = f"{ea} * {eb} = 0 (zero factor)"
    else:
        ex = f"{ea} * {eb} = " + " + ".join([str(ea)] * eb) + f" = {eans}"
    if a == 0 or b == 0:
        return (f"Example: {ex}\n{a} * {b} = 0 (zero factor)\n#### 0")
    pieces = " + ".join([str(a)] * b)
    return (f"Example: {ex}\n"
            f"{a} * {b} = {pieces}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_doubling(a, b, ans):
    """Halve b until manageable; halving b odd -> fall back to table."""
    ea, eb, eans = _ex_mul_1d(a, b)
    if eb == 0 or ea == 0:
        ex = f"{ea} * {eb} = 0"
    elif eb % 2 == 0:
        ehalf = eb // 2
        ex = f"{ea} * {eb} = {ea}*{ehalf} doubled = {ea*ehalf}*2 = {eans}"
    else:
        ex = f"{ea} * {eb} = {eans}"
    if b == 0 or a == 0:
        return (f"Example: {ex}\n{a} * {b} = 0\n#### 0")
    if b % 2 == 0 and b > 0:
        half = b // 2
        first = a * half
        return (f"Example: {ex}\n"
                f"{a} * {b} = {a}*{half} doubled\n"
                f"= {first} * 2\n"
                f"= {ans}\n"
                f"#### {ans}")
    return (f"Example: {ex}\n{a} * {b} = {ans}\n#### {ans}")


def variant_near_round(a, b, ans):
    """If b >= 5, do a*10 - a*(10-b). Else do a*5 - a*(5-b)."""
    ea, eb, eans = _ex_mul_1d(a, b)
    if ea == 0 or eb == 0:
        ex = f"{ea} * {eb} = 0"
    elif eb >= 5:
        ed = 10 - eb
        ex = f"{ea} * {eb} = {ea}*10 - {ea}*{ed} = {ea*10} - {ea*ed} = {eans}"
    else:
        ed = 5 - eb
        ex = f"{ea} * {eb} = {ea}*5 - {ea}*{ed} = {ea*5} - {ea*ed} = {eans}"
    if a == 0 or b == 0:
        return (f"Example: {ex}\n{a} * {b} = 0\n#### 0")
    if b >= 5:
        delta = 10 - b
        full = a * 10
        sub = a * delta
        return (f"Example: {ex}\n"
                f"{a} * {b} = {a}*10 - {a}*{delta}\n"
                f"= {full} - {sub}\n"
                f"= {ans}\n"
                f"#### {ans}")
    delta = 5 - b
    half = a * 5
    sub = a * delta
    return (f"Example: {ex}\n"
            f"{a} * {b} = {a}*5 - {a}*{delta}\n"
            f"= {half} - {sub}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_commute(a, b, ans):
    ea, eb, eans = _ex_mul_1d(a, b)
    es, el = (eb, ea) if ea >= eb else (ea, eb)
    ex = f"{ea} * {eb} = {el} * {es} (commute) = {eans}"
    s, l = (b, a) if a >= b else (a, b)
    return (f"Example: {ex}\n"
            f"{a} * {b} = {l} * {s} (commute)\n"
            f"= {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'table': variant_table,
    'count': variant_count,
    'doubling': variant_doubling,
    'near_round': variant_near_round,
    'commute': variant_commute,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} * {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'mul_1d',
        'stage': 1, 'level': 1,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(holdout_frac=0.2, seed=42):
    pairs = gen_pairs()
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
        'solution': str(ans), 'answer': str(ans),
        'answer_real': ans, 'concept': 'mul_1d',
        'stage': 1, 'level': 1,
    } for a, b, ans in val_pairs]
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
    train_p = out_dir / f'mul_1d_rich_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'mul_1d_rich_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//5} pairs × 5 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    print('\nSample 7 * 8:')
    for vname, vfn in VARIANTS.items():
        print(f'  [{vname}]')
        for line in vfn(7, 8, 56).split('\n'):
            print(f'    {line}')


if __name__ == '__main__':
    main()
