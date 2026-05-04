"""
cot_add_1d.py — Atomic add_1d skill (single-digit + single-digit).

Tiny universe: a, b ∈ [0,9] = 100 pairs. Answer ∈ [0, 18].

Five rich-CoT variants:
  table       : direct lookup
  count_up    : 3 + 4 = 4, 5, 6, 7
  ten_complement : 8 + 5 = 8 + 2 + 3 = 10 + 3 = 13
  commute     : a + b = b + a
  doubles     : 6 + 7 = 6 + 6 + 1 = 12 + 1 = 13 (when one operand is close to a "double")
"""
import argparse, json, datetime, random
from pathlib import Path


def gen_pairs():
    return [(a, b, a + b) for a in range(10) for b in range(10)]


def _ex_add_1d(a, b):
    """Deterministic sibling example, always different from (a, b)."""
    for ea, eb in [(6, 7), (3, 5), (8, 4), (2, 9), (1, 8), (0, 5)]:
        if (ea, eb) != (a, b):
            return ea, eb, ea + eb
    return 0, 1, 1


def variant_table(a, b, ans):
    ea, eb, eans = _ex_add_1d(a, b)
    return (f"Example: {ea} + {eb} = {eans}\n"
            f"{a} + {b} = {ans}\n"
            f"#### {ans}")


def variant_count_up(a, b, ans):
    ea, eb, eans = _ex_add_1d(a, b)
    if eb == 0:
        eseq = f"{ea} + 0 = {ea}"
    else:
        eseq = (f"{ea} + {eb}: count up {eb} = "
                + ", ".join(str(ea + i) for i in range(1, eb + 1))
                + f" = {eans}")
    if b == 0:
        return (f"Example: {eseq}\n{a} + 0 = {a}\n#### {a}")
    if a == 0:
        return (f"Example: {eseq}\n0 + {b} = {b}\n#### {b}")
    seq = ", ".join(str(a + i) for i in range(1, b + 1))
    return (f"Example: {eseq}\n"
            f"{a} + {b}: starting at {a}, count up {b} = {seq}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_ten_complement(a, b, ans):
    """If a + b >= 10, decompose b = (10-a) + r so a + b = 10 + r."""
    ea, eb, eans = _ex_add_1d(a, b)
    if ea + eb < 10 or ea == 0 or eb == 0:
        ex = f"{ea} + {eb} = {eans}"
    else:
        ed = 10 - ea
        er = eb - ed
        ex = f"{ea} + {eb} = {ea} + {ed} + {er} = 10 + {er} = {eans}"
    if a + b < 10 or a == 0 or b == 0:
        return (f"Example: {ex}\n{a} + {b} = {ans}\n#### {ans}")
    delta = 10 - a
    rem = b - delta
    return (f"Example: {ex}\n"
            f"{a} + {b} = {a} + {delta} + {rem}\n"
            f"= 10 + {rem}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_commute(a, b, ans):
    ea, eb, eans = _ex_add_1d(a, b)
    ex = (f"{ea} + {eb} = {eans}" if ea >= eb
          else f"{ea} + {eb} = {eb} + {ea} = {eans}")
    if a >= b:
        return (f"Example: {ex}\n{a} + {b} = {ans}\n#### {ans}")
    return (f"Example: {ex}\n"
            f"{a} + {b} = {b} + {a} (commute)\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_doubles(a, b, ans):
    """Express in terms of doubles: a + b = double(min) + |a-b|."""
    ea, eb, eans = _ex_add_1d(a, b)
    elo, ehi = min(ea, eb), max(ea, eb)
    if ea == eb:
        ex = f"{ea} + {eb} = double {ea} = {eans}"
    elif ea == 0 or eb == 0:
        ex = f"{ea} + {eb} = {eans}"
    else:
        ex = (f"{ea} + {eb} = {elo} + {elo} + {ehi - elo}"
              f" = {2 * elo} + {ehi - elo} = {eans}")
    lo, hi = min(a, b), max(a, b)
    if a == b:
        return (f"Example: {ex}\n{a} + {b} = double {a} = {ans}\n#### {ans}")
    if a == 0 or b == 0:
        return (f"Example: {ex}\n{a} + {b} = {ans}\n#### {ans}")
    return (f"Example: {ex}\n"
            f"{a} + {b} = {lo} + {lo} + {hi - lo}\n"
            f"= {2 * lo} + {hi - lo}\n"
            f"= {ans}\n"
            f"#### {ans}")


VARIANTS = {
    'table': variant_table,
    'count_up': variant_count_up,
    'ten_complement': variant_ten_complement,
    'commute': variant_commute,
    'doubles': variant_doubles,
}


def to_record(a, b, ans, vname, vfn):
    return {
        'problem': f'{a} + {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'add_1d',
        'stage': 1, 'level': 1,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
    }


def build(holdout_frac=0.0, seed=42):
    """add_1d universe is finite (100 pairs); default is no holdout (memorize the table)."""
    pairs = gen_pairs()
    rng = random.Random(seed)
    # We replicate 3× to get more training samples per epoch (matches mul_1d_full recipe)
    train_records = []
    for _ in range(3):
        for a, b, ans in pairs:
            for vname, vfn in VARIANTS.items():
                train_records.append(to_record(a, b, ans, vname, vfn))
    rng.shuffle(train_records)

    # Val = full universe (sanity check on table)
    val_records = [{
        'problem': f'{a} + {b}',
        'solution': str(ans),
        'answer': str(ans),
        'answer_real': ans,
        'concept': 'add_1d',
        'stage': 1, 'level': 1,
    } for a, b, ans in pairs]

    if holdout_frac > 0:
        # If a holdout is requested, hold out random pairs from train universe
        rng2 = random.Random(seed + 1)
        all_pairs = list(pairs)
        rng2.shuffle(all_pairs)
        n_val = max(1, int(len(all_pairs) * holdout_frac))
        held = set(all_pairs[:n_val])
        train_pairs = [p for p in all_pairs if p not in held]
        # Rebuild
        train_records = []
        for _ in range(3):
            for a, b, ans in train_pairs:
                for vname, vfn in VARIANTS.items():
                    train_records.append(to_record(a, b, ans, vname, vfn))
        rng.shuffle(train_records)
        val_records = [{
            'problem': f'{a} + {b}', 'solution': str(ans), 'answer': str(ans),
            'answer_real': ans, 'concept': 'add_1d', 'stage': 1, 'level': 1,
        } for a, b, ans in held]

    return train_records, val_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='math_lab/results/skills')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--holdout-frac', type=float, default=0.0)
    args = p.parse_args()

    train, val = build(holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'add_1d_full_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'add_1d_full_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records (~{len(train)//5} pair-instances × 5 variants)')
    print(f'Val:   {len(val)} pairs')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    print('\nSample 6 + 7:')
    for vname, vfn in VARIANTS.items():
        print(f'  [{vname}]')
        for line in vfn(6, 7, 13).split('\n'):
            print(f'    {line}')


if __name__ == '__main__':
    main()
