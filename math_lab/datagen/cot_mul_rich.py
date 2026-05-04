"""
cot_mul_rich.py — Rich-CoT data for atomic mul_2d (counterpart to cot_div_rich).

mul_2d is the next-weakest atomic skill (1-3% on prior tests). Apply the
same data-scaling recipe that took div_1d from 22% → 95.4%.

Coverage: 1000 unique (a,b) pairs sampled from [10,99] × [10,99] = 8100.
Variants per pair: 4 reasoning paths.

  partial_products  : 82 * 37 = 82*(30+7) = 82*30 + 82*7 = 2460+574 = 3034
  distributive_a    : 82 * 37 = (80+2)*37 = 80*37 + 2*37 = 2960+74 = 3034
  area_decomp       : 82 * 37 = (80+2)*(30+7) = 80*30 + 80*7 + 2*30 + 2*7
  commutative       : 37 * 82 (commute) → use partial-products on smaller
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


def _ex_mul_2d(a, b):
    for ea, eb in [(13, 24), (32, 21), (41, 13), (22, 34), (12, 31)]:
        if (ea, eb) != (a, b):
            return ea, eb, ea * eb
    return 11, 12, 132


def variant_partial_products(a, b, ans):
    """Decompose b: a*b = a*(round(b/10)*10 + b%10) = a*round + a*rem"""
    ea, eb, eans = _ex_mul_2d(a, b)
    eb_t = (eb // 10) * 10; eb_o = eb - eb_t
    ex = (f"{ea} * {eb} = {ea}*({eb_t}+{eb_o})"
          f" = {ea*eb_t}+{ea*eb_o} = {eans}")
    b_tens = (b // 10) * 10
    b_ones = b - b_tens
    p1 = a * b_tens
    p2 = a * b_ones
    return (f"Example: {ex}\n"
            f"{a} * {b} = {a}*({b_tens}+{b_ones})\n"
            f"= {a}*{b_tens} + {a}*{b_ones}\n"
            f"= {p1} + {p2}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_distributive_a(a, b, ans):
    """Decompose a: a*b = (round(a/10)*10 + a%10)*b = round*b + rem*b"""
    ea, eb, eans = _ex_mul_2d(a, b)
    ea_t = (ea // 10) * 10; ea_o = ea - ea_t
    ex = (f"{ea} * {eb} = ({ea_t}+{ea_o})*{eb}"
          f" = {ea_t*eb}+{ea_o*eb} = {eans}")
    a_tens = (a // 10) * 10
    a_ones = a - a_tens
    p1 = a_tens * b
    p2 = a_ones * b
    return (f"Example: {ex}\n"
            f"{a} * {b} = ({a_tens}+{a_ones})*{b}\n"
            f"= {a_tens}*{b} + {a_ones}*{b}\n"
            f"= {p1} + {p2}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_area_decomp(a, b, ans):
    """Full grid: (a_t+a_o)*(b_t+b_o) = a_t*b_t + a_t*b_o + a_o*b_t + a_o*b_o"""
    ea, eb, eans = _ex_mul_2d(a, b)
    eat = (ea//10)*10; eao = ea-eat; ebt = (eb//10)*10; ebo = eb-ebt
    ex = (f"{ea}*{eb} = ({eat}+{eao})*({ebt}+{ebo})"
          f" = {eat*ebt}+{eat*ebo}+{eao*ebt}+{eao*ebo} = {eans}")
    a_t = (a // 10) * 10; a_o = a - a_t
    b_t = (b // 10) * 10; b_o = b - b_t
    p1 = a_t * b_t; p2 = a_t * b_o; p3 = a_o * b_t; p4 = a_o * b_o
    return (f"Example: {ex}\n"
            f"{a} * {b} = ({a_t}+{a_o})*({b_t}+{b_o})\n"
            f"= {a_t}*{b_t} + {a_t}*{b_o} + {a_o}*{b_t} + {a_o}*{b_o}\n"
            f"= {p1} + {p2} + {p3} + {p4}\n"
            f"= {ans}\n"
            f"#### {ans}")


def variant_commutative(a, b, ans):
    """Use commutative property to reorient, then partial products on the smaller-tens."""
    ea, eb, eans = _ex_mul_2d(a, b)
    es, el = (eb, ea) if ea > eb else (ea, eb)
    est = (es//10)*10; eso = es-est
    ex = (f"{ea}*{eb} = {el}*{es} (commute)"
          f" = {el}*({est}+{eso}) = {el*est}+{el*eso} = {eans}")
    s, l = (b, a) if a > b else (a, b)
    s_tens = (s // 10) * 10; s_ones = s - s_tens
    p1 = l * s_tens; p2 = l * s_ones
    return (f"Example: {ex}\n"
            f"{a} * {b} = {l} * {s} (commute)\n"
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
    p.add_argument('--n-unique', type=int, default=1000)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    args = p.parse_args()

    train, val = build(n_unique=args.n_unique, holdout_frac=args.holdout_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p = out_dir / f'mul_2d_rich_train_seed{args.seed}_{ts}.jsonl'
    val_p = out_dir / f'mul_2d_rich_val_seed{args.seed}_{ts}.jsonl'
    with open(train_p, 'w') as f:
        for r in train: f.write(json.dumps(r) + '\n')
    with open(val_p, 'w') as f:
        for r in val: f.write(json.dumps(r) + '\n')
    print(f'Train: {len(train)} records ({len(train)//4} pairs × 4 variants)')
    print(f'Val:   {len(val)} novel-pair holdout')
    print(f'  -> {train_p.name}')
    print(f'  -> {val_p.name}')

    # Sanity sample
    print('\nSample variants for "82 * 37":')
    for vname, vfn in VARIANTS.items():
        print(f'  [{vname}]')
        for line in vfn(82, 37, 82*37).split('\n'):
            print(f'    {line}')
        print()


if __name__ == '__main__':
    main()
