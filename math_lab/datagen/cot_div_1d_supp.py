"""
cot_div_1d_supp.py — Supplementary div_1d data targeting the q>9 off-by-one gap.

Specifically adds 3 variants that explicitly:
  1. decade_anchor  : (d*10 + remainder) / d = 10 + remainder/d
  2. table_walk     : d*10=A, d*11=B, d*12=C ... until we hit dividend
  3. step_up        : "got 11 groups = A, need B-A more, B-A/d = k more, total = 11+k"

Focuses entirely on d=2-12, q=9-15 (the failing tier).
~400 train records, all tier 4.
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def gen_pairs(max_d=12, q_min=9, q_max=15):
    seen = {}
    for d in range(2, max_d + 1):
        for q in range(q_min, q_max + 1):
            a = d * q
            if a <= 180:
                seen[(a, d)] = q
    return [(a, d, q) for (a, d), q in seen.items()]


_POOL = None
def _sib(a, d, rng):
    global _POOL
    if _POOL is None:
        _POOL = [(a_, d_, q_) for a_, d_, q_ in gen_pairs() if q_ >= 10 and a_ >= 20]
    return rng.choice([t for t in _POOL if (t[0], t[1]) != (a, d)])


def v_decade_anchor(a, d, q, rng):
    """Split: d*10=base, remainder = a - base, steps = remainder/d."""
    base = d * 10
    rem  = a - base
    steps = q - 10
    ea, ed, eq = _sib(a, d, rng)
    ebase = ed*10; erem = ea-ebase; esteps = eq-10
    ex = (f"{ea}/{ed}: {ed}×10={ebase}, {ea}-{ebase}={erem}, "
          f"{erem}/{ed}={esteps}, total=10+{esteps}={eq}")
    return (f"Example: {ex}\n"
            f"{a}/{d}: {d}×10={base}\n"
            f"remainder: {a}-{base}={rem}\n"
            f"extra steps: {rem}/{d}={steps}\n"
            f"total: 10+{steps}={q}\n"
            f"{a}/{d}={q}\n#### {q}")


def v_table_walk(a, d, q, rng):
    """Walk the times table from d*9 upward until we hit a."""
    ea, ed, eq = _sib(a, d, rng)
    # Show walk from 9 to eq
    ex_walk = "; ".join(f"{ed}×{k}={ed*k}" for k in range(9, eq+1))
    ex = f"{ea}/{ed}: {ex_walk} → {eq}"
    walk = "; ".join(f"{d}×{k}={d*k}" for k in range(9, q+1))
    return (f"Example: {ex}\n"
            f"{a}/{d}: walk from 9:\n"
            f"{walk}\n"
            f"hit {a} at k={q}\n"
            f"{a}/{d}={q}\n#### {q}")


def v_step_up(a, d, q, rng):
    """Know d*10, step up until we reach a."""
    ea, ed, eq = _sib(a, d, rng)
    ebase = ed*10
    ex = (f"{ea}/{ed}: know {ed}×10={ebase}, "
          f"step up: {ebase}+{ed}={ed*11}...={ea} at {eq}")
    base = d * 10
    steps = []
    cur = base
    for k in range(10, q+1):
        steps.append(f"{d}×{k}={d*k}")
    return (f"Example: {ex}\n"
            f"{a}/{d}: know {d}×10={base}\n"
            + "\n".join(steps) + "\n"
            f"reached {a} at {q}\n"
            f"{a}/{d}={q}\n#### {q}")


def v_verify_up(a, d, q, rng):
    """Guess q, compute d*q, verify equals a."""
    ea, ed, eq = _sib(a, d, rng)
    wrong = q - 1
    ex = (f"{ea}/{ed}: try {eq-1}→{ed*(eq-1)}≠{ea}; "
          f"try {eq}→{ed*eq}={ea}✓ → {eq}")
    return (f"Example: {ex}\n"
            f"{a}/{d}: try {wrong}→{d*wrong}"
            + (f"={a}✓" if d*wrong==a else f"≠{a}") +
            f"\ntry {q}→{d*q}={a}✓\n"
            f"{a}/{d}={q}\n#### {q}")


def v_fraction_simplify(a, d, q, rng):
    """Express as improper fraction, simplify by d."""
    ea, ed, eq = _sib(a, d, rng)
    ex = f"{ea}/{ed}: divide top by {ed}: {ea}÷{ed}={eq} → {eq}"
    return (f"Example: {ex}\n"
            f"{a}/{d}: numerator {a} ÷ denominator {d}\n"
            f"{a} ÷ {d} = {q}\n"
            f"{a}/{d} = {q}\n#### {q}")


VARIANTS = {
    'decade_anchor':     v_decade_anchor,
    'table_walk':        v_table_walk,
    'step_up':           v_step_up,
    'verify_up':         v_verify_up,
    'fraction_simplify': v_fraction_simplify,
}


def build(holdout_frac=0.12, seed=42):
    pairs = gen_pairs()
    rng = random.Random(seed); rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * holdout_frac))
    val_p, train_p = pairs[-n_val:], pairs[:-n_val]

    train_recs, val_recs = [], []
    for a, d, q in train_p:
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, d, q, rng)
                assert '####' in cot and cot.count('####')==1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception:
                continue
            train_recs.append({
                'problem': f'{a} / {d}', 'solution': str(q),
                'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': 4,
                'solution_cot': cot, 'cot_source': f'supp_{vn}',
                'tier': 4, 'hold_out': False,
            })
    for a, d, q in val_p:
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, d, q, rng)
            except Exception:
                continue
            val_recs.append({
                'problem': f'{a} / {d}', 'solution': str(q),
                'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': 4,
                'solution_cot': cot, 'cot_source': f'supp_{vn}',
                'tier': 4, 'hold_out': True,
            })
    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"div_1d_supp_train_{tag}.jsonl"
    vf = out / f"div_1d_supp_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))
    print(f"Train: {len(train):>4} records  ({len(set(r['problem'] for r in train))} unique q>9 pairs)")
    print(f"Val:   {len(val):>4} records")
    rng = random.Random(42)
    print("\nSample decade_anchor 96/8:")
    print(v_decade_anchor(96, 8, 12, rng))


if __name__ == '__main__':
    main()
