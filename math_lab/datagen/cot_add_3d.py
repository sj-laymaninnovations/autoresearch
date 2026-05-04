"""
cot_add_3d.py — 3-digit + 3-digit / 3-digit - 3-digit CoT generator (v2 format).

Targets the exact gap found in mul_2d: partial products like 920+138 or 2460+574
require 3-digit intermediate addition. Four reasoning variants:
  column       : standard column addition with carry
  decompose    : split addend into hundreds+tens+ones
  partial      : add hundreds first, then tens+ones
  complement   : round up, subtract overshoot

Generates ~4800 train + 600 val records.
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def gen_pairs(n_unique=1200, seed=42):
    rng = random.Random(seed)
    universe = [(a, b) for a in range(100, 999) for b in range(100, 999) if a + b <= 1999]
    rng.shuffle(universe)
    return [(a, b, a + b) for a, b in universe[:n_unique]]


def _ex_add_3d(a, b):
    for ea, eb in [(354, 278), (612, 489), (783, 157), (456, 348), (920, 138)]:
        if (ea, eb) != (a, b):
            return ea, eb, ea + eb
    return 500, 300, 800


# ── Variants ──────────────────────────────────────────────────────────────────

def variant_column(a, b, ans):
    ea, eb, eans = _ex_add_3d(a, b)
    ea_h, ea_t, ea_o = ea//100, (ea%100)//10, ea%10
    eb_h, eb_t, eb_o = eb//100, (eb%100)//10, eb%10
    eo = ea_o + eb_o; ec1 = eo // 10; eo_ = eo % 10
    et = ea_t + eb_t + ec1; ec2 = et // 10; et_ = et % 10
    eh = ea_h + eb_h + ec2
    ex = (f"{ea}+{eb}: ones {ea_o}+{eb_o}={eo}(c{ec1},w{eo_}); "
          f"tens {ea_t}+{eb_t}+{ec1}={et}(c{ec2},w{et_}); "
          f"hund {ea_h}+{eb_h}+{ec2}={eh}; ={eans}")
    a_h, a_t, a_o = a//100, (a%100)//10, a%10
    b_h, b_t, b_o = b//100, (b%100)//10, b%10
    o = a_o + b_o; c1 = o // 10; o_ = o % 10
    t = a_t + b_t + c1; c2 = t // 10; t_ = t % 10
    h = a_h + b_h + c2
    return (f"Example: {ex}\n"
            f"{a} + {b}:\n"
            f"ones: {a_o}+{b_o}={o} → carry {c1}, write {o_}\n"
            f"tens: {a_t}+{b_t}+{c1}={t} → carry {c2}, write {t_}\n"
            f"hundreds: {a_h}+{b_h}+{c2}={h}\n"
            f"= {ans}\n#### {ans}")


def variant_decompose(a, b, ans):
    ea, eb, eans = _ex_add_3d(a, b)
    eb_h = (eb//100)*100; eb_t = ((eb%100)//10)*10; eb_o = eb%10
    es1 = ea + eb_h; es2 = es1 + eb_t
    ex = f"{ea}+{eb} = {ea}+{eb_h}+{eb_t}+{eb_o} = {es1}+{eb_t}+{eb_o} = {es2}+{eb_o} = {eans}"
    b_h = (b//100)*100; b_t = ((b%100)//10)*10; b_o = b%10
    s1 = a + b_h; s2 = s1 + b_t
    return (f"Example: {ex}\n"
            f"{a} + {b} = {a} + {b_h} + {b_t} + {b_o}\n"
            f"= {s1} + {b_t} + {b_o}\n"
            f"= {s2} + {b_o}\n"
            f"= {ans}\n#### {ans}")


def variant_partial(a, b, ans):
    ea, eb, eans = _ex_add_3d(a, b)
    ea_h = (ea//100)*100; ea_r = ea - ea_h
    eb_h = (eb//100)*100; eb_r = eb - eb_h
    esum_h = ea_h + eb_h; esum_r = ea_r + eb_r
    ex = f"{ea}+{eb} = ({ea_h}+{ea_r})+({eb_h}+{eb_r}) = {esum_h}+{esum_r} = {eans}"
    a_h = (a//100)*100; a_r = a - a_h
    b_h = (b//100)*100; b_r = b - b_h
    sum_h = a_h + b_h; sum_r = a_r + b_r
    return (f"Example: {ex}\n"
            f"{a} + {b} = ({a_h}+{a_r}) + ({b_h}+{b_r})\n"
            f"= {sum_h} + {sum_r}\n"
            f"= {ans}\n#### {ans}")


def variant_complement(a, b, ans):
    ea, eb, eans = _ex_add_3d(a, b)
    eb_r = ((eb // 100) + 1) * 100; eos = eb_r - eb; ei = ea + eb_r
    ex = f"{ea}+{eb} = {ea}+{eb_r}-{eos} = {ei}-{eos} = {eans}"
    b_round = ((b // 100) + 1) * 100; overshoot = b_round - b; inter = a + b_round
    return (f"Example: {ex}\n"
            f"{a} + {b} = {a} + {b_round} - {overshoot}\n"
            f"= {inter} - {overshoot}\n"
            f"= {ans}\n#### {ans}")


VARIANTS = {
    'column':     variant_column,
    'decompose':  variant_decompose,
    'partial':    variant_partial,
    'complement': variant_complement,
}


def to_record(a, b, ans, vname, vfn, hold_out=False):
    return {
        'problem': f'{a} + {b}',
        'solution': str(ans), 'answer': str(ans), 'answer_real': ans,
        'concept': 'add_3d', 'stage': 2, 'level': 3,
        'solution_cot': vfn(a, b, ans),
        'cot_source': f'rule_based_{vname}',
        'hold_out': hold_out,
    }


def build(n_unique=1200, holdout_frac=0.125, seed=42):
    pairs = gen_pairs(n_unique, seed=seed)
    n_hold = max(1, int(len(pairs) * holdout_frac))
    train_p, val_p = pairs[n_hold:], pairs[:n_hold]
    train_recs, val_recs = [], []
    for a, b, ans in train_p:
        for vn, vf in VARIANTS.items():
            train_recs.append(to_record(a, b, ans, vn, vf, False))
    for a, b, ans in val_p:
        for vn, vf in VARIANTS.items():
            val_recs.append(to_record(a, b, ans, vn, vf, True))
    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-unique', type=int, default=1200)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--out-dir',  default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(args.n_unique, seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"add_3d_train_{tag}.jsonl"
    vf = out / f"add_3d_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))
    print(f"Train: {len(train)} records  → {tf.name}")
    print(f"Val:   {len(val)} records  → {vf.name}")
    print(f"\nSample {train[0]['problem']}:")
    print(f"  {train[0]['solution_cot'][:200]}")


if __name__ == '__main__':
    main()
