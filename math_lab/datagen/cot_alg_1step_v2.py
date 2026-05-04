"""
cot_alg_1step_v2.py — Curriculum-sequenced ax=b generator (v2).

Fixes the 20% EM plateau by dramatically expanding the universe and sequencing:
  Tier 1: a=1-3,  x=1-5   → trivial (a=1: identity; a=2,3: small table)
  Tier 2: a=2-5,  x=1-10  → core multiplication table
  Tier 3: a=5-9,  x=1-15  → mid-range
  Tier 4: a=10-15, x=1-20 → extended (larger coefficients)
  Tier 5: negative x:  ax = b where x<0, b<0  (intro negative solutions)

6 variants:
  isolate, balance, inverse, verify (retained)
  + word_frame, number_sense (new)

Also includes a=1 (identity: 1x=b → x=b) as scaffold for tier 1.
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def tier(a, x):
    if a <= 3  and 1 <= x <= 5:  return 1
    if a <= 5  and 1 <= x <= 10: return 2
    if a <= 9  and 1 <= x <= 15: return 3
    if a <= 15 and 1 <= x <= 20: return 4
    return 5  # negative x


def gen_pairs():
    seen = {}
    # Positive x tiers 1-4
    for a in range(1, 16):
        for x in range(1, 21):
            b = a * x
            if b <= 300 and (a, b) not in seen:
                seen[(a, b)] = (x, tier(a, x))
    # Negative x tier 5
    for a in range(2, 10):
        for x in range(-10, 0):
            b = a * x
            if (a, b) not in seen:
                seen[(a, b)] = (x, 5)
    return [(a, b, x, t) for (a, b), (x, t) in seen.items()]


_SIBS = None
def _sib(a, b, rng):
    global _SIBS
    if _SIBS is None:
        _SIBS = [(a_, b_, x_) for a_, b_, x_, _ in gen_pairs()
                 if x_ > 0 and a_ >= 2 and x_ >= 2]
    return rng.choice([t for t in _SIBS if (t[0], t[1]) != (a, b)])


# ── 6 Variants ────────────────────────────────────────────────────────────────

def v_isolate(a, b, x, rng):
    ea, eb, ex = _sib(a, b, rng)
    return (f"Example: {ea}x={eb} → ÷{ea} both sides → x={eb}/{ea}={ex}\n"
            f"{a}x = {b}\n"
            f"divide both sides by {a}:\n"
            f"x = {b} / {a} = {x}\n#### {x}")


def v_balance(a, b, x, rng):
    ea, eb, ex = _sib(a, b, rng)
    return (f"Example: {ea}x={eb} → {ea} equal groups of {ex} = {eb}\n"
            f"{a}x = {b}: {a} equal groups sum to {b}\n"
            f"each group = {b}÷{a} = {x}\n"
            f"x = {x}\n#### {x}")


def v_inverse(a, b, x, rng):
    ea, eb, ex = _sib(a, b, rng)
    return (f"Example: {ea}x={eb} → undo ×{ea} with ÷{ea} → x={ex}\n"
            f"{a}x = {b}\n"
            f"undo ×{a}: divide by {a}\n"
            f"x = {b}÷{a} = {x}\n#### {x}")


def v_verify(a, b, x, rng):
    ea, eb, ex = _sib(a, b, rng)
    return (f"Example: {ea}x={eb} → x={ex}; check {ea}×{ex}={ea*ex}={eb}✓\n"
            f"{a}x = {b}\n"
            f"x = {b}÷{a} = {x}\n"
            f"check: {a}×{x} = {a*x} = {b}✓\n#### {x}")


def v_word_frame(a, b, x, rng):
    """Reframe as a real-world sharing sentence."""
    ea, eb, ex = _sib(a, b, rng)
    return (f"Example: {ea} items in {ex} equal packs of {ea//ex if ex!=0 else '?'} → {ea}÷{ea//ex if ex!=0 else '?'}={ex}\n"
            f"{a}x = {b}: {b} split into {a} equal parts\n"
            f"each part = {b}÷{a} = {x}\n"
            f"x = {x}\n#### {x}")


def v_number_sense(a, b, x, rng):
    """Anchor to nearest known fact."""
    ea, eb, ex = _sib(a, b, rng)
    # Find nearest anchor
    anchor_x = round(x / 2) if x > 2 else 1
    anchor_b = a * anchor_x
    return (f"Example: {ea}x={eb}: near {ea}×{ex-1}={ea*(ex-1)}, step up → {ex}\n"
            f"{a}x = {b}\n"
            f"anchor: {a}×{anchor_x}={anchor_b}\n"
            f"scale: {a}×{x}={b}\n"
            f"x = {x}\n#### {x}")


VARIANTS = {
    'isolate':      v_isolate,
    'balance':      v_balance,
    'inverse':      v_inverse,
    'verify':       v_verify,
    'word_frame':   v_word_frame,
    'number_sense': v_number_sense,
}


def build(holdout_frac=0.12, seed=42):
    all_pairs = gen_pairs()
    rng = random.Random(seed)

    # Curriculum: sort by tier, shuffle within tier
    tiers = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for item in all_pairs:
        tiers[item[3]].append(item)
    for t in tiers.values():
        rng.shuffle(t)
    ordered = tiers[1]+tiers[2]+tiers[3]+tiers[4]+tiers[5]

    n_val = max(1, int(len(ordered) * holdout_frac))
    val_pairs, train_pairs = ordered[-n_val:], ordered[:-n_val]

    train_recs, val_recs = [], []
    for a, b, x, t in train_pairs:
        prob = f"{a}x = {b}"
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, b, x, rng)
                assert '####' in cot and cot.count('####')==1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception:
                continue
            train_recs.append({
                'problem': prob, 'solution': str(x),
                'answer': str(x), 'answer_real': x,
                'concept': 'alg_1step', 'stage': 3, 'level': t,
                'solution_cot': cot, 'cot_source': f'v2_{vn}',
                'tier': t, 'hold_out': False,
            })

    for a, b, x, t in val_pairs:
        prob = f"{a}x = {b}"
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, b, x, rng)
            except Exception:
                continue
            val_recs.append({
                'problem': prob, 'solution': str(x),
                'answer': str(x), 'answer_real': x,
                'concept': 'alg_1step', 'stage': 3, 'level': t,
                'solution_cot': cot, 'cot_source': f'v2_{vn}',
                'tier': t, 'hold_out': True,
            })

    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"alg_1step_v2_train_{tag}.jsonl"
    vf = out / f"alg_1step_v2_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))

    tier_counts = {1:0,2:0,3:0,4:0,5:0}
    for r in train: tier_counts[r['tier']] += 1
    print(f"Train: {len(train):>4} records  ({len(set(r['problem'] for r in train))} unique)")
    for t, n in tier_counts.items():
        labels = {1:"a1-3 x1-5",2:"a2-5 x1-10",3:"a5-9 x1-15",4:"a10-15 x1-20",5:"negative x"}
        print(f"  Tier {t} ({labels[t]}): {n}")
    print(f"Val:   {len(val):>4} records → {vf.name}")


if __name__ == '__main__':
    main()
