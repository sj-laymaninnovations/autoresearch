"""
cot_alg_2step_hard.py — Harder 2-step algebra: ax+b=c with extended range,
negatives, and variables on both sides (ax+b = cx+d).

Curriculum tiers:
  Tier 1: ax+b=c,  a=2-4,  x=1-8,  b small positive
  Tier 2: ax+b=c,  a=2-7,  x=1-12, b positive/negative
  Tier 3: ax+b=c,  a=2-9,  x negative solution
  Tier 4: ax+b=cx+d  (variables both sides, collect terms)

6 variants: isolate, balance, inverse, verify, annotated, both_sides
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Pair generators ───────────────────────────────────────────────────────────

def gen_2step_pairs():
    """ax + b = c, integer x solutions."""
    seen = set()
    items = []
    for a in range(2, 10):
        for x in range(-12, 20):
            for b in range(-15, 16):
                if b == 0: continue
                c = a * x + b
                if abs(c) > 120: continue
                key = (a, b, c)
                if key not in seen:
                    seen.add(key)
                    if a <= 4 and 1 <= x <= 8 and b > 0:
                        t = 1
                    elif a <= 7 and 1 <= x <= 12:
                        t = 2
                    elif x < 0:
                        t = 3
                    else:
                        t = 2
                    items.append((a, b, c, x, t))
    return items


def gen_both_sides_pairs():
    """ax+b = cx+d, collect → (a-c)x = d-b → x = (d-b)/(a-c)."""
    seen = set()
    items = []
    for a in range(3, 10):
        for c in range(1, a):  # a > c so a-c > 0
            diff_a = a - c
            for x in range(1, 12):
                for b in range(-8, 9):
                    if b == 0: continue
                    d = diff_a * x + b
                    if abs(d) > 50: continue
                    key = (a, b, c, d)
                    if key not in seen:
                        seen.add(key); items.append((a, b, c, d, x, 4))
    return items


# ── Sibling examples ──────────────────────────────────────────────────────────
_SIBS2 = None
def _sib2(a, b, c, rng):
    global _SIBS2
    if _SIBS2 is None:
        _SIBS2 = [(a_,b_,c_,x_) for a_,b_,c_,x_,_ in gen_2step_pairs()
                  if x_ > 0 and a_ >= 2]
    return rng.choice([t for t in _SIBS2 if (t[0],t[1],t[2]) != (a,b,c)])


def _sgn(v):
    return f"+ {v}" if v >= 0 else f"- {abs(v)}"


# ── 5 Variants for ax+b=c ─────────────────────────────────────────────────────

def v2_isolate(a, b, c, x, rng):
    ea,eb,ec,ex = _sib2(a,b,c,rng)
    return (f"Example: {ea}x{_sgn(eb)}={ec} → subtract {eb}: {ea}x={ec-eb} → x={ec-eb}/{ea}={ex}\n"
            f"{a}x{_sgn(b)} = {c}\n"
            f"subtract {b}: {a}x = {c-b}\n"
            f"divide by {a}: x = {c-b}/{a} = {x}\n#### {x}")


def v2_balance(a, b, c, x, rng):
    ea,eb,ec,ex = _sib2(a,b,c,rng)
    return (f"Example: {ea}x{_sgn(eb)}={ec} → remove {eb}: {ea}x={ec-eb}; {ea} groups={ex}\n"
            f"{a}x{_sgn(b)} = {c}\n"
            f"remove {b} from both sides: {a}x = {c-b}\n"
            f"{a} equal groups = {c-b}: x = {x}\n#### {x}")


def v2_inverse(a, b, c, x, rng):
    ea,eb,ec,ex = _sib2(a,b,c,rng)
    op = "add" if b < 0 else "subtract"
    return (f"Example: {ea}x{_sgn(eb)}={ec} → undo {'+' if eb>=0 else '-'}{abs(eb)}, ÷{ea} → x={ex}\n"
            f"{a}x{_sgn(b)} = {c}\n"
            f"undo {op} {abs(b)}: {a}x = {c-b}\n"
            f"undo ×{a}: x = {c-b}÷{a} = {x}\n#### {x}")


def v2_verify(a, b, c, x, rng):
    ea,eb,ec,ex = _sib2(a,b,c,rng)
    return (f"Example: {ea}x{_sgn(eb)}={ec} → x={ex}; check {ea}({ex}){_sgn(eb)}={ec}✓\n"
            f"{a}x{_sgn(b)} = {c}\n"
            f"step 1: {a}x = {c} - {b} = {c-b}\n"
            f"step 2: x = {c-b} / {a} = {x}\n"
            f"verify: {a}({x}){_sgn(b)} = {a*x}{_sgn(b)} = {a*x+b} = {c}✓\n#### {x}")


def v2_annotated(a, b, c, x, rng):
    """Show the 'why' at each line."""
    ea,eb,ec,ex = _sib2(a,b,c,rng)
    return (f"Example: {ea}x{_sgn(eb)}={ec} → [sub {eb}]{ea}x={ec-eb} → [÷{ea}]x={ex}\n"
            f"{a}x{_sgn(b)} = {c}    [original]\n"
            f"{a}x = {c-b}           [subtract {b} from both sides]\n"
            f"x = {x}                [divide both sides by {a}]\n#### {x}")


# ── Variant for ax+b=cx+d (both sides) ────────────────────────────────────────

def v_both_sides_isolate(a, b, c, d, x, rng):
    diff = a - c; rhs = d - b
    bsign_b = _sgn(b); bsign_d = _sgn(d)
    return (f"Example: {a+1}x{_sgn(b+1)}={c+1}x{_sgn(d+1)} → collect: {diff+1}x={rhs+1} → x={x}\n"
            f"{a}x{bsign_b} = {c}x{bsign_d}\n"
            f"subtract {c}x: {diff}x{bsign_b} = {d}\n"
            f"subtract {b}: {diff}x = {d-b}\n"
            f"divide by {diff}: x = {d-b}/{diff} = {x}\n#### {x}")


def v_both_sides_verify(a, b, c, d, x, rng):
    diff = a - c
    bsign_b = _sgn(b); bsign_d = _sgn(d)
    lhs = a*x+b; rhs_check = c*x+d
    return (f"Example: {a+1}x{_sgn(b+1)}={c+1}x{_sgn(d+1)} → x={x}; check LHS=RHS\n"
            f"{a}x{bsign_b} = {c}x{bsign_d}\n"
            f"collect x terms: {diff}x = {d} - {b} = {d-b}\n"
            f"x = {d-b}/{diff} = {x}\n"
            f"verify: {a}({x}){bsign_b}={lhs}; {c}({x}){bsign_d}={rhs_check}; {lhs}={rhs_check}✓\n#### {x}")


VARIANTS_2STEP = {
    'isolate':   v2_isolate,
    'balance':   v2_balance,
    'inverse':   v2_inverse,
    'verify':    v2_verify,
    'annotated': v2_annotated,
}

VARIANTS_BOTH = {
    'both_isolate': v_both_sides_isolate,
    'both_verify':  v_both_sides_verify,
}


def build(holdout_frac=0.12, seed=42):
    rng = random.Random(seed)
    pairs_2step = gen_2step_pairs()
    pairs_both  = gen_both_sides_pairs()

    def sort_and_split(items):
        tiers = {}
        for it in items:
            t = it[-1]
            tiers.setdefault(t, []).append(it)
        for v in tiers.values(): rng.shuffle(v)
        ordered = sum([tiers[k] for k in sorted(tiers.keys())], [])
        n_val = max(1, int(len(ordered) * holdout_frac))
        return ordered[:-n_val], ordered[-n_val:]

    train_2s, val_2s = sort_and_split(pairs_2step)
    train_bs, val_bs = sort_and_split(pairs_both)

    train_recs, val_recs = [], []

    def make_record(prob, x, t, cot, source, hold):
        return {'problem': prob, 'solution': str(x), 'answer': str(x),
                'answer_real': x, 'concept': 'alg_2step_hard',
                'stage': 3, 'level': t, 'solution_cot': cot,
                'cot_source': source, 'tier': t, 'hold_out': hold}

    for a, b, c, x, t in train_2s:
        prob = f"{a}x {_sgn(b)} = {c}"
        for vn, vf in VARIANTS_2STEP.items():
            try:
                cot = vf(a, b, c, x, rng)
                assert '####' in cot and cot.count('####')==1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception: continue
            train_recs.append(make_record(prob, x, t, cot, f'hard_{vn}', False))

    for a, b, c, d, x, t in train_bs[:300]:  # cap to avoid bloat
        prob = f"{a}x {_sgn(b)} = {c}x {_sgn(d)}"
        for vn, vf in VARIANTS_BOTH.items():
            try:
                cot = vf(a, b, c, d, x, rng)
                assert '####' in cot and cot.count('####')==1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception: continue
            train_recs.append(make_record(prob, x, t, cot, f'hard_{vn}', False))

    for a, b, c, x, t in val_2s:
        prob = f"{a}x {_sgn(b)} = {c}"
        for vn, vf in VARIANTS_2STEP.items():
            try: cot = vf(a, b, c, x, rng)
            except Exception: continue
            val_recs.append(make_record(prob, x, t, cot, f'hard_{vn}', True))

    rng.shuffle(train_recs)
    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"alg_2step_hard_train_{tag}.jsonl"
    vf = out / f"alg_2step_hard_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))
    print(f"Train: {len(train):>4} records → {tf.name}")
    print(f"Val:   {len(val):>4} records → {vf.name}")
    print("\nSample tier 4 (both sides):")
    rng = random.Random(42)
    pairs_b = gen_both_sides_pairs()
    if pairs_b:
        a,b,c,d,x,t = pairs_b[0]
        print(v_both_sides_verify(a,b,c,d,x,rng))


if __name__ == '__main__':
    main()
