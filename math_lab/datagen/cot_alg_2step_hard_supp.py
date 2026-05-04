"""
cot_alg_2step_hard_supp.py — Targeted supplement for alg_2step_hard failures:

  1. Negative x solutions  (2x + 9 = 1 → x = -4 was failing)
  2. Both-sides equations  (3x + 4 = 2x + 9 → x = 5 was 0%)

Root cause of both-sides failure: the previous variant had arithmetic bugs
in the example text. This generator writes clean, verified variants with
explicit step labels.

Curriculum:
  Tier 1: both-sides, coefficient diff = 1 (x isolated trivially)
  Tier 2: both-sides, coefficient diff = 2-3
  Tier 3: both-sides with negative constants
  Tier 4: negative x solutions from ax+b=c (targeted remediation)

5 variants, all verified at generation time.
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sgn(v):
    if v >= 0: return f"+ {v}"
    return f"- {abs(v)}"

def _sgn_compact(v):
    if v >= 0: return f"+{v}"
    return f"-{abs(v)}"


# ── Both-sides pair generator ─────────────────────────────────────────────────
# Form: ax + b = cx + d  →  (a-c)x = d - b  →  x = (d-b)/(a-c)

def gen_both_sides():
    seen = set(); items = []
    for a in range(2, 10):
        for c in range(1, a):          # a > c so (a-c) > 0
            diff = a - c
            for x in range(-8, 13):
                for b in list(range(-10, 11)):
                    if b == 0: continue
                    d = diff * x + b   # ensures integer solution
                    if abs(d) > 40: continue
                    # Verify
                    lhs = a * x + b
                    rhs = c * x + d
                    if lhs != rhs: continue
                    key = (a, b, c, d)
                    if key in seen: continue
                    seen.add(key)
                    t = 1 if diff == 1 and 0 <= x <= 6 else \
                        2 if diff <= 3 and x >= 0 else \
                        3 if b < 0 or d < 0 else 2
                    items.append((a, b, c, d, x, t))
    return items


# ── Negative-x supplement from ax+b=c ────────────────────────────────────────

def gen_neg_x():
    seen = set(); items = []
    for a in range(2, 8):
        for x in range(-10, -1):
            for b in range(1, 20):   # b positive keeps equation interesting
                c = a * x + b
                if abs(c) > 30: continue
                key = (a, b, c)
                if key in seen: continue
                seen.add(key)
                items.append((a, b, c, x, 4))
    return items


# ── Sibling pickers ───────────────────────────────────────────────────────────
_SIBS_BS = None
_SIBS_NEG = None

def _sib_bs(a, b, c, d, rng):
    global _SIBS_BS
    if _SIBS_BS is None:
        _SIBS_BS = [(a_,b_,c_,d_,x_)
                    for a_,b_,c_,d_,x_,_ in gen_both_sides()
                    if x_ >= 1 and a_ >= 3]
    return rng.choice([t for t in _SIBS_BS if (t[0],t[1],t[2],t[3]) != (a,b,c,d)])

def _sib_neg(a, b, c, rng):
    global _SIBS_NEG
    if _SIBS_NEG is None:
        _SIBS_NEG = [(a_,b_,c_,x_) for a_,b_,c_,x_,_ in gen_neg_x()]
    return rng.choice([t for t in _SIBS_NEG if (t[0],t[1],t[2]) != (a,b,c)])


# ── Both-sides variants ───────────────────────────────────────────────────────

def bs_collect_verify(a, b, c, d, x, rng):
    """Collect x terms left, constants right, verify."""
    ea,eb,ec,ed,ex = _sib_bs(a,b,c,d,rng)
    ediff = ea-ec; erhs = ed-eb
    ex_str = (f"{ea}x {_sgn(eb)} = {ec}x {_sgn(ed)}\n"
              f"subtract {ec}x: {ediff}x {_sgn(eb)} = {ed}\n"
              f"subtract {eb}: {ediff}x = {erhs}\n"
              f"x = {erhs}/{ediff} = {ex}")
    diff = a-c; rhs_val = d-b
    lhs_check = a*x+b; rhs_check = c*x+d
    return (f"Example: {ex_str}\n"
            f"{a}x {_sgn(b)} = {c}x {_sgn(d)}\n"
            f"subtract {c}x: {diff}x {_sgn(b)} = {d}\n"
            f"subtract {b}: {diff}x = {rhs_val}\n"
            f"x = {rhs_val} / {diff} = {x}\n"
            f"verify: {a}({x}){_sgn(b)} = {lhs_check}; {c}({x}){_sgn(d)} = {rhs_check}; {lhs_check} = {rhs_check} ✓\n"
            f"#### {x}")


def bs_rearrange(a, b, c, d, x, rng):
    """Move all x to left, all constants to right."""
    ea,eb,ec,ed,ex = _sib_bs(a,b,c,d,rng)
    ediff = ea-ec; erhs = ed-eb
    diff = a-c; rhs_val = d-b
    return (f"Example: {ea}x{_sgn_compact(eb)}={ec}x{_sgn_compact(ed)} → {ediff}x={erhs} → x={ex}\n"
            f"{a}x {_sgn(b)} = {c}x {_sgn(d)}\n"
            f"move x terms left: ({a} - {c})x = {d} - {b}\n"
            f"{diff}x = {rhs_val}\n"
            f"x = {x}\n"
            f"#### {x}")


def bs_annotated(a, b, c, d, x, rng):
    """Each step explicitly labeled."""
    ea,eb,ec,ed,ex = _sib_bs(a,b,c,d,rng)
    ediff = ea-ec; erhs = ed-eb
    diff = a-c; rhs_val = d-b
    return (f"Example: {ea}x{_sgn_compact(eb)}={ec}x{_sgn_compact(ed)}: collect→{ediff}x={erhs}→x={ex}\n"
            f"{a}x {_sgn(b)} = {c}x {_sgn(d)}   [original]\n"
            f"{diff}x {_sgn(b)} = {d}             [subtract {c}x]\n"
            f"{diff}x = {rhs_val}                 [subtract {b}]\n"
            f"x = {x}                             [divide by {diff}]\n"
            f"#### {x}")


def bs_substitution(a, b, c, d, x, rng):
    """Try x, verify both sides equal."""
    ea,eb,ec,ed,ex = _sib_bs(a,b,c,d,rng)
    ediff = ea-ec; erhs = ed-eb
    diff = a-c; rhs_val = d-b
    lhs = a*x+b; rhs = c*x+d
    return (f"Example: {ea}x{_sgn_compact(eb)}={ec}x{_sgn_compact(ed)} → collect {ediff}x={erhs} → x={ex}\n"
            f"{a}x {_sgn(b)} = {c}x {_sgn(d)}\n"
            f"collect: {diff}x = {d} - {b} = {rhs_val}\n"
            f"x = {rhs_val} / {diff} = {x}\n"
            f"check LHS: {a}×{x} {_sgn(b)} = {a*x} {_sgn(b)} = {lhs}\n"
            f"check RHS: {c}×{x} {_sgn(d)} = {c*x} {_sgn(d)} = {rhs}\n"
            f"LHS = RHS = {lhs} ✓\n"
            f"#### {x}")


def bs_inverse_ops(a, b, c, d, x, rng):
    """Undo operations one at a time."""
    ea,eb,ec,ed,ex = _sib_bs(a,b,c,d,rng)
    ediff = ea-ec; erhs = ed-eb
    diff = a-c; rhs_val = d-b
    return (f"Example: {ea}x{_sgn_compact(eb)}={ec}x{_sgn_compact(ed)}: undo {ec}x, undo {eb} → x={ex}\n"
            f"{a}x {_sgn(b)} = {c}x {_sgn(d)}\n"
            f"undo +{c}x (subtract {c}x both sides): {diff}x {_sgn(b)} = {d}\n"
            f"undo {_sgn(b)} (subtract {b} both sides): {diff}x = {rhs_val}\n"
            f"undo ×{diff} (divide by {diff}): x = {x}\n"
            f"#### {x}")


# ── Negative-x variants ───────────────────────────────────────────────────────

def neg_isolate(a, b, c, x, rng):
    ea,eb,ec,ex = _sib_neg(a,b,c,rng)
    return (f"Example: {ea}x {_sgn(eb)} = {ec} → subtract {eb}: {ea}x = {ec-eb} → x = {(ec-eb)//ea} = {ex}\n"
            f"{a}x {_sgn(b)} = {c}\n"
            f"subtract {b}: {a}x = {c - b}\n"
            f"divide by {a}: x = {c - b} / {a} = {x}\n"
            f"verify: {a}({x}) {_sgn(b)} = {a*x} {_sgn(b)} = {a*x+b} = {c} ✓\n"
            f"#### {x}")


def neg_sign_aware(a, b, c, x, rng):
    """Explicitly shows the sign of the result."""
    ea,eb,ec,ex = _sib_neg(a,b,c,rng)
    return (f"Example: {ea}x {_sgn(eb)} = {ec}: {ea}x = {ec-eb}, x = {ex} (negative)\n"
            f"{a}x {_sgn(b)} = {c}\n"
            f"isolate {a}x: {a}x = {c} - {b} = {c - b}\n"
            f"x = {c - b} ÷ {a} = {x}   ← negative result\n"
            f"check: {a}×({x}) {_sgn(b)} = {a*x} {_sgn(b)} = {a*x+b} = {c} ✓\n"
            f"#### {x}")


VARIANTS_BS = {
    'collect_verify': bs_collect_verify,
    'rearrange':      bs_rearrange,
    'annotated':      bs_annotated,
    'substitution':   bs_substitution,
    'inverse_ops':    bs_inverse_ops,
}

VARIANTS_NEG = {
    'isolate':    neg_isolate,
    'sign_aware': neg_sign_aware,
}


def build(holdout_frac=0.12, seed=42):
    rng = random.Random(seed)
    bs_pairs  = gen_both_sides()
    neg_pairs = gen_neg_x()

    def sort_split(items, key_fn):
        tiers = {}
        for it in items:
            t = key_fn(it); tiers.setdefault(t,[]).append(it)
        for v in tiers.values(): rng.shuffle(v)
        ordered = sum([tiers[k] for k in sorted(tiers)], [])
        n_val = max(1, int(len(ordered)*holdout_frac))
        return ordered[:-n_val], ordered[-n_val:]

    train_bs, val_bs   = sort_split(bs_pairs,  lambda it: it[-1])
    train_neg, val_neg = sort_split(neg_pairs, lambda it: it[-1])

    def make_recs(items, variants, form, hold):
        recs = []
        for item in items:
            if form == 'bs':
                a,b,c,d,x,t = item
                prob = f"{a}x {_sgn(b)} = {c}x {_sgn(d)}"
                for vn, vf in variants.items():
                    try:
                        cot = vf(a,b,c,d,x,rng)
                        assert cot.count('####') == 1
                        assert cot.startswith('Example:')
                        assert cot.strip().split('\n')[-1].startswith('####')
                        lhs = a*x+b; rhs = c*x+d
                        assert lhs == rhs, f"verify fail: {lhs}≠{rhs}"
                    except Exception as e:
                        continue
                    recs.append({'problem':prob,'solution':str(x),'answer':str(x),
                                 'answer_real':x,'concept':'alg_2step_hard','stage':3,
                                 'level':t,'solution_cot':cot,'cot_source':f'supp_bs_{vn}',
                                 'tier':t,'hold_out':hold})
            else:
                a,b,c,x,t = item
                prob = f"{a}x {_sgn(b)} = {c}"
                for vn, vf in variants.items():
                    try:
                        cot = vf(a,b,c,x,rng)
                        assert cot.count('####') == 1
                        assert cot.startswith('Example:')
                        assert cot.strip().split('\n')[-1].startswith('####')
                        assert a*x+b == c, f"neg verify fail"
                    except Exception as e:
                        continue
                    recs.append({'problem':prob,'solution':str(x),'answer':str(x),
                                 'answer_real':x,'concept':'alg_2step_hard','stage':3,
                                 'level':t,'solution_cot':cot,'cot_source':f'supp_neg_{vn}',
                                 'tier':t,'hold_out':hold})
        return recs

    train_recs = make_recs(train_bs,  VARIANTS_BS,  'bs',  False) + \
                 make_recs(train_neg, VARIANTS_NEG, 'neg', False)
    val_recs   = make_recs(val_bs,    VARIANTS_BS,  'bs',  True)  + \
                 make_recs(val_neg,   VARIANTS_NEG, 'neg', True)
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
    tf = out / f"alg_2step_hard_supp_train_{tag}.jsonl"
    vf = out / f"alg_2step_hard_supp_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))

    bs_count  = sum(1 for r in train if 'bs' in r['cot_source'])
    neg_count = sum(1 for r in train if 'neg' in r['cot_source'])
    print(f"Train: {len(train):>4} records")
    print(f"  both-sides:  {bs_count}")
    print(f"  negative-x:  {neg_count}")
    print(f"Val:   {len(val):>4} records → {vf.name}")

    rng = random.Random(42)
    print("\nSample both-sides (3x+4=2x+9, x=5):")
    print(bs_collect_verify(3, 4, 2, 9, 5, rng))
    print("\nSample negative-x (2x+9=1, x=-4):")
    print(neg_sign_aware(2, 9, 1, -4, rng))


if __name__ == '__main__':
    main()
