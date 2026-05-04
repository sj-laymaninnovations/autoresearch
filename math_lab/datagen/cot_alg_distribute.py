"""
cot_alg_distribute.py — Distributive property & combining like terms generator.

Two skill families:
  distribute : a(x+b) = c  → ax+ab=c → x=(c-ab)/a
  combine    : ax + bx = c → (a+b)x = c → x = c/(a+b)

Curriculum tiers:
  Tier 1: distribute, small a (2-3), small b (1-4), x=1-6
  Tier 2: distribute, a=2-6, b=-5 to 8, x=1-10
  Tier 3: distribute, a=2-9, negative x solutions
  Tier 4: combine like terms, a+b ∈ 2-12, x=1-12
  Tier 5: mixed: distribute then combine

5 variants per family: expand, reverse, verify, annotated, word_frame
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sgn(v):
    return f"+ {v}" if v >= 0 else f"- {abs(v)}"


# ── Pair generators ───────────────────────────────────────────────────────────

def gen_distribute():
    """a(x+b) = c, integer x."""
    seen = set(); items = []
    for a in range(2, 10):
        for b in range(-6, 9):
            if b == 0: continue
            for x in range(-8, 16):
                c = a * (x + b)
                if abs(c) > 100: continue
                key = (a, b, c)
                if key not in seen:
                    seen.add(key)
                    t = 1 if a<=3 and b>0 and 1<=x<=6 else \
                        2 if a<=6 and 1<=x<=10 else \
                        3 if x<0 else 2
                    items.append((a, b, c, x, t))
    return items


def gen_combine():
    """ax + bx = c → (a+b)x = c."""
    seen = set(); items = []
    for a in range(1, 8):
        for b in range(1, 8):
            if a == b: continue
            ab = a + b
            for x in range(1, 13):
                c = ab * x
                if c > 120: continue
                key = (a, b, c)
                if key not in seen:
                    seen.add(key)
                    items.append((a, b, c, x, 4))
    return items


# ── Sibling pickers ───────────────────────────────────────────────────────────
_SIBD, _SIBC = None, None

def _sibd(a, b, c, rng):
    global _SIBD
    if _SIBD is None:
        _SIBD = [(a_,b_,c_,x_) for a_,b_,c_,x_,_ in gen_distribute() if x_>0 and a_>=2]
    return rng.choice([t for t in _SIBD if (t[0],t[1],t[2])!=(a,b,c)])


def _sibc(a, b, c, rng):
    global _SIBC
    if _SIBC is None:
        _SIBC = [(a_,b_,c_,x_) for a_,b_,c_,x_,_ in gen_combine() if x_>=2]
    return rng.choice([t for t in _SIBC if (t[0],t[1],t[2])!=(a,b,c)])


# ── Distribute variants ───────────────────────────────────────────────────────

def vd_expand(a, b, c, x, rng):
    ea,eb,ec,ex = _sibd(a,b,c,rng)
    return (f"Example: {ea}(x{_sgn(eb)})={ec} → {ea}x{_sgn(ea*eb)}={ec} → x={ex}\n"
            f"{a}(x{_sgn(b)}) = {c}\n"
            f"expand: {a}x {_sgn(a*b)} = {c}\n"
            f"subtract {a*b}: {a}x = {c-a*b}\n"
            f"x = {c-a*b}/{a} = {x}\n#### {x}")


def vd_reverse(a, b, c, x, rng):
    """Divide first, then subtract."""
    ea,eb,ec,ex = _sibd(a,b,c,rng)
    return (f"Example: {ea}(x{_sgn(eb)})={ec} → ÷{ea}: x{_sgn(eb)}={ec//ea} → x={ex}\n"
            f"{a}(x{_sgn(b)}) = {c}\n"
            f"divide both sides by {a}: x{_sgn(b)} = {c/a:.0f}\n"
            f"subtract {b}: x = {c//a-b} = {x}\n#### {x}")


def vd_verify(a, b, c, x, rng):
    ea,eb,ec,ex = _sibd(a,b,c,rng)
    return (f"Example: {ea}(x{_sgn(eb)})={ec} → x={ex}; check {ea}({ex}{_sgn(eb)})={ec}✓\n"
            f"{a}(x{_sgn(b)}) = {c}\n"
            f"expand: {a}x {_sgn(a*b)} = {c}\n"
            f"x = ({c}{_sgn(-a*b)}) / {a} = {c-a*b}/{a} = {x}\n"
            f"verify: {a}({x}{_sgn(b)}) = {a}×{x+b} = {a*(x+b)} = {c}✓\n#### {x}")


def vd_annotated(a, b, c, x, rng):
    ea,eb,ec,ex = _sibd(a,b,c,rng)
    return (f"Example: {ea}(x{_sgn(eb)})={ec} → [expand]{ea}x{_sgn(ea*eb)}={ec} → [{ex}]\n"
            f"{a}(x{_sgn(b)}) = {c}      [original]\n"
            f"{a}x {_sgn(a*b)} = {c}     [distribute {a}]\n"
            f"{a}x = {c-a*b}             [subtract {a*b}]\n"
            f"x = {x}                    [divide by {a}]\n#### {x}")


def vd_word_frame(a, b, c, x, rng):
    ea,eb,ec,ex = _sibd(a,b,c,rng)
    return (f"Example: {ea} groups of (x{_sgn(eb)}) = {ec}; each group = {ec//ea}; x={ex}\n"
            f"{a} groups, each of size (x{_sgn(b)}), total = {c}\n"
            f"each group size = {c}/{a} = {c//a}\n"
            f"x{_sgn(b)} = {c//a}\n"
            f"x = {c//a} - {b} = {x}\n#### {x}")


# ── Combine like terms variants ───────────────────────────────────────────────

def vc_collect(a, b, c, x, rng):
    ea,eb,ec,ex = _sibc(a,b,c,rng)
    return (f"Example: {ea}x+{eb}x={ec} → ({ea}+{eb})x={ec} → {ea+eb}x={ec} → x={ex}\n"
            f"{a}x + {b}x = {c}\n"
            f"collect: ({a}+{b})x = {c}\n"
            f"{a+b}x = {c}\n"
            f"x = {c}/{a+b} = {x}\n#### {x}")


def vc_verify(a, b, c, x, rng):
    ea,eb,ec,ex = _sibc(a,b,c,rng)
    return (f"Example: {ea}x+{eb}x={ec} → {ea+eb}x={ec} → x={ex}; check {ea*ex}+{eb*ex}={ec}✓\n"
            f"{a}x + {b}x = {c}\n"
            f"{a+b}x = {c}\n"
            f"x = {c}/{a+b} = {x}\n"
            f"verify: {a}×{x}+{b}×{x} = {a*x}+{b*x} = {c}✓\n#### {x}")


def vc_annotated(a, b, c, x, rng):
    ea,eb,ec,ex = _sibc(a,b,c,rng)
    return (f"Example: {ea}x+{eb}x={ec} → [{ea+eb}x={ec}] → x={ex}\n"
            f"{a}x + {b}x = {c}   [original]\n"
            f"({a}+{b})x = {c}    [combine coefficients]\n"
            f"{a+b}x = {c}        [simplify]\n"
            f"x = {x}             [divide by {a+b}]\n#### {x}")


def vc_word(a, b, c, x, rng):
    ea,eb,ec,ex = _sibc(a,b,c,rng)
    return (f"Example: {ea}x+{eb}x={ec}: {ea+eb} lots of x = {ec}, x={ex}\n"
            f"{a}x + {b}x = {c}\n"
            f"{a} lots plus {b} lots = {a+b} lots total\n"
            f"{a+b} × x = {c}\n"
            f"x = {c} ÷ {a+b} = {x}\n#### {x}")


def vc_factor(a, b, c, x, rng):
    ea,eb,ec,ex = _sibc(a,b,c,rng)
    return (f"Example: {ea}x+{eb}x={ec}: factor x({ea}+{eb})={ec} → x={ex}\n"
            f"{a}x + {b}x = {c}\n"
            f"factor: x({a}+{b}) = {c}\n"
            f"x × {a+b} = {c}\n"
            f"x = {c}/{a+b} = {x}\n#### {x}")


VARIANTS_DIST = {'expand':vd_expand,'reverse':vd_reverse,'verify':vd_verify,
                 'annotated':vd_annotated,'word_frame':vd_word_frame}
VARIANTS_COMB = {'collect':vc_collect,'verify':vc_verify,'annotated':vc_annotated,
                 'word':vc_word,'factor':vc_factor}


def build(holdout_frac=0.12, seed=42):
    rng = random.Random(seed)
    dist_pairs = gen_distribute()
    comb_pairs = gen_combine()

    def sort_split(items):
        tiers = {}
        for it in items: tiers.setdefault(it[-1],[]).append(it)
        for v in tiers.values(): rng.shuffle(v)
        ordered = sum([tiers[k] for k in sorted(tiers)], [])
        n_val = max(1, int(len(ordered)*holdout_frac))
        return ordered[:-n_val], ordered[-n_val:]

    train_d, val_d = sort_split(dist_pairs)
    train_c, val_c = sort_split(comb_pairs)

    def records(items, variants, concept, hold):
        recs = []
        for item in items:
            if len(item) == 5:
                a,b,c,x,t = item
                prob = f"{a}(x {_sgn(b)}) = {c}"
            else:
                a,b,c,x,t = item
                prob = f"{a}x + {b}x = {c}"
            for vn, vf in variants.items():
                try:
                    if len(item)==5 and concept=='distribute':
                        cot = vf(a,b,c,x,rng)
                    else:
                        cot = vf(a,b,c,x,rng)
                    assert '####' in cot and cot.count('####')==1
                    assert cot.startswith('Example:')
                    assert cot.strip().split('\n')[-1].startswith('####')
                except Exception: continue
                recs.append({
                    'problem': prob, 'solution': str(x), 'answer': str(x),
                    'answer_real': x, 'concept': concept, 'stage': 3, 'level': t,
                    'solution_cot': cot, 'cot_source': f'dist_{vn}', 'tier': t,
                    'hold_out': hold,
                })
        return recs

    train_recs = records(train_d, VARIANTS_DIST, 'distribute', False) + \
                 records(train_c, VARIANTS_COMB, 'combine', False)
    val_recs   = records(val_d,   VARIANTS_DIST, 'distribute', True)  + \
                 records(val_c,   VARIANTS_COMB, 'combine', True)
    rng.shuffle(train_recs)
    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"alg_distribute_train_{tag}.jsonl"
    vf = out / f"alg_distribute_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))
    print(f"Train: {len(train):>4} records → {tf.name}")
    print(f"Val:   {len(val):>4} records → {vf.name}")
    # samples
    rng = random.Random(42)
    print("\nSample distribute:")
    print(vd_verify(3, 2, 18, 4, rng))
    print("\nSample combine:")
    print(vc_verify(3, 2, 20, 4, rng))


if __name__ == '__main__':
    main()
