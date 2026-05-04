"""
cot_alg_linear.py — Beginning algebra CoT generator (v2 format).

Two skill tiers:
  alg_1step : ax = b  →  x = b/a           (integer solutions, a∈2-9, b∈2-81)
  alg_2step : ax + b = c  →  x = (c-b)/a   (integer solutions)

Four reasoning variants per tier:
  isolate   : standard "divide both sides"
  balance   : balance-scale language
  inverse   : inverse operations explicitly stated
  verify    : solve then substitute back to verify

Format follows v2: Example → reasoning → #### answer
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ── 1-step: ax = b ────────────────────────────────────────────────────────────

def gen_1step(seed=42):
    rng = random.Random(seed)
    pairs = {}
    for a in range(2, 10):
        for x in range(1, 13):
            b = a * x
            if b <= 81:
                pairs[(a, b)] = x
    items = [(a, b, x) for (a, b), x in pairs.items()]
    rng.shuffle(items)
    return items


def _ex1(a, b, x):
    for ea, ex_, ex_val in [(3, 12, 4), (5, 35, 7), (4, 28, 7), (6, 42, 7), (2, 16, 8)]:
        eb = ea * ex_val
        if (ea, eb) != (a, b):
            return ea, eb, ex_val
    return 3, 12, 4


def v1_isolate(a, b, x):
    ea, eb, ex = _ex1(a, b, x)
    return (f"Example: {ea}x = {eb} → divide both sides by {ea} → x = {eb}/{ea} = {ex}\n"
            f"{a}x = {b}\n"
            f"divide both sides by {a}:\n"
            f"x = {b} / {a}\n"
            f"x = {x}\n#### {x}")


def v1_balance(a, b, x):
    ea, eb, ex = _ex1(a, b, x)
    return (f"Example: {ea}x = {eb} → {ea} groups of x = {eb} → x = {eb}÷{ea} = {ex}\n"
            f"{a}x = {b}: {a} equal groups sum to {b}\n"
            f"each group = {b} ÷ {a} = {x}\n"
            f"x = {x}\n#### {x}")


def v1_inverse(a, b, x):
    ea, eb, ex = _ex1(a, b, x)
    return (f"Example: {ea}x = {eb} → inverse of ×{ea} is ÷{ea} → x = {eb}÷{ea} = {ex}\n"
            f"{a}x = {b}\n"
            f"inverse of ×{a} is ÷{a}\n"
            f"x = {b} ÷ {a} = {x}\n#### {x}")


def v1_verify(a, b, x):
    ea, eb, ex = _ex1(a, b, x)
    return (f"Example: {ea}x = {eb} → x = {eb}÷{ea} = {ex}; check: {ea}×{ex} = {eb} ✓\n"
            f"{a}x = {b}\n"
            f"x = {b} ÷ {a} = {x}\n"
            f"check: {a} × {x} = {a*x} = {b} ✓\n#### {x}")


VARIANTS_1STEP = {
    'isolate': v1_isolate, 'balance': v1_balance,
    'inverse': v1_inverse, 'verify': v1_verify,
}


# ── 2-step: ax + b = c ────────────────────────────────────────────────────────

def gen_2step(seed=42):
    rng = random.Random(seed)
    items = []
    for a in range(2, 8):
        for x in range(1, 10):
            for b in range(-12, 13):
                if b == 0: continue
                c = a * x + b
                if -50 <= c <= 100 and c != b:
                    items.append((a, b, c, x))
    rng.shuffle(items)
    seen = set()
    unique = []
    for item in items:
        key = (item[0], item[1], item[2])
        if key not in seen:
            seen.add(key); unique.append(item)
    return unique[:600]


def _ex2(a, b, c, x):
    candidates = [(2, 3, 11, 4), (3, -2, 10, 4), (4, 5, 21, 4), (2, -1, 7, 4), (5, 2, 22, 4)]
    for ea, eb, ec, ex in candidates:
        if (ea, eb, ec) != (a, b, c):
            return ea, eb, ec, ex
    return 2, 3, 11, 4


def v2_isolate(a, b, c, x):
    ea, eb, ec, ex = _ex2(a, b, c, x)
    step1 = ec - eb
    bsign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    esign = f"+ {eb}" if eb >= 0 else f"- {abs(eb)}"
    return (f"Example: {ea}x {esign} = {ec} → subtract {eb}: {ea}x = {step1} → x = {step1}÷{ea} = {ex}\n"
            f"{a}x {bsign} = {c}\n"
            f"subtract {b} from both sides: {a}x = {c} - {b} = {c-b}\n"
            f"divide both sides by {a}: x = {c-b} / {a} = {x}\n#### {x}")


def v2_balance(a, b, c, x):
    ea, eb, ec, ex = _ex2(a, b, c, x)
    bsign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    esign = f"+ {eb}" if eb >= 0 else f"- {abs(eb)}"
    return (f"Example: {ea}x {esign} = {ec} → balance: remove {eb}, {ea} groups = {ec-eb}, x = {ex}\n"
            f"{a}x {bsign} = {c}\n"
            f"remove {b} from both sides → {a}x = {c-b}\n"
            f"{a} equal groups = {c-b} → x = {x}\n#### {x}")


def v2_inverse(a, b, c, x):
    ea, eb, ec, ex = _ex2(a, b, c, x)
    bsign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    esign = f"+ {eb}" if eb >= 0 else f"- {abs(eb)}"
    return (f"Example: {ea}x {esign} = {ec} → undo +{eb} (subtract {eb}), undo ×{ea} (÷{ea}) → x = {ex}\n"
            f"{a}x {bsign} = {c}\n"
            f"undo {'add' if b>=0 else 'subtract'} {abs(b)}: {a}x = {c-b}\n"
            f"undo multiply {a}: x = {c-b} ÷ {a} = {x}\n#### {x}")


def v2_verify(a, b, c, x):
    ea, eb, ec, ex = _ex2(a, b, c, x)
    bsign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    esign = f"+ {eb}" if eb >= 0 else f"- {abs(eb)}"
    return (f"Example: {ea}x {esign} = {ec} → x = {ex}; check: {ea}({ex}){esign} = {ec} ✓\n"
            f"{a}x {bsign} = {c}\n"
            f"step 1: {a}x = {c} - {b} = {c-b}\n"
            f"step 2: x = {c-b} / {a} = {x}\n"
            f"verify: {a}({x}) {bsign} = {a*x} {bsign} = {a*x+b} = {c} ✓\n#### {x}")


VARIANTS_2STEP = {
    'isolate': v2_isolate, 'balance': v2_balance,
    'inverse': v2_inverse, 'verify': v2_verify,
}


# ── Builder ───────────────────────────────────────────────────────────────────

def build(seed=42, holdout_frac=0.15):
    items_1 = gen_1step(seed)
    items_2 = gen_2step(seed)

    def split_build(items, variants, concept, level, prob_fn, ans_fn):
        n_hold = max(1, int(len(items) * holdout_frac))
        val_items, train_items = items[:n_hold], items[n_hold:]
        train_recs, val_recs = [], []
        for item in train_items:
            for vn, vf in variants.items():
                train_recs.append({
                    'problem': prob_fn(item), 'solution': str(ans_fn(item)),
                    'answer': str(ans_fn(item)), 'answer_real': ans_fn(item),
                    'concept': concept, 'stage': 3, 'level': level,
                    'solution_cot': vf(*item), 'cot_source': f'rule_{vn}', 'hold_out': False,
                })
        for item in val_items:
            for vn, vf in variants.items():
                val_recs.append({
                    'problem': prob_fn(item), 'solution': str(ans_fn(item)),
                    'answer': str(ans_fn(item)), 'answer_real': ans_fn(item),
                    'concept': concept, 'stage': 3, 'level': level,
                    'solution_cot': vf(*item), 'cot_source': f'rule_{vn}', 'hold_out': True,
                })
        return train_recs, val_recs

    def prob1(item): return f"{item[0]}x = {item[1]}"
    def prob2(item):
        a, b, c, x = item
        return f"{a}x {'+ ' if b>=0 else '- '}{abs(b)} = {c}"

    t1, v1 = split_build(items_1, VARIANTS_1STEP, 'alg_1step', 3, prob1, lambda i: i[2])
    t2, v2 = split_build(items_2, VARIANTS_2STEP, 'alg_2step', 4, prob2, lambda i: i[3])
    return t1, v1, t2, v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    t1, v1, t2, v2 = build(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"

    files = [
        (f"alg_1step_train_{tag}.jsonl", t1),
        (f"alg_1step_val_{tag}.jsonl",   v1),
        (f"alg_2step_train_{tag}.jsonl", t2),
        (f"alg_2step_val_{tag}.jsonl",   v2),
    ]
    for name, recs in files:
        (out / name).write_text("\n".join(json.dumps(r) for r in recs))
        print(f"  {name}: {len(recs)} records")

    print(f"\nSample alg_1step: {t1[0]['problem']}")
    print(t1[0]['solution_cot'])
    print(f"\nSample alg_2step: {t2[0]['problem']}")
    print(t2[0]['solution_cot'])


if __name__ == '__main__':
    main()
