"""
cot_div_1d_v3.py — Expanded division CoT generator (v3, answer-at-end format).

Addresses the critical data starvation in div_1d (160 train records → ~700+):

Expansions vs v2:
  1. Quotient range: q ∈ 0-12  (was 0-9)  → 104 unique pairs (was 80)
  2. Variants: 8 total (was 4):
       factorization, buildup, countdown, inverse (retained)
       + guess_check, sharing, times_blank, number_line (new)
  3. Dynamic sibling example: chosen per-call from pool, never same pair as main
  4. Holdout: 15% (was 50%) — small universe needs more train coverage
  5. All CoTs: Example first, #### last (v2 format)

Final size: ~700 train, ~110 val
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def gen_pairs(max_q=12):
    """All (a, d, q) where a = d*q, d ∈ 2-9, q ∈ 0-max_q."""
    pairs = {}
    for d in range(2, 10):
        for q in range(0, max_q + 1):
            a = d * q
            pairs[(a, d)] = q
    return [(a, d, q) for (a, d), q in pairs.items()]


# ── Sibling example picker ────────────────────────────────────────────────────

_POOL = None

def _pick_ex(a, d, rng):
    """Return (ea, ed, eq) — a sibling pair different from (a, d)."""
    global _POOL
    if _POOL is None:
        _POOL = [(a_, d_, q_) for a_, d_, q_ in gen_pairs()
                 if q_ >= 2 and a_ >= 6]
    candidates = [t for t in _POOL if (t[0], t[1]) != (a, d)]
    return rng.choice(candidates)


# ── Existing variants (updated to dynamic examples + v2 format) ───────────────

def variant_factorization(a, d, q, rng):
    ea, ed, eq = _pick_ex(a, d, rng)
    ex = f"{ea} = {ed} × {eq}; {ea}/{ed} = {eq}"
    return (f"Example: {ex}\n"
            f"{a} = {d} × {q}\n"
            f"{a} / {d} = {q}\n"
            f"#### {q}")


def variant_buildup(a, d, q, rng):
    ea, ed, eq = _pick_ex(a, d, rng)
    ex_steps = "; ".join(f"{ed}×{k}={ed*k}" for k in range(1, min(eq+1, 5)))
    if eq > 4:
        ex_steps += f"...{ed}×{eq}={ea}"
    ex = f"{ea}/{ed}: {ex_steps} → {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{d} × 0 = 0\n"
                f"{a} / {d} = 0\n#### 0")
    multiples = "\n".join(f"{d} × {k} = {d*k}" for k in range(1, min(q+1, 6)))
    if q > 5:
        multiples += f"\n... {d} × {q} = {a}"
    return (f"Example: {ex}\n"
            f"{multiples}\n"
            f"{a} / {d} = {q}\n"
            f"#### {q}")


def variant_countdown(a, d, q, rng):
    ea, ed, eq = _pick_ex(a, d, rng)
    ex_steps = "; ".join(f"{ea-ed*k+ed}-{ed}={ea-ed*k}"
                         for k in range(1, min(eq+1, 3)))
    if eq > 2:
        ex_steps += "..."
    ex = f"{ea}/{ed}: {ex_steps} ({eq} steps) = {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{a} = 0 (none fit)\n"
                f"{a} / {d} = 0\n#### 0")
    steps = []
    cur = a
    for k in range(1, min(q+1, 6)):
        nxt = cur - d
        steps.append(f"{cur} - {d} = {nxt}")
        cur = nxt
    if q > 5:
        steps.append(f"... ({q} steps total)")
    return (f"Example: {ex}\n"
            + "\n".join(steps) + "\n"
            f"step count = {q}\n"
            f"{a} / {d} = {q}\n"
            f"#### {q}")


def variant_inverse(a, d, q, rng):
    ea, ed, eq = _pick_ex(a, d, rng)
    ex = f"{ea}/{ed}: how many {ed}s in {ea}? {eq}×{ed}={ea} → {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{a}/{d}: how many {d}s in {a}?\n"
                f"0 × {d} = 0 = {a}\n"
                f"{a} / {d} = 0\n#### 0")
    return (f"Example: {ex}\n"
            f"{a}/{d}: how many {d}s in {a}?\n"
            f"{q} × {d} = {a}\n"
            f"{a} / {d} = {q}\n"
            f"#### {q}")


# ── New variants ──────────────────────────────────────────────────────────────

def variant_guess_check(a, d, q, rng):
    """Guess, check, adjust — models explicit self-correction."""
    ea, ed, eq = _pick_ex(a, d, rng)
    # build a wrong guess to show correction
    guess = max(0, q - 1) if rng.random() < 0.5 else min(q + 1, 12)
    g_prod = d * guess
    ex = (f"{ea}/{ed}: try {eq-1}→{ed*(eq-1)}≠{ea}; "
          f"try {eq}→{ea} ✓ → {eq}")
    if q == 0:
        return (f"Example: {ex}\n"
                f"{a}/{d}: try 1 → {d}×1={d} ≠ {a} → go lower\n"
                f"0 × {d} = 0 = {a} ✓\n"
                f"{a}/{d} = 0\n#### 0")
    wrong_guess = q - 1 if q > 0 else 1
    wg_prod = d * wrong_guess
    return (f"Example: {ex}\n"
            f"{a}/{d}: try {wrong_guess} → {d}×{wrong_guess}={wg_prod}"
            f" {'✓' if wg_prod==a else '≠ '+str(a)+', adjust up'}\n"
            + (f"try {q} → {d}×{q}={a} ✓\n" if wg_prod != a else "")
            + f"{a}/{d} = {q}\n#### {q}")


def variant_sharing(a, d, q, rng):
    """Equal-sharing / grouping language."""
    ea, ed, eq = _pick_ex(a, d, rng)
    ex = f"Share {ea} equally among {ed} groups → {eq} each ({eq}×{ed}={ea})"
    if q == 0:
        return (f"Example: {ex}\n"
                f"Share {a} among {d} groups\n"
                f"0 items each (nothing to share)\n"
                f"{a} / {d} = 0\n#### 0")
    return (f"Example: {ex}\n"
            f"Share {a} equally among {d} groups\n"
            f"each group gets {q}\n"
            f"check: {q} × {d} = {q*d} = {a} ✓\n"
            f"{a} / {d} = {q}\n#### {q}")


def variant_times_blank(a, d, q, rng):
    """Fill-in-the-blank multiplication frame."""
    ea, ed, eq = _pick_ex(a, d, rng)
    ex = f"{ed} × __ = {ea}; __ = {eq}; so {ea}/{ed} = {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{d} × __ = {a}\n"
                f"__ = 0 (since {a} = 0)\n"
                f"{a} / {d} = 0\n#### 0")
    return (f"Example: {ex}\n"
            f"{d} × __ = {a}\n"
            f"__ = {q}\n"
            f"because {d} × {q} = {a}\n"
            f"{a} / {d} = {q}\n#### {q}")


def variant_number_line(a, d, q, rng):
    """Jump along number line in steps of d."""
    ea, ed, eq = _pick_ex(a, d, rng)
    ex_jumps = "→".join(str(ed*k) for k in range(0, min(eq+1, 5)))
    if eq > 4:
        ex_jumps += f"→...→{ea}"
    ex = f"{ea}/{ed}: jumps of {ed}: {ex_jumps} ({eq} jumps) = {eq}"
    if q == 0:
        return (f"Example: {ex}\n"
                f"{a}/{d}: no jumps of {d} needed to reach {a}\n"
                f"{a}/{d} = 0\n#### 0")
    jumps = "→".join(str(d*k) for k in range(0, min(q+1, 6)))
    if q > 5:
        jumps += f"→...→{a}"
    return (f"Example: {ex}\n"
            f"{a}/{d}: count jumps of {d} to reach {a}\n"
            f"{jumps}\n"
            f"{q} jumps → {a}/{d} = {q}\n"
            f"#### {q}")


VARIANTS = {
    'factorization': variant_factorization,
    'buildup':       variant_buildup,
    'countdown':     variant_countdown,
    'inverse':       variant_inverse,
    'guess_check':   variant_guess_check,
    'sharing':       variant_sharing,
    'times_blank':   variant_times_blank,
    'number_line':   variant_number_line,
}


# ── Builder ───────────────────────────────────────────────────────────────────

def build(max_q=12, holdout_frac=0.15, seed=42):
    pairs = gen_pairs(max_q)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * holdout_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    train_records = []
    for a, d, q in train_pairs:
        for vname, vfn in VARIANTS.items():
            try:
                cot = vfn(a, d, q, rng)
                assert '####' in cot and cot.count('####') == 1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception as e:
                print(f"  SKIP {a}/{d}={q} [{vname}]: {e}")
                continue
            train_records.append({
                'problem': f'{a} / {d}',
                'solution': str(q), 'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': 2,
                'solution_cot': cot, 'cot_source': f'rule_v3_{vname}',
                'hold_out': False,
            })

    val_records = []
    for a, d, q in val_pairs:
        for vname, vfn in VARIANTS.items():
            try:
                cot = vfn(a, d, q, rng)
            except Exception:
                continue
            val_records.append({
                'problem': f'{a} / {d}',
                'solution': str(q), 'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': 2,
                'solution_cot': cot, 'cot_source': f'rule_v3_{vname}',
                'hold_out': True,
            })

    rng.shuffle(train_records)
    return train_records, val_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-q',   type=int, default=12)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(args.max_q, seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"div_1d_v3_train_{tag}.jsonl"
    vf = out / f"div_1d_v3_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))

    unique_pairs = set((r['problem']) for r in train)
    print(f"Train: {len(train):>4} records  ({len(unique_pairs)} unique pairs × up to 8 variants) → {tf.name}")
    print(f"Val:   {len(val):>4} records  → {vf.name}")

    print("\nSample variants for '24 / 4':")
    rng = random.Random(args.seed)
    for vname, vfn in VARIANTS.items():
        print(f"\n  [{vname}]")
        for line in vfn(24, 4, 6, rng).split('\n'):
            print(f"    {line}")


if __name__ == '__main__':
    main()
