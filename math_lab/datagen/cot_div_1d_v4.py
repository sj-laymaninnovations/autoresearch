"""
cot_div_1d_v4.py — Curriculum-sequenced division generator (v4).

Fixes the 60% EM plateau by:
  1. Curriculum ordering: records sorted easy→hard within JSONL
     Tier 1: d=2-3, q=1-5   (small table facts)
     Tier 2: d=4-6, q=1-9   (mid table)
     Tier 3: d=7-9, q=1-12  (hard table)
     Tier 4: d=2-12, q=10-15 (extended quotients)
  2. Extended universe: d ∈ 2-12, q ∈ 0-15
  3. 10 variants (was 8): adds lookup_table + scaled_known
  4. Holdout: 12% of pairs (more train coverage)
  5. v2 format: Example first, #### last line only
"""
import argparse, json, datetime, random
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def tier(d, q):
    if d <= 3 and q <= 5:  return 1
    if d <= 6 and q <= 9:  return 2
    if d <= 9 and q <= 12: return 3
    return 4


def gen_pairs(max_d=12, max_q=15):
    seen = {}
    for d in range(2, max_d + 1):
        for q in range(0, max_q + 1):
            a = d * q
            if a <= 180:
                seen[(a, d)] = (q, tier(d, q))
    return [(a, d, q, t) for (a, d), (q, t) in seen.items()]


# Pool of sibling examples (never same pair as main)
_SIBLINGS = None
def _sib(a, d, rng):
    global _SIBLINGS
    if _SIBLINGS is None:
        _SIBLINGS = [(a_, d_, q_) for a_, d_, q_, _ in gen_pairs() if q_ >= 2 and a_ >= 6]
    return rng.choice([t for t in _SIBLINGS if (t[0], t[1]) != (a, d)])


# ── 10 Variants ───────────────────────────────────────────────────────────────

def v_factorization(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    return (f"Example: {ea} = {ed}×{eq}; {ea}/{ed} = {eq}\n"
            f"{a} = {d} × {q}\n{a}/{d} = {q}\n#### {q}")


def v_buildup(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    ex = "; ".join(f"{ed}×{k}={ed*k}" for k in range(1, min(eq+1, 5)))
    if eq > 4: ex += f"...{ed}×{eq}={ea}"
    if q == 0:
        return (f"Example: {ex} → {eq}\n{d}×0=0; {a}/{d}=0\n#### 0")
    mults = "\n".join(f"{d}×{k}={d*k}" for k in range(1, min(q+1, 7)))
    if q > 6: mults += f"\n...{d}×{q}={a}"
    return (f"Example: {ex} → {eq}\n{mults}\n{a}/{d}={q}\n#### {q}")


def v_countdown(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    ex_s = "; ".join(f"{ea-ed*k+ed}-{ed}={ea-ed*k}" for k in range(1, min(eq+1, 3)))
    ex = f"{ea}/{ed}: {ex_s}{'...' if eq>2 else ''} ({eq} steps)"
    if q == 0:
        return (f"Example: {ex}\n{a}/{d}: 0 steps needed\n#### 0")
    steps = []
    cur = a
    for k in range(1, min(q+1, 7)):
        nxt = cur - d; steps.append(f"{cur}-{d}={nxt}"); cur = nxt
    if q > 6: steps.append(f"...({q} steps)")
    return (f"Example: {ex}\n" + "\n".join(steps) +
            f"\nstep count={q}\n{a}/{d}={q}\n#### {q}")


def v_inverse(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    return (f"Example: {ea}/{ed}: {eq}×{ed}={ea} → {eq}\n"
            f"{a}/{d}: {q}×{d}={a}\n{a}/{d}={q}\n#### {q}")


def v_guess_check(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    ex = f"{ea}/{ed}: try {eq-1}→{ed*(eq-1)}≠{ea}; {eq}→{ea}✓"
    if q == 0:
        return (f"Example: {ex}\n{a}/{d}: try 1→{d}≠{a}; 0→{a}✓\n{a}/{d}=0\n#### 0")
    wrong = q-1; wp = d*wrong
    return (f"Example: {ex}\n"
            f"{a}/{d}: try {wrong}→{d}×{wrong}={wp}"
            + (f"={a}✓" if wp==a else f"≠{a}; try {q}→{d}×{q}={a}✓") +
            f"\n{a}/{d}={q}\n#### {q}")


def v_sharing(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    return (f"Example: share {ea} among {ed} → {eq} each ({eq}×{ed}={ea})\n"
            f"share {a} equally among {d} groups\n"
            f"each group gets {q}\ncheck: {q}×{d}={q*d}={a}✓\n{a}/{d}={q}\n#### {q}")


def v_times_blank(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    return (f"Example: {ed}×__={ea}; __={eq}; {ea}/{ed}={eq}\n"
            f"{d}×__={a}\n__={q}\nbecause {d}×{q}={a}\n{a}/{d}={q}\n#### {q}")


def v_number_line(a, d, q, rng):
    ea, ed, eq = _sib(a, d, rng)
    ex_j = "→".join(str(ed*k) for k in range(0, min(eq+1, 5)))
    if eq > 4: ex_j += f"→...→{ea}"
    jumps = "→".join(str(d*k) for k in range(0, min(q+1, 7)))
    if q > 6: jumps += f"→...→{a}"
    return (f"Example: {ea}/{ed}: {ex_j} ({eq} jumps)\n"
            f"{a}/{d}: jumps of {d}\n{jumps}\n{q} jumps → {a}/{d}={q}\n#### {q}")


def v_lookup_table(a, d, q, rng):
    """Scan the × table for d until we hit a."""
    ea, ed, eq = _sib(a, d, rng)
    ex_row = ", ".join(f"{ed}×{k}={ed*k}" for k in range(max(1,eq-2), eq+1))
    row = ", ".join(f"{d}×{k}={d*k}" for k in range(max(1,q-2), q+1))
    if q == 0:
        return (f"Example: {ed} table: ... → {ea}; {ea}/{ed}={eq}\n"
                f"{d} table: {d}×0=0={a}\n{a}/{d}=0\n#### 0")
    return (f"Example: {ed} table: {ex_row} → {ea}/{ed}={eq}\n"
            f"{d} table: {row} → {d}×{q}={a}\n{a}/{d}={q}\n#### {q}")


def v_scaled_known(a, d, q, rng):
    """Anchor to a known simpler fact then scale."""
    ea, ed, eq = _sib(a, d, rng)
    if q <= 1 or q % 2 != 0:
        # Fallback to factorization for q≤1 or odd q
        return v_factorization(a, d, q, rng)
    half_q = q // 2; half_a = d * half_q
    return (f"Example: {ea}/{ed}={eq}; know {ed}×{eq//2}={ea//2}, double → {eq}\n"
            f"know {d}×{half_q}={half_a}\n"
            f"double: {d}×{q}={a}\n{a}/{d}={q}\n#### {q}")


VARIANTS = {
    'factorization': v_factorization,
    'buildup':       v_buildup,
    'countdown':     v_countdown,
    'inverse':       v_inverse,
    'guess_check':   v_guess_check,
    'sharing':       v_sharing,
    'times_blank':   v_times_blank,
    'number_line':   v_number_line,
    'lookup_table':  v_lookup_table,
    'scaled_known':  v_scaled_known,
}


def build(max_d=12, max_q=15, holdout_frac=0.12, seed=42):
    all_pairs = gen_pairs(max_d, max_q)
    rng = random.Random(seed)

    # Sort by tier then shuffle within tier (curriculum order)
    tiers = {1: [], 2: [], 3: [], 4: []}
    for item in all_pairs:
        tiers[item[3]].append(item)
    for t in tiers.values():
        rng.shuffle(t)

    ordered = tiers[1] + tiers[2] + tiers[3] + tiers[4]
    n_val = max(1, int(len(ordered) * holdout_frac))
    # Val from last (hardest) items
    val_pairs   = ordered[-n_val:]
    train_pairs = ordered[:-n_val]

    train_recs, val_recs = [], []
    for a, d, q, t in train_pairs:
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, d, q, rng)
                assert '####' in cot and cot.count('####') == 1
                assert cot.startswith('Example:')
                assert cot.strip().split('\n')[-1].startswith('####')
            except Exception as e:
                continue
            train_recs.append({
                'problem': f'{a} / {d}', 'solution': str(q),
                'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': t,
                'solution_cot': cot, 'cot_source': f'v4_{vn}',
                'tier': t, 'hold_out': False,
            })

    for a, d, q, t in val_pairs:
        for vn, vf in VARIANTS.items():
            try:
                cot = vf(a, d, q, rng)
            except Exception:
                continue
            val_recs.append({
                'problem': f'{a} / {d}', 'solution': str(q),
                'answer': str(q), 'answer_real': q,
                'concept': 'div_1d', 'stage': 1, 'level': t,
                'solution_cot': cot, 'cot_source': f'v4_{vn}',
                'tier': t, 'hold_out': True,
            })

    return train_recs, val_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-d',   type=int, default=12)
    ap.add_argument('--max-q',   type=int, default=15)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out-dir', default='math_lab/results/skills')
    args = ap.parse_args()

    train, val = build(args.max_d, args.max_q, seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_{DATESTAMP}"
    tf = out / f"div_1d_v4_train_{tag}.jsonl"
    vf = out / f"div_1d_v4_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train))
    vf.write_text("\n".join(json.dumps(r) for r in val))

    tier_counts = {1:0,2:0,3:0,4:0}
    for r in train: tier_counts[r['tier']] += 1
    unique = len(set(r['problem'] for r in train))
    print(f"Train: {len(train):>4} records  ({unique} unique pairs × 10 variants)")
    for t, n in tier_counts.items():
        labels = {1:"d2-3 q1-5", 2:"d4-6 q1-9", 3:"d7-9 q1-12", 4:"extended q10-15"}
        print(f"  Tier {t} ({labels[t]}): {n} records")
    print(f"Val:   {len(val):>4} records → {vf.name}")


if __name__ == '__main__':
    main()
