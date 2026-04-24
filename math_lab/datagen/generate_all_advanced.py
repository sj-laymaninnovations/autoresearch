"""
generate_all_advanced.py — Batch Orchestrator for Advanced Math QA Generators

Calls all 9 domain generators + number_formats, distributing pairs across
difficulty levels using a weighted curriculum inspired by Polaris 4B:

  L1  elementary  (basic):       1x weight
  L2  intermediate (medium):     2x weight
  L3  advanced    (difficult):   3x weight
  L4  expert      (PhD-level):   4x weight
  L5  quant/research (frontier): 4x weight

So for --total 10000:
  L1 → ~714   pairs   (1/14 of total)
  L2 → ~1429  pairs   (2/14)
  L3 → ~2143  pairs   (3/14)
  L4 → ~2857  pairs   (4/14)
  L5 → ~2857  pairs   (4/14)

Each domain gets an equal share of each level's budget.

Usage:
  python math_lab/datagen/generate_all_advanced.py --total 10000 --seed 42

Output:
  math_lab/results/harder_qa_<domain>_<tag>_<timestamp>.jsonl  (one per domain)
  math_lab/results/generate_all_summary_<tag>_<timestamp>.txt  (run summary)
"""

import sys
import json
import datetime
import argparse
from pathlib import Path
from collections import Counter

# ── Generator imports ─────────────────────────────────────────────────────────
# Each module exports generate_*_pairs(n_pairs, levels, seed) -> list[dict]
# and save_pairs(pairs, run_tag) -> Path

DATAGEN_DIR = Path(__file__).parent
sys.path.insert(0, str(DATAGEN_DIR))

from algebra_qa          import generate_algebra_pairs,      save_pairs as save_algebra
from geometry_qa         import generate_geometry_pairs,     save_pairs as save_geometry
from trigonometry_qa     import generate_trig_pairs,         save_pairs as save_trig
from calculus_qa         import generate_calculus_pairs,     save_pairs as save_calculus
from linear_algebra_qa   import generate_linalg_pairs,       save_pairs as save_linalg
from statistics_qa       import generate_stats_pairs,        save_pairs as save_stats
from linear_regression_qa import generate_linreg_pairs,      save_pairs as save_linreg
from differential_eq_qa  import generate_diffeq_pairs,       save_pairs as save_diffeq
from stochastic_qa       import generate_stochastic_pairs,   save_pairs as save_stochastic
from number_formats      import generate_format_pairs

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Level weight scheme ───────────────────────────────────────────────────────
# 1:2:3:4:4 → trend toward challenging; tiny easy base, heavy expert/quant top

LEVEL_WEIGHTS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4}
LEVEL_LABELS  = {
    1: "elementary",
    2: "intermediate",
    3: "advanced",
    4: "expert/PhD",
    5: "quant/research",
}

DOMAINS = [
    ("algebra",       generate_algebra_pairs,       save_algebra),
    ("geometry",      generate_geometry_pairs,       save_geometry),
    ("trigonometry",  generate_trig_pairs,           save_trig),
    ("calculus",      generate_calculus_pairs,       save_calculus),
    ("linalg",        generate_linalg_pairs,         save_linalg),
    ("statistics",    generate_stats_pairs,          save_stats),
    ("linreg",        generate_linreg_pairs,         save_linreg),
    ("diffeq",        generate_diffeq_pairs,         save_diffeq),
    ("stochastic",    generate_stochastic_pairs,     save_stochastic),
]


def compute_level_counts(total: int, levels: list[int]) -> dict[int, int]:
    """
    Compute how many pairs to generate per level given weights.
    The total is distributed proportionally, then remainder added to L4.

    Returns {level: count}
    """
    weights = {l: LEVEL_WEIGHTS[l] for l in levels}
    total_weight = sum(weights.values())
    counts = {}
    allocated = 0
    for i, level in enumerate(levels):
        if i == len(levels) - 1:
            counts[level] = total - allocated
        else:
            c = int(total * weights[level] / total_weight)
            counts[level] = c
            allocated += c
    return counts


def run_domain(name: str, gen_fn, save_fn, level_counts: dict,
               seed: int, run_tag: str) -> tuple:
    """
    Generate pairs for one domain across all levels, save JSONL.
    Calls gen_fn once per level with the right count.

    Returns (domain_name, total_pairs, out_path, level_summary)
    """
    print(f"\n  [{name}] generating...", flush=True)
    all_pairs = []
    level_summary = {}

    for level, count in sorted(level_counts.items()):
        if count < 1:
            continue
        label = LEVEL_LABELS.get(level, f"L{level}")
        # Use a per-domain per-level seed so runs are reproducible
        level_seed = (seed * 1000 + level * 17 + hash(name)) % (2**31)
        try:
            pairs = gen_fn(
                n_pairs=count,
                levels=[level],
                seed=level_seed,
            )
        except Exception as e:
            print(f"    WARNING: {name} L{level} failed: {e}")
            pairs = []
        level_summary[level] = len(pairs)
        print(f"    L{level} ({label}): {len(pairs)}/{count} pairs", flush=True)
        all_pairs.extend(pairs)

    out_path = save_fn(all_pairs, run_tag)
    return name, len(all_pairs), out_path, level_summary


def run_numfmt(total_numfmt: int, seed: int, run_tag: str) -> tuple:
    """Generate number-format equivalence pairs separately."""
    print(f"\n  [number_formats] generating {total_numfmt} pairs...", flush=True)
    pairs = generate_format_pairs(n_pairs=total_numfmt, seed=seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    out = RESULTS_DIR / f"harder_qa_numfmt{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"    Saved {len(pairs)} number-format pairs -> {out}")
    return "number_formats", len(pairs), out, {}


def print_summary(results: list, level_counts: dict, total: int,
                  elapsed: float, out_summary: Path):
    """Print and save a run summary table."""
    lines = []
    lines.append("=" * 70)
    lines.append("  Advanced Math QA Generation Summary")
    lines.append(f"  Target total: {total:,}  |  Elapsed: {elapsed:.1f}s")
    lines.append("")
    lines.append("  Level weights (Polaris-inspired curriculum):")
    for lvl, w in LEVEL_WEIGHTS.items():
        n = level_counts.get(lvl, 0)
        label = LEVEL_LABELS.get(lvl, f"L{lvl}")
        lines.append(f"    L{lvl} {label:20s} weight={w}  target={n:>6,}")
    lines.append("")
    lines.append(f"  {'Domain':<20} {'Total':>7}  Per-level breakdown")
    lines.append("  " + "-" * 65)
    grand_total = 0
    for name, count, path, lvl_summary in results:
        lvl_str = "  ".join(f"L{l}:{n}" for l, n in sorted(lvl_summary.items()))
        lines.append(f"  {name:<20} {count:>7,}  {lvl_str}")
        grand_total += count
    lines.append("  " + "-" * 65)
    lines.append(f"  {'GRAND TOTAL':<20} {grand_total:>7,}")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  Summary saved -> {out_summary}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate all advanced math Q/A pairs with Polaris-style "
            "difficulty weighting (1:2:3:4:4 across L1-L5)."
        )
    )
    parser.add_argument(
        "--total", type=int, default=10000,
        help="Total pairs across all domains (default: 10000)"
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Difficulty levels to generate (default: all 1-5)"
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Subset of domains to run (default: all). "
             "Choices: algebra geometry trigonometry calculus linalg "
             "statistics linreg diffeq stochastic number_formats"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility"
    )
    parser.add_argument(
        "--run-tag", default="",
        help="Optional tag appended to output filenames"
    )
    parser.add_argument(
        "--numfmt-ratio", type=float, default=0.05,
        help="Fraction of --total to allocate to number-format pairs (default: 0.05)"
    )
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or ts
    t0 = datetime.datetime.now()

    # Number-format pairs budget (cross-cutting concern)
    numfmt_total = max(100, int(args.total * args.numfmt_ratio))
    domain_total = args.total - numfmt_total

    # Filter domains
    selected_domains = DOMAINS
    if args.domains:
        name_set = set(args.domains)
        selected_domains = [(n, g, s) for n, g, s in DOMAINS if n in name_set]

    per_domain_total = max(1, domain_total // len(selected_domains))
    level_counts = compute_level_counts(per_domain_total, args.levels)

    print(f"\n{'='*70}")
    print(f"  Advanced Math QA Batch Generator")
    print(f"  Total target: {args.total:,}  |  Domains: {len(selected_domains)}  |  Seed: {args.seed}")
    print(f"  Per-domain: ~{per_domain_total:,}  |  Number-format: {numfmt_total:,}")
    print(f"  Level distribution: {level_counts}")
    print(f"{'='*70}", flush=True)

    results = []

    # Run domain generators
    for name, gen_fn, save_fn in selected_domains:
        result = run_domain(name, gen_fn, save_fn,
                            level_counts, args.seed, tag)
        results.append(result)

    # Number-format equivalence pairs
    if not args.domains or "number_formats" in (args.domains or []):
        nf_result = run_numfmt(numfmt_total, args.seed, tag)
        results.append(nf_result)

    elapsed = (datetime.datetime.now() - t0).total_seconds()

    # Summary
    summary_path = RESULTS_DIR / f"generate_all_summary_{tag}.txt"
    print_summary(results, level_counts, args.total, elapsed, summary_path)


if __name__ == "__main__":
    main()
