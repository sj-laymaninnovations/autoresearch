"""
merge_results.py — Merge and deduplicate all harder_qa JSONL files

Reads every harder_qa_*.jsonl in math_lab/results/, deduplicates by
(problem, answer), and writes a single master training_data.jsonl.

Usage:
    python math_lab/merge_results.py                      # merge all, default output
    python math_lab/merge_results.py --domain algebra     # only algebra pairs
    python math_lab/merge_results.py --min-level 3        # only level 3+ problems
    python math_lab/merge_results.py --out custom.jsonl   # custom output path
    python math_lab/merge_results.py --stats              # print stats only, no write
"""

import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

RESULTS_DIR  = Path(__file__).parent / "results"
DEFAULT_OUT  = RESULTS_DIR / "training_data.jsonl"
STATS_FILE   = RESULTS_DIR / "merge_stats.json"


def load_all_pairs(results_dir: Path, domain: str = None) -> list[dict]:
    """Load every pair from every harder_qa_*.jsonl in results_dir."""
    pattern = f"harder_qa_{domain}_*.jsonl" if domain else "harder_qa_*.jsonl"
    files   = sorted(results_dir.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' in {results_dir}")
        return []

    all_pairs = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    all_pairs.append((json.loads(line), f.name))
        except Exception as e:
            print(f"  WARNING: could not read {f.name}: {e}")

    return all_pairs


def dedup(raw: list[tuple[dict, str]]) -> tuple[list[dict], dict]:
    """
    Deduplicate by (problem.strip().lower(), str(answer)).
    Returns (unique_pairs, stats_dict).
    When two records share a key, keep the one with the higher level.
    """
    seen     = {}   # key -> best record
    sources  = defaultdict(int)
    domains  = defaultdict(int)
    levels   = defaultdict(int)
    dupes    = 0

    for record, source_file in raw:
        problem = record.get("problem", "").strip().lower()
        answer  = str(record.get("answer", ""))
        key     = (problem, answer)

        if key in seen:
            dupes += 1
            # Keep the higher-level version
            if record.get("level", 0) > seen[key].get("level", 0):
                seen[key] = record
        else:
            seen[key] = record

        sources[source_file] += 1
        domains[record.get("domain", "unknown")] += 1
        levels[str(record.get("level", "?"))] += 1

    unique = list(seen.values())

    # Sort: by domain then level (ascending) for clean training curriculum
    unique.sort(key=lambda r: (r.get("domain", ""), r.get("level", 0)))

    stats = {
        "total_raw":    len(raw),
        "total_unique": len(unique),
        "duplicates":   dupes,
        "by_domain":    dict(domains),
        "by_level":     dict(levels),
        "source_files": len(sources),
    }
    return unique, stats


def apply_filters(pairs: list[dict], min_level: int = None,
                  domain: str = None) -> list[dict]:
    """Apply optional post-dedup filters."""
    out = pairs
    if domain:
        out = [p for p in out if p.get("domain") == domain]
    if min_level is not None:
        out = [p for p in out if p.get("level", 0) >= min_level]
    return out


def write_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def print_stats(stats: dict, output_path: Path = None) -> None:
    print(f"\n{'='*55}")
    print(f"  merge_results summary")
    print(f"{'='*55}")
    print(f"  Source files : {stats['source_files']}")
    print(f"  Raw pairs    : {stats['total_raw']}")
    print(f"  Duplicates   : {stats['duplicates']}  "
          f"({100*stats['duplicates']/max(stats['total_raw'],1):.1f}%)")
    print(f"  Unique pairs : {stats['total_unique']}")
    if output_path:
        print(f"  Output       : {output_path}")
    print()
    print("  By domain:")
    for d, n in sorted(stats["by_domain"].items()):
        print(f"    {d:<20} {n:>4} pairs")
    print()
    print("  By difficulty level:")
    for lv, n in sorted(stats["by_level"].items(), key=lambda x: x[0]):
        bar = "#" * min(n, 40)
        print(f"    L{lv}  {bar}  {n}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and deduplicate all harder_qa JSONL files")
    parser.add_argument("--domain",    default=None,
                        help="Filter to one domain (arithmetic, algebra, etc.)")
    parser.add_argument("--min-level", type=int, default=None,
                        help="Only include problems at or above this difficulty level")
    parser.add_argument("--out",       default=str(DEFAULT_OUT),
                        help="Output JSONL path (default: results/training_data.jsonl)")
    parser.add_argument("--stats",     action="store_true",
                        help="Print stats only, do not write output file")
    args = parser.parse_args()

    out_path = Path(args.out)

    # Load
    raw = load_all_pairs(RESULTS_DIR, domain=args.domain)
    if not raw:
        return

    # Dedup
    unique, stats = dedup(raw)

    # Filter
    filtered = apply_filters(unique, min_level=args.min_level, domain=args.domain)
    if len(filtered) < len(unique):
        stats["after_filter"] = len(filtered)

    # Stats
    print_stats(stats, None if args.stats else out_path)

    if args.stats:
        return

    # Write
    write_jsonl(filtered, out_path)
    print(f"Wrote {len(filtered)} pairs -> {out_path}")

    # Save stats alongside
    stats["generated_at"] = datetime.datetime.now().isoformat()
    stats["output"] = str(out_path)
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
