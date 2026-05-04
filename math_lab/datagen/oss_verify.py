"""
oss_verify.py — Diagnostic report for the curated OSS JSONL corpus.

Usage:
    python3 math_lab/datagen/oss_verify.py --data math_lab/results/oss/oss_train_*.jsonl
    python3 math_lab/datagen/oss_verify.py --manifest math_lab/results/oss/manifest.json
"""
import json, re, sys, argparse, random
from pathlib import Path
from collections import Counter

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     help="Path to train JSONL (glob ok)")
    ap.add_argument("--manifest", help="Path to manifest.json (auto-resolves data path)")
    ap.add_argument("--samples",  type=int, default=3, help="Sample records to print per source")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Resolve data path
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        data_path = Path(manifest["train_file"])
    elif args.data:
        # support glob
        import glob
        matches = sorted(glob.glob(args.data))
        if not matches:
            print(f"No files matched: {args.data}"); sys.exit(1)
        data_path = Path(matches[-1])   # latest
    else:
        # auto-find latest in default location
        candidates = sorted(Path("math_lab/results/oss").glob("oss_train_*.jsonl"))
        if not candidates:
            print("No OSS train file found. Run oss_ingest.py first."); sys.exit(1)
        data_path = candidates[-1]

    print(f"Verifying: {data_path}  ({data_path.stat().st_size/1e6:.1f} MB)\n")
    records = load_jsonl(data_path)
    print(f"Total records: {len(records)}\n")

    # ── Per-source breakdown ──────────────────────────────────────────────────
    src_counts  = Counter(r["source"]       for r in records)
    tier_counts = Counter(r["tier"]         for r in records)
    cot_lens    = [len(r.get("solution_cot","")) for r in records]

    print("By source:")
    for src, n in sorted(src_counts.items()):
        pct = n/len(records)*100
        print(f"  {src:<30} {n:>6}  ({pct:.1f}%)")

    print("\nBy tier:")
    tier_labels = {1: "simple arithmetic word problem",
                   2: "multi-step",
                   3: "algebraic / ratio / percent"}
    for t, n in sorted(tier_counts.items()):
        print(f"  Tier {t} ({tier_labels.get(t,'?')}): {n}")

    print(f"\nCoT length (chars): min={min(cot_lens)} mean={sum(cot_lens)//len(cot_lens)} max={max(cot_lens)}")

    # ── Answer distribution ───────────────────────────────────────────────────
    answers = [r.get("answer_real", 0) for r in records if isinstance(r.get("answer_real"), int)]
    buckets = Counter()
    for a in answers:
        if a < 0:            buckets["negative"] += 1
        elif a == 0:         buckets["zero"]     += 1
        elif a <= 10:        buckets["1-10"]     += 1
        elif a <= 100:       buckets["11-100"]   += 1
        elif a <= 1000:      buckets["101-1000"] += 1
        elif a <= 10000:     buckets["1001-10K"] += 1
        else:                buckets[">10K"]     += 1
    print("\nAnswer distribution:")
    for k in ["negative","zero","1-10","11-100","101-1000","1001-10K",">10K"]:
        if k in buckets:
            print(f"  {k:<12} {buckets[k]:>6}  ({buckets[k]/len(answers)*100:.1f}%)")

    # ── CoT format check ─────────────────────────────────────────────────────
    missing_commit = [r for r in records if "####" not in r.get("solution_cot","")]
    wrong_answer   = []
    for r in records:
        cot = r.get("solution_cot","")
        m = re.search(r'####\s*(-?\d+)', cot)
        if m and int(m.group(1)) != r.get("answer_real"):
            wrong_answer.append(r)

    print(f"\nCoT format checks:")
    print(f"  Missing '####' commit token: {len(missing_commit)}")
    print(f"  '####' answer ≠ answer_real: {len(wrong_answer)}")
    if wrong_answer[:3]:
        print("  First mismatch examples:")
        for r in wrong_answer[:3]:
            print(f"    problem: {r['problem'][:60]}")
            print(f"    answer_real={r['answer_real']}  cot snippet: ...{r['solution_cot'][-40:]}")

    # ── Sample records per source ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Sample records ({args.samples} per source):")
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    for src, recs in sorted(by_source.items()):
        print(f"\n  ── {src} ──")
        sample = rng.sample(recs, min(args.samples, len(recs)))
        for r in sample:
            print(f"  [Tier {r['tier']}] {r['problem'][:80]}")
            print(f"  CoT snippet: {r['solution_cot'][:120].replace(chr(10), ' ')}")
            print(f"  Answer: {r['answer_real']}\n")

if __name__ == "__main__":
    main()
