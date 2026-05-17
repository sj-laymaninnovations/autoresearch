"""
build_balanced_corpus.py — upsample minority concepts to match majority,
producing a concept-balanced training set for v3.

Strategy: for each concept, sample with replacement up to the max
concept's count (asm_kernel_arm64 at 614). All concepts then appear in
batches at roughly equal rate during PyTorch DataLoader shuffle.

Source:  training_data/bet_a_asm_v2_train.jsonl   (1087 unbalanced)
Output:  training_data/bet_a_asm_v3_balanced_train.jsonl

Val set unchanged (we want val to reflect natural distribution so we can
compare val_loss across v1/v2/v3 on the same basis).
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "training_data" / "bet_a_asm_v2_train.jsonl"
OUT  = ROOT / "training_data" / "bet_a_asm_v3_balanced_train.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cap-to", type=int, default=None,
                    help="Override target count per concept (default: max concept count)")
    args = ap.parse_args()

    records = [json.loads(l) for l in SRC.open()]
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r)

    counts = {k: len(v) for k, v in by_concept.items()}
    target = args.cap_to or max(counts.values())

    rng = random.Random(args.seed)
    out_records = []
    upsample_log = []
    for concept, group in by_concept.items():
        n_orig = len(group)
        if n_orig >= target:
            out_records.extend(group)
            upsample_log.append((concept, n_orig, n_orig, 1.0))
            continue
        # Take all originals first, then sample additional copies with replacement
        out_records.extend(group)
        deficit = target - n_orig
        extras = [rng.choice(group) for _ in range(deficit)]
        out_records.extend(extras)
        upsample_log.append((concept, n_orig, target, target / n_orig))

    rng.shuffle(out_records)
    with OUT.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_records)} records to {OUT.name}\n")
    print(f"{'concept':<24s} {'before':>8s}  {'after':>6s}  {'multiplier':>10s}")
    for c, n_orig, n_new, mult in upsample_log:
        print(f"  {c:<22s} {n_orig:>8d}  {n_new:>6d}  {mult:>9.2f}x")
    # Sanity: verify distribution post-balance
    post = Counter(r["concept"] for r in out_records)
    print(f"\npost-balance concept counts: {dict(post.most_common())}")


if __name__ == "__main__":
    main()
