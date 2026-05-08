"""
extract_prehistoric.py — Filter enriched shards for genuine prehistoric-era records.

Reads every parquet in sources_enriched/<dataset>/, selects rows where
corrected_era == "prehistoric", and writes them into sources_prehistoric/
as consolidated parquet files (~50K rows each for manageable shard size).

Only rows that passed the content-signal gate (corrected_era is not None,
date is -30000 or -10000) are included. This guarantees the output is
genuine prehistoric content — no humanities false-positives.

Usage:
    python pipeline/extract_prehistoric.py
    python pipeline/extract_prehistoric.py --dataset fineweb-edu --max-per-shard 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR     = Path(__file__).parent.parent
ENRICHED_DIR = BASE_DIR / "sources_enriched"
OUT_DIR      = BASE_DIR / "sources_prehistoric"
COMPRESSION  = "snappy"

# Era filter — must exactly match corrected_era values from enrich_record.py
ERA_FILTER   = "prehistoric"


def extract(dataset: str, max_per_shard: int, dry_run: bool) -> None:
    source_dir = ENRICHED_DIR / dataset
    if not source_dir.exists():
        print(f"ERROR: {source_dir} not found. Run batch_enrich.py first.")
        sys.exit(1)

    out_dir = OUT_DIR / dataset
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(source_dir.glob("*.parquet"))
    print(f"\nExtracting '{ERA_FILTER}' era records")
    print(f"  source : {source_dir}  ({len(shards)} shards)")
    print(f"  output : {out_dir}")
    print(f"  max rows/shard : {max_per_shard:,}")
    if dry_run:
        print("  [DRY RUN]\n")
    else:
        print()

    total_in   = 0
    total_out  = 0
    shard_idx  = 0
    buf: list[dict] = []
    t0 = time.time()

    def flush_buf(buf: list[dict], idx: int) -> None:
        if not buf or dry_run:
            return
        cols = {k: [r.get(k) for r in buf] for k in buf[0].keys()}
        table = pa.table(cols)
        path  = out_dir / f"prehistoric_{idx:04d}.parquet"
        pq.write_table(table, path, compression=COMPRESSION)
        print(f"  → wrote {path.name}  ({len(buf):,} rows)")

    for shard_path in shards:
        print(f"  scanning {shard_path.name} …", end=" ", flush=True)
        pf = pq.ParquetFile(shard_path)
        shard_hit = 0

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rows_dict = rg.to_pydict()
            n = len(next(iter(rows_dict.values())))
            era_col = rows_dict.get("corrected_era", [None] * n)

            for i in range(n):
                total_in += 1
                if era_col[i] == ERA_FILTER:
                    buf.append({k: rows_dict[k][i] for k in rows_dict})
                    shard_hit += 1
                    total_out += 1

                    if len(buf) >= max_per_shard:
                        flush_buf(buf, shard_idx)
                        shard_idx += 1
                        buf = []

        print(f"{shard_hit:,} hits")

    # Flush remainder
    if buf:
        flush_buf(buf, shard_idx)
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\n{'─'*56}")
    print(f"  Scanned       : {total_in:,} rows across {len(shards)} shards")
    print(f"  Prehistoric   : {total_out:,} rows  ({total_out/max(1,total_in)*100:.2f}%)")
    print(f"  Output shards : {shard_idx}")
    print(f"  Time          : {elapsed:.1f}s")
    if total_out == 0:
        print("\n  ⚠️  No prehistoric rows found — check corrected_era values in enriched data.")
    else:
        print(f"\n  ✅ Output written to: {out_dir}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Extract prehistoric-era records from enriched shards")
    ap.add_argument("--dataset",       default="fineweb-edu",
                    help="Subdirectory under sources_enriched/ (default: fineweb-edu)")
    ap.add_argument("--max-per-shard", type=int, default=50_000,
                    help="Max rows per output parquet shard (default: 50000)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Scan and count matches without writing output")
    args = ap.parse_args()
    extract(args.dataset, args.max_per_shard, args.dry_run)


if __name__ == "__main__":
    main()
