"""
fetch_wikipedia_prehistoric.py — Stream-filter Wikipedia for prehistoric-era articles.

Uses the HuggingFace `datasets` library to stream the English Wikipedia snapshot
(~6.5M articles) and filter for genuine 30,000–10,000 BCE content using the same
two-tier keyword system as enrich_record.py.

Two-tier filtering:
  Tier 1 — MUST match at least one STRONG prehistoric keyword (high specificity)
  Tier 2 — Title OR first 500 chars must contain a topic anchor keyword

This dual gate keeps precision high — a general history article that mentions
"paleolithic" once in passing does not qualify.

Output: sources_prehistoric/fineweb-edu/wikipedia_prehistoric_NNNN.parquet
        (same schema as enriched FineWeb-Edu shards so they merge cleanly)

Usage:
    python pipeline/fetch_wikipedia_prehistoric.py
    python pipeline/fetch_wikipedia_prehistoric.py --max-articles 20000 --min-tokens 80
    python pipeline/fetch_wikipedia_prehistoric.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "sources_prehistoric" / "fineweb-edu"

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Tier 1 — strong prehistoric signals (any one is sufficient)
STRONG_SIGNALS = [
    "paleolithic", "palaeolithic", "upper paleolithic", "lower paleolithic",
    "middle paleolithic", "neolithic", "mesolithic", "epipaleolithic",
    "aurignacian", "gravettian", "solutrean", "magdalenian", "châtelperronian",
    "chatelperronian", "mousterian", "oldowan", "acheulean", "acheulian",
    "cave painting", "cave art", "rock art", "parietal art",
    "lascaux", "altamira", "chauvet", "el castillo", "pech merle",
    "font-de-gaume", "trois-frères", "dolni vestonice", "sungir",
    "homo heidelbergensis", "homo antecessor", "homo naledi",
    "neanderthal", "neanderthals", "denisovan", "denisovans",
    "homo sapiens sapiens", "anatomically modern human",
    "pleistocene", "late pleistocene", "glacial maximum", "last glacial maximum",
    "ice age", "würm glaciation", "weichsel glaciation", "devensian",
    "cro-magnon", "cro magnon",
    "stone age", "hunter-gatherer", "hunter gatherer",
    "venus figurine", "venus of", "willendorf", "laussel",
    "prehistoric", "prehistory", "pre-historic",
    "megalith", "dolmen", "menhir", "göbekli tepe", "gobekli tepe",
    "jericho", "natufian", "pre-pottery neolithic",
    "beringia", "land bridge", "out of africa", "human migration",
    "flint knapping", "lithic technology", "blade technology",
    "mammoth", "woolly mammoth", "woolly rhinoceros", "cave bear",
    "megafauna", "pleistocene megafauna",
    "30,000", "20,000", "15,000", "10,000 bce", "10000 bc",
    "40,000 years ago", "35,000 years ago", "25,000 years ago",
    "archaic homo", "hominid", "hominins", "homo erectus",
    "atlatl", "hafting", "microlith", "burin",
]

# Tier 2 — topic anchor keywords that must appear in title or opening
TOPIC_ANCHORS = [
    "prehistoric", "paleolithic", "palaeolithic", "neolithic", "mesolithic",
    "stone age", "cave", "ancient human", "early human", "hominid", "hominin",
    "pleistocene", "ice age", "hunter", "archaeological", "archaeology",
    "excavation", "fossil", "neanderthal", "denisovan", "homo ", "flint",
    "mammoth", "megafauna", "burial", "migration", "rock art", "figurine",
    "dolmen", "megalith", "menhir", "gobekli", "natufian", "jericho",
    "lascaux", "altamira", "chauvet", "aurignacian", "gravettian",
]


def _text_lower(text: str, chars: int = 4000) -> str:
    return text[:chars].lower()


def _title_lower(title: str) -> str:
    return title.lower()


def is_prehistoric(title: str, text: str) -> bool:
    """Return True if this article is genuinely prehistoric-era content."""
    tl = _title_lower(title)
    tx = _text_lower(text)
    opening = (tl + " " + text[:500].lower())

    # Tier 1: must contain at least one strong prehistoric signal
    has_strong = any(kw in tx for kw in STRONG_SIGNALS)
    if not has_strong:
        return False

    # Tier 2: title or opening must have a topic anchor
    has_anchor = any(kw in opening for kw in TOPIC_ANCHORS)
    return has_anchor


def _dedup_hash(text: str) -> str:
    return hashlib.sha256(text[:500].encode("utf-8", errors="replace")).hexdigest()


def _make_record(article: dict) -> dict:
    """Convert a Wikipedia article dict to pipeline-compatible record schema."""
    text  = article.get("text") or ""
    title = article.get("title") or ""
    url   = article.get("url") or ""
    wiki_id = str(article.get("id") or "")

    return {
        # Core fields matching enriched FineWeb-Edu schema
        "text":                    text,
        "id":                      f"wiki_{wiki_id}",
        "dump":                    "wikipedia-en-20231101",
        "url":                     url,
        "file_path":               f"wikipedia/{title}",
        "language":                "en",
        "language_score":          1.0,
        "token_count":             len(text.split()),
        "score":                   None,
        "int_score":               None,
        # Enrichment fields (pre-populated — skip re-enrichment overhead)
        "fk_score":                None,
        "est_token_count":         int(len(text.split()) * 1.3),
        "grade_level":             None,
        "age_range_min":           None,
        "age_range_max":           None,
        "edu_score":               None,
        "edu_score_estimated":     True,
        "dedup_hash":              _dedup_hash(text),
        "domain_slug":             "anthropology",
        "domain_name":             "Anthropology",
        "domain_category":         "Social Sciences",
        "domain_confidence":       0.9,
        "age_epoch_of_history":    "Prehistoric Era",
        "earliest_historical_year": -30000,
        "corrected_era":           "prehistoric",
        "corrected_date":          -30000,
    }


def fetch(
    max_articles: int,
    min_tokens: int,
    max_per_shard: int,
    dry_run: bool,
    streaming: bool,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Wikipedia prehistoric filter]")
    print(f"  max_articles   : {max_articles:,}")
    print(f"  min_tokens     : {min_tokens}")
    print(f"  max_per_shard  : {max_per_shard:,}")
    print(f"  output dir     : {OUT_DIR}")
    if dry_run:
        print(f"  [DRY RUN — no files written]\n")
    else:
        print()

    print(f"  Loading Wikipedia dataset (streaming={streaming}) …")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=streaming,
    )

    buf: list[dict] = []
    shard_idx      = 0
    scanned        = 0
    matched        = 0
    skipped_short  = 0
    t0             = time.time()

    def flush(buf: list[dict], idx: int) -> None:
        if not buf or dry_run:
            return
        cols  = {k: [r.get(k) for r in buf] for k in buf[0].keys()}
        table = pa.table(cols)
        path  = OUT_DIR / f"wikipedia_prehistoric_{idx:04d}.parquet"
        pq.write_table(table, path, compression="snappy")
        print(f"\n  → wrote {path.name}  ({len(buf):,} rows)")

    for article in ds:
        scanned += 1

        title = article.get("title", "")
        text  = article.get("text", "") or ""

        # Minimum content gate
        if len(text.split()) < min_tokens:
            skipped_short += 1
            continue

        if is_prehistoric(title, text):
            rec = _make_record(article)
            buf.append(rec)
            matched += 1

            if len(buf) >= max_per_shard:
                flush(buf, shard_idx)
                shard_idx += 1
                buf = []

            if matched % 100 == 0:
                elapsed = time.time() - t0
                rate = scanned / elapsed
                print(f"  scanned {scanned:,} | matched {matched:,} | "
                      f"{rate:.0f} art/s", end="\r", flush=True)

        if matched >= max_articles:
            print(f"\n  Hit max_articles limit ({max_articles:,})")
            break

        # Progress heartbeat every 100k articles
        if scanned % 100_000 == 0:
            elapsed = time.time() - t0
            rate = scanned / elapsed
            print(f"  scanned {scanned:,} | matched {matched:,} | "
                  f"{rate:.0f} art/s        ", end="\r", flush=True)

    # Flush remainder
    if buf:
        flush(buf, shard_idx)
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\n\n{'─'*56}")
    print(f"  Scanned         : {scanned:,} articles")
    print(f"  Skipped (short) : {skipped_short:,}")
    print(f"  Matched         : {matched:,}")
    print(f"  Output shards   : {shard_idx}")
    print(f"  Time            : {elapsed:.1f}s  ({scanned/elapsed:.0f} art/s)")
    if matched == 0:
        print("\n  ⚠️  No articles matched — check keyword lists.")
    else:
        print(f"\n  ✅ Wikipedia prehistoric data written to: {OUT_DIR}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Filter Wikipedia for prehistoric-era articles"
    )
    ap.add_argument("--max-articles",  type=int, default=20_000,
                    help="Max matching articles to collect (default: 20000)")
    ap.add_argument("--min-tokens",    type=int, default=80,
                    help="Min word count for an article to qualify (default: 80)")
    ap.add_argument("--max-per-shard", type=int, default=5_000,
                    help="Max rows per output parquet (default: 5000)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Count matches without writing output")
    ap.add_argument("--no-streaming",  action="store_true",
                    help="Download full dataset instead of streaming (faster filter, more RAM)")
    args = ap.parse_args()

    fetch(
        max_articles  = args.max_articles,
        min_tokens    = args.min_tokens,
        max_per_shard = args.max_per_shard,
        dry_run       = args.dry_run,
        streaming     = not args.no_streaming,
    )


if __name__ == "__main__":
    main()
