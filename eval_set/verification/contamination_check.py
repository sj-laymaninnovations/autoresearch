"""
contamination_check.py — Task 7: stream 10K rows from each training corpus
and compare normalized fingerprints to our eval problems.

Usage:
    pip install datasets scikit-learn
    python verification/contamination_check.py

Output: verification/contamination_report.md
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time
from typing import Iterator

PROBLEMS_DIR = pathlib.Path(__file__).parent.parent / "problems"
OUT_REPORT   = pathlib.Path(__file__).parent / "contamination_report.md"
SAMPLE_SIZE  = 10_000

# Corpora to stream. Each is a (HuggingFace dataset id, config, split, text_field) tuple.
CORPORA = [
    ("open-r1/OpenR1-Math-220k",   "default",  "train", "problem"),
    ("NovaSky-UC-Berkeley/Sky-T1-Data-17K", None, "train", "conversations"),
    ("AI-MO/NuminaMath-CoT",       None,       "train", "problem"),
    ("qfq/openai-math-s1k",        None,       "train", "problem"),
]


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def fingerprint(text: str) -> str:
    """Normalized fingerprint: lowercase, collapse whitespace, strip punctuation."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_set(fp: str) -> set[str]:
    return set(fp.split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_corpus(dataset_id: str, config: str | None,
                  split: str, text_field: str,
                  max_rows: int) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] `datasets` not installed. Run: pip install datasets")
        return

    try:
        ds = load_dataset(
            dataset_id, config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        for row in ds:
            raw = row.get(text_field, "")
            if isinstance(raw, list):
                # conversations field: concat all content values
                raw = " ".join(
                    item.get("content", "") if isinstance(item, dict) else str(item)
                    for item in raw
                )
            text = str(raw).strip()
            if text:
                yield text
                count += 1
            if count >= max_rows:
                break
    except Exception as e:
        print(f"  [!] Could not stream {dataset_id}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load our eval problems
    eval_probs: list[dict] = []
    for p in sorted(PROBLEMS_DIR.glob("EVAL-*.json")):
        with open(p, encoding="utf-8") as f:
            prob = json.load(f)
        if prob.get("status") == "needs_verification":
            continue
        eval_probs.append(prob)

    if not eval_probs:
        print("No verified problems to check.")
        return

    eval_fps  = [fingerprint(p["problem_text"]) for p in eval_probs]
    eval_sets = [token_set(fp) for fp in eval_fps]

    print(f"Checking {len(eval_probs)} problems against {len(CORPORA)} corpora "
          f"({SAMPLE_SIZE:,} rows each)...\n")

    # Results: (problem_id, corpus, similarity_score, corpus_text_snippet)
    flags: list[tuple[str, str, float, str]] = []

    for ds_id, config, split, text_field in CORPORA:
        print(f"  Streaming {ds_id} ...")
        t0 = time.time()
        count = 0
        for row_text in stream_corpus(ds_id, config, split, text_field, SAMPLE_SIZE):
            row_fp  = fingerprint(row_text)
            row_set = token_set(row_fp)
            for i, (eval_s, prob) in enumerate(zip(eval_sets, eval_probs)):
                sim = jaccard(eval_s, row_set)
                if sim >= 0.70:
                    flags.append((
                        prob["id"],
                        ds_id,
                        sim,
                        row_text[:300],
                    ))
            count += 1
        print(f"    {count:,} rows scanned in {time.time()-t0:.1f}s")

    # Write report
    lines = ["# Contamination Check Report\n"]
    lines += [f"- Problems checked: {len(eval_probs)}"]
    lines += [f"- Corpora sampled: {[c[0] for c in CORPORA]}"]
    lines += [f"- Sample size per corpus: {SAMPLE_SIZE:,}"]
    lines += [f"- Similarity threshold: 0.70 (Jaccard on token sets)\n"]

    if not flags:
        lines += ["\n## Result: NO matches found above threshold ✓\n"]
        lines += ["All eval problems appear clean against the sampled training rows."]
    else:
        lines += [f"\n## Result: {len(flags)} potential match(es) flagged\n"]
        for pid, corpus, score, snippet in sorted(flags, key=lambda x: -x[2]):
            lines += [
                f"### {pid} — {corpus}  (Jaccard={score:.2f})",
                f"```\n{snippet}\n```\n",
            ]

    lines += ["\n---\n*Note: this is a streaming sample (10K rows), not exhaustive.*\n"]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT_REPORT}")
    if flags:
        print(f"⚠ {len(flags)} problems flagged for review — see contamination_report.md")
    else:
        print("✓ No contamination detected in sampled rows")


if __name__ == "__main__":
    main()
