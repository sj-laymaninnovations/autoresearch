"""
preprocessor.py — Bet A distillation preprocessing.

Two responsibilities:
  1. classify_body(body) -> "process" | "type_D" | "too_short"
     Heuristic catch for bare-benchmark Type-D commits that frontier
     teachers will hallucinate rationale for rather than refusing.
  2. dedupe_commits(records) -> ordered list of unique commits, aggregating
     hunks/file_paths that share a SHA.

Run as a module for self-test:
    python -m preprocessor  --selftest
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

# ---- Type-D classifier ----------------------------------------------------

# A line is a "benchmark line" if it looks like any of:
#   "label: N N N N"           (label-colon-numbers)
#   "  N N N N"                (whitespace-leading multi-numeric)
#   anywhere has >=3 numeric tokens separated by whitespace
_RE_LABEL_COLON_NUMS = re.compile(r'^\s*\S+:\s+[\d.]+(\s+[\d.]+)?')
_RE_THREE_PLUS_NUMS  = re.compile(r'^\s*[\d.+\-]+(\s+[\d.+\-]+){2,}\s*$')

def _is_benchmark_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _RE_LABEL_COLON_NUMS.match(s):
        return True
    if _RE_THREE_PLUS_NUMS.match(s):
        return True
    # `Before:` / `After:` lines used by ffmpeg / x264 conventions
    if re.match(r'^\s*(Before|After|Pre|Post)\s*:\s*', s):
        return True
    # Lines that are mostly numeric tokens (>=50% numeric)
    toks = s.split()
    if len(toks) >= 3:
        n_num = sum(1 for t in toks if re.fullmatch(r'[\d.+\-]+[%x×]?', t))
        if n_num / len(toks) >= 0.5:
            return True
    return False


# A line is a "prose line" if it has >= 3 alphabetic words longer than 2 chars,
# no leading digit, and isn't a benchmark line.
_RE_WORD = re.compile(r'\b[A-Za-z][A-Za-z\-]{2,}\b')

def _is_prose_line(line: str) -> bool:
    s = line.strip()
    if not s or _is_benchmark_line(s):
        return False
    if s[0].isdigit():
        return False
    words = _RE_WORD.findall(s)
    return len(words) >= 3


def classify_body(body: str) -> tuple[str, dict]:
    """
    Returns (status, stats).
    status in {"process", "type_D", "too_short"}.
    stats has per-line counts for the audit trail.
    """
    body = body or ""
    if len(body.strip()) < 80:
        return "too_short", {"body_chars": len(body.strip())}

    lines = [l for l in body.split('\n') if l.strip()]
    n_total = len(lines)
    if n_total == 0:
        return "too_short", {"body_chars": 0}

    n_bench = sum(1 for l in lines if _is_benchmark_line(l))
    n_prose = sum(1 for l in lines if _is_prose_line(l))
    bench_ratio = n_bench / n_total

    stats = {
        "body_chars":   len(body),
        "n_lines":      n_total,
        "n_bench":      n_bench,
        "n_prose":      n_prose,
        "bench_ratio":  round(bench_ratio, 2),
    }

    # Type-D: heavy benchmark, light prose
    if bench_ratio > 0.5 and n_prose < 3:
        return "type_D", stats
    return "process", stats


# ---- Commit deduplication -------------------------------------------------

def dedupe_commits(hunk_records: list[dict]) -> list[dict]:
    """
    Group hunk records by SHA. One output record per unique commit:
      {
        "sha":          ...,
        "subject":      first non-empty subject seen,
        "file_paths":   sorted unique list of file paths across all hunks,
        "n_hunks":      count of input hunks,
        "n_added_total": sum of n_added across hunks,
        "n_comment_in_added_total": sum,
        "body_substantive": all True,
      }
    """
    by_sha = defaultdict(list)
    for r in hunk_records:
        by_sha[r['sha']].append(r)

    out = []
    for sha, hunks in by_sha.items():
        out.append({
            "sha":           sha,
            "subject":       next((h.get('subject') for h in hunks if h.get('subject')), ''),
            "file_paths":    sorted({h['file_path'] for h in hunks}),
            "n_hunks":       len(hunks),
            "n_added_total": sum(h.get('n_added', 0) for h in hunks),
            "n_comment_in_added_total": sum(h.get('n_comment_in_added', 0) for h in hunks),
            "body_substantive": all(h.get('body_substantive', False) for h in hunks),
            "body_chars":    max(h.get('body_chars', 0) for h in hunks),
        })
    # Order: commit SHA (stable, deterministic for git_show)
    out.sort(key=lambda r: r['sha'])
    return out


# ---- Self-test ------------------------------------------------------------

def _selftest():
    cases = [
        # Type-D — synthetic bare benchmark
        ("aarch64: Add fast path",
         "        Cortex A53   A72   A73   X1\n"
         "Before: 250          165   170   140\n"
         "After:  240          158   163   135",
         "type_D"),
        # Real x264 4664f5aa-style — has prose AND benchmarks
        ("aarch64: Improve scheduling",
         "                Cortex A53    A72    A73\n"
         "Before:\n"
         "sad_x3_4x4_neon:      580    303    204\n"
         "After:\n"
         "sad_x3_4x4_neon:      477    298    206\n"
         "\n"
         "Thus, this is around a 20-25% speedup on Cortex A53 for the small\n"
         "sizes (much smaller difference for bigger sizes though), while it\n"
         "doesn't make much of a difference at all (mostly within measurement\n"
         "noise) for the out-of-order cores (A72 and A73).",
         "process"),
        # Too short
        ("Fix typo", "fix typo", "too_short"),
        # Pure prose — clearly process
        ("Optimize lane writes",
         "Partial register writes can create long dependency chains, which "
         "can reduce performance on out-of-order CPUs. This patch removes "
         "most of these kinds of problems in MC functions by filling the "
         "full register before other lane loading instructions. Most lane "
         "extracting stores can also be optimized using FP scalar stores "
         "when the 0th lane would be extracted.",
         "process"),
    ]
    print("classify_body self-test:")
    n_ok = 0
    for subj, body, expected in cases:
        status, stats = classify_body(body)
        ok = (status == expected)
        n_ok += ok
        marker = "✓" if ok else "✗"
        print(f"  {marker} {subj[:40]:40s} -> {status}  (expected {expected})  {stats}")
    print(f"\n  {n_ok}/{len(cases)} cases pass\n")
    return 0 if n_ok == len(cases) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())


if __name__ == "__main__":
    main()
