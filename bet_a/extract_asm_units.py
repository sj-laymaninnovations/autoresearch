"""
extract_asm_units.py — Multi-granularity extractor for ARM asm source files.

For each .S file, emits records at multiple granularities (per Sean's
design: block / comment / function / [method] / [class]). asm files have
no class/method structure, so 3 of 5 granularities apply here. The C/
C++/Python equivalent extractor (TODO) will cover all five.

Per-record schema (one JSONL line each):
    {
      "id":              "<sha1-of-text-12>",
      "file_path":       "<repo-relative path>",
      "repo":            "pytorch/pytorch",
      "org":             "pytorch",
      "language":        "asm_aarch64" | "asm_aarch32",
      "license":         "BSD-2-Clause",
      "granularity":     "comment_block" | "block" | "function",
      "name":            "<symbol or label, or null>",
      "context":         "<containing function name, or null>",
      "start_line":      <int>,
      "end_line":        <int>,
      "n_lines":         <int>,
      "text":            "<full unit content>"
    }

Extraction rules:
  - comment_block: consecutive '#' or '//' lines (≥2 lines, ≥40 chars
    substantive content after stripping the comment-leader). Skip
    license/copyright headers at the very top of the file.
  - function: BEGIN_FUNCTION <name> ... END_FUNCTION <name> (QNNPACK
    convention), OR '.global X' / 'X:' to 'ret' / '.size X, .-X'.
  - block: within a function, labelled subsection ('.LBLname:' or
    'name_label:') to the next label. Skipped if <3 instruction lines.

Usage:
    python extract_asm_units.py \\
        --root '/path/to/repos/pytorch/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack' \\
        --pattern '*.S' \\
        --out curated/qnnpack_asm_units.jsonl \\
        --repo pytorch/pytorch \\
        --license BSD-2-Clause

Run self-test:
    python extract_asm_units.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# A comment line starts with whitespace + ('#' or '//' or ';')
_COMMENT_RE = re.compile(r"^\s*(?:#(?!include|\s*define|\s*if|\s*endif|\s*else|\s*elif)|//|;)\s?(.*)$")
# .global / .globl declarations
_GLOBAL_RE  = re.compile(r"^\s*\.glob(?:al|l)\s+(\S+)")
# A label definition (function or local block). Allow leading `.` for
# GCC-style local labels like `.Lloop_start:` and `.Lend:`.
_LABEL_RE   = re.compile(r"^\s*(\.?[A-Za-z_][\w\.$]*)\s*:")
# BEGIN_FUNCTION / END_FUNCTION (QNNPACK macros)
_BEGIN_FN   = re.compile(r"^\s*BEGIN_FUNCTION\s+(\S+)")
_END_FN     = re.compile(r"^\s*END_FUNCTION\s+(\S+)")
# .size X, .-X — function-end marker in GCC-style asm
_SIZE_END   = re.compile(r"^\s*\.size\s+(\S+)\s*,\s*\.\-\1")
# A ret instruction (ARM and x86 variants)
_RET_RE     = re.compile(r"^\s*(?:ret|retq)\s*$")
# License/copyright detection (skip these as comment_block)
_LICENSE_HINT = re.compile(r"copyright|license|all rights reserved", re.I)

# Map filename to language tag
def language_of(path: Path) -> str:
    s = path.name.lower()
    if "aarch64" in s or "arm64" in s: return "asm_aarch64"
    if "aarch32" in s or "arm/" in str(path).lower(): return "asm_aarch32"
    if "x86" in s or "x86_64" in s:    return "asm_x86_64"
    return "asm_unknown"


# ---------------------------------------------------------------------------
# Comment-line accessor — returns the comment text after the leader, or None
# ---------------------------------------------------------------------------

def _comment_text(line: str) -> str | None:
    m = _COMMENT_RE.match(line)
    if m is None:
        return None
    return m.group(1).rstrip()


# ---------------------------------------------------------------------------
# Per-granularity extractors
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    granularity: str
    name:        str | None
    context:     str | None
    start_line:  int
    end_line:    int
    text:        str


def extract_comment_blocks(lines: list[str]) -> list[Unit]:
    """Comment_block: consecutive comment-leader lines (≥2 lines)."""
    units: list[Unit] = []
    in_block = False
    block_start = 0
    block_lines: list[str] = []
    block_comment_lines: list[str] = []

    # We also skip the license header at the very top.
    # Heuristic: the first comment block in the file containing "copyright" or
    # "license" is dropped.
    license_block_seen = False

    for i, line in enumerate(lines, start=1):
        c = _comment_text(line)
        if c is not None:
            if not in_block:
                in_block = True
                block_start = i
                block_lines = []
                block_comment_lines = []
            block_lines.append(line)
            block_comment_lines.append(c)
        else:
            if in_block:
                _flush_block(units, block_start, i - 1, block_lines,
                              block_comment_lines, license_block_seen)
                if not license_block_seen and any(_LICENSE_HINT.search(c)
                                                    for c in block_comment_lines):
                    license_block_seen = True
                in_block = False
    # Tail
    if in_block:
        _flush_block(units, block_start, len(lines), block_lines,
                      block_comment_lines, license_block_seen)
    return units


def _flush_block(out: list[Unit], start: int, end: int,
                  raw: list[str], cmt: list[str],
                  license_seen: bool):
    # Filter: ≥2 lines AND ≥40 substantive chars after stripping leaders
    if len(raw) < 2:
        return
    substantive = "\n".join(cmt).strip()
    if len(substantive) < 40:
        return
    # Skip license headers (first block containing copyright/license is treated
    # as the license, dropped here even if substantive)
    if not license_seen and any(_LICENSE_HINT.search(c) for c in cmt):
        return
    out.append(Unit(
        granularity="comment_block",
        name=None,
        context=None,
        start_line=start,
        end_line=end,
        text="\n".join(raw).rstrip(),
    ))


def extract_functions(lines: list[str]) -> list[Unit]:
    """Function: BEGIN_FUNCTION/END_FUNCTION pair OR .global/ret region."""
    units: list[Unit] = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        # QNNPACK-style
        m = _BEGIN_FN.match(line)
        if m:
            name = m.group(1)
            start = i + 1
            j = i + 1
            while j < n and not _END_FN.match(lines[j]):
                j += 1
            end = j + 1 if j < n else n
            body = "\n".join(lines[i:end])
            units.append(Unit("function", name, None, start, end, body))
            i = end
            continue
        # GCC-style .global X / X: ... .size X, .-X
        m = _GLOBAL_RE.match(line)
        if m:
            name = m.group(1)
            # Find the label line and the end (.size or ret followed by blank)
            label_re = re.compile(rf"^\s*{re.escape(name)}\s*:")
            j = i + 1
            start = i + 1
            while j < n and not label_re.match(lines[j]):
                j += 1
            if j >= n:
                i += 1
                continue
            label_at = j
            # Find end — first .size matching, OR ret followed by blank, OR
            # next .global
            k = label_at + 1
            end = None
            while k < n:
                if _SIZE_END.match(lines[k]) and _SIZE_END.match(lines[k]).group(1) == name:
                    end = k + 1
                    break
                if _GLOBAL_RE.match(lines[k]):
                    end = k
                    break
                k += 1
            if end is None:
                end = n
            body = "\n".join(lines[i:end])
            units.append(Unit("function", name, None, start, end, body))
            i = end
            continue
        i += 1
    return units


def extract_blocks(lines: list[str], functions: list[Unit]) -> list[Unit]:
    """Block: label-delimited subsection inside a function."""
    units: list[Unit] = []
    for fn in functions:
        # Walk the function body and find labels (excluding the function's own
        # name label).
        fn_lines = fn.text.split("\n")
        labels: list[tuple[int, str]] = []  # (line-offset, name)
        for offset, ln in enumerate(fn_lines):
            m = _LABEL_RE.match(ln)
            if m:
                label_name = m.group(1)
                if label_name == fn.name:
                    continue
                # Skip if line is a directive disguised as label, etc.
                labels.append((offset, label_name))
        # For each label, the block is from this label to the next label or end
        for idx, (off, name) in enumerate(labels):
            next_off = labels[idx + 1][0] if idx + 1 < len(labels) else len(fn_lines)
            body_lines = fn_lines[off:next_off]
            # Filter: ≥3 instruction-ish lines (not comment/blank/label-only)
            inst_count = sum(
                1 for l in body_lines[1:]  # skip the label line itself
                if l.strip() and not _COMMENT_RE.match(l) and not _LABEL_RE.match(l)
            )
            if inst_count < 3:
                continue
            block_text = "\n".join(body_lines).rstrip()
            units.append(Unit(
                granularity="block",
                name=name,
                context=fn.name,
                start_line=fn.start_line + off,
                end_line=fn.start_line + next_off - 1,
                text=block_text,
            ))
    return units


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def extract_from_file(path: Path, repo: str, license_: str,
                       root: Path | None = None) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    comments = extract_comment_blocks(lines)
    functions = extract_functions(lines)
    blocks    = extract_blocks(lines, functions)

    rel_path = str(path.relative_to(root)) if root else str(path)
    org = repo.split("/")[0] if "/" in repo else repo

    records: list[dict] = []
    for u in comments + functions + blocks:
        records.append({
            "id":          hash_id(u.text),
            "file_path":   rel_path,
            "repo":        repo,
            "org":         org,
            "language":    language_of(path),
            "license":     license_,
            "granularity": u.granularity,
            "name":        u.name,
            "context":     u.context,
            "start_line":  u.start_line,
            "end_line":    u.end_line,
            "n_lines":     u.text.count("\n") + 1,
            "text":        u.text,
        })
    return records


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SELFTEST_ASM = '''\
/*
 * Copyright (c) Facebook
 * BSD-2 license
 */

#include <foo.h>

#  Args passed via 8 registers (64 bytes)
#  x0: mr
#  x1: nr
#  x2: kc

# void example_function(int mr, int nr, int kc);
BEGIN_FUNCTION example_function
    # Save FP and LR
    stp x29, x30, [sp, #-16]!

.Lloop_start:
    # main loop body
    ldp q0, q1, [x4], #32
    fmla v8.4s, v0.4s, v16.4s
    fmla v9.4s, v1.4s, v16.4s
    cbnz x2, .Lloop_start

.Lepilogue:
    # restore FP/LR and return
    ldp x29, x30, [sp], #16
    ret
END_FUNCTION example_function
'''


def _selftest():
    import tempfile
    print("=== extract_asm_units self-test ===\n")
    with tempfile.NamedTemporaryFile("w", suffix=".S", delete=False) as f:
        f.write(_SELFTEST_ASM)
        path = Path(f.name)

    recs = extract_from_file(path, repo="test/test", license_="TEST")

    by_gran = {}
    for r in recs:
        by_gran.setdefault(r["granularity"], []).append(r)

    expected_min = {"comment_block": 1, "function": 1, "block": 1}
    ok = True
    for gran, need in expected_min.items():
        got = len(by_gran.get(gran, []))
        marker = "✓" if got >= need else "✗"
        ok &= got >= need
        print(f"  {marker} {gran:<14s} count={got} (expected ≥{need})")

    print(f"\n  Total units: {len(recs)}\n")
    print("=== unit details ===")
    for gran in ("comment_block", "function", "block"):
        for r in by_gran.get(gran, []):
            print(f"\n[{gran}] name={r['name']!r} context={r['context']!r}  "
                  f"lines {r['start_line']}-{r['end_line']} ({r['n_lines']}L)")
            preview = r["text"][:300]
            print(f"  {preview}{'...' if len(r['text']) > 300 else ''}")

    # License header should be dropped from comment_block list
    for r in by_gran.get("comment_block", []):
        if "Copyright" in r["text"] or "license" in r["text"].lower():
            print(f"\n  ✗ FAIL: license header leaked into comment_block")
            ok = False
            break
    else:
        print(f"\n  ✓ license header correctly dropped")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root",      type=Path, help="Directory to walk")
    ap.add_argument("--pattern",   default="*.S", help="Glob (default: *.S)")
    ap.add_argument("--out",       type=Path)
    ap.add_argument("--repo",      help="e.g. pytorch/pytorch")
    ap.add_argument("--license",   dest="license_", default="UNKNOWN")
    ap.add_argument("--root-for-rel", type=Path, default=None,
                    help="Make file_path relative to this dir (default: --root)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    assert args.root and args.out and args.repo, \
        "--root, --out, --repo required (or use --selftest)"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rel_root = args.root_for_rel or args.root

    files = sorted(args.root.rglob(args.pattern))
    print(f"  scanning {len(files)} files matching {args.pattern!r} under {args.root}")

    n_units = 0
    n_by_gran: dict[str, int] = {}
    with args.out.open("w") as f:
        for fp in files:
            recs = extract_from_file(fp, args.repo, args.license_, root=rel_root)
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_units += 1
                n_by_gran[r["granularity"]] = n_by_gran.get(r["granularity"], 0) + 1
    print(f"  wrote {n_units} units -> {args.out}")
    for g, n in sorted(n_by_gran.items()):
        print(f"    {g:<14s} {n}")


if __name__ == "__main__":
    main()
