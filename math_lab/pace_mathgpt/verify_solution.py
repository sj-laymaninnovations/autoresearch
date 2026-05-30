#!/usr/bin/env python3
"""
verify_solution.py — Arithmetic auditor for mathgpt training jsonl.

Why this exists:
    The merged training_data.jsonl contains rows from older generators whose
    `solution` field is a fake trace (e.g. problem "730x + 418 = 305195" with
    solution "730 + 418 = 417.5"). Feeding those rows into PACE Stage 3 or
    Stage 6 bakes the wrong reasoning into the weights with much higher
    gradient signal than the flat format gives. PACE's canvas explicitly
    warns about this — "erroneous chains will corrupt them quickly".

What this does:
    - Parses each line of the solution into steps (lines containing `=`).
    - Evaluates the LHS arithmetic and confirms it equals the declared RHS.
    - Confirms the `####` final value matches the `answer` field (within
      tolerance for floats).
    - Classifies every row as `clean` or `dirty` with a reason.
    - Writes <out>/clean.jsonl and <out>/rejected.jsonl. Originals untouched.

Policy: REJECT, do not auto-repair. Repair could introduce new wrong
arithmetic, which is exactly the failure mode we are trying to avoid.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

# Match "LHS = RHS" lines. The LHS may contain integers, decimals, and the
# operators + - * / and parentheses. We deliberately do NOT support algebra
# variables (x, y, ...) — those steps are treated as opaque and skipped from
# arithmetic verification, but the final `####` value is still checked.
_STEP_RE = re.compile(r"^\s*([0-9+\-*/.() ]+?)\s*=\s*(-?[0-9.]+)\s*$")
_FINAL_RE = re.compile(r"^####\s*(-?[0-9.]+)\s*$")

# Tolerance for float comparisons. Mathgpt traces are typically integer or
# 2-3 decimal places.
_FLOAT_TOL = 1e-3


def _safe_eval_expr(expr: str) -> Optional[float]:
    """Evaluate an arithmetic expression using only +-*/() and numeric literals.
    Returns None if the expression contains anything else."""
    # Hard reject anything that is not in the safe alphabet.
    if not re.fullmatch(r"[0-9+\-*/.() ]+", expr):
        return None
    try:
        # eval with empty namespaces — safe because the alphabet check above
        # forbids names, attribute access, comprehensions, and function calls.
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return float(result)
        return None
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError):
        return None


def _approx_equal(a: float, b: float, tol: float = _FLOAT_TOL) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol or abs(a - b) <= tol * max(abs(a), abs(b))


def classify_row(row: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Classify a single mathgpt row.

    Returns (is_clean, reason, diagnostics).
    """
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    declared_answer = row.get("answer", None)

    if not problem.strip():
        return False, "missing_problem", {}
    if not solution.strip():
        return False, "missing_solution", {}
    if declared_answer is None:
        return False, "missing_answer", {}

    lines = [ln.strip() for ln in solution.split("\n") if ln.strip()]
    if not lines:
        return False, "empty_solution", {}

    final_value: Optional[float] = None
    step_errors: List[Dict[str, Any]] = []
    verifiable_steps = 0
    suspicious_lines: List[str] = []

    for ln in lines:
        # Final marker
        m_final = _FINAL_RE.match(ln)
        if m_final:
            try:
                final_value = float(m_final.group(1))
            except ValueError:
                return False, "final_not_numeric", {"line": ln}
            continue

        # Placeholder-CoT detection: generators that emit "671 ... 276542 = 412"
        # produce lines containing ellipses / placeholder words / arrows that
        # are not real reasoning. Treat any such line as junk.
        if "..." in ln or "…" in ln:
            suspicious_lines.append(ln)
            continue
        if re.search(r"\b(Result|Step|Calc|Working|Answer|TODO|FIXME)\b\s*[:=]", ln):
            # "Result = 250.08" with no LHS expression — placeholder.
            suspicious_lines.append(ln)
            continue
        if "→" in ln or "->" in ln:
            # Arrow notation often replaces real arithmetic in placeholder traces.
            suspicious_lines.append(ln)
            continue

        # Step "LHS = RHS"
        m_step = _STEP_RE.match(ln)
        if not m_step:
            # Any line with `=` that didn't match the strict arithmetic step
            # regex is suspect — could be `x = ...`, `Result = 250`, or other
            # placeholder. Real CoT either uses pure arithmetic on the LHS or
            # has no `=` at all (narrative).
            if "=" in ln:
                suspicious_lines.append(ln)
            continue

        lhs_expr = m_step.group(1)
        try:
            declared_rhs = float(m_step.group(2))
        except ValueError:
            step_errors.append({"line": ln, "issue": "non_numeric_rhs"})
            continue

        evaluated = _safe_eval_expr(lhs_expr)
        if evaluated is None:
            # LHS in safe alphabet but eval failed — skip silently.
            continue

        verifiable_steps += 1
        if not _approx_equal(evaluated, declared_rhs):
            step_errors.append({
                "line": ln,
                "lhs_evaluated": evaluated,
                "declared_rhs": declared_rhs,
            })

    diagnostics: Dict[str, Any] = {
        "verifiable_steps": verifiable_steps,
        "step_errors": step_errors,
        "suspicious_lines": suspicious_lines,
    }

    if final_value is None:
        return False, "missing_final_marker", diagnostics

    try:
        decl = float(declared_answer)
    except (TypeError, ValueError):
        return False, "answer_not_numeric", diagnostics

    if not _approx_equal(final_value, decl):
        diagnostics["final_value"] = final_value
        diagnostics["declared_answer"] = decl
        return False, "final_mismatches_answer", diagnostics

    if step_errors:
        return False, "step_arithmetic_error", diagnostics

    if suspicious_lines:
        # The solution contained placeholder-CoT markers (`...`, `Result =`,
        # arrows, or `=` lines that didn't parse). These trace patterns are
        # what we are trying to keep out of the training set.
        return False, "placeholder_cot", diagnostics

    if verifiable_steps == 0:
        # No arithmetic to verify and no suspicious markers — the solution
        # is just the `####` marker. Probably a one-shot answer. Accept but
        # tag so the adapter can choose to route these to Stage 1/2 only
        # (no CoT scaffolding) rather than Stage 3/6 (which need real steps).
        diagnostics["answer_only"] = True
        return True, "clean_answer_only", diagnostics

    return True, "clean_verified", diagnostics


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def audit_file(in_path: Path, out_dir: Path,
               sample: Optional[int] = None) -> Dict[str, Any]:
    """Audit `in_path`, writing clean.jsonl and rejected.jsonl to `out_dir`.
    Returns a stats dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    stats_path = out_dir / "audit_stats.json"

    reason_counts: Dict[str, int] = {}
    by_domain: Dict[str, Dict[str, int]] = {}
    total = 0
    clean_count = 0
    rejected_count = 0

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(clean_path, "w", encoding="utf-8") as f_clean, \
         open(rejected_path, "w", encoding="utf-8") as f_rej:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if sample is not None and total > sample:
                total -= 1
                break

            is_clean, reason, diag = classify_row(row)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            domain = row.get("domain", "unknown")
            by_domain.setdefault(domain, {"clean": 0, "dirty": 0})
            if is_clean:
                clean_count += 1
                by_domain[domain]["clean"] += 1
                # Attach a verification stamp without mutating the original
                # keys downstream consumers may rely on.
                row["_verify_reason"] = reason
                f_clean.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                rejected_count += 1
                by_domain[domain]["dirty"] += 1
                row["_reject_reason"] = reason
                row["_reject_diagnostics"] = diag
                f_rej.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input": str(in_path),
        "total": total,
        "clean": clean_count,
        "rejected": rejected_count,
        "clean_pct": round(100 * clean_count / total, 2) if total else 0.0,
        "by_reason": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "by_domain": by_domain,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    print("\n=== Audit Stats ===")
    print(f"Input: {stats['input']}")
    print(f"Total: {stats['total']}  |  Clean: {stats['clean']} "
          f"({stats['clean_pct']}%)  |  Rejected: {stats['rejected']}")
    print("\nReasons:")
    for reason, n in stats["by_reason"].items():
        print(f"  {n:>6}  {reason}")
    print("\nBy domain:")
    for domain, counts in sorted(stats["by_domain"].items()):
        total = counts["clean"] + counts["dirty"]
        pct = round(100 * counts["clean"] / total, 1) if total else 0.0
        print(f"  {domain:<20}  clean={counts['clean']:>5}  "
              f"dirty={counts['dirty']:>5}  ({pct}% clean)")


def main():
    parser = argparse.ArgumentParser(
        description="Arithmetic auditor for mathgpt training jsonl.")
    parser.add_argument("input", type=str,
                        help="Path to the source jsonl (e.g. training_data.jsonl)")
    parser.add_argument("--out-dir", type=str, default="audit_out",
                        help="Output directory for clean/rejected files "
                             "(default: audit_out)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional row cap for quick previews")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    stats = audit_file(in_path, out_dir, sample=args.sample)
    print_stats(stats)
    print(f"\nClean:    {out_dir / 'clean.jsonl'}")
    print(f"Rejected: {out_dir / 'rejected.jsonl'}")
    print(f"Stats:    {out_dir / 'audit_stats.json'}")


if __name__ == "__main__":
    main()
