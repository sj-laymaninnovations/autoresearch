"""
verify_answer.py — Programmatic answer verification for MathGPT eval set.

Verification hierarchy:
  1. SymPy symbolic equivalence  (for expressions, fractions, surds)
  2. Numeric near-equality       (for floating-point representations)
  3. Normalized string match     (for integers and clean forms)

Usage:
    from verify_answer import verify, VerifyResult

    result = verify(problem_record, candidate_answer="47")
    print(result.passed, result.reason)

Run tests:
    python verify_answer.py --test
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional


# ── SymPy is optional; fall back gracefully ─────────────────────────────────
try:
    import sympy
    from sympy import (
        sympify, simplify, nsimplify, Rational, sqrt, pi, E,
        latex, parse_expr, Symbol
    )
    from sympy.parsing.sympy_parser import (
        standard_transformations, implicit_multiplication_application
    )
    SYMPY_AVAILABLE = True
    _TRANSFORMATIONS = (
        standard_transformations + (implicit_multiplication_application,)
    )
except ImportError:
    SYMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    passed: bool
    method_used: str          # "sympy" | "numeric" | "string" | "error"
    reason: str
    candidate_normalized: str
    expected_normalized: str
    tolerance: float = 1e-9

    def __repr__(self) -> str:
        symbol = "✓" if self.passed else "✗"
        return (
            f"VerifyResult({symbol} {self.method_used}: "
            f"expected={self.expected_normalized!r}, "
            f"got={self.candidate_normalized!r}) — {self.reason}"
        )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_string(s: str) -> str:
    """Collapse whitespace, strip, lowercase."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _strip_answer_prefix(s: str) -> str:
    """Remove common answer prefixes like 'the answer is', 'x =', etc."""
    s = s.strip()
    for pat in (
        r"^the answer is[:\s]+",
        r"^answer[:\s]+",
        r"^=\s*",
        r"^x\s*=\s*",
        r"^y\s*=\s*",
        r"^n\s*=\s*",
    ):
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
    return s


def _try_fraction(s: str) -> Optional[Fraction]:
    """Try to parse s as a fraction (int, 'a/b', or '-a/b')."""
    s = s.strip()
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            pass
    return None


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SymPy path
# ---------------------------------------------------------------------------

def _sympy_equivalent(expected: str, candidate: str) -> VerifyResult:
    """Use SymPy to check symbolic equivalence."""
    if not SYMPY_AVAILABLE:
        return VerifyResult(
            passed=False,
            method_used="error",
            reason="SymPy not installed",
            candidate_normalized=candidate,
            expected_normalized=expected,
        )
    try:
        exp_expr = parse_expr(expected, transformations=_TRANSFORMATIONS)
        cand_expr = parse_expr(candidate, transformations=_TRANSFORMATIONS)
        diff = simplify(exp_expr - cand_expr)
        if diff == 0:
            return VerifyResult(
                passed=True,
                method_used="sympy",
                reason="Symbolic difference is 0",
                candidate_normalized=str(cand_expr),
                expected_normalized=str(exp_expr),
            )
        # Try numeric evaluation
        try:
            diff_val = float(diff.evalf())
            if abs(diff_val) < 1e-9:
                return VerifyResult(
                    passed=True,
                    method_used="sympy",
                    reason=f"Numeric difference |{diff_val:.2e}| < 1e-9",
                    candidate_normalized=str(cand_expr),
                    expected_normalized=str(exp_expr),
                )
        except Exception:
            pass
        return VerifyResult(
            passed=False,
            method_used="sympy",
            reason=f"Symbolic difference = {diff}",
            candidate_normalized=str(cand_expr),
            expected_normalized=str(exp_expr),
        )
    except Exception as e:
        return VerifyResult(
            passed=False,
            method_used="error",
            reason=f"SymPy parse error: {e}",
            candidate_normalized=candidate,
            expected_normalized=expected,
        )


# ---------------------------------------------------------------------------
# Main verify function
# ---------------------------------------------------------------------------

def verify(
    problem: dict | object,
    candidate_answer: str,
) -> VerifyResult:
    """
    Verify a candidate answer against the recorded answer in a problem record.

    Args:
        problem:          dict or EvalProblem object with at least
                          'answer', 'answer_type', 'verification_method'.
        candidate_answer: the answer to check.

    Returns:
        VerifyResult with .passed, .method_used, .reason.
    """
    # Normalize inputs
    if hasattr(problem, "answer"):
        expected_raw   = str(problem.answer)
        answer_type    = str(problem.answer_type)
        verify_method  = str(problem.verification_method)
    else:
        expected_raw   = str(problem["answer"])
        answer_type    = str(problem.get("answer_type", "integer"))
        verify_method  = str(problem.get("verification_method", "exact_match"))

    cand_raw = _strip_answer_prefix(str(candidate_answer))
    exp_raw  = _strip_answer_prefix(expected_raw)

    cand_norm = _normalize_string(cand_raw)
    exp_norm  = _normalize_string(exp_raw)

    # ── Level 1: direct normalized string match ──────────────────────────
    if cand_norm == exp_norm:
        return VerifyResult(
            passed=True,
            method_used="string",
            reason="Exact normalized match",
            candidate_normalized=cand_norm,
            expected_normalized=exp_norm,
        )

    # ── Level 2: fraction equivalence ────────────────────────────────────
    exp_frac  = _try_fraction(exp_raw)
    cand_frac = _try_fraction(cand_raw)
    if exp_frac is not None and cand_frac is not None:
        if exp_frac == cand_frac:
            return VerifyResult(
                passed=True,
                method_used="numeric",
                reason="Fraction equivalence",
                candidate_normalized=str(cand_frac),
                expected_normalized=str(exp_frac),
            )
        return VerifyResult(
            passed=False,
            method_used="numeric",
            reason=f"Fractions differ: {exp_frac} ≠ {cand_frac}",
            candidate_normalized=str(cand_frac),
            expected_normalized=str(exp_frac),
        )

    # ── Level 3: floating-point near-equality ─────────────────────────────
    exp_f  = _try_float(exp_raw)
    cand_f = _try_float(cand_raw)
    if exp_f is not None and cand_f is not None:
        if abs(exp_f - cand_f) < 1e-9 * max(1.0, abs(exp_f)):
            return VerifyResult(
                passed=True,
                method_used="numeric",
                reason=f"Float near-equal: |{exp_f - cand_f:.2e}|",
                candidate_normalized=str(cand_f),
                expected_normalized=str(exp_f),
            )
        return VerifyResult(
            passed=False,
            method_used="numeric",
            reason=f"Float differs: expected {exp_f}, got {cand_f}",
            candidate_normalized=str(cand_f),
            expected_normalized=str(exp_f),
        )

    # ── Level 4: SymPy symbolic equivalence ───────────────────────────────
    if (
        verify_method == "sympy_equivalence"
        or answer_type in ("expression", "fraction")
        or any(c in exp_raw for c in ("sqrt", "^", "*", "/", "pi", "e"))
    ):
        result = _sympy_equivalent(exp_raw, cand_raw)
        if result.passed:
            return result
        # Still fall through to final failure
        return result

    # ── Final: string mismatch ────────────────────────────────────────────
    return VerifyResult(
        passed=False,
        method_used="string",
        reason=f"No match at any level",
        candidate_normalized=cand_norm,
        expected_normalized=exp_norm,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests() -> int:
    """Run built-in test suite. Returns exit code (0 = all pass)."""
    tests = [
        # (label, expected, candidate, answer_type, should_pass)
        # --- Integers ---
        ("int exact",       "47",     "47",     "integer", True),
        ("int prefixed",    "47",     "The answer is 47", "integer", True),
        ("int mismatch",    "47",     "48",     "integer", False),
        ("int leading 0",   "7",      "07",     "integer", True),   # numerically equal — valid in competition context
        # --- Fractions ---
        ("frac exact",      "3/4",    "3/4",    "fraction", True),
        ("frac reduced",    "6/8",    "3/4",    "fraction", True),
        ("frac negative",   "-1/2",   "-1/2",   "fraction", True),
        ("frac mismatch",   "3/4",    "5/6",    "fraction", False),
        # --- Decimals ---
        ("decimal",         "0.5",    "1/2",    "fraction", True),
        ("decimal match",   "3.14",   "3.14",   "expression", True),
        # --- Algebraic expressions ---
        ("sympy simple",    "x**2+2*x+1",  "(x+1)**2",  "expression", True),
        ("sympy surd",      "sqrt(2)",  "2**(1/2)",  "expression", True),
        ("sympy inequiv",   "x**2+1",  "x**2+2",  "expression", False),
        # --- Common competition answer forms ---
        ("aime int 0",      "0",      "0",      "integer", True),
        ("aime int 999",    "999",    "999",    "integer", True),
        ("fraction form",   "7/16",   "7/16",   "fraction", True),
    ]
    passed = 0
    failed = 0
    print("=== verify_answer.py unit tests ===\n")
    for label, expected, candidate, atype, should_pass in tests:
        prob = {"answer": expected, "answer_type": atype,
                "verification_method": "sympy_equivalence" if atype == "expression" else "exact_match"}
        result = verify(prob, candidate)
        ok = result.passed == should_pass
        if ok:
            passed += 1
            print(f"  ✓ {label}")
        else:
            failed += 1
            print(f"  ✗ {label} — expected passed={should_pass}, got {result}")
    print(f"\n{passed}/{passed+failed} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_tests())
    # Manual test
    prob = {"answer": sys.argv[1], "answer_type": "integer",
            "verification_method": "exact_match"}
    cand = sys.argv[2] if len(sys.argv) > 2 else input("Candidate answer: ")
    print(verify(prob, cand))
