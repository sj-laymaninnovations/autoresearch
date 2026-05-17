"""
schema.py — Pydantic v2 schema and validator for MathGPT eval set problems.

Usage:
    from schema import EvalProblem, load_problem, validate_all

    p = load_problem("problems/EVAL-001.json")   # returns EvalProblem or raises
    validate_all("problems/")                    # validate every .json in dir
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enum types
# ---------------------------------------------------------------------------

class AnswerType(str, Enum):
    integer    = "integer"
    fraction   = "fraction"
    expression = "expression"
    proof      = "proof"

class VerificationMethod(str, Enum):
    exact_match       = "exact_match"
    sympy_equivalence = "sympy_equivalence"
    known_key         = "known_key"

class Difficulty(str, Enum):
    easy   = "easy"
    medium = "medium"
    hard   = "hard"

class Category(str, Enum):
    algebra        = "algebra"
    geometry       = "geometry"
    number_theory  = "number_theory"
    combinatorics  = "combinatorics"
    calculus       = "calculus"
    logic          = "logic"
    word_problem   = "word_problem"

class ContaminationRisk(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"

class Status(str, Enum):
    verified            = "verified"
    needs_verification  = "needs_verification"
    flagged             = "flagged"


# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

class EvalProblem(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier, format EVAL-NNN",
        pattern=r"^EVAL-\d{3,}$",
    )
    source: str = Field(
        ...,
        description="Human-readable source, e.g. 'AIME 2026 I, Problem 3' or 'Hand-authored'",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="URL to official problem statement; None for hand-authored",
    )
    problem_text: str = Field(
        ...,
        min_length=10,
        description="Full problem statement, verbatim from official source or authored text",
    )
    answer: str = Field(
        ...,
        description="Ground-truth answer. Must come from official key or SymPy-verified derivation",
    )
    answer_type: AnswerType
    verification_method: VerificationMethod
    difficulty_estimate: Difficulty
    category: Category
    contamination_risk: ContaminationRisk = ContaminationRisk.low
    status: Status = Status.verified
    notes: Optional[str] = Field(
        default=None,
        description="Any caveats, alternate forms, or reviewer flags",
    )

    # ── validators ────────────────────────────────────────────────────────

    @field_validator("id")
    @classmethod
    def id_prefix(cls, v: str) -> str:
        if not v.startswith("EVAL-"):
            raise ValueError(f"id must start with 'EVAL-', got {v!r}")
        return v

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be blank")
        return v.strip()

    @model_validator(mode="after")
    def proof_needs_note(self) -> "EvalProblem":
        if self.answer_type == AnswerType.proof and not self.notes:
            raise ValueError(
                "Proof-type problems must include notes explaining what constitutes "
                "a correct proof (e.g. 'show that N is odd')."
            )
        return self

    @model_validator(mode="after")
    def known_key_implies_source(self) -> "EvalProblem":
        if (
            self.verification_method == VerificationMethod.known_key
            and self.source == "Hand-authored"
        ):
            raise ValueError(
                "verification_method='known_key' implies an official answer key; "
                "hand-authored problems must use 'exact_match' or 'sympy_equivalence'."
            )
        return self

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problem(path: str | Path) -> EvalProblem:
    """Load and validate a single problem JSON file. Raises on invalid data."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalProblem.model_validate(data)


def validate_all(problems_dir: str | Path) -> dict[str, list[str]]:
    """
    Validate every .json file in problems_dir.

    Returns a dict mapping filename → list of error strings.
    An empty list means the file passed validation.
    """
    problems_dir = Path(problems_dir)
    results: dict[str, list[str]] = {}
    for p in sorted(problems_dir.glob("*.json")):
        try:
            load_problem(p)
            results[p.name] = []
        except Exception as e:
            results[p.name] = [str(e)]
    return results


# ---------------------------------------------------------------------------
# CLI usage: python schema.py problems/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("problems")
    if target.is_dir():
        results = validate_all(target)
        n_ok = sum(1 for v in results.values() if not v)
        n_fail = len(results) - n_ok
        print(f"Validated {len(results)} problems: {n_ok} OK, {n_fail} FAILED")
        for fname, errors in results.items():
            if errors:
                print(f"  ✗ {fname}")
                for e in errors:
                    print(f"      {e}")
        sys.exit(0 if n_fail == 0 else 1)
    else:
        try:
            prob = load_problem(target)
            print(f"✓ {prob.id}  [{prob.source}]  {prob.category}  {prob.difficulty_estimate}")
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            sys.exit(1)
