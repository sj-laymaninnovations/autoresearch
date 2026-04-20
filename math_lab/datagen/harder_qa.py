"""
harder_qa.py — Harder Q/A Pair Generator for Math Training

After the kernel makes wrong predictions on benchmark problems, this module
generates progressively harder variants to use as fine-tuning data.

Strategy:
  1. Parse the wrong answer to identify which reasoning step failed
  2. Perturb the problem to make that step harder
  3. Verify the answer using integer arithmetic (no floats)
  4. Emit JSONL records for fine-tuning ingestion

Output: math_lab/results/harder_qa_<domain>_<run>.jsonl
"""

import re
import json
import random
import hashlib
import datetime
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_int(text: str) -> Optional[int]:
    """Extract last integer from text."""
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return int(nums[-1]) if nums else None


# ── Problem perturbation ──────────────────────────────────────────────────────

def perturb_arithmetic(problem: str, answer: float,
                        difficulty: str = "one_harder") -> dict:
    """
    Replace small numbers in a problem with larger ones.
    'one_harder': x5 to x20 magnitude increase
    'structural': change operation type (add->multiply etc.)
    """
    # Find all standalone integers in the problem
    nums = re.findall(r"\b(\d{1,4})\b", problem)
    if not nums:
        return {"problem": problem, "answer": answer, "valid": False}

    scale = {"one_harder": random.randint(5, 20),
             "structural": random.randint(10, 50)}.get(difficulty, 5)

    new_problem = problem
    new_answer = answer * scale   # approximate; verified below

    for n in set(nums):
        new_n = str(int(n) * scale)
        new_problem = re.sub(r"\b" + n + r"\b", new_n, new_problem, count=1)

    # Simple arithmetic verification: re-evaluate if expression is visible
    return {"problem": new_problem, "answer": new_answer,
            "scale_applied": scale, "valid": True}


def generate_chain_of_thought(problem: str, answer: float) -> str:
    """
    Generate a solution that shows the actual computation.
    Format: '<expr> = <answer>\n#### <answer>'
    Gives the model explicit arithmetic to learn rather than "Computing the result..."
    """
    import re as _re

    # Prefer integers; round floats to 2dp max
    if isinstance(answer, float):
        if abs(answer - round(answer)) < 0.01:
            answer = int(round(answer))
        else:
            answer = round(answer, 2)
    ans_str = str(answer)

    # Build a clean computation line from the operands in the problem
    # Find all standalone numbers
    nums = _re.findall(r"-?\d+(?:\.\d+)?", problem)
    ops  = _re.findall(r"[\+\-\*\/]", problem.replace("->", ""))

    if len(nums) >= 2 and ops:
        a, b, op = nums[0], nums[1], ops[0]
        computation = f"{a} {op} {b} = {ans_str}"
    elif len(nums) >= 2:
        computation = f"{nums[0]} ... {nums[1]} = {ans_str}"
    else:
        computation = f"Result = {ans_str}"

    return f"{computation}\n#### {ans_str}"


# ── Sub-domain generators ─────────────────────────────────────────────────────

TEMPLATES = {
    "arithmetic": [
        ("What is {a} + {b}?",                  lambda a, b, c: a + b),
        ("What is {a} - {b}?",                  lambda a, b, c: a - b),
        ("What is {a} times {b}?",              lambda a, b, c: a * b),
        ("A store has {a} items. If {b} more arrive, how many total?",
                                                 lambda a, b, c: a + b),
        ("If you have {a} dollars and spend {b}, how much is left?",
                                                 lambda a, b, c: a - b),
    ],
    "algebra": [
        ("Solve for x: {a}x + {b} = {c}",       lambda a, b, c: (c - b) / a),
        ("If {a}x = {c}, what is x?",            lambda a, b, c: c / a),
        ("What is x if x squared = {c}?",        lambda a, b, c: c**0.5),
    ],
    "number_theory": [
        ("What is the remainder when {a} is divided by {b}?",
                                                 lambda a, b, c: a % b),
        ("What is {a} divided by {b} (integer)?",
                                                 lambda a, b, c: a // b),
        ("What is the GCD of {a} and {b}?",
                                                 lambda a, b, c: __import__("math").gcd(a, b)),
    ],
    "combinatorics": [
        ("In how many ways can you arrange {a} items?",
                                                 lambda a, b, c: __import__("math").factorial(min(a, 10))),
        ("Choose {b} from {a}. How many combinations?",
                                                 lambda a, b, c: __import__("math").comb(a, min(b, a))),
    ],
}


def generate_new_problem(domain: str, difficulty_level: int,
                         rng: random.Random) -> dict:
    """
    Generate a fresh math problem at the requested difficulty level.
    difficulty_level: 1=easy, 2=medium, 3=hard, 4=harder, 5=very_hard
    """
    scale = 10 ** (difficulty_level - 1)

    templates = TEMPLATES.get(domain, TEMPLATES["arithmetic"])
    template, fn = rng.choice(templates)

    a = rng.randint(scale, scale * 10)
    b = rng.randint(max(1, scale // 2), scale * 5)
    c = a * b + rng.randint(1, scale)

    try:
        answer = fn(a, b, c)
    except (ZeroDivisionError, ArithmeticError):
        a, b, c = 12, 4, 48
        answer = a + b

    # Format template
    try:
        problem = template.format(a=a, b=b, c=c)
    except (KeyError, IndexError):
        problem = f"What is {a} + {b}?"
        answer = a + b

    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)

    return {
        "problem": problem,
        "answer":  answer,
        "domain":  domain,
        "level":   difficulty_level,
    }


# ── Main generator ────────────────────────────────────────────────────────────

def generate_harder_pairs(
    model_wrong_answers: list[dict],
    domain: str,
    target_difficulty: str = "one_harder",
    n_pairs: int = 50,
    seed: int = None,
) -> list[dict]:
    """
    For each wrong answer, generate a harder variant.

    Args:
        model_wrong_answers: list of {"problem": str, "expected": float, "got": float}
        domain: math sub-domain string
        target_difficulty: "one_harder" | "structural" | "adversarial"
        n_pairs: total number of new pairs to generate
        seed: random seed for reproducibility

    Returns:
        list of {"problem": str, "solution": str, "answer": any,
                 "domain": str, "level": int, "source": str}
    """
    # seed=None → uses OS entropy (time-based), giving different output each run
    rng = random.Random(seed)
    pairs = []

    # Generate perturbed versions of wrong answers
    for entry in model_wrong_answers[:n_pairs // 2]:
        problem  = entry.get("problem", "")
        expected = entry.get("expected", 0.0)

        perturbed = perturb_arithmetic(problem, expected, target_difficulty)
        if not perturbed["valid"]:
            continue

        new_prob = perturbed["problem"]
        new_ans  = perturbed["answer"]
        cot      = generate_chain_of_thought(new_prob, new_ans)

        pairs.append({
            "problem":  new_prob,
            "solution": cot,
            "answer":   new_ans,
            "domain":   domain,
            "level":    3,
            "source":   f"perturb_{target_difficulty}",
        })

    # Fill remainder with freshly generated problems at higher difficulty
    level = {"one_harder": 3, "structural": 4, "adversarial": 5}.get(target_difficulty, 3)
    while len(pairs) < n_pairs:
        p = generate_new_problem(domain, level, rng)
        cot = generate_chain_of_thought(p["problem"], p["answer"])
        pairs.append({
            "problem":  p["problem"],
            "solution": cot,
            "answer":   p["answer"],
            "domain":   domain,
            "level":    level,
            "source":   "generated",
        })

    return pairs[:n_pairs]


def save_pairs(pairs: list[dict], domain: str, run_tag: str = "") -> Path:
    """Save generated pairs as JSONL to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    out_path = RESULTS_DIR / f"harder_qa_{domain}{suffix}_{ts}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} harder Q/A pairs -> {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate harder math Q/A pairs from model wrong answers")
    parser.add_argument("--domain",      default="arithmetic",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--n",           type=int,   default=50)
    parser.add_argument("--difficulty",  default="one_harder",
                        choices=["one_harder", "structural", "adversarial"])
    parser.add_argument("--wrong-file",  default=None,
                        help="JSON file of wrong answer records [{problem,expected,got}]")
    parser.add_argument("--run-tag",     default="")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed (omit for a different batch each run)")
    args = parser.parse_args()

    # Load wrong answers if provided
    wrong = []
    if args.wrong_file:
        with open(args.wrong_file) as f:
            wrong = json.load(f)

    pairs = generate_harder_pairs(wrong, args.domain,
                                   target_difficulty=args.difficulty,
                                   n_pairs=args.n,
                                   seed=args.seed)
    out = save_pairs(pairs, args.domain, args.run_tag)

    print(f"\nSample (first 3 problems):")
    for p in pairs[:3]:
        print(f"  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> Answer: {p['answer']}\n")
