"""
eval_harness.py — Math Domain Evaluation Harness for ATTN11-x86 Kernel

Loads the compiled kernel.dll (or kernel.so) via ctypes, runs it on
math benchmark problems, and returns structured accuracy metrics.

Usage (direct):
    python eval_harness.py --domain gsm8k --n 100
    python eval_harness.py --domain algebra --n 200 --kernel ../kernel/kernel.dll

Usage (autoresearch loop):
    from math_lab.benchmarks.eval_harness import eval_math_domain
    result = eval_math_domain("kernel.dll", "gsm8k", num_problems=200)
"""

import os
import sys
import re
import time
import argparse
import ctypes
from ctypes import c_int32, c_float, POINTER, byref
from pathlib import Path
from typing import Optional

# ── Kernel ctypes interface ───────────────────────────────────────────────────

Q16_ONE  = 65536    # 1.0 in Q8.16
Q16_VOCAB = 131     # qsparser vocab size (char-level ASCII)

def load_kernel(kernel_path: str) -> ctypes.CDLL:
    """Load the ATTN11-x86 kernel shared library."""
    path = os.path.abspath(kernel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kernel not found: {path}")

    lib = ctypes.CDLL(path)

    # fxmath
    lib.fxmul.restype   = c_int32
    lib.fxmul.argtypes  = [c_int32, c_int32]
    lib.fxdiv.restype   = c_int32
    lib.fxdiv.argtypes  = [c_int32, c_int32]

    # qsparser
    lib.qsparse.restype  = c_int32
    lib.qsparse.argtypes = [ctypes.c_char_p, POINTER(c_int32), c_int32]
    lib.qsdecode.restype  = None
    lib.qsdecode.argtypes = [POINTER(c_int32), c_int32, ctypes.c_char_p, c_int32]

    return lib


def tokenize(lib: ctypes.CDLL, text: str, max_tokens: int = 512) -> list[int]:
    """Tokenize ASCII text using the kernel's qsparse function."""
    token_buf = (c_int32 * max_tokens)()
    n = lib.qsparse(text.encode("ascii", errors="replace"), token_buf, max_tokens)
    return list(token_buf[:n])


def decode(lib: ctypes.CDLL, token_ids: list[int]) -> str:
    """Decode token IDs back to string using kernel's qsdecode."""
    n = len(token_ids)
    token_arr = (c_int32 * n)(*token_ids)
    buf = ctypes.create_string_buffer(n + 1)
    lib.qsdecode(token_arr, n, buf, n + 1)
    return buf.value.decode("ascii", errors="replace")


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_number(text: str) -> Optional[float]:
    """
    Extract the last number from a model response string.
    Handles integers, decimals, negatives, fractions.
    """
    # Look for "#### N" pattern (GSM8K style)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    # Fall back: last number in text
    nums = re.findall(r"-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)", text)
    if nums:
        return float(nums[-1].replace(",", ""))
    return None


def answers_match(predicted: Optional[float], expected: Optional[float],
                  tol: float = 1e-3) -> bool:
    if predicted is None or expected is None:
        return False
    if abs(expected) < 1e-9:
        return abs(predicted) < tol
    return abs(predicted - expected) / (abs(expected) + 1e-9) < tol


# ── Benchmark loaders ─────────────────────────────────────────────────────────

def load_gsm8k(n: int) -> list[dict]:
    """Load n GSM8K problems. Downloads from HuggingFace if not cached."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        problems = []
        for row in ds:
            if len(problems) >= n:
                break
            # Extract numeric answer from "#### N" at end of answer
            ans_text = row["answer"]
            m = re.search(r"####\s*(-?[\d,]+)", ans_text)
            if m:
                expected = float(m.group(1).replace(",", ""))
                problems.append({
                    "problem": row["question"],
                    "answer": expected,
                    "domain": "gsm8k",
                })
        return problems
    except ImportError:
        print("WARNING: 'datasets' package not installed. Using synthetic problems.")
        return _synthetic_math(n, "gsm8k")


def load_math_domain(domain: str, n: int) -> list[dict]:
    """Load MATH dataset problems for a specific domain."""
    domain_map = {
        "algebra":      "algebra",
        "number_theory":"number_theory",
        "geometry":     "geometry",
        "combinatorics":"counting_and_probability",
    }
    hf_name = domain_map.get(domain, domain)
    try:
        from datasets import load_dataset
        ds = load_dataset("competition_math", split="test")
        problems = [
            {"problem": row["problem"], "answer": row["solution"], "domain": domain}
            for row in ds
            if row.get("type", "").lower().replace(" ", "_") == hf_name
        ][:n]
        return problems
    except Exception:
        return _synthetic_math(n, domain)


def _synthetic_math(n: int, domain: str) -> list[dict]:
    """Generate simple synthetic math problems (fallback when datasets unavailable)."""
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        ans = eval(f"{a} {op} {b}")
        problems.append({
            "problem": f"What is {a} {op} {b}?",
            "answer": float(ans),
            "domain": domain,
        })
    return problems


# ── Inference stub ────────────────────────────────────────────────────────────

def run_inference(lib: ctypes.CDLL, problem: str,
                  max_new_tokens: int = 20) -> str:
    """
    Run the integer attention kernel on a math problem.

    NOTE: This is the integration stub. Full inference requires:
      1. A trained weight file converted via turboquant
      2. attn_forward() wired to a language-model loop
      3. kvcache for autoregressive generation

    For now this uses the kernel's tokenizer to verify the pipeline
    and returns a placeholder response based on tokenization.
    The autoresearch agent will implement the full loop in train.py.
    """
    tokens = tokenize(lib, problem[:256])  # truncate to 256 chars
    # Placeholder: extract any digits already in the problem as the "answer"
    m = re.search(r"\d+", problem[::-1])   # last number in problem (naive)
    if m:
        return f"#### {m.group()[::-1]}"
    return "#### 0"


# ── Main eval function ────────────────────────────────────────────────────────

def eval_math_domain(
    kernel_path: str,
    domain: str,
    num_problems: int = 200,
    timeout_s: float = 120.0,
    verbose: bool = False,
) -> dict:
    """
    Evaluate a kernel variant on a math sub-domain.

    Returns a dict compatible with results.tsv logging:
    {
        "domain": str,
        "accuracy": float,
        "num_problems": int,
        "avg_tokens": float,
        "kernel_size_bytes": int,
        "wall_time_s": float,
    }
    """
    t0 = time.time()

    # Load kernel
    try:
        lib = load_kernel(kernel_path)
    except FileNotFoundError as e:
        return {"error": str(e), "accuracy": 0.0, "domain": domain}

    # Load problems
    if domain == "gsm8k":
        problems = load_gsm8k(num_problems)
    else:
        problems = load_math_domain(domain, num_problems)

    if not problems:
        return {"error": "no problems loaded", "accuracy": 0.0, "domain": domain}

    # Run inference + score
    correct = 0
    total   = 0
    total_tokens = 0

    for p in problems:
        if time.time() - t0 > timeout_s:
            break

        problem_text = p["problem"]
        expected     = p.get("answer")

        try:
            response  = run_inference(lib, problem_text)
            predicted = extract_number(response)
            total_tokens += len(tokenize(lib, problem_text))

            if isinstance(expected, (int, float)):
                match = answers_match(predicted, expected)
            else:
                # For non-numeric (MATH dataset), do string match
                match = str(predicted) in str(expected)

            correct += int(match)
            total   += 1

            if verbose:
                status = "✓" if match else "✗"
                print(f"  {status} pred={predicted}  exp={expected}  | {problem_text[:60]}...")

        except Exception as e:
            total += 1
            if verbose:
                print(f"  ERROR: {e}")

    kernel_size = os.path.getsize(kernel_path) if os.path.exists(kernel_path) else 0
    accuracy    = correct / total if total > 0 else 0.0
    wall_time   = time.time() - t0

    result = {
        "domain":            domain,
        "accuracy":          accuracy,
        "correct":           correct,
        "total":             total,
        "avg_tokens":        total_tokens / max(total, 1),
        "kernel_size_bytes": kernel_size,
        "wall_time_s":       wall_time,
    }
    return result


def format_tsv_row(commit: str, results: dict) -> str:
    """Format an eval result as a results.tsv row."""
    return (
        f"{commit}\t"
        f"{results.get('accuracy', 0.0):.6f}\t"
        f"{results.get('accuracy', 0.0):.6f}\t"    # placeholder for second domain
        f"{results.get('kernel_size_bytes', 0) / 1024:.1f}\t"
        f"{results.get('wall_time_s', 0):.1f}\t"
        f"pending\t"
        f"{results.get('domain', '?')} n={results.get('total', 0)}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math benchmark eval harness")
    parser.add_argument("--kernel", default="../kernel/kernel.dll",
                        help="Path to compiled kernel.dll or kernel.so")
    parser.add_argument("--domain", default="gsm8k",
                        choices=["gsm8k", "algebra", "number_theory",
                                 "geometry", "combinatorics"],
                        help="Math domain to evaluate")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of problems to evaluate")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\nEvaluating kernel: {args.kernel}")
    print(f"Domain: {args.domain}  |  n={args.n}\n")

    result = eval_math_domain(args.kernel, args.domain, args.n,
                              verbose=args.verbose)

    print(f"\n{'=' * 50}")
    print(f"Domain:    {result['domain']}")
    print(f"Accuracy:  {result.get('accuracy', 0):.2%}  "
          f"({result.get('correct', 0)}/{result.get('total', 0)})")
    print(f"Avg tokens:{result.get('avg_tokens', 0):.1f}")
    print(f"Kernel:    {result.get('kernel_size_bytes', 0) / 1024:.1f} KB")
    print(f"Time:      {result.get('wall_time_s', 0):.1f}s")
    print(f"{'=' * 50}\n")
