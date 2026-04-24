"""
claude_datagen.py — Generate math QA pairs via Claude on Vertex AI

Calls Claude with adaptive thinking to produce high-quality chain-of-thought
math problem/solution pairs for each domain. System prompt is prefix-cached.

Usage:
    python math_lab/claude_datagen.py --domain arithmetic --n 200
    python math_lab/claude_datagen.py --domain all --n 100
    python math_lab/claude_datagen.py --domain algebra --levels 3,4,5 --n 300

Environment:
    GOOGLE_CLOUD_PROJECT  — GCP project ID (or pass --project)
    ANTHROPIC_VERTEX_PROJECT_ID — alternative env var for project
    GOOGLE_CLOUD_REGION   — region (or pass --region, default us-east5)

Output: math_lab/results/claude_qa_{domain}_{timestamp}.jsonl
Each line: {"problem": "...", "solution": "...", "domain": "...", "level": N}
"""

import os
import re
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

DOMAINS = [
    "arithmetic",
    "algebra",
    "calculus",
    "geometry",
    "trigonometry",
    "number_theory",
    "combinatorics",
    "diffeq",
    "linalg",
    "linreg",
    "statistics",
    "stochastic",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert mathematics educator and problem designer.
Your task is to generate challenging, pedagogically sound math problems with
complete worked solutions.

Each problem must:
1. Be clearly and unambiguously stated
2. Have a single correct numerical or symbolic answer
3. Come with a full step-by-step solution showing all work
4. Match the specified difficulty level (1=elementary, 2=intermediate,
   3=advanced, 4=expert/olympiad, 5=research/frontier)

Solution format: Show all intermediate steps, then end with "#### <answer>"
on its own line (GSM8K-style). The answer should be the final numerical or
symbolic result.

Example:
{
  "problem": "If 3x + 7 = 22, find x.",
  "solution": "3x + 7 = 22\\n3x = 22 - 7 = 15\\nx = 15/3 = 5\\n#### 5",
  "level": 2
}

Output ONLY a valid JSON array. No prose before or after the array."""

DOMAIN_CONTEXT = {
    "arithmetic": "integer and decimal arithmetic: addition, subtraction, multiplication, division, order of operations, word problems",
    "algebra": "algebraic equations, polynomials, factoring, systems of equations, quadratics, inequalities, functions",
    "calculus": "limits, derivatives, integrals, chain rule, product rule, optimization, related rates, series",
    "geometry": "Euclidean geometry, areas, volumes, angles, similar triangles, coordinate geometry, conic sections",
    "trigonometry": "trig functions, identities, inverse trig, law of sines/cosines, polar coordinates, complex numbers",
    "number_theory": "divisibility, prime factorization, GCD, LCM, modular arithmetic, Diophantine equations, congruences",
    "combinatorics": "counting, permutations, combinations, pigeonhole principle, inclusion-exclusion, generating functions",
    "diffeq": "ODEs, separation of variables, integrating factors, linear ODEs, initial value problems, systems of ODEs",
    "linalg": "vectors, matrices, determinants, eigenvalues, linear transformations, vector spaces, orthogonality",
    "linreg": "regression equations, least squares, correlation, residuals, prediction intervals, multiple regression",
    "statistics": "probability, distributions, expected value, variance, hypothesis testing, confidence intervals, Bayes theorem",
    "stochastic": "random walks, Markov chains, Poisson processes, martingales, stochastic integrals",
}


def build_user_message(domain: str, level: int, n: int) -> str:
    ctx = DOMAIN_CONTEXT.get(domain, domain)
    return (
        f"Generate exactly {n} math problems for the domain: {domain} ({ctx}).\n"
        f"Difficulty level: {level} (1=elementary to 5=research-frontier).\n"
        f"Output a JSON array of {n} objects, each with keys: "
        f'"problem" (string), "solution" (string with steps ending #### answer), '
        f'"level" (integer {level}).\n'
        f"Vary the specific subtopics within {domain}. Make problems distinct."
    )


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

def make_client(project: str, region: str):
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: uv add anthropic")
        sys.exit(1)

    return anthropic.AnthropicVertex(region=region, project_id=project)


def generate_batch(client, domain: str, level: int, n: int,
                   model: str = "claude-opus-4-7",
                   max_retries: int = 3) -> list[dict]:
    """Call Claude and parse a JSON array of problems. Returns parsed list."""
    import anthropic

    user_msg = build_user_message(domain, level, n)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )

            # Extract text from response (skip thinking blocks)
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Parse JSON array - find outermost [ ... ]
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON array found in response")

            pairs = json.loads(match.group())
            if not isinstance(pairs, list):
                raise ValueError("Expected JSON array at top level")

            # Validate and annotate each record
            valid = []
            for p in pairs:
                if not isinstance(p, dict):
                    continue
                problem = str(p.get("problem", "")).strip()
                solution = str(p.get("solution", p.get("answer", ""))).strip()
                if problem and solution:
                    valid.append({
                        "problem": problem,
                        "solution": solution,
                        "domain": domain,
                        "level": int(p.get("level", level)),
                    })
            return valid

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Parse error on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                wait = 30 * attempt
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error on attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(5)

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate math QA pairs via Claude Vertex AI")
    parser.add_argument("--domain", default="arithmetic",
                        help=f"Domain or 'all'. Choices: {', '.join(DOMAINS)}")
    parser.add_argument("--n", type=int, default=100,
                        help="Total pairs per domain")
    parser.add_argument("--levels", default="1,2,3,4,5",
                        help="Comma-separated difficulty levels to generate (e.g. 3,4,5)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Problems per Claude call (10-20 is reliable)")
    parser.add_argument("--project", default=None,
                        help="GCP project ID (env: GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--region", default=None,
                        help="Vertex AI region (env: GOOGLE_CLOUD_REGION, default: us-east5)")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Claude model ID")
    args = parser.parse_args()

    # Resolve project / region
    project = (args.project
               or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
               or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if not project:
        print("ERROR: GCP project required. Set GOOGLE_CLOUD_PROJECT or pass --project")
        sys.exit(1)

    region = (args.region
              or os.environ.get("GOOGLE_CLOUD_REGION")
              or "us-east5")

    levels = [int(x) for x in args.levels.split(",")]
    domains = DOMAINS if args.domain == "all" else [args.domain]

    print(f"Claude datagen: model={args.model}  project={project}  region={region}")
    print(f"Domains: {domains}")
    print(f"Levels: {levels}  total_per_domain={args.n}  batch_size={args.batch_size}")
    print()

    client = make_client(project, region)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        print(f"── {domain} ────────────────────────────────")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"claude_qa_{domain}_{ts}.jsonl"
        all_pairs = []

        # Distribute n across levels evenly
        per_level = max(1, args.n // len(levels))
        for level in levels:
            collected = 0
            target = per_level
            while collected < target:
                batch_n = min(args.batch_size, target - collected)
                print(f"  L{level}: requesting {batch_n} problems...", end=" ", flush=True)
                batch = generate_batch(client, domain, level, batch_n, model=args.model)
                print(f"got {len(batch)}")
                all_pairs.extend(batch)
                collected += len(batch)
                if not batch:
                    print(f"  WARNING: empty batch, skipping remaining L{level}")
                    break
                time.sleep(0.5)  # brief pause between calls

        with open(out_path, "w", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"  Saved {len(all_pairs)} pairs -> {out_path}")
        print()


if __name__ == "__main__":
    main()
