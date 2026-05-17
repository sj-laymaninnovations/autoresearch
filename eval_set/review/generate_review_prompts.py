"""
generate_review_prompts.py — Task 6: build LLM review prompts.
Usage: python review/generate_review_prompts.py
"""
import json, pathlib, sys

PROBLEMS_DIR = pathlib.Path(__file__).parent.parent / "problems"
OUT_DIR      = pathlib.Path(__file__).parent

PREAMBLE = """You are acting as an independent mathematical reviewer for a held-out evaluation set.

Instructions:
1. For each problem below, solve it INDEPENDENTLY — do not look at any answer key.
2. Write your solution and final answer clearly.
3. After solving, note if any problem is:
   (a) Ambiguous or ill-posed (multiple valid interpretations)
   (b) Has an answer you consider incorrect or incomplete
   (c) Appears to be a problem you recognize from a published source or training dataset
      (flag the source if known)
4. Do NOT guess — if you cannot solve a problem, say so explicitly.

Format your response for each problem as:
---
PROBLEM ID: [ID]
MY ANSWER: [your computed answer]
WORKING: [brief step-by-step]
FLAGS: [none | ambiguous | answer-suspect | contamination-risk: {description}]
---

The recorded answers will be compared to yours after you submit. We are looking for
problems where your answer differs from ours — this is a quality-control check.
"""

def load_problems():
    problems = []
    for p in sorted(PROBLEMS_DIR.glob("EVAL-*.json")):
        with open(p) as f:
            prob = json.load(f)
        if prob.get("status") == "needs_verification":
            continue  # skip unfilled skeletons
        problems.append(prob)
    return problems

def format_problem(prob: dict) -> str:
    lines = [
        f"PROBLEM {prob['id']} [{prob.get('category','?')} / {prob.get('difficulty_estimate','?')}]",
        prob["problem_text"],
        f"(Source: {prob['source']})",
        "",
    ]
    return "\n".join(lines)

def build_prompt(model_name: str, problems: list[dict]) -> str:
    header = f"# MathGPT Eval Set — LLM Review ({model_name})\n\n"
    body = PREAMBLE + "\n\n---\n\n"
    body += "## Problems\n\n"
    for prob in problems:
        body += format_problem(prob) + "\n"
    return header + body

def main():
    problems = load_problems()
    if not problems:
        print("No verified problems found. Check problems/ directory.")
        sys.exit(1)
    print(f"Generating review prompts for {len(problems)} verified problems...")
    for model in ("chatgpt", "gemini", "claude"):
        prompt = build_prompt(model.capitalize(), problems)
        out = OUT_DIR / f"review_prompt_{model}.md"
        out.write_text(prompt, encoding="utf-8")
        print(f"  wrote {out}")

if __name__ == "__main__":
    main()
