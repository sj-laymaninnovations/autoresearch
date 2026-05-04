"""
distill_wordproblems.py — long-running synthetic-data generator.

Loops asking an Abacus teacher (default: gpt-4o-mini) to produce ONE
word-problem record per call across our 7 atomic skill types. Output goes
to a single appendable JSONL file with the same schema as our other skill
data — so it can be fed straight into finetune.py once enough records
accumulate.

Designed for unattended overnight / workday runs. Resilient to:
- transient teacher errors (handled by Teacher's retry loop)
- bad outputs (filtered by validation; failures logged but not fatal)
- crashes (output is append-only; rerun resumes silently)

Usage:
    python math_lab/datagen/distill_wordproblems.py \
        --out math_lab/results/skills/wordproblems.jsonl \
        --max-records 5000 \
        --model gpt-4o-mini

The output file gets one JSON object per line:
    {"problem": "...", "solution_cot": "...", "answer": "7",
     "concept": "word_problem_add_1d", "operator": "+",
     "stage": 3, "level": 3, "cot_source": "abacus_<model>"}
"""
from __future__ import annotations

import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher_abacus import Teacher


# ---------------------------------------------------------------------------
# Skill specs — what to ask the teacher for each atomic skill
# ---------------------------------------------------------------------------

SKILL_SPECS = {
    "add_1d": {
        "operator": "+",
        "constraint": "Use exactly two single-digit numbers (each 0-9). Result is 0..18.",
        "examples": "3 + 4, 7 + 5, 9 + 9",
    },
    "add_2d": {
        "operator": "+",
        "constraint": "Use exactly two two-digit numbers (each 10-99). Result is 20..198. Either with or without carry.",
        "examples": "23 + 45, 58 + 47, 19 + 33",
    },
    "sub_2d_no_borrow": {
        "operator": "-",
        "constraint": "Subtract two two-digit numbers (10-99 each) with the larger first AND the ones-digit of the larger >= ones-digit of the smaller (no borrow needed). Result is positive.",
        "examples": "67 - 23, 98 - 41, 75 - 32",
    },
    "sub_2d_borrow": {
        "operator": "-",
        "constraint": "Subtract two two-digit numbers (10-99 each) with the larger first AND the ones-digit of the larger < ones-digit of the smaller (borrow IS needed). Result is positive.",
        "examples": "53 - 27, 82 - 39, 91 - 47",
    },
    "mul_1d": {
        "operator": "*",
        "constraint": "Multiply two single-digit numbers (each 0-9).",
        "examples": "7 * 8, 3 * 9, 6 * 6",
    },
    "mul_2d": {
        "operator": "*",
        "constraint": "Multiply two two-digit numbers (each 10-99). Result is up to 9801.",
        "examples": "12 * 34, 47 * 23, 65 * 89",
    },
    "div_1d": {
        "operator": "/",
        "constraint": "Clean division: divisor d in [2, 99], quotient q in [0, 9], dividend a = q*d. So a/d gives an exact single-digit answer.",
        "examples": "24 / 6, 81 / 9, 0 / 5",
    },
}


PROMPT_TEMPLATE = """\
You are generating ONE math word problem for a small student model.

Skill required: **{skill}** ({operator})
{constraint}

Output STRICT JSON with exactly these three keys (no extra text):
{{"problem": "<one short word problem, 1-2 sentences>",
 "cot": "<2-5 lines of natural reasoning that ENDS with a line in this exact form: '#### N' where N is the numeric answer>",
 "answer": "<just the integer answer as a string, no units>"}}

Variety: invent a fresh scenario (apples / marbles / coins / books / dogs / players / kilometers / minutes / pages / etc.). Avoid making the same problem twice. Vary the names, objects, and verbs each time.

Math must be CORRECT. Numbers in the problem must match the worked CoT. The numeric answer in the '#### N' line MUST equal the answer field. Do NOT include units or commas in the answer. Operands within the constraint range only.

Example (different skill, just for format):
{example_json}

Now generate a fresh problem for skill **{skill}**:
"""


EXAMPLE_JSON = ('{"problem": "Tom has 5 marbles and finds 3 more in the grass. '
                'How many marbles does Tom have now?", '
                '"cot": "Tom starts with 5.\\nHe finds 3 more.\\n5 + 3 = 8\\n#### 8", '
                '"answer": "8"}')


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r"####\s*(-?\d+)\s*$", re.MULTILINE)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(content: str) -> dict | None:
    """Extract & validate the JSON record. Returns None on failure."""
    m = JSON_RE.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in obj for k in ("problem", "cot", "answer")):
        return None
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in ("problem", "cot", "answer")):
        return None
    # Pull the #### N from the CoT and confirm it matches `answer`.
    am = ANSWER_RE.search(obj["cot"])
    if not am:
        return None
    cot_ans = am.group(1).strip()
    declared_ans = obj["answer"].strip()
    # Strip leading + / zeros for comparison
    try:
        if int(cot_ans) != int(declared_ans):
            return None
    except ValueError:
        return None
    return obj


def to_record(obj: dict, skill: str, model: str) -> dict:
    spec = SKILL_SPECS[skill]
    return {
        "problem": obj["problem"].strip(),
        "solution_cot": obj["cot"].strip(),
        "answer": obj["answer"].strip(),
        "answer_real": int(obj["answer"]),
        "concept": f"word_problem_{skill}",
        "operator": spec["operator"],
        "stage": 3, "level": 3,
        "cot_source": f"abacus_{model}",
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output JSONL file (append-only)")
    ap.add_argument("--max-records", type=int, default=5000,
                    help="stop after this many ACCEPTED records")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="teacher model id (any from RouteLLM /v1/models)")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="higher = more diverse problems")
    ap.add_argument("--cooldown", type=float, default=0.4,
                    help="seconds between calls (rate-limit hygiene)")
    ap.add_argument("--skills", nargs="+", default=list(SKILL_SPECS.keys()),
                    help="restrict to these skills")
    ap.add_argument("--report-every", type=int, default=20,
                    help="emit a status line every N accepted records")
    args = ap.parse_args()

    skills = [s for s in args.skills if s in SKILL_SPECS]
    if not skills:
        raise SystemExit(f"No valid skills selected. Available: {list(SKILL_SPECS.keys())}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count existing lines if file already exists
    existing = 0
    if out_path.exists():
        existing = sum(1 for _ in out_path.open() if _.strip())
        print(f"  resuming: {existing} records already in {out_path.name}")

    target = args.max_records
    if existing >= target:
        print(f"  already at target ({existing} ≥ {target}); nothing to do")
        return

    teacher = Teacher(model=args.model)
    print(f"  teacher : {args.model}  via Abacus RouteLLM")
    print(f"  skills  : {len(skills)} ({', '.join(skills)})")
    print(f"  target  : {target} records  (currently {existing})")
    print(f"  out     : {out_path}")
    print()

    rng = random.Random()
    accepted = existing
    rejected = 0
    skill_counts = {s: 0 for s in skills}
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as f:
        while accepted < target:
            skill = rng.choice(skills)
            spec = SKILL_SPECS[skill]
            prompt = PROMPT_TEMPLATE.format(
                skill=skill, operator=spec["operator"],
                constraint=spec["constraint"],
                example_json=EXAMPLE_JSON,
            )
            try:
                content = teacher.chat([
                    {"role": "user", "content": prompt}
                ], temperature=args.temperature, max_tokens=400)
            except Exception as e:
                rejected += 1
                print(f"  [error] {skill}: {str(e)[:120]}", flush=True)
                time.sleep(min(30.0, args.cooldown * 8))
                continue

            obj = parse_response(content)
            if obj is None:
                rejected += 1
                if rejected % 10 == 1:
                    print(f"  [reject] {skill}: bad/invalid output (last 80c): {content[-80:]!r}", flush=True)
                time.sleep(args.cooldown)
                continue

            record = to_record(obj, skill, args.model)
            f.write(json.dumps(record) + "\n")
            f.flush()
            accepted += 1
            skill_counts[skill] += 1

            if accepted % args.report_every == 0:
                elapsed = time.time() - t0
                rate = (accepted - existing) / max(elapsed, 1e-3)
                eta_s = (target - accepted) / max(rate, 1e-3)
                eta_m = eta_s / 60
                dist = " ".join(f"{s}={skill_counts[s]}" for s in skills)
                print(f"  [{accepted:5d}/{target}] reject={rejected}  "
                      f"rate={rate:.2f}/s  eta={eta_m:.1f}m  | {dist}",
                      flush=True)

            time.sleep(args.cooldown)

    print()
    print(f"  done — {accepted} records, {rejected} rejected")
    print(f"  per-skill: {skill_counts}")


if __name__ == "__main__":
    main()
