"""
generate_targeted_data.py — Generate targeted training examples for the
four problem categories where v2-25M is failing.

Strategy:
  1. Generate problem parameters in Python (guarantees correct answers)
  2. Ask Mathstral (via LM Studio) to write step-by-step working
  3. Verify #### answer matches our computed answer
  4. Save verified records in canonical format

Categories:
  A. divisors     — (e1+1)(e2+1)... formula, must show zero exponent
  B. word_price   — two-variable price/quantity system of equations
  C. rectangle    — P=2(l+w) + quadratic from perimeter+area
  D. work_rates   — rate=1/time principle, rates add

Usage:
  python generate_targeted_data.py --category all --n 100 --out external_data/targeted_v1.jsonl
  python generate_targeted_data.py --category divisors --n 50
"""

from __future__ import annotations
import argparse, json, random, re, math, sys, time
from pathlib import Path
import requests

random.seed(42)

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "mathstral-7b-v0.1"

# ── LM Studio call ────────────────────────────────────────────────────────────

def call_mathstral(system: str, user: str, max_tokens: int = 500) -> str:
    """Call Mathstral via LM Studio OpenAI-compatible API."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    try:
        r = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


# ── Problem generators ────────────────────────────────────────────────────────

def make_divisors_problem() -> dict | None:
    """Generate a divisors counting problem with 2-4 prime factors."""
    primes = [2, 3, 5, 7, 11, 13]
    n_primes = random.choice([2, 2, 3, 3, 4])
    chosen = random.sample(primes, n_primes)
    exponents = [random.randint(1, 4) for _ in range(n_primes)]
    answer = math.prod(e + 1 for e in exponents)

    # Build number representation like "2^3 * 5^2 * 7"
    parts = []
    for p, e in zip(chosen, exponents):
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    num_str = " * ".join(parts)

    problem = f"Find the number of positive divisors of {num_str}."

    # Build explicit factor string so Mathstral knows the target
    factor_parts = " * ".join(f"({e}+1)" for e in exponents)
    system = (
        f"You are a precise math tutor. Write a complete step-by-step solution.\n"
        f"The correct answer is {answer}. Write working that arrives at {answer}.\n"
        f"CRITICAL RULES:\n"
        f"1. State the divisor formula: if n = p1^e1 * p2^e2 * ..., "
        f"   then number of divisors = (e1+1)(e2+1)...\n"
        f"2. For each prime p^e, list ALL exponents from 0 to e: "
        f"   '0, 1, 2, ..., e' — that gives (e+1) choices.\n"
        f"3. Show the multiplication explicitly: {factor_parts} = {answer}\n"
        f"4. End with exactly: #### {answer}\n"
        f"Do NOT use 'The answer is'. End with #### only."
    )
    user = f"Q: {problem}\nA: "

    response = call_mathstral(system, user)
    if response.startswith("ERROR"): return None

    normed = verify_and_normalize(response, str(answer))
    if normed is None:
        return None
    response = normed  # reject if Mathstral got it wrong

    return {
        "problem": problem,
        "solution": response,
        "solution_cot": response,
        "answer": str(answer),
        "answer_real": answer,
        "concept": "divisors_counting",
        "stage": 3,
        "level": 3,
        "source": "generated_targeted_v1",
        "category": "divisors",
    }


def make_word_price_problem() -> dict | None:
    """Generate a two-item price/quantity word problem."""
    items = [
        ("apple", "orange"), ("pen", "pencil"), ("cookie", "brownie"),
        ("mango", "banana"), ("notebook", "eraser"), ("stamp", "sticker"),
    ]
    item1, item2 = random.choice(items)

    # Generate prices and quantities with integer solution
    # a*p1 + (total-a)*p2 = total_cost  →  a = (total_cost - total*p2) / (p1-p2)
    total = random.randint(8, 15)
    p2_cents = random.choice([50, 60, 75, 80, 90])      # cheaper item
    p1_cents = p2_cents + random.choice([25, 30, 40, 50])  # more expensive item
    # Pick a (number of item1) to be a nice integer
    a = random.randint(2, total - 2)
    total_cost_cents = a * p1_cents + (total - a) * p2_cents
    if total_cost_cents % 100 != 0:
        # Adjust total_cost to be whole dollars
        total_cost_cents = ((total_cost_cents // 100) + 1) * 100
        a = (total_cost_cents - total * p2_cents) // (p1_cents - p2_cents)
        if a <= 0 or a >= total:
            return None

    p1 = p1_cents / 100
    p2 = p2_cents / 100
    total_cost = total_cost_cents / 100
    answer = a

    problem = (
        f"A store sells {item1}s for ${p1:.2f} each and {item2}s for ${p2:.2f} each. "
        f"A customer spends exactly ${total_cost:.2f} buying a total of {total} items. "
        f"How many {item1}s did the customer buy?"
    )

    b = total - answer  # number of item2s
    cost_eq = f"{p1:.2f}*a + {p2:.2f}*({total}-a) = {total_cost:.2f}"
    system = (
        f"You are a precise math tutor. Write a complete step-by-step solution.\n"
        f"The correct answer is {answer}. Write working that arrives at {answer}.\n"
        f"CRITICAL RULES:\n"
        f"1. Define: 'Let a = number of {item1}s, then ({total}-a) = number of {item2}s'\n"
        f"2. Write the cost equation: {cost_eq}\n"
        f"3. Solve step by step to get a = {answer}\n"
        f"4. End with exactly: #### {answer}\n"
        f"Do NOT use 'The answer is'. End with #### only."
    )
    user = f"Q: {problem}\nA: "

    response = call_mathstral(system, user)
    if response.startswith("ERROR"): return None

    normed = verify_and_normalize(response, str(answer))
    if normed is None:
        return None
    response = normed

    return {
        "problem": problem,
        "solution": response,
        "solution_cot": response,
        "answer": str(answer),
        "answer_real": answer,
        "concept": "word_problem_price",
        "stage": 2,
        "level": 2,
        "source": "generated_targeted_v1",
        "category": "word_price",
    }


def make_rectangle_problem() -> dict | None:
    """Generate a rectangle perimeter+area → longer side problem."""
    # Choose l > w with integer l, w
    w = random.randint(4, 15)
    l = random.randint(w + 1, w + 15)
    perimeter = 2 * (l + w)
    area = l * w
    answer = l  # longer side

    problem = (
        f"A rectangle has perimeter {perimeter} and area {area}. "
        f"What is the length of the longer side?"
    )

    s = l + w  # sum of sides = perimeter/2
    system = (
        f"You are a precise math tutor. Write a complete step-by-step solution.\n"
        f"The correct answer is {answer}. Write working that arrives at {answer}.\n"
        f"CRITICAL RULES:\n"
        f"1. State: perimeter = 2*(l+w), so l+w = {perimeter}/2 = {s}\n"
        f"2. State: l*w = {area}\n"
        f"3. Set up the quadratic: x^2 - {s}x + {area} = 0\n"
        f"4. Solve it and identify the larger root as the longer side: {answer}\n"
        f"5. End with exactly: #### {answer}\n"
        f"Do NOT use 'The answer is'. End with #### only."
    )
    user = f"Q: {problem}\nA: "

    response = call_mathstral(system, user)
    if response.startswith("ERROR"): return None

    normed = verify_and_normalize(response, str(answer))
    if normed is None:
        return None
    response = normed

    return {
        "problem": problem,
        "solution": response,
        "solution_cot": response,
        "answer": str(answer),
        "answer_real": answer,
        "concept": "rectangle_perimeter_area",
        "stage": 3,
        "level": 3,
        "source": "generated_targeted_v1",
        "category": "rectangle",
    }


def make_work_rate_problem() -> dict | None:
    """Generate a work rates problem where rates add."""
    # A alone = a days, B alone = b days
    # Together = T where 1/T = 1/a + 1/b
    a = random.randint(4, 20)
    b = random.randint(4, 20)
    if a == b: return None

    # Together time T = ab/(a+b)
    num = a * b
    den = a + b
    g = math.gcd(num, den)
    T_num, T_den = num // g, den // g

    if T_den == 1:
        problem_type = "integer"
        answer = T_num
        problem = (
            f"Worker A can complete a job alone in {a} days. "
            f"Worker B can complete the same job alone in {b} days. "
            f"How many days will it take them to complete the job working together?"
        )
        answer_str = str(answer)
    else:
        problem_type = "fraction"
        answer = T_num + T_den  # p+q form
        problem = (
            f"Worker A can complete a job alone in {a} days. "
            f"Worker B can complete the same job alone in {b} days. "
            f"Working together, they finish in p/q days where p/q is in lowest terms. "
            f"Find p+q."
        )
        answer_str = str(answer)

    system = (
        f"You are a precise math tutor. Write a complete step-by-step solution.\n"
        f"The correct answer is {answer_str}. Write working that arrives at {answer_str}.\n"
        f"CRITICAL RULES:\n"
        f"1. State: if a worker takes d days alone, their RATE is 1/d jobs per day\n"
        f"2. Combined rate = 1/{a} + 1/{b} = {den}/({a}*{b}) = {den}/{num}\n"
        f"3. Time together = {num}/{den} days = {T_num}/{T_den} days (reduced)\n"
        f"4. Show all fraction arithmetic step by step\n"
        f"5. End with exactly: #### {answer_str}\n"
        f"Do NOT use 'The answer is'. End with #### only."
    )
    user = f"Q: {problem}\nA: "

    response = call_mathstral(system, user)
    if response.startswith("ERROR"): return None

    normed = verify_and_normalize(response, str(answer))
    if normed is None:
        return None
    response = normed

    return {
        "problem": problem,
        "solution": response,
        "solution_cot": response,
        "answer": answer_str,
        "answer_real": answer,
        "concept": "work_rates",
        "stage": 3,
        "level": 3,
        "source": "generated_targeted_v1",
        "category": "work_rates",
    }


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract final answer — checks ####, boxed, or last number in text."""
    m = re.search(r'####\s*([^\n]{1,30})', text)
    if m: return m.group(1).strip()
    # Last \boxed{} in the text
    boxes = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxes: return boxes[-1].strip()
    # Last integer in the text
    nums = re.findall(r'\b(\d+)\b', text)
    return nums[-1] if nums else None


def verify_and_normalize(response: str, expected: str) -> str | None:
    """Return normalized response with #### appended if answer is correct,
    or None if Mathstral got it wrong."""
    extracted = extract_answer(response)
    if extracted is None: return None
    # Normalize: strip whitespace, compare as strings
    if str(extracted).strip() != str(expected).strip():
        return None
    # If #### not already present, append it
    if '####' not in response:
        return response.rstrip() + f'\n#### {expected}'
    return response


# ── Main ───────────────────────────────────────────────────────────────────────

GENERATORS = {
    'divisors':   make_divisors_problem,
    'word_price': make_word_price_problem,
    'rectangle':  make_rectangle_problem,
    'work_rates': make_work_rate_problem,
}


def generate_category(category: str, n: int, out_path: Path,
                       existing: list | None = None) -> list[dict]:
    gen_fn = GENERATORS[category]
    records = list(existing or [])
    seen_problems = {r['problem'] for r in records}  # dedup against pre-loaded
    attempts = 0
    target = (len(records) + n)

    print(f"\n[{category}] Generating {n} examples (have {len(records)} already)...")

    while len(records) < target:
        attempts += 1
        rec = gen_fn()
        if rec is None:
            if attempts % 10 == 0:
                print(f"  {len(records)}/{target}  (attempts={attempts}, "
                      f"accept_rate={len(records)/max(attempts,1)*100:.0f}%)", flush=True)
            continue

        if rec['problem'] in seen_problems:
            continue  # skip duplicate problem statement
        seen_problems.add(rec['problem'])
        records.append(rec)
        pct = len(records) / target * 100
        print(f"  [{pct:5.1f}%] ✓  {rec['problem'][:70]}  → #### {rec['answer']}", flush=True)

        # Save incrementally
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

        if attempts > n * 10:
            print(f"  Stopping: too many attempts ({attempts}) for {len(records)} records")
            break

    print(f"  Done: {len(records)} examples, {attempts} attempts "
          f"({len(records)/max(attempts,1)*100:.0f}% accept rate)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='all',
                    choices=['all'] + list(GENERATORS.keys()))
    ap.add_argument('--n', type=int, default=100,
                    help='Target examples per category')
    ap.add_argument('--out', type=Path,
                    default=Path('external_data/targeted_training_v1.jsonl'))
    ap.add_argument('--test', action='store_true',
                    help='Generate just 3 per category to test connection')
    args = ap.parse_args()

    n = 3 if args.test else args.n
    categories = list(GENERATORS.keys()) if args.category == 'all' else [args.category]

    # Check LM Studio is running
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        models = [m['id'] for m in r.json().get('data', [])]
        if MODEL not in models:
            print(f"WARNING: {MODEL} not found. Available: {models[:3]}")
        else:
            print(f"LM Studio connected. Using {MODEL}")
    except Exception as e:
        print(f"ERROR: LM Studio not reachable: {e}")
        sys.exit(1)

    all_records = []
    t0 = time.time()

    for cat in categories:
        out_cat = args.out.parent / f"targeted_{cat}.jsonl"
        # Load existing if any
        existing = []
        if out_cat.exists():
            existing = [json.loads(l) for l in open(out_cat)]
            print(f"  Loaded {len(existing)} existing {cat} records")
        recs = generate_category(cat, n, out_cat, existing)
        all_records.extend(recs)

    # Write combined file
    if len(categories) > 1:
        with open(args.out, 'w') as f:
            for r in all_records:
                f.write(json.dumps(r) + '\n')
        print(f"\nCombined → {args.out}  ({len(all_records)} total records)")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min  ({len(all_records)} records)")
    print(f"\nTo add to training pipeline:")
    print(f"  1. Append {args.out} to external_data/mathgpt_train_v1.jsonl")
    print(f"  2. Re-run build_mathgpt_dataset.py")
    print(f"  3. Retrain or continue from current checkpoint with augmented data")


if __name__ == '__main__':
    main()
