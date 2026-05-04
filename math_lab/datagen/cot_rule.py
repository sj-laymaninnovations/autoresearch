"""
cot_rule.py — Rule-based CoT generator for arithmetic training data

Deterministic algorithmic decomposition for each (op, tier) combo. Uniform
coverage, perfect format, no teacher needed. Used as fallback when teacher
LLM (cot_teacher.py via LM Studio) isn't available.

Decomposition recipes:
  add: place-value (a + round_b + remainder)
  sub: place-value (a - round_b - remainder)
  mul: distributive (a * (round_b + remainder))
  div: divisor*quotient = dividend (since our div is backward-generated)

Output JSONL with same fields as input + solution_cot field.
"""

import argparse
import json
from pathlib import Path


def add_cot(a: int, b: int) -> str:
    """38 + 32 → 38+30=68, 68+2=70, #### 70"""
    ans = a + b
    if b < 10 or b % 10 == 0:
        # Single direct step suffices
        return f"{a}+{b}={ans}\n#### {ans}"
    round_b = (b // 10) * 10
    rem = b - round_b
    intermediate = a + round_b
    return f"{a}+{round_b}={intermediate}\n{intermediate}+{rem}={ans}\n#### {ans}"


def sub_cot(a: int, b: int) -> str:
    """89 - 52 → 89-50=39, 39-2=37, #### 37"""
    ans = a - b
    if b < 10 or b % 10 == 0:
        return f"{a}-{b}={ans}\n#### {ans}"
    round_b = (b // 10) * 10
    rem = b - round_b
    intermediate = a - round_b
    return f"{a}-{round_b}={intermediate}\n{intermediate}-{rem}={ans}\n#### {ans}"


def mul_cot(a: int, b: int) -> str:
    """82 * 37 → 82*37 = 82*(30+7) = 82*30 + 82*7 = 2460 + 574 = 3034, #### 3034"""
    ans = a * b
    if b < 10 or a < 10 or b % 10 == 0:
        return f"{a}*{b}={ans}\n#### {ans}"
    round_b = (b // 10) * 10
    rem = b - round_b
    p1 = a * round_b
    p2 = a * rem
    return (f"{a}*{b}={a}*({round_b}+{rem})\n"
            f"={a}*{round_b}+{a}*{rem}\n"
            f"={p1}+{p2}\n"
            f"={ans}\n"
            f"#### {ans}")


def div_cot(dividend: int, divisor: int, quotient: int) -> str:
    """288 / 12 = 24 → 12*24=288 so 288/12=24, #### 24"""
    return f"{divisor}*{quotient}={dividend}\n{dividend}/{divisor}={quotient}\n#### {quotient}"


def make_cot(item: dict) -> str:
    """Pick the right recipe based on the operator in the problem string."""
    prob = item['problem'].strip()
    answer = int(item['answer_real'])

    # Parse like "12 + 34" or "1716 / 66"
    parts = prob.split()
    if len(parts) != 3:
        # Couldn't parse; fall back to direct equation
        return f"{prob}={answer}\n#### {answer}"
    a_str, op, b_str = parts
    a = int(a_str)
    b = int(b_str)

    if op == '+':
        return add_cot(a, b)
    elif op == '-':
        return sub_cot(a, b)
    elif op == '*':
        return mul_cot(a, b)
    elif op == '/':
        return div_cot(a, b, answer)
    else:
        return f"{prob}={answer}\n#### {answer}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    items = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    out_f = open(args.output, 'w')
    n_ok = 0
    for it in items:
        try:
            cot = make_cot(it)
            new_it = dict(it)
            new_it['solution_cot'] = cot
            new_it['cot_source'] = 'rule_based'
            out_f.write(json.dumps(new_it) + '\n')
            n_ok += 1
        except Exception as e:
            print(f"  skip {it.get('problem','?')}: {e}")
    out_f.close()
    print(f"Generated rule-based CoT for {n_ok}/{len(items)} pairs")
    print(f"Output: {args.output}")

    # Show samples
    print("\nSamples:")
    with open(args.output) as f:
        for i, line in enumerate(f):
            if i >= 5: break
            d = json.loads(line)
            print(f"  Q: {d['problem']:14s}")
            for cot_line in d['solution_cot'].split('\n'):
                print(f"     {cot_line}")
            print()


if __name__ == "__main__":
    main()
