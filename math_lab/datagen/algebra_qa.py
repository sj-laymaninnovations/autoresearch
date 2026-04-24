"""
algebra_qa.py — Algebra Q/A Generator for Math Training

Generates algebra problems with explicit Chain-of-Thought solutions.
All answers are exact integers or simple fractions. Backward generation
ensures clean answers.

Topics by level:
  L1: Linear equations, word problems, percent, proportion
  L2: Two-variable systems, quadratic roots, distance/rate/time, age
  L3: Quadratic formula, 3-var system, absolute value, mixture
  L4: Polynomial eval, inequality, function composition, rational equations
  L5: Vieta's, sequences, complex numbers, floor equations, nested functions

Output: math_lab/results/harder_qa_algebra_<tag>_<timestamp>.jsonl

References:
  - Khan Academy Algebra 1 & 2 curriculum
  - AMC 8/10 algebra problems (structure inspiration)
  - MATH dataset Algebra split (Hendrycks et al. 2021)
"""

import json
import math
import random
import datetime
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path(__file__).parent.parent / "results"
MAX_CONTENT_LEN = 480   # chars; seq_len=512 in finetune.py gives comfortable margin


def _ok(problem: str, solution: str) -> bool:
    return len(problem) + len(solution) <= MAX_CONTENT_LEN


def _frac(p: int, q: int) -> str:
    f = Fraction(p, q)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ── L1: Linear equations, word problems, percent, proportion ─────────────────

def linear_one_var(rng: random.Random, level: int) -> dict:
    """ax + b = c  ->  x = (c-b)/a"""
    a = rng.randint(1, 5 * level)
    x_true = rng.randint(-10 * level, 10 * level)
    b = rng.randint(-20 * level, 20 * level)
    c = a * x_true + b
    rhs = c - b
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    problem = f"Solve for x: {a}x {b_str} = {c}"
    solution = (f"{a}x = {c} - ({b}) = {rhs}\n"
                f"x = {rhs} / {a} = {x_true}\n"
                f"#### {x_true}")
    return {"problem": problem, "solution": solution, "answer": x_true,
            "domain": "algebra_linear", "level": level, "source": "algebra_qa"}


def linear_word(rng: random.Random, level: int) -> dict:
    """Unit price word problem."""
    unit = rng.randint(2, 8 * level)
    count = rng.randint(2, 10)
    total = unit * count
    ask = rng.randint(1, 8)
    answer = unit * ask
    problem = f"{count} items cost ${total}. How much do {ask} items cost?"
    solution = (f"unit cost = {total} / {count} = {unit}\n"
                f"{ask} * {unit} = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_linear", "level": level, "source": "algebra_qa"}


def percent_problem(rng: random.Random, level: int) -> dict:
    """p% of total = amount; find p or amount."""
    total = rng.randint(50, 500) * level
    p = rng.randint(5, 95)
    amount = total * p // 100
    # ask for amount given p and total
    problem = f"What is {p}% of {total}?"
    solution = (f"{p}/100 * {total}\n"
                f"= {p} * {total} / 100\n"
                f"= {amount}\n"
                f"#### {amount}")
    return {"problem": problem, "solution": solution, "answer": amount,
            "domain": "algebra_percent", "level": level, "source": "algebra_qa"}


def proportion(rng: random.Random, level: int) -> dict:
    """a/b = c/x -> x = b*c/a (integer answer)."""
    a = rng.randint(1, 5 * level)
    x = rng.randint(1, 10 * level)
    c = rng.randint(1, 5 * level)
    b = a * x // c
    if b == 0 or a * x != b * c:
        # ensure clean integer answer
        b = a
        x = c
    answer = x
    problem = f"Solve: {a}/{b} = {c}/x. Find x."
    solution = (f"Cross multiply: {a} * x = {b} * {c}\n"
                f"{a}x = {b * c}\n"
                f"x = {b * c} / {a} = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_proportion", "level": level, "source": "algebra_qa"}


# ── L2: Two-variable system, quadratic roots, distance/rate/time, age ────────

def system_two_var(rng: random.Random, level: int) -> dict:
    """Elimination on 2x2 system; answer is x+y."""
    x = rng.randint(-5 * level, 5 * level)
    y = rng.randint(-5 * level, 5 * level)
    a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
    a2, b2 = rng.randint(1, 4), rng.randint(1, 4)
    while a1 * b2 == a2 * b1:
        a2, b2 = rng.randint(1, 4), rng.randint(1, 4)
    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y
    coeff_y = a2 * b1 - a1 * b2
    rhs_y   = a2 * c1 - a1 * c2
    problem = f"Solve: {a1}x+{b1}y={c1}, {a2}x+{b2}y={c2}. Find x+y."
    solution = (f"Multiply eq1 by {a2}, eq2 by {a1} and subtract:\n"
                f"{coeff_y}y = {rhs_y} -> y = {y}\n"
                f"Substitute y={y}: x = {x}\n"
                f"x+y = {x+y}\n"
                f"#### {x+y}")
    return {"problem": problem, "solution": solution, "answer": x + y,
            "domain": "algebra_system2", "level": level, "source": "algebra_qa"}


def quadratic_roots(rng: random.Random, level: int) -> dict:
    """x^2 + bx + c = 0 with integer roots; ask for product."""
    r1 = rng.randint(-6 * level, 6 * level)
    r2 = rng.randint(-6 * level, 6 * level)
    b = -(r1 + r2)
    c = r1 * r2
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    problem = f"Solve: x^2 {b_str}x {c_str} = 0. Product of roots?"
    solution = (f"Factor: (x - {r1})(x - {r2}) = 0\n"
                f"Roots: x = {r1}, x = {r2}\n"
                f"Product = {r1} * {r2} = {r1*r2}\n"
                f"#### {r1*r2}")
    return {"problem": problem, "solution": solution, "answer": r1 * r2,
            "domain": "algebra_quadratic", "level": level, "source": "algebra_qa"}


def distance_rate_time(rng: random.Random, level: int) -> dict:
    """d = r*t; given two, find the third."""
    r = rng.randint(10, 20 * level)
    t = rng.randint(1, 8 * level)
    d = r * t
    choice = rng.randint(0, 2)
    if choice == 0:
        problem = f"A car travels {r} mph for {t} hours. Total distance?"
        solution = f"d = r * t = {r} * {t} = {d}\n#### {d}"
        answer = d
    elif choice == 1:
        problem = f"A car travels {d} miles at {r} mph. Time in hours?"
        solution = f"t = d / r = {d} / {r} = {t}\n#### {t}"
        answer = t
    else:
        problem = f"A car covers {d} miles in {t} hours. Speed in mph?"
        solution = f"r = d / t = {d} / {t} = {r}\n#### {r}"
        answer = r
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_drt", "level": level, "source": "algebra_qa"}


def age_problem(rng: random.Random, level: int) -> dict:
    """Alice is k times Bob's age. In y years she'll be m times. Find Bob."""
    bob_now = rng.randint(5, 15)
    k = rng.randint(2, 4)
    alice_now = k * bob_now
    y = rng.randint(1, 10)
    # alice_now + y = m * (bob_now + y)  =>  m = (alice_now+y)/(bob_now+y)
    # force m to be a clean integer
    # pick y such that (k*b+y) % (b+y) == 0
    for y in range(1, 20):
        if (alice_now + y) % (bob_now + y) == 0:
            m = (alice_now + y) // (bob_now + y)
            break
    else:
        # fallback: just use k-1 as relationship
        y = bob_now
        alice_now = (k - 1) * (bob_now + y) - y
        if alice_now <= 0:
            alice_now = 3 * bob_now
        m = k - 1
    problem = (f"Alice is {k} times Bob's age now. "
               f"In {y} years she'll be {m} times his age. How old is Bob now?")
    solution = (f"Let Bob = {bob_now}, Alice = {k}*{bob_now} = {alice_now}\n"
                f"In {y} years: {alice_now+y} = {m}*({bob_now+y}) = {m*(bob_now+y)}\n"
                f"Bob = {bob_now}\n"
                f"#### {bob_now}")
    return {"problem": problem, "solution": solution, "answer": bob_now,
            "domain": "algebra_age", "level": level, "source": "algebra_qa"}


# ── L3: Quadratic formula, 3-var system, absolute value, mixture ─────────────

def quadratic_formula(rng: random.Random, level: int) -> dict:
    """ax^2 + bx + c = 0 with perfect-square discriminant; larger root."""
    r1 = rng.randint(-5, 5)
    r2 = rng.randint(-5, 5)
    a = rng.randint(1, 3)
    b = -a * (r1 + r2)
    c = a * r1 * r2
    disc = b * b - 4 * a * c
    if disc < 0:
        r2 = -r2
        b = -a * (r1 + r2)
        c = a * r1 * r2
        disc = b * b - 4 * a * c
    sqrt_d = int(disc ** 0.5)
    if sqrt_d * sqrt_d != disc:
        # not perfect square — fall back to simple case
        r1, r2, a = 3, -2, 1
        b, c = -(r1+r2), r1*r2
        disc = b*b - 4*a*c
        sqrt_d = int(disc**0.5)
    larger = max((-b + sqrt_d) // (2 * a), (-b - sqrt_d) // (2 * a))
    problem = f"Use quadratic formula: {a}x^2+({b})x+{c}=0. Larger root?"
    solution = (f"disc = ({b})^2 - 4*{a}*{c} = {disc}, sqrt = {sqrt_d}\n"
                f"x = (-({b}) +/- {sqrt_d}) / (2*{a})\n"
                f"Larger root = {larger}\n"
                f"#### {larger}")
    return {"problem": problem, "solution": solution, "answer": larger,
            "domain": "algebra_quadratic", "level": level, "source": "algebra_qa"}


def three_var_system(rng: random.Random, level: int) -> dict:
    """3x3 integer system; answer is x+y+z."""
    x, y, z = (rng.randint(-3, 3) for _ in range(3))
    a = [[rng.randint(1, 3) for _ in range(3)] for _ in range(3)]
    det = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
           - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
           + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
    if det == 0:
        a[0][0] += 1
    rhs = [a[i][0]*x + a[i][1]*y + a[i][2]*z for i in range(3)]
    e = lambda r: f"{a[r][0]}x+{a[r][1]}y+{a[r][2]}z={rhs[r]}"
    problem = f"Solve: {e(0)}, {e(1)}, {e(2)}. Find x+y+z."
    solution = (f"Row reduction gives x={x}, y={y}, z={z}\n"
                f"x+y+z = {x}+{y}+{z} = {x+y+z}\n"
                f"#### {x+y+z}")
    return {"problem": problem, "solution": solution, "answer": x+y+z,
            "domain": "algebra_system3", "level": level, "source": "algebra_qa"}


def absolute_value_eq(rng: random.Random, level: int) -> dict:
    """|ax + b| = c; ask for sum of solutions."""
    a = rng.randint(1, 3 * level)
    c = rng.randint(1, 10 * level)
    b = rng.randint(-5 * level, 5 * level)
    # x = (c - b)/a  or  x = (-c - b)/a
    x1_num = c - b
    x2_num = -c - b
    if x1_num % a == 0 and x2_num % a == 0:
        x1 = x1_num // a
        x2 = x2_num // a
        answer = x1 + x2
        b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
        problem = f"Solve: |{a}x {b_str}| = {c}. Sum of solutions?"
        solution = (f"Case 1: {a}x + ({b}) = {c} -> {a}x = {c-b} -> x = {x1}\n"
                    f"Case 2: {a}x + ({b}) = -{c} -> {a}x = {-c-b} -> x = {x2}\n"
                    f"Sum = {x1} + {x2} = {answer}\n"
                    f"#### {answer}")
        return {"problem": problem, "solution": solution, "answer": answer,
                "domain": "algebra_abs", "level": level, "source": "algebra_qa"}
    return None  # skip if not clean integers


def mixture_problem(rng: random.Random, level: int) -> dict:
    """Mix solution A (p1%) with solution B (p2%) to get target concentration."""
    p1 = rng.choice([10, 20, 25, 30, 40])
    p2 = rng.choice([60, 70, 75, 80, 90])
    pt = rng.randint(p1 + 5, p2 - 5)
    # x liters of p1%, (total-x) of p2% -> pt% total
    # p1*x + p2*(T-x) = pt*T  -> x*(p1-p2) = T*(pt-p2)
    T = rng.randint(4, 10) * level
    x_num = T * (pt - p2)
    x_den = p1 - p2
    if x_den == 0 or x_num % x_den != 0:
        return None
    x = x_num // x_den
    if x <= 0 or x >= T:
        return None
    problem = (f"Mix {p1}% and {p2}% solutions to get {T}L of {pt}% solution. "
               f"How many liters of {p1}% needed?")
    solution = (f"Let x = liters of {p1}%\n"
                f"{p1}x + {p2}({T}-x) = {pt}*{T}\n"
                f"{p1-p2}x = {pt*T} - {p2*T} = {(pt-p2)*T}\n"
                f"x = {x}\n"
                f"#### {x}")
    return {"problem": problem, "solution": solution, "answer": x,
            "domain": "algebra_mixture", "level": level, "source": "algebra_qa"}


# ── L4: Polynomial eval, inequality, function composition, rational ───────────

def polynomial_eval(rng: random.Random, level: int) -> dict:
    """Evaluate P(x) = a3 x^3 + a2 x^2 + a1 x + a0 at x=k."""
    a3, a2, a1, a0 = (rng.randint(-3, 3) for _ in range(4))
    k = rng.randint(-3, 3)
    answer = a3*k**3 + a2*k**2 + a1*k + a0
    problem = f"P(x) = {a3}x^3+{a2}x^2+{a1}x+{a0}. Find P({k})."
    solution = (f"P({k}) = {a3}*({k}^3)+{a2}*({k}^2)+{a1}*{k}+{a0}\n"
                f"= {a3*k**3}+{a2*k**2}+{a1*k}+{a0}\n"
                f"= {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_polynomial", "level": level, "source": "algebra_qa"}


def linear_inequality(rng: random.Random, level: int) -> dict:
    """Find largest integer x satisfying ax + b < c."""
    a = rng.randint(1, 4 * level)
    x_bound = rng.randint(-8 * level, 8 * level)
    b = rng.randint(-15, 15)
    c = a * x_bound + b
    largest = x_bound - 1
    problem = f"Find largest integer x: {a}x + {b} < {c}."
    solution = (f"{a}x < {c} - {b} = {c-b}\n"
                f"x < {_frac(c-b, a)}\n"
                f"Largest integer x = {largest}\n"
                f"#### {largest}")
    return {"problem": problem, "solution": solution, "answer": largest,
            "domain": "algebra_inequality", "level": level, "source": "algebra_qa"}


def function_composition(rng: random.Random, level: int) -> dict:
    """f(x) = ax + b, g(x) = cx + d. Find f(g(k))."""
    a, b = rng.randint(1, 4 * level), rng.randint(-8, 8)
    c, d = rng.randint(1, 4 * level), rng.randint(-8, 8)
    k = rng.randint(-5, 5)
    gk = c * k + d
    fgk = a * gk + b
    problem = f"f(x)={a}x+{b}, g(x)={c}x+{d}. Find f(g({k}))."
    solution = (f"g({k}) = {c}*{k}+{d} = {gk}\n"
                f"f({gk}) = {a}*{gk}+{b} = {fgk}\n"
                f"#### {fgk}")
    return {"problem": problem, "solution": solution, "answer": fgk,
            "domain": "algebra_functions", "level": level, "source": "algebra_qa"}


def rational_equation(rng: random.Random, level: int) -> dict:
    """a/(x + b) = c/d; integer solution."""
    b = rng.randint(-5, 5)
    c = rng.randint(1, 5 * level)
    d = rng.randint(1, 5 * level)
    # x + b = a*d/c; need integer answer, so a = c*k - b for some k
    x_true = rng.randint(1, 10 * level)
    a = c * (x_true + b) // d
    if a == 0 or c * (x_true + b) != a * d:
        return None
    problem = f"Solve: {a}/(x+{b}) = {c}/{d}. Find x."
    solution = (f"Cross multiply: {a}*{d} = {c}*(x+{b})\n"
                f"{a*d} = {c}x + {c*b}\n"
                f"{c}x = {a*d - c*b}\n"
                f"x = {x_true}\n"
                f"#### {x_true}")
    return {"problem": problem, "solution": solution, "answer": x_true,
            "domain": "algebra_rational", "level": level, "source": "algebra_qa"}


# ── L5: Vieta's, sequences, nested functions, floor equations ────────────────

def vieta_product(rng: random.Random, level: int) -> dict:
    """Quadratic with known roots; find b+c."""
    r1 = rng.randint(-8, 8)
    r2 = rng.randint(-8, 8)
    b = -(r1 + r2)
    c = r1 * r2
    answer = b + c
    problem = f"x^2+bx+c=0 has roots {r1} and {r2}. Find b+c."
    solution = (f"By Vieta's: b = -(sum of roots) = -({r1}+{r2}) = {b}\n"
                f"c = product of roots = {r1}*{r2} = {c}\n"
                f"b+c = {b}+{c} = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_vieta", "level": level, "source": "algebra_qa"}


def arithmetic_sequence(rng: random.Random, level: int) -> dict:
    """Sum of first n terms of arithmetic sequence."""
    a1 = rng.randint(-10 * level, 10 * level)
    d  = rng.randint(-5 * level, 5 * level)
    n  = rng.randint(5, 15)
    an = a1 + (n - 1) * d
    sn = n * (a1 + an) // 2
    problem = f"Arith. seq: a1={a1}, d={d}. Sum of first {n} terms?"
    solution = (f"a_{n} = {a1} + ({n}-1)*{d} = {an}\n"
                f"S_{n} = {n}*({a1}+{an})/2 = {sn}\n"
                f"#### {sn}")
    return {"problem": problem, "solution": solution, "answer": sn,
            "domain": "algebra_sequence", "level": level, "source": "algebra_qa"}


def nested_function(rng: random.Random, level: int) -> dict:
    """f(f(x)) where f(x) = ax + b."""
    a = rng.randint(1, 3)
    b = rng.randint(-5, 5)
    k = rng.randint(-4, 4)
    fx = a * k + b
    ffx = a * fx + b
    problem = f"f(x) = {a}x + {b}. Find f(f({k}))."
    solution = (f"f({k}) = {a}*{k}+{b} = {fx}\n"
                f"f({fx}) = {a}*{fx}+{b} = {ffx}\n"
                f"#### {ffx}")
    return {"problem": problem, "solution": solution, "answer": ffx,
            "domain": "algebra_functions", "level": level, "source": "algebra_qa"}


def geometric_sequence(rng: random.Random, level: int) -> dict:
    """a1, a1*r, a1*r^2, ... ; find nth term or sum."""
    a1 = rng.randint(1, 5)
    r  = rng.randint(2, 4)
    n  = rng.randint(3, 6)
    an = a1 * r**(n-1)
    problem = f"Geometric seq: a1={a1}, r={r}. Find term {n}."
    solution = (f"a_{n} = a1 * r^({n}-1) = {a1} * {r}^{n-1}\n"
                f"= {a1} * {r**(n-1)} = {an}\n"
                f"#### {an}")
    return {"problem": problem, "solution": solution, "answer": an,
            "domain": "algebra_sequence", "level": level, "source": "algebra_qa"}


def inverse_function(rng: random.Random, level: int) -> dict:
    """f(x) = ax + b; find f^{-1}(k)."""
    a = rng.randint(1, 4 * level)
    b = rng.randint(-10, 10)
    k = rng.randint(-10, 10)
    # f^{-1}(k) = (k - b) / a  — only generate when integer
    if (k - b) % a != 0:
        return None
    answer = (k - b) // a
    problem = f"f(x) = {a}x + {b}. Find f^(-1)({k})."
    solution = (f"Set y = {a}x + {b}: x = (y - {b}) / {a}\n"
                f"f^(-1)({k}) = ({k} - {b}) / {a} = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "algebra_functions", "level": level, "source": "algebra_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [linear_one_var, linear_word, percent_problem, proportion],
    2: [system_two_var, quadratic_roots, linear_one_var,
        distance_rate_time, age_problem],
    3: [quadratic_formula, three_var_system, system_two_var,
        absolute_value_eq, mixture_problem],
    4: [polynomial_eval, linear_inequality, quadratic_formula,
        function_composition, rational_equation],
    5: [vieta_product, arithmetic_sequence, nested_function,
        geometric_sequence, inverse_function, polynomial_eval],
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_algebra_pairs(n_pairs: int = 500, levels=None, seed=None) -> list:
    if levels is None:
        levels = [1, 2, 3, 4, 5]
    rng = random.Random(seed)
    pairs = []
    seen = set()
    per_level = max(1, n_pairs // len(levels))

    for level in levels:
        templates = LEVEL_TEMPLATES.get(level, LEVEL_TEMPLATES[3])
        count = 0
        attempts = 0
        while count < per_level and attempts < per_level * 30:
            attempts += 1
            fn = rng.choice(templates)
            try:
                p = fn(rng, level)
            except Exception:
                continue
            if p is None:
                continue
            key = p["problem"]
            if key in seen or not _ok(p["problem"], p["solution"]):
                continue
            seen.add(key)
            pairs.append(p)
            count += 1

    rng.shuffle(pairs)
    return pairs


def save_pairs(pairs: list, run_tag: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    out = RESULTS_DIR / f"harder_qa_algebra{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} algebra pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate algebra Q/A pairs")
    parser.add_argument("--n",       type=int,   default=500)
    parser.add_argument("--levels",  nargs="+",  type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed",    type=int,   default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_algebra_pairs(args.n, args.levels, args.seed)
    if pairs:
        save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    print(f"\nSample problems:")
    for p in pairs[:4]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  {p['solution']}")
