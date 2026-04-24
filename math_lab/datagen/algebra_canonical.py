"""
algebra_canonical.py — Canonical Algebra Curriculum Dataset

Deterministic, concept-complete algebra QA generator for curriculum training.
Each of 48 concepts is instantiated N times with seeded RNG-generated params,
then split into train and val halves (each variant uses the SAME template,
testing whether the model learned the concept structure vs memorizing numbers).

Design principles:
  1. Every concept covered — no missing primitives
  2. N variants per concept (default 6 → 3 train + 3 val per concept)
  3. All answers are integers or simple fractions (no float noise)
  4. Solutions show every step — no elided reasoning
  5. `concept` tag on every record for per-concept eval analysis
  6. `stage` tag (1-5) marks curriculum difficulty
  7. Fully deterministic: same seed + n_variants → same dataset

Five stages:
  Stage 1: Foundations (solve simple equations, distribute, combine terms)
  Stage 2: Two-step intermediate (vars both sides, percents, proportions, inequalities)
  Stage 3: Systems, quadratics, functions, sequences
  Stage 4: Word problems (DRT, age, mixture, unit price, work, percent change)
  Stage 5: Higher-level (polynomials, exponents, radicals, logs, completing square)

Usage:
    python math_lab/datagen/algebra_canonical.py                       # default N=6
    python math_lab/datagen/algebra_canonical.py --n-variants 10
    python math_lab/datagen/algebra_canonical.py --run-tag v2 --seed 42

Output:
    math_lab/results/canonical_algebra_train_<tag>_<ts>.jsonl
    math_lab/results/canonical_algebra_val_<tag>_<ts>.jsonl
"""

import json
import random
import datetime
from math import gcd
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ===========================================================================
# STAGE 1 — Foundations
# ===========================================================================

def c1_solve_x_plus_a(v, rng):
    """x + a = b"""
    a = rng.randint(2, 20)
    x = rng.randint(1, 25)
    b = x + a
    return dict(
        problem=f"Solve for x: x + {a} = {b}",
        solution=f"x = {b} - {a}\nx = {x}\n#### {x}",
        answer=x, concept="solve_x_plus_a", stage=1, variant=v)


def c1_solve_x_minus_a(v, rng):
    """x - a = b"""
    a = rng.randint(2, 15)
    b = rng.randint(1, 25)
    x = a + b
    return dict(
        problem=f"Solve for x: x - {a} = {b}",
        solution=f"x = {b} + {a}\nx = {x}\n#### {x}",
        answer=x, concept="solve_x_minus_a", stage=1, variant=v)


def c1_solve_ax(v, rng):
    """ax = b"""
    a = rng.randint(2, 9)
    x = rng.randint(2, 12)
    b = a * x
    return dict(
        problem=f"Solve for x: {a}x = {b}",
        solution=f"x = {b} / {a}\nx = {x}\n#### {x}",
        answer=x, concept="solve_ax", stage=1, variant=v)


def c1_solve_x_over_a(v, rng):
    """x/a = b"""
    a = rng.randint(2, 9)
    b = rng.randint(2, 12)
    x = a * b
    return dict(
        problem=f"Solve for x: x/{a} = {b}",
        solution=f"x = {a} * {b}\nx = {x}\n#### {x}",
        answer=x, concept="solve_x_over_a", stage=1, variant=v)


def c1_solve_ax_plus_b(v, rng):
    """ax + b = c"""
    a = rng.randint(2, 7)
    x = rng.randint(1, 10)
    b = rng.randint(1, 15)
    c = a * x + b
    return dict(
        problem=f"Solve for x: {a}x + {b} = {c}",
        solution=(f"{a}x = {c} - {b}\n"
                  f"{a}x = {c-b}\n"
                  f"x = {c-b} / {a}\n"
                  f"x = {x}\n#### {x}"),
        answer=x, concept="solve_ax_plus_b", stage=1, variant=v)


def c1_solve_ax_minus_b(v, rng):
    """ax - b = c"""
    a = rng.randint(2, 7)
    x = rng.randint(1, 10)
    b = rng.randint(1, 15)
    c = a * x - b
    return dict(
        problem=f"Solve for x: {a}x - {b} = {c}",
        solution=(f"{a}x = {c} + {b}\n"
                  f"{a}x = {c+b}\n"
                  f"x = {c+b} / {a}\n"
                  f"x = {x}\n#### {x}"),
        answer=x, concept="solve_ax_minus_b", stage=1, variant=v)


def c1_solve_a_minus_x(v, rng):
    """a - x = b"""
    b = rng.randint(1, 15)
    a = rng.randint(b + 1, b + 20)
    x = a - b
    return dict(
        problem=f"Solve for x: {a} - x = {b}",
        solution=f"x = {a} - {b}\nx = {x}\n#### {x}",
        answer=x, concept="solve_a_minus_x", stage=1, variant=v)


def c1_solve_neg_x_plus_b(v, rng):
    """-x + b = c"""
    c = rng.randint(1, 10)
    b = rng.randint(c + 1, c + 15)
    x = b - c
    return dict(
        problem=f"Solve for x: -x + {b} = {c}",
        solution=f"-x = {c} - {b} = {c-b}\nx = {x}\n#### {x}",
        answer=x, concept="solve_neg_x_plus_b", stage=1, variant=v)


def c1_combine_like_terms(v, rng):
    """ax + bx = (a+b)x"""
    a = rng.randint(2, 10)
    b = rng.randint(2, 10)
    s = a + b
    return dict(
        problem=f"Simplify: {a}x + {b}x",
        solution=f"= ({a} + {b})x\n= {s}x\n#### {s}x",
        answer=f"{s}x", concept="combine_like_terms", stage=1, variant=v)


def c1_combine_with_constants(v, rng):
    """ax + b + cx + d"""
    a = rng.randint(1, 8)
    b = rng.randint(1, 10)
    c = rng.randint(1, 8)
    d = rng.randint(1, 10)
    coef = a + c
    const = b + d
    return dict(
        problem=f"Simplify: {a}x + {b} + {c}x + {d}",
        solution=(f"Combine x terms: {a}x + {c}x = {coef}x\n"
                  f"Combine constants: {b} + {d} = {const}\n"
                  f"= {coef}x + {const}\n#### {coef}x + {const}"),
        answer=f"{coef}x + {const}", concept="combine_with_constants",
        stage=1, variant=v)


def c1_distribute_positive(v, rng):
    """a(x + b)"""
    a = rng.randint(2, 8)
    b = rng.randint(2, 10)
    return dict(
        problem=f"Expand: {a}(x + {b})",
        solution=(f"= {a}*x + {a}*{b}\n"
                  f"= {a}x + {a*b}\n#### {a}x + {a*b}"),
        answer=f"{a}x + {a*b}", concept="distribute_positive",
        stage=1, variant=v)


def c1_distribute_negative(v, rng):
    """-a(x - b)"""
    a = rng.randint(2, 8)
    b = rng.randint(2, 10)
    return dict(
        problem=f"Expand: -{a}(x - {b})",
        solution=(f"= -{a}*x + (-{a})*(-{b})\n"
                  f"= -{a}x + {a*b}\n#### -{a}x + {a*b}"),
        answer=f"-{a}x + {a*b}", concept="distribute_negative",
        stage=1, variant=v)


def c1_eval_at_value(v, rng):
    """Evaluate ax + b at x = k"""
    a = rng.randint(2, 8)
    b = rng.randint(1, 12)
    k = rng.randint(-5, 8)
    ans = a * k + b
    return dict(
        problem=f"If x = {k}, find {a}x + {b}.",
        solution=f"{a}*{k} + {b} = {a*k} + {b} = {ans}\n#### {ans}",
        answer=ans, concept="eval_at_value", stage=1, variant=v)


def c1_factor_common(v, rng):
    """Factor out GCD from ax + b"""
    g = rng.randint(2, 7)
    ai = rng.randint(2, 9)
    bi = rng.randint(1, 9)
    while gcd(ai, bi) != 1:
        ai = rng.randint(2, 9)
        bi = rng.randint(1, 9)
    a = g * ai
    b = g * bi
    return dict(
        problem=f"Factor: {a}x + {b}",
        solution=(f"GCD({a}, {b}) = {g}\n"
                  f"= {g}({ai}x + {bi})\n#### {g}({ai}x + {bi})"),
        answer=f"{g}({ai}x + {bi})", concept="factor_common",
        stage=1, variant=v)


# ===========================================================================
# STAGE 2 — Two-step intermediate
# ===========================================================================

def c2_var_both_sides(v, rng):
    """ax + b = cx + d, a > c (so a-c > 0)"""
    c = rng.randint(1, 4)
    a = rng.randint(c + 1, c + 5)
    x = rng.randint(1, 8)
    b = rng.randint(1, 10)
    d = (a - c) * x + b
    return dict(
        problem=f"Solve for x: {a}x + {b} = {c}x + {d}",
        solution=(f"Move x terms to left: {a}x - {c}x = {d} - {b}\n"
                  f"{a-c}x = {d-b}\n"
                  f"x = {d-b} / {a-c}\n"
                  f"x = {x}\n#### {x}"),
        answer=x, concept="var_both_sides", stage=2, variant=v)


def c2_paren_then_solve(v, rng):
    """a(x + b) = c"""
    a = rng.randint(2, 6)
    b = rng.randint(1, 8)
    x = rng.randint(1, 10)
    c = a * (x + b)
    return dict(
        problem=f"Solve for x: {a}(x + {b}) = {c}",
        solution=(f"Distribute: {a}x + {a*b} = {c}\n"
                  f"{a}x = {c} - {a*b} = {c-a*b}\n"
                  f"x = {c-a*b} / {a} = {x}\n#### {x}"),
        answer=x, concept="paren_then_solve", stage=2, variant=v)


def c2_frac_coefficient(v, rng):
    """x/a + b = c"""
    a = rng.randint(2, 6)
    x = a * rng.randint(2, 8)  # ensure x/a is integer
    q = x // a
    b = rng.randint(1, 10)
    c = q + b
    return dict(
        problem=f"Solve for x: x/{a} + {b} = {c}",
        solution=(f"x/{a} = {c} - {b} = {c-b}\n"
                  f"x = {a} * {c-b} = {x}\n#### {x}"),
        answer=x, concept="frac_coefficient", stage=2, variant=v)


def c2_proportion(v, rng):
    """a/b = c/x → x = b*c/a with integer x"""
    a = rng.randint(2, 8)
    k = rng.randint(2, 10)
    b = rng.randint(2, 10)
    c = a * k  # so b*c/a = b*k which is integer
    x = b * k
    return dict(
        problem=f"Solve: {a}/{b} = {c}/x",
        solution=(f"Cross multiply: {a}*x = {b}*{c}\n"
                  f"{a}x = {b*c}\n"
                  f"x = {b*c} / {a} = {x}\n#### {x}"),
        answer=x, concept="proportion", stage=2, variant=v)


def c2_percent_of(v, rng):
    """p% of n with clean integer result"""
    p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
    # choose n so p*n/100 is integer
    n_factor = rng.randint(2, 10)
    n = (100 // gcd(p, 100)) * n_factor
    ans = p * n // 100
    return dict(
        problem=f"What is {p}% of {n}?",
        solution=(f"{p}/100 * {n}\n"
                  f"= {p}*{n}/100 = {p*n}/100 = {ans}\n#### {ans}"),
        answer=ans, concept="percent_of", stage=2, variant=v)


def c2_percent_is_what(v, rng):
    """a is what percent of b? (integer answer)"""
    p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
    b_factor = rng.randint(2, 8)
    b = (100 // gcd(p, 100)) * b_factor
    a = p * b // 100
    return dict(
        problem=f"{a} is what percent of {b}?",
        solution=(f"{a}/{b} * 100\n"
                  f"= {a*100}/{b} = {p}\n#### {p}"),
        answer=p, concept="percent_is_what", stage=2, variant=v)


def c2_inequality_basic(v, rng):
    """ax + b < c, find largest integer x"""
    a = rng.randint(2, 5)
    x_bound = rng.randint(1, 8)
    b = rng.randint(1, 10)
    c = a * x_bound + b + 1  # strict inequality: x < x_bound+something
    # largest integer x such that ax + b < c
    # x < (c-b)/a → largest integer is x_bound (since ax_bound + b = c - 1 < c)
    largest = x_bound
    return dict(
        problem=f"Find largest integer x: {a}x + {b} < {c}",
        solution=(f"{a}x < {c} - {b} = {c-b}\n"
                  f"x < {c-b} / {a} = {Fraction(c-b, a)}\n"
                  f"Largest integer: {largest}\n#### {largest}"),
        answer=largest, concept="inequality_basic", stage=2, variant=v)


def c2_inequality_sign_flip(v, rng):
    """-ax > b; flip sign when dividing"""
    a = rng.randint(2, 5)
    k = rng.randint(2, 8)
    b = a * k  # so -a*(-k-1) > ak: -a*x > b means x < -b/a = -k
    bound = -k
    largest = bound - 1
    return dict(
        problem=f"Find largest integer x: -{a}x > {b}",
        solution=(f"Divide both sides by -{a} (flip inequality):\n"
                  f"x < -{b}/{a} = {bound}\n"
                  f"Largest integer: {largest}\n#### {largest}"),
        answer=largest, concept="inequality_sign_flip", stage=2, variant=v)


def c2_abs_value(v, rng):
    """|x + b| = c; sum of two solutions"""
    b = rng.randint(1, 10)
    c = rng.randint(b + 1, b + 12)
    # Solutions: x = c-b and x = -c-b, sum = -2b
    x1 = c - b
    x2 = -c - b
    ans = x1 + x2
    return dict(
        problem=f"|x + {b}| = {c}. Find the sum of all solutions.",
        solution=(f"Case 1: x + {b} = {c}  =>  x = {x1}\n"
                  f"Case 2: x + {b} = -{c}  =>  x = {x2}\n"
                  f"Sum = {x1} + ({x2}) = {ans}\n#### {ans}"),
        answer=ans, concept="abs_value", stage=2, variant=v)


# ===========================================================================
# STAGE 3 — Systems, quadratics, functions, sequences
# ===========================================================================

def c3_system_sub(v, rng):
    """System: x+y=s, x-y=d; find x"""
    x = rng.randint(1, 10)
    y = rng.randint(1, 10)
    s, d = x + y, x - y
    return dict(
        problem=f"Solve: x + y = {s}, x - y = {d}. Find x.",
        solution=(f"Add the equations: 2x = {s} + {d} = {s+d}\n"
                  f"x = {s+d}/2 = {x}\n#### {x}"),
        answer=x, concept="system_sub", stage=3, variant=v)


def c3_system_elim(v, rng):
    """2x2 system by elimination"""
    x = rng.randint(1, 8)
    y = rng.randint(1, 8)
    a, b = 2, 1
    c, d = 1, 1
    e1 = a * x + b * y
    e2 = c * x + d * y
    return dict(
        problem=f"Solve: {a}x + {b}y = {e1}, {c}x + {d}y = {e2}. Find x.",
        solution=(f"Subtract eq2 from eq1: ({a}-{c})x = {e1}-{e2}\n"
                  f"{a-c}x = {e1-e2}\n"
                  f"x = {e1-e2}/{a-c} = {x}\n#### {x}"),
        answer=x, concept="system_elim", stage=3, variant=v)


def c3_quad_factor(v, rng):
    """x^2 + bx + c = 0 with integer roots; sum of roots"""
    r1 = rng.randint(1, 6)
    r2 = rng.randint(1, 6)
    b = -(r1 + r2)
    c = r1 * r2
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    return dict(
        problem=f"Solve: x^2 {b_str}x + {c} = 0. Sum of roots?",
        solution=(f"Factor: (x - {r1})(x - {r2}) = 0\n"
                  f"Roots: x = {r1}, x = {r2}\n"
                  f"Sum = {r1} + {r2} = {r1+r2}\n#### {r1+r2}"),
        answer=r1+r2, concept="quad_factor", stage=3, variant=v)


def c3_quad_diff_squares(v, rng):
    """x^2 - a^2 = 0; positive root"""
    a = rng.randint(2, 9)
    return dict(
        problem=f"Solve: x^2 - {a*a} = 0. Positive root?",
        solution=(f"x^2 = {a*a}\n"
                  f"x = +/- {a}\n"
                  f"Positive root: {a}\n#### {a}"),
        answer=a, concept="quad_diff_squares", stage=3, variant=v)


def c3_quad_formula(v, rng):
    """x^2 + bx + c = 0 with integer roots (one negative); larger root"""
    r1 = rng.randint(-6, -1)
    r2 = rng.randint(2, 7)
    b = -(r1 + r2)
    c = r1 * r2
    disc = b * b - 4 * c
    sqrt_d = int(disc ** 0.5)
    larger = max(r1, r2)
    return dict(
        problem=f"Solve using quadratic formula: x^2 + ({b})x + ({c}) = 0. Larger root?",
        solution=(f"disc = ({b})^2 - 4*1*({c}) = {disc}\n"
                  f"sqrt({disc}) = {sqrt_d}\n"
                  f"x = (-{b} +/- {sqrt_d}) / 2\n"
                  f"Larger root = {larger}\n#### {larger}"),
        answer=larger, concept="quad_formula", stage=3, variant=v)


def c3_func_eval(v, rng):
    """f(x) = ax + b, find f(k)"""
    a = rng.randint(2, 7)
    b = rng.randint(-5, 10)
    k = rng.randint(-4, 8)
    ans = a * k + b
    return dict(
        problem=f"f(x) = {a}x + {b}. Find f({k}).",
        solution=f"f({k}) = {a}*{k} + {b} = {a*k} + {b} = {ans}\n#### {ans}",
        answer=ans, concept="func_eval", stage=3, variant=v)


def c3_func_compose(v, rng):
    """f(g(x))"""
    a = rng.randint(2, 5)
    b = rng.randint(-3, 5)
    c = rng.randint(2, 5)
    d = rng.randint(-3, 5)
    k = rng.randint(-3, 5)
    gk = c * k + d
    fgk = a * gk + b
    return dict(
        problem=f"f(x) = {a}x + {b}, g(x) = {c}x + {d}. Find f(g({k})).",
        solution=(f"g({k}) = {c}*{k} + {d} = {gk}\n"
                  f"f({gk}) = {a}*{gk} + {b} = {fgk}\n#### {fgk}"),
        answer=fgk, concept="func_compose", stage=3, variant=v)


def c3_func_inverse(v, rng):
    """f(x) = ax + b, find f^{-1}(k); require integer result"""
    a = rng.randint(2, 6)
    b = rng.randint(-8, 8)
    x = rng.randint(1, 10)
    k = a * x + b  # so (k-b)/a = x, integer
    return dict(
        problem=f"f(x) = {a}x + {b}. Find f^-1({k}).",
        solution=(f"y = {a}x + {b}\n"
                  f"x = (y - {b}) / {a}\n"
                  f"f^-1({k}) = ({k} - {b}) / {a} = {k-b}/{a} = {x}\n#### {x}"),
        answer=x, concept="func_inverse", stage=3, variant=v)


def c3_arith_sequence_sum(v, rng):
    """Sum of first n terms of arithmetic sequence"""
    a1 = rng.randint(1, 8)
    d = rng.randint(1, 5)
    n = rng.randint(4, 10)
    an = a1 + (n - 1) * d
    s = n * (a1 + an) // 2
    return dict(
        problem=f"Arithmetic sequence: a1 = {a1}, d = {d}. Sum of first {n} terms?",
        solution=(f"a_{n} = {a1} + ({n}-1)*{d} = {an}\n"
                  f"S = {n} * ({a1} + {an}) / 2 = {n}*{a1+an}/2 = {s}\n#### {s}"),
        answer=s, concept="arith_sequence_sum", stage=3, variant=v)


def c3_geom_sequence_nth(v, rng):
    """nth term of geometric sequence"""
    a1 = rng.randint(1, 5)
    r = rng.randint(2, 4)
    n = rng.randint(3, 6)
    an = a1 * r ** (n - 1)
    return dict(
        problem=f"Geometric sequence: a1 = {a1}, r = {r}. Find term {n}.",
        solution=(f"a_{n} = {a1} * {r}^({n}-1)\n"
                  f"= {a1} * {r}^{n-1} = {a1} * {r**(n-1)} = {an}\n#### {an}"),
        answer=an, concept="geom_sequence_nth", stage=3, variant=v)


# ===========================================================================
# STAGE 4 — Word problems
# ===========================================================================

def c4_word_drt(v, rng):
    """distance = rate * time"""
    r = rng.randint(30, 80)
    t = rng.randint(2, 6)
    d = r * t
    return dict(
        problem=f"A car travels {r} mph for {t} hours. Total distance in miles?",
        solution=f"distance = rate * time = {r} * {t} = {d}\n#### {d}",
        answer=d, concept="word_drt", stage=4, variant=v)


def c4_word_age(v, rng):
    """Alice is k years older than Bob; sum = s. Find Bob"""
    k = rng.randint(3, 12)
    bob = rng.randint(5, 18)
    alice = bob + k
    s = alice + bob
    return dict(
        problem=f"Alice is {k} years older than Bob. Their ages sum to {s}. How old is Bob?",
        solution=(f"Let Bob = x, Alice = x + {k}\n"
                  f"x + (x + {k}) = {s}\n"
                  f"2x + {k} = {s}\n"
                  f"2x = {s-k}\n"
                  f"x = {bob}\n#### {bob}"),
        answer=bob, concept="word_age", stage=4, variant=v)


def c4_word_unit_price(v, rng):
    """n items cost T; cost of k items?"""
    n = rng.randint(2, 8)
    unit = rng.randint(3, 15)
    k = rng.randint(2, 10)
    T = n * unit
    ans = k * unit
    return dict(
        problem=f"{n} items cost ${T}. Cost of {k} items?",
        solution=(f"Unit price = {T} / {n} = {unit}\n"
                  f"Cost of {k} items = {k} * {unit} = {ans}\n#### {ans}"),
        answer=ans, concept="word_unit_price", stage=4, variant=v)


def c4_word_percent_change(v, rng):
    """Old price, new price; percent increase"""
    # Choose such that (new-old)/old * 100 is integer
    pct = rng.choice([10, 20, 25, 50, 75])
    old = rng.choice([40, 50, 60, 80, 100]) if pct in (20, 50) else 100 * rng.randint(1, 4)
    new = old + old * pct // 100
    return dict(
        problem=f"Price was ${old}, now ${new}. Percent increase?",
        solution=(f"Change = {new} - {old} = {new-old}\n"
                  f"Percent = {new-old}/{old} * 100 = {pct}\n#### {pct}"),
        answer=pct, concept="word_percent_change", stage=4, variant=v)


def c4_word_mixture(v, rng):
    """Mix x L of p1% with (T-x) L of p2% to get pt%. Find x with integer solution."""
    # Fixed param set that always produces integer x
    presets = [
        (20, 80, 10, 50),   # x = 5
        (30, 70, 20, 40),   # x = 15
        (10, 60, 10, 40),   # x = 4
        (25, 75, 8, 50),    # x = 4
        (20, 60, 10, 40),   # x = 5
        (10, 50, 12, 30),   # x = 6 (works: 10x + 50*(12-x) = 30*12, -40x+600=360, 40x=240, x=6)
        (25, 65, 10, 45),   # x = 5 (works: 25*5 + 65*5 = 125+325 = 450 = 45*10)
        (30, 90, 6, 50),    # x = 4 (works: 30*4+90*2 = 120+180 = 300 = 50*6)
    ]
    p1, p2, T, pt = rng.choice(presets)
    x = T * (pt - p2) // (p1 - p2)
    return dict(
        problem=(f"Mix {p1}% and {p2}% solutions to make {T}L of {pt}% solution. "
                 f"How many liters of {p1}% solution?"),
        solution=(f"Let x = liters of {p1}% solution\n"
                  f"{p1}x + {p2}({T}-x) = {pt}*{T}\n"
                  f"({p1}-{p2})x = ({pt}-{p2})*{T}\n"
                  f"{p1-p2}x = {(pt-p2)*T}\n"
                  f"x = {x}\n#### {x}"),
        answer=x, concept="word_mixture", stage=4, variant=v)


def c4_word_work(v, rng):
    """Combined work rate; choose (a,b) so a*b is divisible by (a+b)"""
    presets = [
        (6, 3),    # 2
        (10, 15),  # 6
        (12, 4),   # 3
        (12, 6),   # 4
        (4, 12),   # 3
        (6, 12),   # 4
        (20, 5),   # 4
        (15, 10),  # 6
    ]
    a, b = rng.choice(presets)
    ans = a * b // (a + b)
    return dict(
        problem=f"A finishes a job in {a} hours, B in {b} hours. Together (in hours)?",
        solution=(f"A's rate = 1/{a}, B's rate = 1/{b}\n"
                  f"Combined rate = 1/{a} + 1/{b} = ({b}+{a})/{a*b} = {a+b}/{a*b}\n"
                  f"Time = {a*b}/{a+b} = {ans}\n#### {ans}"),
        answer=ans, concept="word_work", stage=4, variant=v)


# ===========================================================================
# STAGE 5 — Higher-level
# ===========================================================================

def c5_polynomial_eval(v, rng):
    """Evaluate P(x) = ax^3 + bx + c at k"""
    a = rng.choice([1, 2])
    b = rng.randint(-3, 3)
    c = rng.randint(-5, 5)
    k = rng.randint(-3, 4)
    ans = a * k**3 + b * k + c
    return dict(
        problem=f"P(x) = {a}x^3 + ({b})x + ({c}). Find P({k}).",
        solution=(f"P({k}) = {a}*{k}^3 + ({b})*{k} + ({c})\n"
                  f"= {a*k**3} + ({b*k}) + ({c})\n"
                  f"= {ans}\n#### {ans}"),
        answer=ans, concept="polynomial_eval", stage=5, variant=v)


def c5_polynomial_factor(v, rng):
    """x^2 + bx + c → (x+p)(x+q)"""
    p = rng.randint(1, 6)
    q = rng.randint(1, 6)
    b, c = p + q, p * q
    return dict(
        problem=f"Factor: x^2 + {b}x + {c}",
        solution=(f"Find p, q where p*q = {c} and p+q = {b}\n"
                  f"p = {p}, q = {q}\n"
                  f"= (x + {p})(x + {q})\n#### (x + {p})(x + {q})"),
        answer=f"(x + {p})(x + {q})", concept="polynomial_factor",
        stage=5, variant=v)


def c5_difference_of_squares(v, rng):
    """(x+a)(x-a) = x^2 - a^2"""
    a = rng.randint(2, 10)
    return dict(
        problem=f"Expand: (x + {a})(x - {a})",
        solution=(f"Difference of squares pattern: (x+{a})(x-{a}) = x^2 - {a}^2\n"
                  f"= x^2 - {a*a}\n#### x^2 - {a*a}"),
        answer=f"x^2 - {a*a}", concept="difference_of_squares",
        stage=5, variant=v)


def c5_exponent_multiply(v, rng):
    """x^a * x^b = x^(a+b)"""
    a = rng.randint(2, 6)
    b = rng.randint(1, 5)
    return dict(
        problem=f"Simplify: x^{a} * x^{b}",
        solution=(f"When multiplying powers with same base, add exponents:\n"
                  f"x^({a}+{b}) = x^{a+b}\n#### x^{a+b}"),
        answer=f"x^{a+b}", concept="exponent_multiply", stage=5, variant=v)


def c5_exponent_divide(v, rng):
    """x^a / x^b = x^(a-b), a > b"""
    b = rng.randint(1, 5)
    a = rng.randint(b + 1, b + 6)
    return dict(
        problem=f"Simplify: x^{a} / x^{b}",
        solution=(f"When dividing powers with same base, subtract exponents:\n"
                  f"x^({a}-{b}) = x^{a-b}\n#### x^{a-b}"),
        answer=f"x^{a-b}", concept="exponent_divide", stage=5, variant=v)


def c5_exponent_power(v, rng):
    """(x^a)^b = x^(a*b)"""
    a = rng.randint(2, 5)
    b = rng.randint(2, 5)
    return dict(
        problem=f"Simplify: (x^{a})^{b}",
        solution=(f"Power of a power: multiply exponents\n"
                  f"x^({a}*{b}) = x^{a*b}\n#### x^{a*b}"),
        answer=f"x^{a*b}", concept="exponent_power", stage=5, variant=v)


def c5_exponential_eq(v, rng):
    """b^x = n where n is a power of b"""
    b = rng.choice([2, 3, 5])
    x = rng.randint(2, 5)
    n = b ** x
    return dict(
        problem=f"Solve: {b}^x = {n}",
        solution=(f"{b}^x = {n} = {b}^{x}\n"
                  f"So x = {x}\n#### {x}"),
        answer=x, concept="exponential_eq", stage=5, variant=v)


def c5_log_basic(v, rng):
    """log_b(n)"""
    b = rng.choice([2, 3, 5])
    x = rng.randint(2, 5)
    n = b ** x
    return dict(
        problem=f"Evaluate: log base {b} of {n}",
        solution=(f"log_{b}({n}) = x means {b}^x = {n}\n"
                  f"{b}^{x} = {n}, so x = {x}\n#### {x}"),
        answer=x, concept="log_basic", stage=5, variant=v)


def c5_completing_square(v, rng):
    """x^2 + bx + c → (x + h)^2 + k. Find h + k."""
    h = rng.randint(2, 8)
    k = rng.randint(1, 10)
    b = 2 * h
    c = h * h + k
    return dict(
        problem=f"Write x^2 + {b}x + {c} in the form (x + h)^2 + k. Find h + k.",
        solution=(f"Take half of {b}: h = {b}/2 = {h}\n"
                  f"(x + {h})^2 = x^2 + {b}x + {h*h}\n"
                  f"So k = {c} - {h*h} = {k}\n"
                  f"h + k = {h} + {k} = {h+k}\n#### {h+k}"),
        answer=h+k, concept="completing_square", stage=5, variant=v)


# ===========================================================================
# Concept registry
# ===========================================================================

CONCEPTS = [
    # Stage 1: Foundations
    c1_solve_x_plus_a, c1_solve_x_minus_a, c1_solve_ax, c1_solve_x_over_a,
    c1_solve_ax_plus_b, c1_solve_ax_minus_b, c1_solve_a_minus_x,
    c1_solve_neg_x_plus_b, c1_combine_like_terms, c1_combine_with_constants,
    c1_distribute_positive, c1_distribute_negative, c1_eval_at_value,
    c1_factor_common,
    # Stage 2: Two-step intermediate
    c2_var_both_sides, c2_paren_then_solve, c2_frac_coefficient,
    c2_proportion, c2_percent_of, c2_percent_is_what, c2_inequality_basic,
    c2_inequality_sign_flip, c2_abs_value,
    # Stage 3: Systems, quadratics, functions, sequences
    c3_system_sub, c3_system_elim, c3_quad_factor, c3_quad_diff_squares,
    c3_quad_formula, c3_func_eval, c3_func_compose, c3_func_inverse,
    c3_arith_sequence_sum, c3_geom_sequence_nth,
    # Stage 4: Word problems
    c4_word_drt, c4_word_age, c4_word_unit_price, c4_word_percent_change,
    c4_word_mixture, c4_word_work,
    # Stage 5: Higher-level
    c5_polynomial_eval, c5_polynomial_factor, c5_difference_of_squares,
    c5_exponent_multiply, c5_exponent_divide, c5_exponent_power,
    c5_exponential_eq, c5_log_basic, c5_completing_square,
]


# ===========================================================================
# Main
# ===========================================================================

def generate_all(n_variants: int = 6, seed: int = 42):
    """Return (train_pairs, val_pairs). Each concept instantiated n_variants
    times with deterministic seeding. First ceil(N/2) variants → train,
    rest → val. Within a concept, all variants test the same template
    with different numbers."""
    master = random.Random(seed)
    # Per-concept master seeds (stable across runs given same seed)
    concept_seeds = [master.randint(0, 10**9) for _ in CONCEPTS]

    train, val = [], []
    train_n = (n_variants + 1) // 2
    for fn_idx, fn in enumerate(CONCEPTS):
        concept_rng = random.Random(concept_seeds[fn_idx])
        variant_seeds = [concept_rng.randint(0, 10**9) for _ in range(n_variants)]
        seen_problems = set()
        records = []
        for v in range(n_variants):
            # Resample if we hit an exact problem duplicate for this concept
            for attempt in range(20):
                rng = random.Random(variant_seeds[v] + attempt)
                rec = fn(v, rng)
                if rec["problem"] not in seen_problems:
                    seen_problems.add(rec["problem"])
                    records.append(rec)
                    break
            else:
                records.append(rec)  # accept duplicate if we can't find unique

        for i, rec in enumerate(records):
            rec["source"] = "algebra_canonical"
            rec["split"] = "train" if i < train_n else "val"
            if i < train_n:
                train.append(rec)
            else:
                val.append(rec)
    return train, val


def save(pairs: list, split: str, tag: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out = RESULTS_DIR / f"canonical_algebra_{split}{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {split}: {len(pairs)} pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-variants", type=int, default=6,
                    help="Total variants per concept (half train, half val)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-tag", default="v2")
    args = ap.parse_args()

    train, val = generate_all(n_variants=args.n_variants, seed=args.seed)

    print(f"Generated {len(CONCEPTS)} concepts × {args.n_variants} variants")
    print(f"  {len(train)} train pairs, {len(val)} val pairs\n")
    print("Per-stage counts (train):")
    by_stage = Counter(p["stage"] for p in train)
    for s in sorted(by_stage):
        print(f"  Stage {s}: {by_stage[s]} pairs")
    print()

    save(train, "train", args.run_tag)
    save(val, "val", args.run_tag)

    # Overlap check
    train_probs = {p["problem"] for p in train}
    val_probs = {p["problem"] for p in val}
    overlap = train_probs & val_probs
    print(f"\nOverlap (train ∩ val problems): {len(overlap)} (should be 0)")

    # Sample variants for one concept
    print("\n=== Sample variants for concept 'solve_x_plus_a' ===")
    for rec in train + val:
        if rec["concept"] == "solve_x_plus_a":
            print(f"  [{rec['split']} v{rec['variant']}] {rec['problem']} → {rec['answer']}")
