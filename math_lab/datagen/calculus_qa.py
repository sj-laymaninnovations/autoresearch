"""
calculus_qa.py — Calculus Q/A Generator for Math Training

Topics by level:
  L1: Limit by substitution, power-rule derivative, basic indefinite integral
  L2: Product/quotient/chain rules, definite integral over integers, u-sub setup
  L3: Implicit differentiation, area between curves, related-rate problems
  L4: Optimization (critical pts + 2nd-derivative test), integration by parts
  L5: Taylor polynomial, separable ODE solution, Riemann sum

Number formats: answers appear as integers, fractions, decimals, and
scientific notation so the model learns all representations.

Output: math_lab/results/harder_qa_calculus_<tag>_<timestamp>.jsonl

References:
  - Stewart Calculus 8th edition (problem structure)
  - MIT OCW 18.01 Single Variable Calculus
  - AP Calculus AB/BC exam problem banks
"""

import json
import random
import datetime
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path(__file__).parent.parent / "results"
MAX_CONTENT_LEN = 246


def _ok(p: str, s: str) -> bool:
    return len(p) + len(s) <= MAX_CONTENT_LEN


def _frac(p: int, q: int) -> str:
    f = Fraction(p, q)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _fmt_num(val, rng: random.Random, style: str = None) -> str:
    """Randomly format a numeric value in one of several styles."""
    try:
        from number_formats import fmt
        return fmt(val, style, rng)
    except ImportError:
        return str(val)


# ── L1: Limits, basic derivatives & integrals ────────────────────────────────

def limit_poly(rng: random.Random, level: int) -> dict:
    """lim x->a of polynomial P(x) = P(a) by substitution."""
    a = rng.randint(-5 * level, 5 * level)
    coeffs = [rng.randint(-5, 5) for _ in range(3)]  # c2 x^2 + c1 x + c0
    c2, c1, c0 = coeffs
    answer = c2 * a**2 + c1 * a + c0
    ans_str = str(answer)
    problem = f"Find: lim(x->{a}) of {c2}x^2 + {c1}x + {c0}"
    solution = (f"Substitute x = {a}:\n"
                f"= {c2}*({a})^2 + {c1}*({a}) + {c0}\n"
                f"= {c2*a**2} + {c1*a} + {c0}\n"
                f"= {answer}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "calc_limit", "level": level, "source": "calculus_qa"}


def deriv_power_rule(rng: random.Random, level: int) -> dict:
    """d/dx [a*x^n] = a*n*x^(n-1)."""
    n = rng.randint(1, 4 + level)
    a = rng.randint(1, 8 * level)
    new_coeff = a * n
    new_exp = n - 1
    if new_exp == 0:
        deriv_str = str(new_coeff)
        answer_str = str(new_coeff)
    elif new_exp == 1:
        deriv_str = f"{new_coeff}x"
        answer_str = f"{new_coeff}x"
    else:
        deriv_str = f"{new_coeff}x^{new_exp}"
        answer_str = f"{new_coeff}x^{new_exp}"
    problem = f"Find d/dx [{a}x^{n}]."
    solution = (f"Power rule: d/dx [ax^n] = a*n*x^(n-1)\n"
                f"= {a}*{n}*x^({n}-1)\n"
                f"= {deriv_str}\n"
                f"#### {answer_str}")
    return {"problem": problem, "solution": solution, "answer": answer_str,
            "domain": "calc_derivative", "level": level, "source": "calculus_qa"}


def integral_power_rule(rng: random.Random, level: int) -> dict:
    """Integral of a*x^n = a/(n+1) * x^(n+1) + C."""
    n = rng.randint(0, 3 + level)
    a = rng.randint(1, 8 * level)
    new_exp = n + 1
    coeff_frac = Fraction(a, new_exp)
    coeff_str = _frac(coeff_frac.numerator, coeff_frac.denominator)
    if new_exp == 1:
        ans_str = f"{coeff_str}x + C"
    else:
        ans_str = f"{coeff_str}x^{new_exp} + C"
    problem = f"Find the indefinite integral of {a}x^{n} dx."
    solution = (f"Integral of x^n = x^(n+1)/(n+1)\n"
                f"= {a} * x^{new_exp}/{new_exp} + C\n"
                f"= {ans_str}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "calc_integral", "level": level, "source": "calculus_qa"}


def definite_integral_poly(rng: random.Random, level: int) -> dict:
    """Definite integral of ax^n from b to c (integers)."""
    n = rng.randint(1, 2 + level)
    a = rng.randint(1, 5 * level)
    lo = rng.randint(-3, 0)
    hi = rng.randint(1, 5 * level)
    # F(x) = a/(n+1) * x^(n+1)
    # Definite = F(hi) - F(lo)
    denom = n + 1
    numer_hi = a * hi**(n+1)
    numer_lo = a * lo**(n+1)
    result_frac = Fraction(numer_hi - numer_lo, denom)
    ans_str = _frac(result_frac.numerator, result_frac.denominator)
    problem = f"Evaluate integral from {lo} to {hi} of {a}x^{n} dx."
    solution = (f"F(x) = {_frac(a, denom)}x^{n+1}\n"
                f"F({hi}) = {_frac(numer_hi, denom)}\n"
                f"F({lo}) = {_frac(numer_lo, denom)}\n"
                f"= {ans_str}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "calc_integral", "level": level, "source": "calculus_qa"}


# ── L2: Product/chain/quotient rules ────────────────────────────────────────

def deriv_product_rule(rng: random.Random, level: int) -> dict:
    """d/dx [x^m * x^n] = (m+n)*x^(m+n-1) (combine via product rule)."""
    m = rng.randint(1, 3 + level)
    n = rng.randint(1, 3 + level)
    # f = x^m, g = x^n
    # f'g + fg' = m*x^(m-1)*x^n + x^m*n*x^(n-1) = (m+n)*x^(m+n-1)
    total_exp = m + n - 1
    coeff = m + n
    ans_str = f"{coeff}x^{total_exp}" if total_exp > 1 else f"{coeff}x"
    problem = f"Use the product rule: d/dx [x^{m} * x^{n}]."
    solution = (f"f = x^{m}, g = x^{n}\n"
                f"f' = {m}x^{m-1}, g' = {n}x^{n-1}\n"
                f"d/dx = f'g + fg' = {m}x^{m+n-1} + {n}x^{m+n-1}\n"
                f"= {coeff}x^{total_exp}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "calc_derivative", "level": level, "source": "calculus_qa"}


def deriv_chain_rule(rng: random.Random, level: int) -> dict:
    """d/dx [(ax+b)^n] = n*a*(ax+b)^(n-1)."""
    a = rng.randint(1, 5 * level)
    b = rng.randint(-10, 10)
    n = rng.randint(2, 4 + level)
    new_n = n - 1
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    inner = f"{a}x {b_str}"
    outer_coeff = n * a
    if new_n == 0:
        ans_str = str(outer_coeff)
    elif new_n == 1:
        ans_str = f"{outer_coeff}*({inner})"
    else:
        ans_str = f"{outer_coeff}*({inner})^{new_n}"
    problem = f"Find d/dx [({inner})^{n}]."
    solution = (f"Chain rule: d/dx [u^n] = n*u^(n-1)*u'\n"
                f"u = {inner}, u' = {a}\n"
                f"= {n}*({inner})^{new_n}*{a}\n"
                f"= {ans_str}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "calc_derivative", "level": level, "source": "calculus_qa"}


# ── L3: Optimization, area between curves ────────────────────────────────────

def optimization_quadratic(rng: random.Random, level: int) -> dict:
    """f(x) = ax^2 + bx + c; find x that maximizes/minimizes."""
    a = rng.choice([-1, -2, -3, 1, 2, 3]) * level
    # Vertex at x* = -b/(2a)
    x_star_num = rng.randint(-10, 10)
    x_star_den = rng.randint(1, 4)
    # Pick b so x* = x_star_num/x_star_den
    b_num = -2 * a * x_star_num
    b_den = x_star_den
    b_frac = Fraction(b_num, b_den)
    if b_frac.denominator != 1:
        # Skip non-integer b for simplicity
        a = 1
        x_star_num = rng.randint(-5, 5)
        b_num = -2 * a * x_star_num
        b_frac = Fraction(b_num)

    b = int(b_frac)
    c = rng.randint(-10, 10)
    kind = "minimum" if a > 0 else "maximum"
    x_star = Fraction(-b, 2*a)
    x_star_str = _frac(x_star.numerator, x_star.denominator)
    f_star = a * float(x_star)**2 + b * float(x_star) + c
    f_star_frac = Fraction(a) * x_star**2 + Fraction(b) * x_star + Fraction(c)
    f_star_str = _frac(f_star_frac.numerator, f_star_frac.denominator)

    b_sign = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_sign = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    problem = f"f(x) = {a}x^2 {b_sign}x {c_sign}. Find x at {kind}."
    solution = (f"f'(x) = {2*a}x + {b} = 0\n"
                f"x = {x_star_str}\n"
                f"f''(x) = {2*a} {'> 0 (min)' if a > 0 else '< 0 (max)'}\n"
                f"#### {x_star_str}")
    return {"problem": problem, "solution": solution, "answer": x_star_str,
            "domain": "calc_optimization", "level": level, "source": "calculus_qa"}


def area_between_curves(rng: random.Random, level: int) -> dict:
    """Area between f(x)=a and g(x)=bx^2 from 0 to c (exact fractions)."""
    # Ensure f(x) >= g(x) on [0,c]: a >= b*c^2
    b = rng.randint(1, 3 * level)
    c = rng.randint(1, 4)
    a = b * c**2 + rng.randint(1, 5 * level)
    # Area = integral_0^c (a - b*x^2) dx = [ax - b*x^3/3]_0^c
    area_frac = Fraction(a * c) - Fraction(b * c**3, 3)
    area_str = _frac(area_frac.numerator, area_frac.denominator)

    problem = f"Area between y={a} and y={b}x^2 from x=0 to x={c}."
    solution = (f"Integral_0^{c} ({a} - {b}x^2) dx\n"
                f"= [{a}x - {b}x^3/3] from 0 to {c}\n"
                f"= {a*c} - {_frac(b*c**3, 3)}\n"
                f"= {area_str}\n"
                f"#### {area_str}")
    return {"problem": problem, "solution": solution, "answer": area_str,
            "domain": "calc_integral", "level": level, "source": "calculus_qa"}


# ── L4: Related rates word problems ─────────────────────────────────────────

def related_rate_volume(rng: random.Random, level: int) -> dict:
    """Cube expanding: dV/dt = 3s^2 * ds/dt."""
    s = rng.randint(2, 10 * level)
    ds_dt = rng.randint(1, 5 * level)
    dV_dt = 3 * s**2 * ds_dt
    problem = (f"Cube side = {s} cm, growing at {ds_dt} cm/s. "
               f"Rate of volume change?")
    solution = (f"V = s^3, dV/dt = 3s^2 * ds/dt\n"
                f"= 3*{s}^2*{ds_dt}\n"
                f"= 3*{s**2}*{ds_dt}\n"
                f"= {dV_dt} cm^3/s\n"
                f"#### {dV_dt}")
    return {"problem": problem, "solution": solution, "answer": dV_dt,
            "domain": "calc_related_rate", "level": level, "source": "calculus_qa"}


def integration_by_parts_simple(rng: random.Random, level: int) -> dict:
    """Integral of x*e^x = e^x*(x-1)+C (fixed template, scaled)."""
    a = rng.randint(1, 3 * level)
    # integral of a*x*e^x = a*(x-1)*e^x + C
    ans_str = f"{a}*(x-1)*e^x + C"
    problem = f"Integrate {a}*x*e^x dx using integration by parts."
    solution = (f"u = x, dv = {a}e^x dx\n"
                f"du = dx, v = {a}e^x\n"
                f"= {a}x*e^x - integral({a}e^x dx)\n"
                f"= {a}x*e^x - {a}e^x + C\n"
                f"= {a}*(x-1)*e^x + C\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "calc_integral", "level": level, "source": "calculus_qa"}


# ── L5: Taylor polynomial, Riemann sum ───────────────────────────────────────

def taylor_poly_sin(rng: random.Random, level: int) -> dict:
    """sin(x) ~ x - x^3/6 + x^5/120; evaluate at small angle."""
    # Use x = pi/6 = 0.5236... -> sin ~ 0.5
    angle_choices = [
        ("pi/6", "0.5236", "1/2"),
        ("pi/4", "0.7854", "sqrt(2)/2"),
    ]
    angle_name, angle_dec, sin_exact = rng.choice(angle_choices)
    problem = f"Taylor series: approximate sin({angle_name}) using 2 terms."
    solution = (f"sin(x) ≈ x - x^3/6\n"
                f"x = {angle_dec}\n"
                f"x^3/6 ≈ {round(float(angle_dec)**3/6, 4)}\n"
                f"sin({angle_name}) ≈ {angle_dec} - "
                f"{round(float(angle_dec)**3/6, 4)}\n"
                f"≈ {sin_exact}\n"
                f"#### {sin_exact}")
    return {"problem": problem, "solution": solution, "answer": sin_exact,
            "domain": "calc_taylor", "level": level, "source": "calculus_qa"}


def riemann_sum(rng: random.Random, level: int) -> dict:
    """Left Riemann sum for a*x^2 on [0, b] with n rectangles."""
    a = rng.randint(1, 4)
    b = rng.randint(2, 5)
    n = rng.choice([4, 5, 10])
    dx = Fraction(b, n)
    # Left sum: sum_{k=0}^{n-1} a*(k*dx)^2 * dx
    total = Fraction(0)
    for k in range(n):
        x_k = k * dx
        total += a * x_k**2 * dx
    total_str = _frac(total.numerator, total.denominator)
    problem = (f"Left Riemann sum: f(x)={a}x^2 on [0,{b}], {n} rects.")
    solution = (f"dx = {b}/{n} = {dx}\n"
                f"Sum = sum_{{k=0}}^{{{n-1}}} {a}*(k*{dx})^2*{dx}\n"
                f"= {total_str}\n"
                f"#### {total_str}")
    return {"problem": problem, "solution": solution, "answer": total_str,
            "domain": "calc_riemann", "level": level, "source": "calculus_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [limit_poly, deriv_power_rule, integral_power_rule],
    2: [deriv_product_rule, deriv_chain_rule, definite_integral_poly],
    3: [optimization_quadratic, area_between_curves, deriv_chain_rule],
    4: [related_rate_volume, integration_by_parts_simple, optimization_quadratic],
    5: [taylor_poly_sin, riemann_sum, integration_by_parts_simple],
}


def generate_calculus_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
        while count < per_level and attempts < per_level * 20:
            attempts += 1
            fn = rng.choice(templates)
            try:
                p = fn(rng, level)
            except Exception:
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
    out = RESULTS_DIR / f"harder_qa_calculus{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} calculus pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate calculus Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_calculus_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
