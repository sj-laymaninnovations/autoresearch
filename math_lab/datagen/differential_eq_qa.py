"""
differential_eq_qa.py — Differential Equations Q/A Generator

Topics by level:
  L1: Exponential ODE dy/dx = ky -> y = Ce^(kx), apply IC
  L2: Separable ODEs: separate variables, integrate both sides, apply IC
  L3: First-order linear ODE with integrating factor
  L4: Second-order constant-coeff: characteristic equation -> general solution
  L5: Euler's method (3 steps shown explicitly), logistic ODE analysis

Number formats: constants C, k, and step values appear as fractions,
decimals, and sci-notation throughout. The CoT always shows arithmetic
explicitly so the model learns both the calculus and the arithmetic.

Output: math_lab/results/harder_qa_diffeq_<tag>_<timestamp>.jsonl

References:
  - Boyce & DiPrima, Elementary Differential Equations (10th ed.)
  - MIT OCW 18.03 Differential Equations
  - Paul's Online Math Notes - Differential Equations
"""

import json
import math
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


def _dec(f, places: int = 4) -> str:
    s = f"{float(f):.{places}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _sci(val: float, places: int = 3) -> str:
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    m = round(val / 10**exp, places)
    ms = f"{m:.{places}f}".rstrip("0").rstrip(".")
    return f"{ms}e{exp}"


# ── L1: Exponential ODE dy/dx = ky ───────────────────────────────────────────

def exponential_ode(rng: random.Random, level: int) -> dict:
    """dy/dx = k*y, y(0) = y0 -> y = y0*e^(kx). Find y(t)."""
    k = rng.randint(-3*level, 3*level)
    while k == 0:
        k = rng.randint(-3, 3)
    y0 = rng.randint(1, 10*level)
    t = rng.randint(1, 3)
    # Answer is symbolic: y0*e^(k*t)
    exp_val = k * t
    ans_sym = f"{y0}*e^({exp_val})"
    ans_approx = round(y0 * math.exp(exp_val), 4)
    ans_sci = _sci(ans_approx)
    k_str = str(k)
    problem = f"Solve: dy/dx = {k_str}y, y(0)={y0}. Find y({t})."
    solution = (f"General: y = C*e^({k_str}x)\n"
                f"IC y(0)={y0}: C = {y0}\n"
                f"y = {y0}*e^({k_str}x)\n"
                f"y({t}) = {ans_sym} ≈ {ans_approx} = {ans_sci}\n"
                f"#### {ans_sym}")
    return {"problem": problem, "solution": solution, "answer": ans_sym,
            "domain": "diffeq_exponential", "level": level, "source": "diffeq_qa"}


def exponential_decay_half_life(rng: random.Random, level: int) -> dict:
    """y = y0*e^(kt); find half-life (t when y = y0/2)."""
    y0 = rng.randint(10, 100 * level)
    k = -rng.randint(1, 5 * level)   # negative for decay
    # half-life: ln(2)/|k|
    hl_approx = round(math.log(2) / abs(k), 4)
    hl_sci = _sci(hl_approx)
    problem = (f"Decay: y = {y0}*e^({k}t). "
               f"Approximate half-life (ln2≈0.6931)?")
    solution = (f"y0/2 = {y0}*e^({k}t)\n"
                f"e^({k}t) = 1/2\n"
                f"{k}t = ln(1/2) = -ln(2) ≈ -0.6931\n"
                f"t = 0.6931 / {abs(k)} ≈ {hl_approx} = {hl_sci}\n"
                f"#### {hl_approx}")
    return {"problem": problem, "solution": solution, "answer": hl_approx,
            "domain": "diffeq_exponential", "level": level, "source": "diffeq_qa"}


# ── L2: Separable ODEs ────────────────────────────────────────────────────────

def separable_simple(rng: random.Random, level: int) -> dict:
    """dy/dx = x/y -> y*dy = x*dx -> y^2/2 = x^2/2 + C -> y^2 - x^2 = K."""
    # y dy = a x dx -> y^2 = a*x^2 + C
    a = rng.randint(1, 5*level)
    # IC: y(0) = y0
    y0 = rng.randint(1, 8*level)
    C = y0**2   # y^2 = a*x^2 + y0^2
    # Evaluate at x=1: y(1)^2 = a + y0^2
    val = a + y0**2
    y1 = round(val**0.5, 4)
    y1_sci = _sci(y1)
    problem = f"Solve: y*dy/dx = {a}x, y(0)={y0}. Find y(1)."
    solution = (f"y dy = {a}x dx\n"
                f"y^2/2 = {a}x^2/2 + C\n"
                f"y^2 = {a}x^2 + C\n"
                f"IC: {y0}^2 = C => C = {C}\n"
                f"y^2 = {a}x^2 + {C}\n"
                f"y(1)^2 = {a} + {C} = {val}\n"
                f"y(1) = sqrt({val}) ≈ {y1} = {y1_sci}\n"
                f"#### {y1}")
    return {"problem": problem, "solution": solution, "answer": y1,
            "domain": "diffeq_separable", "level": level, "source": "diffeq_qa"}


def separable_product(rng: random.Random, level: int) -> dict:
    """dy/dx = (a*x^n) / y -> y*dy = a*x^n*dx."""
    a = rng.randint(1, 4*level)
    n = rng.randint(1, 3)
    y0 = rng.randint(1, 6*level)
    # y^2/2 = a*x^(n+1)/(n+1) + C
    C_num = y0**2
    # At x=1: y^2 = 2a/(n+1) + y0^2
    two_a_over = Fraction(2*a, n+1)
    y1_sq = two_a_over + y0**2
    y1 = round(float(y1_sq)**0.5, 4)
    problem = f"Separable: y*dy/dx = {a}x^{n}, y(0)={y0}. y(1)≈?"
    solution = (f"y dy = {a}x^{n} dx\n"
                f"y^2/2 = {a}x^{n+1}/{n+1} + C\n"
                f"IC: C = {y0}^2/2 = {_frac(y0**2,2)}\n"
                f"y(1)^2 = 2*{a}/{n+1} + {y0}^2 = {float(y1_sq):.4f}\n"
                f"y(1) ≈ {y1}\n"
                f"#### {y1}")
    return {"problem": problem, "solution": solution, "answer": y1,
            "domain": "diffeq_separable", "level": level, "source": "diffeq_qa"}


# ── L3: First-order linear ODE (integrating factor) ───────────────────────────

def integrating_factor(rng: random.Random, level: int) -> dict:
    """y' + P*y = Q (constant P, Q): IF = e^(Px), y = (Q/P) + C*e^(-Px)."""
    P = rng.randint(1, 3*level)
    Q = rng.randint(-5*level, 5*level)
    y0 = rng.randint(-5*level, 5*level)
    # Particular: Q/P; general: C*e^(-Px)
    # IC y(0) = y0: C = y0 - Q/P
    QoverP = Fraction(Q, P)
    C_frac = Fraction(y0) - QoverP
    QP_str = _frac(QoverP.numerator, QoverP.denominator)
    C_str = _frac(C_frac.numerator, C_frac.denominator)
    C_dec = _dec(C_frac)
    problem = f"Solve: y' + {P}y = {Q}, y(0) = {y0}. Find C in general solution."
    solution = (f"Integrating factor: mu = e^({P}x)\n"
                f"d/dx[e^({P}x)*y] = {Q}*e^({P}x)\n"
                f"y = {Q}/{P} + C*e^(-{P}x)\n"
                f"IC: {y0} = {QP_str} + C => C = {C_str} = {C_dec}\n"
                f"#### C={C_dec}")
    answer = f"C={C_dec}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "diffeq_linear", "level": level, "source": "diffeq_qa"}


# ── L4: Second-order constant-coefficient ────────────────────────────────────

def second_order_real_roots(rng: random.Random, level: int) -> dict:
    """a*y'' + b*y' + c*y = 0 with two distinct real roots."""
    # Pick roots r1, r2 -> polynomial (L-r1)(L-r2) = L^2 - (r1+r2)L + r1*r2
    r1 = rng.randint(-4*level, 0)
    r2 = rng.randint(-4*level, 0)
    # Ensure distinct
    while r2 == r1:
        r2 = rng.randint(-4, 0)
    a = 1
    b = -(r1 + r2)
    c = r1 * r2
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    problem = f"Solve: y'' {b_str}y' {c_str}y = 0. General solution?"
    answer = f"y = C1*e^({r1}x) + C2*e^({r2}x)"
    solution = (f"Char eq: r^2 {b_str}r {c_str} = 0\n"
                f"(r-({r1}))(r-({r2})) = 0\n"
                f"r = {r1}, r = {r2}\n"
                f"y = C1*e^({r1}x) + C2*e^({r2}x)\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "diffeq_2ndorder", "level": level, "source": "diffeq_qa"}


def second_order_repeated_root(rng: random.Random, level: int) -> dict:
    """y'' + 2r*y' + r^2*y = 0 -> (r-r0)^2 = 0 -> y = (C1+C2x)*e^(r0x)."""
    r0 = rng.randint(-3*level, -1)
    b = -2 * r0
    c = r0**2
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    answer = f"y = (C1+C2x)*e^({r0}x)"
    problem = f"Solve: y'' {b_str}y' {c_str}y = 0 (repeated root)."
    solution = (f"Char eq: r^2 {b_str}r {c_str} = (r-({r0}))^2 = 0\n"
                f"Repeated root: r = {r0}\n"
                f"y = (C1+C2x)*e^({r0}x)\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "diffeq_2ndorder", "level": level, "source": "diffeq_qa"}


# ── L5: Euler's method ────────────────────────────────────────────────────────

def eulers_method(rng: random.Random, level: int) -> dict:
    """y' = f(x,y) = k*y; 3 Euler steps from x0, y0 with step h."""
    k = rng.choice([1, 2, -1, -2])
    y0_raw = rng.randint(1, 5*level)
    y0 = Fraction(y0_raw)
    h = Fraction(rng.choice([1, 2, 5]), 10)  # 0.1, 0.2, 0.5
    x0 = Fraction(0)
    h_dec = _dec(h)
    h_sci = _sci(float(h))

    # 3 steps
    xs = [x0]
    ys = [y0]
    for i in range(3):
        f_val = k * ys[-1]
        y_next = ys[-1] + h * f_val
        xs.append(xs[-1] + h)
        ys.append(y_next)

    y3 = ys[3]
    y3_frac = _frac(y3.numerator, y3.denominator)
    y3_dec = _dec(y3)
    y3_sci = _sci(float(y3))

    problem = (f"Euler's method: y'={k}y, y(0)={y0_raw}, h={h_dec}={h_sci}. "
               f"y after 3 steps?")
    lines = []
    for i in range(3):
        fi = k * ys[i]
        lines.append(f"Step {i+1}: y = {_dec(ys[i])} + {h_dec}*{k}*{_dec(ys[i])} "
                     f"= {_dec(ys[i+1])}")
    solution = "\n".join(lines) + f"\n= {y3_frac} = {y3_dec} = {y3_sci}\n#### {y3_dec}"
    return {"problem": problem, "solution": solution, "answer": y3_dec,
            "domain": "diffeq_euler", "level": level, "source": "diffeq_qa"}


def logistic_equilibria(rng: random.Random, level: int) -> dict:
    """dP/dt = rP(1-P/K); find equilibria P=0 and P=K."""
    r = rng.randint(1, 5*level)
    K = rng.randint(100, 1000*level)
    problem = f"Logistic ODE: dP/dt = {r}P(1-P/{K}). Find equilibria."
    answer = f"P=0, P={K}"
    solution = (f"dP/dt = {r}P(1-P/{K}) = 0\n"
                f"=> P = 0 or 1-P/{K} = 0\n"
                f"=> P = 0 or P = {K}\n"
                f"Stable: P={K}, Unstable: P=0\n"
                f"#### P=0, P={K}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "diffeq_logistic", "level": level, "source": "diffeq_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [exponential_ode, exponential_decay_half_life],
    2: [separable_simple, separable_product],
    3: [integrating_factor, separable_simple],
    4: [second_order_real_roots, second_order_repeated_root],
    5: [eulers_method, logistic_equilibria, second_order_real_roots],
}


def generate_diffeq_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_diffeq{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} diff eq pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate differential equations Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_diffeq_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
