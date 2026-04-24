"""
linear_regression_qa.py — Linear Regression Q/A Generator

Topics by level:
  L1: Compute x-bar, y-bar for small datasets
  L2: Slope b1 = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2), intercept b0
  L3: Full OLS on 4-5 points: slope, intercept, prediction
  L4: R-squared: SST, SSR, SSE decomposition
  L5: Gradient descent single step (w1 -= lr * dL/dw1)

Number formats: slopes, intercepts, predictions appear as fractions,
decimals, sci-notation, and floats. The model learns that 0.5, 1/2,
5e-1, and 5.0E-1 are all valid representations.

Output: math_lab/results/harder_qa_linreg_<tag>_<timestamp>.jsonl

References:
  - Freedman, Pisani & Purves, Statistics (4th ed.)
  - Andrew Ng, Machine Learning (Coursera)
  - sklearn LinearRegression documentation
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


def _dec(f: Fraction, places: int = 4) -> str:
    s = f"{float(f):.{places}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _sci(val: float, places: int = 3) -> str:
    if val == 0:
        return "0"
    import math as _m
    exp = int(_m.floor(_m.log10(abs(val))))
    m = round(val / 10**exp, places)
    ms = f"{m:.{places}f}".rstrip("0").rstrip(".")
    return f"{ms}e{exp}"


def _ols(xs, ys):
    """Return (b1, b0) as Fraction objects."""
    n = len(xs)
    xbar = Fraction(sum(xs), n)
    ybar = Fraction(sum(ys), n)
    num = sum((Fraction(x) - xbar) * (Fraction(y) - ybar)
              for x, y in zip(xs, ys))
    den = sum((Fraction(x) - xbar) ** 2 for x in xs)
    if den == 0:
        return Fraction(0), ybar
    b1 = num / den
    b0 = ybar - b1 * xbar
    return b1, b0


# ── L1: Compute means ─────────────────────────────────────────────────────────

def compute_means(rng: random.Random, level: int) -> dict:
    """Given n (x,y) pairs, find x-bar and y-bar."""
    n = rng.randint(3, 5 + level)
    xs = [rng.randint(1, 20 * level) for _ in range(n)]
    ys = [rng.randint(1, 20 * level) for _ in range(n)]
    xbar = Fraction(sum(xs), n)
    ybar = Fraction(sum(ys), n)
    xs_str = ", ".join(str(x) for x in xs)
    ys_str = ", ".join(str(y) for y in ys)
    xb_str = _frac(xbar.numerator, xbar.denominator)
    yb_str = _frac(ybar.numerator, ybar.denominator)
    problem = f"x=[{xs_str}], y=[{ys_str}]. Find x-bar and y-bar."
    solution = (f"sum(x)={sum(xs)}, n={n}, x-bar={sum(xs)}/{n}={xb_str}\n"
                f"sum(y)={sum(ys)}, y-bar={sum(ys)}/{n}={yb_str}\n"
                f"#### x-bar={xb_str}, y-bar={yb_str}")
    answer = f"xbar={xb_str}, ybar={yb_str}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linreg_means", "level": level, "source": "linreg_qa"}


def slope_two_points(rng: random.Random, level: int) -> dict:
    """Slope between two points (basic prerequisite)."""
    x1 = rng.randint(-10*level, 10*level)
    x2 = rng.randint(-10*level, 10*level)
    while x2 == x1:
        x2 = rng.randint(-10*level, 10*level)
    y1 = rng.randint(-10*level, 10*level)
    y2 = rng.randint(-10*level, 10*level)
    slope_frac = Fraction(y2 - y1, x2 - x1)
    ans = _frac(slope_frac.numerator, slope_frac.denominator)
    ans_dec = _dec(slope_frac)
    problem = f"Slope through ({x1},{y1}) and ({x2},{y2})?"
    solution = (f"slope = (y2-y1)/(x2-x1)\n"
                f"= ({y2}-{y1})/({x2}-{x1})\n"
                f"= {y2-y1}/{x2-x1} = {ans} = {ans_dec}\n"
                f"#### {ans}")
    return {"problem": problem, "solution": solution, "answer": ans,
            "domain": "linreg_slope", "level": level, "source": "linreg_qa"}


# ── L2: OLS slope and intercept ───────────────────────────────────────────────

def ols_3points(rng: random.Random, level: int) -> dict:
    """Least-squares slope for 3-point dataset."""
    # Use backward generation: pick slope and intercept, derive y
    b1_true = rng.randint(-5*level, 5*level)
    b0_true = rng.randint(-10*level, 10*level)
    xs = [rng.randint(1, 10*level) for _ in range(3)]
    ys = [b1_true * x + b0_true for x in xs]

    xbar = Fraction(sum(xs), 3)
    ybar = Fraction(sum(ys), 3)
    b1_frac, b0_frac = _ols(xs, ys)

    xs_str = ", ".join(str(x) for x in xs)
    ys_str = ", ".join(str(y) for y in ys)
    b1_str = _frac(b1_frac.numerator, b1_frac.denominator)
    b0_str = _frac(b0_frac.numerator, b0_frac.denominator)
    b1_dec = _dec(b1_frac)
    b0_dec = _dec(b0_frac)

    sum_xy = sum((Fraction(x)-xbar)*(Fraction(y)-ybar) for x,y in zip(xs,ys))
    sum_xx = sum((Fraction(x)-xbar)**2 for x in xs)

    problem = f"OLS slope: x=[{xs_str}], y=[{ys_str}]."
    solution = (f"xbar={_frac(xbar.numerator,xbar.denominator)}, "
                f"ybar={_frac(ybar.numerator,ybar.denominator)}\n"
                f"sum(xi-xbar)(yi-ybar)={float(sum_xy):.2f}\n"
                f"sum(xi-xbar)^2={float(sum_xx):.2f}\n"
                f"b1 = {b1_str} = {b1_dec}\n"
                f"#### b1={b1_dec}")
    return {"problem": problem, "solution": solution, "answer": b1_dec,
            "domain": "linreg_ols", "level": level, "source": "linreg_qa"}


def ols_line(rng: random.Random, level: int) -> dict:
    """Full regression line: slope + intercept for 4-point set."""
    b1_true = rng.randint(-3*level, 3*level)
    b0_true = rng.randint(-5*level, 5*level)
    n = 4
    xs = [rng.randint(1, 8*level) for _ in range(n)]
    ys = [b1_true * x + b0_true for x in xs]
    b1_frac, b0_frac = _ols(xs, ys)
    xs_str = ", ".join(str(x) for x in xs)
    ys_str = ", ".join(str(y) for y in ys)
    b1_str = _frac(b1_frac.numerator, b1_frac.denominator)
    b0_str = _frac(b0_frac.numerator, b0_frac.denominator)
    b1_dec = _dec(b1_frac)
    b0_dec = _dec(b0_frac)
    problem = f"Regression line: x=[{xs_str}], y=[{ys_str}]. Find slope and intercept."
    solution = (f"OLS: b1={b1_str} ({b1_dec})\n"
                f"b0 = ybar - b1*xbar = {b0_str} ({b0_dec})\n"
                f"y-hat = {b1_dec}x + {b0_dec}\n"
                f"#### b1={b1_dec}, b0={b0_dec}")
    answer = f"b1={b1_dec}, b0={b0_dec}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linreg_ols", "level": level, "source": "linreg_qa"}


# ── L3: Prediction ────────────────────────────────────────────────────────────

def predict_y(rng: random.Random, level: int) -> dict:
    """y-hat = b1*x_new + b0."""
    b1 = rng.randint(-5*level, 5*level)
    b0 = rng.randint(-10*level, 10*level)
    x_new = rng.randint(1, 20*level)
    y_hat = b1 * x_new + b0
    b1_frac = Fraction(b1)
    b0_frac = Fraction(b0)
    # Show answer in multiple formats
    ans_dec = str(y_hat)
    ans_sci = _sci(float(y_hat)) if y_hat != 0 else "0"
    problem = f"Regression: b1={b1}, b0={b0}. Predict y at x={x_new}."
    solution = (f"y-hat = b1*x + b0\n"
                f"= {b1}*{x_new} + {b0}\n"
                f"= {b1*x_new} + {b0}\n"
                f"= {y_hat} = {ans_sci}\n"
                f"#### {y_hat}")
    return {"problem": problem, "solution": solution, "answer": y_hat,
            "domain": "linreg_predict", "level": level, "source": "linreg_qa"}


def residual_calc(rng: random.Random, level: int) -> dict:
    """Residual e = y - y-hat."""
    b1 = rng.randint(-4*level, 4*level)
    b0 = rng.randint(-8*level, 8*level)
    x = rng.randint(1, 15*level)
    noise = rng.randint(-5*level, 5*level)
    y_hat = b1 * x + b0
    y = y_hat + noise
    residual = y - y_hat
    problem = f"y-hat={b1}x+{b0}, x={x}, y={y}. Residual?"
    solution = (f"y-hat = {b1}*{x}+{b0} = {y_hat}\n"
                f"e = y - y-hat = {y} - {y_hat} = {residual}\n"
                f"#### {residual}")
    return {"problem": problem, "solution": solution, "answer": residual,
            "domain": "linreg_residual", "level": level, "source": "linreg_qa"}


# ── L4: R-squared ─────────────────────────────────────────────────────────────

def r_squared(rng: random.Random, level: int) -> dict:
    """R^2 = 1 - SSE/SST using exact integer arithmetic."""
    b1_true = rng.randint(1, 4*level)
    b0_true = rng.randint(-5*level, 5*level)
    n = 4
    xs = [rng.randint(1, 8*level) for _ in range(n)]
    ys = [b1_true * x + b0_true + rng.randint(-2, 2) for x in xs]
    b1_frac, b0_frac = _ols(xs, ys)
    ybar = Fraction(sum(ys), n)
    y_hats = [b1_frac * Fraction(x) + b0_frac for x in xs]
    SSE = sum((Fraction(y) - yh)**2 for y, yh in zip(ys, y_hats))
    SST = sum((Fraction(y) - ybar)**2 for y in ys)
    R2 = 1 - SSE / SST if SST != 0 else Fraction(1)
    R2_str = _frac(R2.numerator, R2.denominator)
    R2_dec = _dec(R2)
    xs_str = ", ".join(str(x) for x in xs)
    ys_str = ", ".join(str(y) for y in ys)
    problem = f"R^2: x=[{xs_str}], y=[{ys_str}]."
    solution = (f"Fit OLS line, compute SSE={float(SSE):.4f}, SST={float(SST):.4f}\n"
                f"R^2 = 1 - SSE/SST = {R2_str}\n"
                f"= {R2_dec}\n"
                f"#### {R2_dec}")
    return {"problem": problem, "solution": solution, "answer": R2_dec,
            "domain": "linreg_r2", "level": level, "source": "linreg_qa"}


# ── L5: Gradient descent step ─────────────────────────────────────────────────

def gradient_descent_step(rng: random.Random, level: int) -> dict:
    """One GD update: w -= lr * dL/dw for MSE on single sample."""
    # MSE = (y_hat - y)^2 = (w*x - y)^2
    # dL/dw = 2*(w*x-y)*x
    # w_new = w - lr * 2*(w*x-y)*x
    w = Fraction(rng.randint(-5, 5))
    x = Fraction(rng.randint(1, 5*level))
    y = Fraction(rng.randint(-10*level, 10*level))
    lr = Fraction(rng.choice([1, 2, 5, 10]), 100)  # 0.01, 0.02, 0.05, 0.10
    error = w * x - y
    grad = 2 * error * x
    w_new = w - lr * grad
    w_str = _frac(w.numerator, w.denominator)
    lr_str = _dec(lr)
    lr_sci = _sci(float(lr))
    w_new_str = _frac(w_new.numerator, w_new.denominator)
    w_new_dec = _dec(w_new)
    problem = (f"GD step: w={w_str}, x={x}, y={y}, lr={lr_dec}={lr_sci}. "
               f"New w?").replace("lr_dec", lr_str)
    solution = (f"error = w*x - y = {w_str}*{x}-{y} = {_frac(error.numerator,error.denominator)}\n"
                f"grad = 2*error*x = {_frac(grad.numerator,grad.denominator)}\n"
                f"w_new = {w_str} - {lr_str}*{_frac(grad.numerator,grad.denominator)}\n"
                f"= {w_new_str} = {w_new_dec}\n"
                f"#### {w_new_dec}")
    return {"problem": problem, "solution": solution, "answer": w_new_dec,
            "domain": "linreg_gradient_descent", "level": level, "source": "linreg_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [compute_means, slope_two_points],
    2: [ols_3points, ols_line, slope_two_points],
    3: [predict_y, residual_calc, ols_line],
    4: [r_squared, predict_y, ols_line],
    5: [gradient_descent_step, r_squared, predict_y],
}


def generate_linreg_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_linreg{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} linear regression pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate linear regression Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_linreg_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
