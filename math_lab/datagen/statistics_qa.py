"""
statistics_qa.py — Statistics & Probability Q/A Generator

Topics by level:
  L1: Mean, median, mode, basic probability P(A)
  L2: Variance, std dev, conditional probability P(A|B), nCr/nPr
  L3: Bayes theorem, expected value, z-score
  L4: Pearson correlation, binomial PMF, law of total probability
  L5: Confidence interval (CLT), hypothesis test z-statistic

Number formats: stats values appear as fractions, decimals, sci-notation,
and percentages to teach format equivalence in a probability context.

Output: math_lab/results/harder_qa_stats_<tag>_<timestamp>.jsonl

References:
  - Khan Academy Statistics & Probability
  - OpenIntro Statistics (Diez, Barr, Cetinkaya-Rundel)
  - AP Statistics free-response problem bank
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


def _pct(f: Fraction, places: int = 2) -> str:
    return f"{float(f)*100:.{places}f}".rstrip("0").rstrip(".") + "%"


def _sci(val: float, places: int = 4) -> str:
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    m = round(val / 10**exp, places)
    ms = f"{m:.{places}f}".rstrip("0").rstrip(".")
    return f"{ms}e{exp}"


# ── L1: Mean, median, mode, basic probability ─────────────────────────────────

def mean_of_list(rng: random.Random, level: int) -> dict:
    n = rng.randint(3, 6 + level)
    total = rng.randint(n * 5, n * 20 * level)
    # Build list summing to total
    vals = [rng.randint(1, 30 * level) for _ in range(n - 1)]
    vals.append(total - sum(vals))
    if vals[-1] < 0:
        vals = [abs(v) for v in vals]
        total = sum(vals)
    mean_frac = Fraction(total, n)
    # Show answer in multiple formats based on whether exact
    if mean_frac.denominator == 1:
        ans = str(mean_frac.numerator)
    else:
        ans = _frac(mean_frac.numerator, mean_frac.denominator)
    vals_str = ", ".join(str(v) for v in vals)
    problem = f"Mean of [{vals_str}]?"
    solution = (f"sum = {'+'.join(str(v) for v in vals)} = {total}\n"
                f"mean = {total}/{n} = {ans}\n"
                f"#### {ans}")
    return {"problem": problem, "solution": solution, "answer": ans,
            "domain": "stats_mean", "level": level, "source": "stats_qa"}


def median_of_list(rng: random.Random, level: int) -> dict:
    n = rng.choice([3, 5, 7])  # odd for clean median
    vals = sorted(rng.randint(1, 50 * level) for _ in range(n))
    median = vals[n // 2]
    vals_str = ", ".join(str(v) for v in vals)
    problem = f"Median of [{vals_str}]?"
    solution = (f"Sorted: {vals_str}\n"
                f"Middle ({n//2+1} of {n}): {median}\n"
                f"#### {median}")
    return {"problem": problem, "solution": solution, "answer": median,
            "domain": "stats_median", "level": level, "source": "stats_qa"}


def basic_probability(rng: random.Random, level: int) -> dict:
    """P(event) = favorable/total as fraction, decimal, and percentage."""
    total = rng.randint(5, 20 * level)
    fav = rng.randint(1, total)
    prob_frac = Fraction(fav, total)
    prob_dec = _dec(prob_frac)
    prob_pct = _pct(prob_frac)
    # Random format for answer
    fmt = rng.choice(["fraction", "decimal", "percentage"])
    if fmt == "fraction":
        ans = _frac(prob_frac.numerator, prob_frac.denominator)
    elif fmt == "decimal":
        ans = prob_dec
    else:
        ans = prob_pct
    problem = f"P(event): {fav} favorable out of {total}. Express as {fmt}."
    solution = (f"P = {fav}/{total}\n"
                f"= {_frac(prob_frac.numerator, prob_frac.denominator)}\n"
                f"= {prob_dec} = {prob_pct}\n"
                f"#### {ans}")
    return {"problem": problem, "solution": solution, "answer": ans,
            "domain": "stats_probability", "level": level, "source": "stats_qa"}


# ── L2: Variance, std dev, conditional prob, nCr ─────────────────────────────

def variance_std(rng: random.Random, level: int) -> dict:
    """Population variance and std dev of small integer list."""
    n = rng.randint(3, 6)
    vals = [rng.randint(1, 20 * level) for _ in range(n)]
    mean_f = Fraction(sum(vals), n)
    var_f = sum((Fraction(v) - mean_f)**2 for v in vals) / n
    var_str = _frac(var_f.numerator, var_f.denominator)
    std_float = float(var_f) ** 0.5
    std_str = f"{round(std_float, 4)}"
    vals_str = ", ".join(str(v) for v in vals)
    problem = f"Population variance of [{vals_str}]?"
    solution = (f"mean = {_frac(mean_f.numerator,mean_f.denominator)}\n"
                f"var = sum((x-mean)^2)/n = {var_str}\n"
                f"std = sqrt({var_str}) ≈ {std_str}\n"
                f"#### {var_str}")
    return {"problem": problem, "solution": solution, "answer": var_str,
            "domain": "stats_variance", "level": level, "source": "stats_qa"}


def conditional_prob(rng: random.Random, level: int) -> dict:
    """P(A|B) = P(A and B)/P(B) with integer counts."""
    total = rng.randint(20, 100 * level)
    n_b = rng.randint(5, total // 2)
    n_ab = rng.randint(1, n_b)
    p_b = Fraction(n_b, total)
    p_ab = Fraction(n_ab, total)
    p_a_given_b = Fraction(n_ab, n_b)
    fmt = rng.choice(["fraction", "decimal"])
    ans = (_frac(p_a_given_b.numerator, p_a_given_b.denominator)
           if fmt == "fraction" else _dec(p_a_given_b))
    problem = (f"N={total}, P(B)={n_b}/{total}, P(A∩B)={n_ab}/{total}. "
               f"Find P(A|B) as {fmt}.")
    solution = (f"P(A|B) = P(A∩B)/P(B)\n"
                f"= ({n_ab}/{total}) / ({n_b}/{total})\n"
                f"= {n_ab}/{n_b}\n"
                f"= {ans}\n"
                f"#### {ans}")
    return {"problem": problem, "solution": solution, "answer": ans,
            "domain": "stats_probability", "level": level, "source": "stats_qa"}


def combinations(rng: random.Random, level: int) -> dict:
    """C(n,r) = n! / (r! * (n-r)!)."""
    n = rng.randint(4, 12 + 2*level)
    r = rng.randint(1, min(n // 2, 6))
    answer = math.comb(n, r)
    problem = f"How many ways to choose {r} from {n}? (C({n},{r}))"
    solution = (f"C({n},{r}) = {n}! / ({r}! * {n-r}!)\n"
                f"= {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "stats_counting", "level": level, "source": "stats_qa"}


def permutations(rng: random.Random, level: int) -> dict:
    """P(n,r) = n!/(n-r)!."""
    n = rng.randint(4, 10 + 2*level)
    r = rng.randint(1, min(n, 5))
    answer = math.perm(n, r)
    problem = f"Arrangements of {r} from {n} distinct items? (P({n},{r}))"
    solution = (f"P({n},{r}) = {n}! / ({n-r})!\n"
                f"= {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "stats_counting", "level": level, "source": "stats_qa"}


# ── L3: Bayes, expected value, z-score ────────────────────────────────────────

def bayes_theorem(rng: random.Random, level: int) -> dict:
    """P(A|B) via Bayes, given P(B|A), P(A), P(B|~A) as fractions."""
    # Use small denominators to stay in integers
    p_a = Fraction(rng.randint(1, 4), 10)
    p_b_given_a = Fraction(rng.randint(6, 9), 10)
    p_b_given_na = Fraction(rng.randint(1, 4), 10)
    p_na = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_na * p_na
    p_a_given_b = (p_b_given_a * p_a) / p_b
    ans = _frac(p_a_given_b.numerator, p_a_given_b.denominator)
    ans_dec = _dec(p_a_given_b)
    problem = (f"P(A)={_dec(p_a)}, P(B|A)={_dec(p_b_given_a)}, "
               f"P(B|~A)={_dec(p_b_given_na)}. Find P(A|B).")
    solution = (f"P(B) = P(B|A)*P(A) + P(B|~A)*P(~A)\n"
                f"= {_dec(p_b_given_a)}*{_dec(p_a)} + {_dec(p_b_given_na)}*{_dec(p_na)}\n"
                f"= {_dec(p_b)}\n"
                f"P(A|B) = {_dec(p_b_given_a)}*{_dec(p_a)} / {_dec(p_b)}\n"
                f"= {ans_dec}\n"
                f"#### {ans_dec}")
    return {"problem": problem, "solution": solution, "answer": ans_dec,
            "domain": "stats_bayes", "level": level, "source": "stats_qa"}


def expected_value(rng: random.Random, level: int) -> dict:
    """E[X] for discrete distribution given outcomes and probs (sum to 1)."""
    n_outcomes = rng.randint(3, 5)
    # Probabilities: uniform fractions 1/total, 2/total, ...
    total = n_outcomes * rng.randint(2, 4)
    raw = [rng.randint(1, 4) for _ in range(n_outcomes - 1)]
    raw.append(total - sum(raw))
    if raw[-1] <= 0:
        raw = [total // n_outcomes] * n_outcomes
        raw[-1] += total - sum(raw)
    outcomes = [rng.randint(-5*level, 20*level) for _ in range(n_outcomes)]
    ev = Fraction(sum(o * p for o, p in zip(outcomes, raw)), total)
    ev_str = _frac(ev.numerator, ev.denominator)
    ev_dec = _dec(ev)
    probs_str = ", ".join(f"{p}/{total}" for p in raw)
    outs_str = ", ".join(str(o) for o in outcomes)
    problem = f"E[X]: outcomes=[{outs_str}], probs=[{probs_str}]."
    solution = (f"E[X] = sum(x*p)\n"
                + "".join(f"+ {o}*{p}/{total} " for o, p in zip(outcomes, raw))
                + f"\n= {ev_str} = {ev_dec}\n"
                f"#### {ev_dec}")
    return {"problem": problem, "solution": solution, "answer": ev_dec,
            "domain": "stats_expected_value", "level": level, "source": "stats_qa"}


def z_score(rng: random.Random, level: int) -> dict:
    """z = (x - mu) / sigma."""
    mu = rng.randint(-20*level, 20*level)
    sigma = rng.randint(1, 10*level)
    x = mu + rng.randint(-3*sigma, 3*sigma)
    z_frac = Fraction(x - mu, sigma)
    z_str = _frac(z_frac.numerator, z_frac.denominator)
    z_dec = _dec(z_frac)
    problem = f"z-score: x={x}, mean={mu}, std={sigma}."
    solution = (f"z = (x - mu) / sigma\n"
                f"= ({x} - {mu}) / {sigma}\n"
                f"= {x-mu}/{sigma}\n"
                f"= {z_str} = {z_dec}\n"
                f"#### {z_dec}")
    return {"problem": problem, "solution": solution, "answer": z_dec,
            "domain": "stats_zscore", "level": level, "source": "stats_qa"}


# ── L4: Binomial PMF, Pearson r ───────────────────────────────────────────────

def binomial_pmf(rng: random.Random, level: int) -> dict:
    """P(X=k) = C(n,k)*p^k*(1-p)^(n-k)."""
    n = rng.randint(3, 8 + level)
    k = rng.randint(0, n)
    # p as simple fraction 1/2, 1/3, 1/4, 2/3, 3/4
    p_choices = [Fraction(1,2), Fraction(1,3), Fraction(1,4),
                 Fraction(2,3), Fraction(3,4)]
    p = rng.choice(p_choices)
    q = 1 - p
    comb = math.comb(n, k)
    prob_frac = Fraction(comb) * p**k * q**(n-k)
    ans_str = _frac(prob_frac.numerator, prob_frac.denominator)
    ans_dec = _dec(prob_frac)
    ans_sci = _sci(float(prob_frac))
    problem = (f"Binomial: n={n}, p={p}, k={k}. "
               f"P(X={k})? (fraction)")
    solution = (f"P(X={k}) = C({n},{k})*({p})^{k}*(1-{p})^{n-k}\n"
                f"= {comb}*{_frac((p**k).numerator,(p**k).denominator)}"
                f"*{_frac((q**(n-k)).numerator,(q**(n-k)).denominator)}\n"
                f"= {ans_str} = {ans_dec}\n"
                f"#### {ans_str}")
    return {"problem": problem, "solution": solution, "answer": ans_str,
            "domain": "stats_binomial", "level": level, "source": "stats_qa"}


def pearson_r(rng: random.Random, level: int) -> dict:
    """Pearson r for 3-point dataset with exact integer computation."""
    n = 3
    # Pick correlated points
    slope = rng.choice([-2, -1, 1, 2])
    xs = [rng.randint(1, 10*level) for _ in range(n)]
    ys = [slope * x + rng.randint(-3, 3) for x in xs]
    xbar = Fraction(sum(xs), n)
    ybar = Fraction(sum(ys), n)
    num = sum((Fraction(x)-xbar)*(Fraction(y)-ybar) for x, y in zip(xs, ys))
    denom_x = sum((Fraction(x)-xbar)**2 for x in xs)
    denom_y = sum((Fraction(y)-ybar)**2 for y in ys)
    # r = num / sqrt(denom_x * denom_y)
    denom_prod = float(denom_x) * float(denom_y)
    r = float(num) / (denom_prod ** 0.5) if denom_prod > 0 else 0
    r_str = f"{round(r, 4)}"
    xs_str = ", ".join(str(x) for x in xs)
    ys_str = ", ".join(str(y) for y in ys)
    problem = f"Pearson r: x=[{xs_str}], y=[{ys_str}]."
    solution = (f"xbar={_frac(xbar.numerator,xbar.denominator)}, "
                f"ybar={_frac(ybar.numerator,ybar.denominator)}\n"
                f"num={float(num):.4f}, denom=sqrt({float(denom_x):.2f}*{float(denom_y):.2f})\n"
                f"r ≈ {r_str}\n"
                f"#### {r_str}")
    return {"problem": problem, "solution": solution, "answer": r_str,
            "domain": "stats_correlation", "level": level, "source": "stats_qa"}


# ── L5: Confidence interval, hypothesis test ──────────────────────────────────

def confidence_interval(rng: random.Random, level: int) -> dict:
    """CI: x_bar +/- z*(sigma/sqrt(n)), 95% z=1.96."""
    xbar = rng.randint(50, 200 * level)
    sigma = rng.randint(5, 30 * level)
    n = rng.choice([25, 36, 49, 100])
    sqrt_n = int(n ** 0.5)
    z = 1.96
    margin = round(z * sigma / sqrt_n, 4)
    lo = round(xbar - margin, 4)
    hi = round(xbar + margin, 4)
    problem = (f"95% CI: xbar={xbar}, sigma={sigma}, n={n}. "
               f"Find margin of error.")
    solution = (f"SE = sigma/sqrt(n) = {sigma}/{sqrt_n} = {sigma/sqrt_n:.4f}\n"
                f"ME = 1.96 * {sigma/sqrt_n:.4f} = {margin}\n"
                f"CI = ({lo}, {hi})\n"
                f"#### {margin}")
    return {"problem": problem, "solution": solution, "answer": margin,
            "domain": "stats_inference", "level": level, "source": "stats_qa"}


def hypothesis_test_z(rng: random.Random, level: int) -> dict:
    """z-test: z = (xbar - mu0) / (sigma/sqrt(n))."""
    mu0 = rng.randint(50, 200 * level)
    sigma = rng.randint(5, 30 * level)
    n = rng.choice([25, 36, 49, 100])
    sqrt_n = int(n ** 0.5)
    # Pick xbar so z is a clean number
    z_true = rng.choice([-2, -1, 0, 1, 2])
    xbar = mu0 + z_true * sigma // sqrt_n
    z_calc = round((xbar - mu0) / (sigma / sqrt_n), 4)
    decision = "reject H0" if abs(z_calc) > 1.96 else "fail to reject H0"
    problem = (f"z-test: mu0={mu0}, sigma={sigma}, n={n}, xbar={xbar}. "
               f"z-statistic?")
    solution = (f"SE = {sigma}/{sqrt_n} = {sigma/sqrt_n:.4f}\n"
                f"z = ({xbar}-{mu0}) / {sigma/sqrt_n:.4f}\n"
                f"= {z_calc}\n"
                f"({decision} at alpha=0.05)\n"
                f"#### {z_calc}")
    return {"problem": problem, "solution": solution, "answer": z_calc,
            "domain": "stats_inference", "level": level, "source": "stats_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [mean_of_list, median_of_list, basic_probability],
    2: [variance_std, conditional_prob, combinations, permutations],
    3: [bayes_theorem, expected_value, z_score],
    4: [binomial_pmf, pearson_r, conditional_prob],
    5: [confidence_interval, hypothesis_test_z, bayes_theorem],
}


def generate_stats_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_stats{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} statistics pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate statistics Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_stats_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
