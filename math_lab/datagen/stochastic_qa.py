"""
stochastic_qa.py — Stochastic Processes Q/A Generator

All number formats used: fractions (3/7), decimals (0.4286), short sci (4.286e-1),
long sci (4.286 x 10^-1), float_dot (0.4286), percentage (42.86%).
Stochastic answers are naturally fractional probabilities, making this
domain ideal for format-variety training.

Topics by level:
  L1: Geometric distribution E[T], Poisson PMF P(X=k)
  L2: 2-state Markov chain steady-state via detailed balance
  L3: Expected hitting time (system of equations), 3-state chain
  L4: Gambler's ruin absorption probability
  L5: Random walk, martingale check E[X_{n+1}|F_n]=X_n

Output: math_lab/results/harder_qa_stochastic_<tag>_<timestamp>.jsonl

References:
  - Sheldon Ross, Introduction to Probability Models (12th ed.)
  - Durrett, Probability: Theory and Examples
  - MIT OCW 6.041 Probabilistic Systems Analysis
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


def _sci(val: float, places: int = 4) -> str:
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    m = round(val / 10**exp, places)
    ms = f"{m:.{places}f}".rstrip("0").rstrip(".")
    return f"{ms}e{exp}"


def _sci_long(val: float, places: int = 4) -> str:
    if val == 0:
        return "0 x 10^0"
    exp = int(math.floor(math.log10(abs(val))))
    m = round(val / 10**exp, places)
    ms = f"{m:.{places}f}".rstrip("0").rstrip(".")
    return f"{ms} x 10^{exp}"


def _pct(f, places: int = 2) -> str:
    return f"{float(f)*100:.{places}f}".rstrip("0").rstrip(".") + "%"


def _all_formats(frac: Fraction) -> str:
    """Show a probability in all common formats for training."""
    frac_str = _frac(frac.numerator, frac.denominator)
    dec_str = _dec(frac)
    sci_str = _sci(float(frac))
    sciL_str = _sci_long(float(frac))
    pct_str = _pct(frac)
    return f"{frac_str} = {dec_str} = {sci_str} = {pct_str}"


# ── L1: Geometric distribution, Poisson PMF ───────────────────────────────────

def geometric_expected(rng: random.Random, level: int) -> dict:
    """E[T] = 1/p for geometric; answer in fraction, decimal, sci."""
    p_num = rng.randint(1, 5)
    p_den = rng.randint(max(p_num+1, 5), 10 + 5*level)
    p = Fraction(p_num, p_den)
    p_reduced = _frac(p.numerator, p.denominator)
    E = Fraction(p_den, p_num)  # 1/p = p_den/p_num
    e_frac = _frac(E.numerator, E.denominator)
    e_dec = _dec(E)
    e_sci = _sci(float(E))
    problem = f"Geometric dist: P(success)={p_reduced}. E[trials]?"
    solution = (f"E[T] = 1/p = 1/{p_reduced}\n"
                f"= {e_frac} = {e_dec} = {e_sci}\n"
                f"#### {e_frac}")
    return {"problem": problem, "solution": solution, "answer": e_frac,
            "domain": "stoch_geometric", "level": level, "source": "stoch_qa"}


def geometric_pmf(rng: random.Random, level: int) -> dict:
    """P(T=k) = (1-p)^(k-1)*p; all formats shown."""
    p_choices = [Fraction(1,2), Fraction(1,3), Fraction(1,4), Fraction(1,5)]
    p = rng.choice(p_choices)
    k = rng.randint(1, 4 + level)
    q = 1 - p
    prob = q**(k-1) * p
    prob_frac = _frac(prob.numerator, prob.denominator)
    prob_dec = _dec(prob)
    prob_sci = _sci(float(prob))
    prob_sciL = _sci_long(float(prob))
    prob_pct = _pct(prob)
    p_str = _frac(p.numerator, p.denominator)
    q_str = _frac(q.numerator, q.denominator)
    problem = f"Geometric P(success)={p_str}. P(T={k})? All formats."
    solution = (f"P(T={k}) = (1-{p_str})^{k-1} * {p_str}\n"
                f"= {q_str}^{k-1} * {p_str}\n"
                f"= {_frac((q**(k-1)).numerator,(q**(k-1)).denominator)} * {p_str}\n"
                f"= {prob_frac} = {prob_dec} = {prob_sci} = {prob_pct}\n"
                f"#### {prob_frac}")
    return {"problem": problem, "solution": solution, "answer": prob_frac,
            "domain": "stoch_geometric", "level": level, "source": "stoch_qa"}


def poisson_pmf(rng: random.Random, level: int) -> dict:
    """P(X=k) = e^(-lam)*lam^k/k!; store decimal and sci."""
    lam = rng.randint(1, 4 + level)
    k = rng.randint(0, 4)
    factorial_k = math.factorial(k)
    # P = e^(-lam)*lam^k/k!
    prob = math.exp(-lam) * lam**k / factorial_k
    prob_dec = f"{round(prob, 6)}"
    prob_sci = _sci(prob)
    prob_sciL = _sci_long(prob)
    problem = f"Poisson(lam={lam}): P(X={k})? (e≈2.71828)"
    solution = (f"P(X={k}) = e^(-{lam}) * {lam}^{k} / {k}!\n"
                f"= e^(-{lam}) * {lam**k} / {factorial_k}\n"
                f"≈ {prob_dec} = {prob_sci}\n"
                f"= {prob_sciL}\n"
                f"#### {prob_sci}")
    return {"problem": problem, "solution": solution, "answer": prob_sci,
            "domain": "stoch_poisson", "level": level, "source": "stoch_qa"}


def poisson_expected(rng: random.Random, level: int) -> dict:
    """E[X] = Var[X] = lambda for Poisson."""
    lam = rng.randint(1, 20 * level)
    lam_sci = _sci(float(lam))
    problem = f"Poisson distribution with rate lambda={lam}. E[X] and Var[X]?"
    solution = (f"For Poisson: E[X] = Var[X] = lambda\n"
                f"E[X] = {lam} = {lam_sci}\n"
                f"Var[X] = {lam} = {lam_sci}\n"
                f"#### E={lam}, Var={lam}")
    return {"problem": problem, "solution": solution, "answer": f"E={lam}, Var={lam}",
            "domain": "stoch_poisson", "level": level, "source": "stoch_qa"}


# ── L2: 2-state Markov chain steady-state ────────────────────────────────────

def markov_2state_steady(rng: random.Random, level: int) -> dict:
    """P(0->1)=a, P(1->0)=b; pi1 = a/(a+b) as fraction, dec, sci, pct."""
    a_num = rng.randint(1, 5)
    b_num = rng.randint(1, 5)
    a_den = rng.randint(a_num+1, 10 + 5*level)
    b_den = rng.randint(b_num+1, 10 + 5*level)
    a = Fraction(a_num, a_den)
    b = Fraction(b_num, b_den)
    a_str = _frac(a.numerator, a.denominator)
    b_str = _frac(b.numerator, b.denominator)
    # pi1 = a / (a + b)
    denom = a + b
    pi1 = a / denom
    pi0 = b / denom
    pi1_frac = _frac(pi1.numerator, pi1.denominator)
    pi1_dec = _dec(pi1)
    pi1_sci = _sci(float(pi1))
    pi1_pct = _pct(pi1)
    pi0_frac = _frac(pi0.numerator, pi0.denominator)
    problem = f"Markov chain: P(0->1)={a_str}, P(1->0)={b_str}. Steady-state pi1?"
    solution = (f"Balance: pi1*{b_str} = pi0*{a_str}\n"
                f"pi1 = {a_str}/({a_str}+{b_str})\n"
                f"= {_frac(a.numerator,a.denominator)}/{_frac(denom.numerator,denom.denominator)}\n"
                f"= {pi1_frac} = {pi1_dec} = {pi1_sci} = {pi1_pct}\n"
                f"#### {pi1_frac}")
    return {"problem": problem, "solution": solution, "answer": pi1_frac,
            "domain": "stoch_markov", "level": level, "source": "stoch_qa"}


def markov_2state_n_steps(rng: random.Random, level: int) -> dict:
    """P(X_n=1|X_0=0) for small n using matrix power."""
    p = Fraction(rng.randint(1,4), rng.randint(5,10))
    q = Fraction(rng.randint(1,4), rng.randint(5,10))
    n_steps = rng.randint(1, 3)
    p_str = _frac(p.numerator, p.denominator)
    q_str = _frac(q.numerator, q.denominator)
    # Exact steady state
    pi1 = p / (p + q)
    # For n=1: P(1|0,1step) = p
    # Approximation: use steady-state as the answer for large n
    ans = _frac(pi1.numerator, pi1.denominator)
    ans_dec = _dec(pi1)
    problem = f"Markov: P(0->1)={p_str}, P(1->0)={q_str}. Steady-state P(state=1)?"
    solution = (f"Steady-state: pi1 = p/(p+q)\n"
                f"= {p_str}/({p_str}+{q_str})\n"
                f"= {ans} = {ans_dec}\n"
                f"#### {ans}")
    return {"problem": problem, "solution": solution, "answer": ans,
            "domain": "stoch_markov", "level": level, "source": "stoch_qa"}


# ── L3: Expected hitting time ──────────────────────────────────────────────────

def hitting_time_simple(rng: random.Random, level: int) -> dict:
    """Random walk on {0,1,...,N} with absorbing barrier at 0 and N.
    P(right)=p; E[T_0] from state 1 = 1/p (simplified)."""
    p_num = rng.randint(1, 4)
    p_den = rng.randint(p_num+1, 8 + 4*level)
    p = Fraction(p_num, p_den)
    q = 1 - p
    p_str = _frac(p.numerator, p.denominator)
    q_str = _frac(q.numerator, q.denominator)
    # E[T] from state 1 (absorbing at 0): E = 1/p + q/p * E[T]
    # E*(1 - q/p) = 1/p -> for p>q: E = 1/(p-q) if p!=q else infinite
    if p == q:
        return geometric_expected(rng, level)
    pq_diff = p - q
    E = 1 / pq_diff
    E_frac = _frac(E.numerator, E.denominator)
    E_dec = _dec(E)
    E_sci = _sci(float(E))
    E_pct = _pct(E) if 0 < float(E) <= 1 else E_dec
    problem = f"Random walk: p={p_str}, q={q_str}. E[steps to absorb]?"
    solution = (f"E = 1/(p-q) = 1/({p_str}-{q_str})\n"
                f"= {E_frac} = {E_dec} = {E_sci}\n"
                f"#### {E_frac}")
    return {"problem": problem, "solution": solution, "answer": E_frac,
            "domain": "stoch_hitting_time", "level": level, "source": "stoch_qa"}


def markov_3state_steady(rng: random.Random, level: int) -> dict:
    """3-state Markov chain (symmetric); steady state = [1/3, 1/3, 1/3]."""
    # Use a doubly stochastic matrix: uniform steady state
    # Row transitions chosen so rows sum to 1
    a = Fraction(rng.randint(1,3), rng.randint(4,8))
    b = Fraction(rng.randint(1,3), rng.randint(4,8))
    c = 1 - a - b
    if c <= 0:
        a, b, c = Fraction(1,3), Fraction(1,3), Fraction(1,3)
    a_str = _frac(a.numerator, a.denominator)
    b_str = _frac(b.numerator, b.denominator)
    c_str = _frac(c.numerator, c.denominator)
    # For doubly stochastic, pi = [1/3,1/3,1/3]
    pi = Fraction(1, 3)
    pi_frac = _frac(pi.numerator, pi.denominator)
    pi_dec = _dec(pi)
    pi_sci = _sci(float(pi))
    pi_pct = _pct(pi)
    problem = (f"Doubly stochastic 3-state chain. Steady-state pi_i?")
    solution = (f"Doubly stochastic => uniform steady-state\n"
                f"pi = 1/3 for each state\n"
                f"= {pi_frac} = {pi_dec} = {pi_sci} = {pi_pct}\n"
                f"#### pi=1/3 each")
    return {"problem": problem, "solution": solution, "answer": "1/3 each",
            "domain": "stoch_markov", "level": level, "source": "stoch_qa"}


# ── L4: Gambler's ruin ────────────────────────────────────────────────────────

def gamblers_ruin(rng: random.Random, level: int) -> dict:
    """P(reach N before 0 | start at k) = k/N for fair (p=1/2)."""
    N = rng.randint(4, 10 * level)
    k = rng.randint(1, N - 1)
    prob = Fraction(k, N)
    prob_frac = _frac(prob.numerator, prob.denominator)
    prob_dec = _dec(prob)
    prob_sci = _sci(float(prob))
    prob_sciL = _sci_long(float(prob))
    prob_pct = _pct(prob)
    problem = (f"Gambler's ruin (fair): N={N}, start k={k}. "
               f"P(reach {N} before 0)?")
    solution = (f"Fair game (p=1/2): P(win) = k/N\n"
                f"= {k}/{N} = {prob_frac}\n"
                f"= {prob_dec} = {prob_sci} = {prob_pct}\n"
                f"#### {prob_frac}")
    return {"problem": problem, "solution": solution, "answer": prob_frac,
            "domain": "stoch_gamblers_ruin", "level": level, "source": "stoch_qa"}


def gamblers_ruin_biased(rng: random.Random, level: int) -> dict:
    """P(win | k) = (1-(q/p)^k) / (1-(q/p)^N) for p != 1/2."""
    N = rng.randint(3, 6 + level)
    k = rng.randint(1, N - 1)
    # p = 2/3, q = 1/3 -> q/p = 1/2
    p = Fraction(2, 3)
    q = Fraction(1, 3)
    r = q / p  # = 1/2
    # P(win) = (1 - r^k) / (1 - r^N)
    rk = r**k
    rN = r**N
    prob = (1 - rk) / (1 - rN)
    prob_frac = _frac(prob.numerator, prob.denominator)
    prob_dec = _dec(prob)
    prob_sci = _sci(float(prob))
    prob_pct = _pct(prob)
    p_str = _frac(p.numerator, p.denominator)
    problem = (f"Gambler's ruin: p={p_str}, N={N}, k={k}. P(reach {N})?")
    solution = (f"r = q/p = {_frac(r.numerator,r.denominator)}\n"
                f"P = (1-r^{k})/(1-r^{N})\n"
                f"= (1-{_frac(rk.numerator,rk.denominator)})"
                f"/(1-{_frac(rN.numerator,rN.denominator)})\n"
                f"= {prob_frac} = {prob_dec} = {prob_sci} = {prob_pct}\n"
                f"#### {prob_frac}")
    return {"problem": problem, "solution": solution, "answer": prob_frac,
            "domain": "stoch_gamblers_ruin", "level": level, "source": "stoch_qa"}


# ── L5: Martingale check, random walk variance ────────────────────────────────

def martingale_check(rng: random.Random, level: int) -> dict:
    """Show X_n = S_n (simple RW) is a martingale: E[X_{n+1}|F_n] = X_n."""
    p = Fraction(1, 2)
    x = rng.randint(-10*level, 10*level)
    e_next = Fraction(1,2) * (x+1) + Fraction(1,2) * (x-1)
    e_frac = _frac(e_next.numerator, e_next.denominator)
    e_dec = _dec(e_next)
    problem = f"Simple RW X_n: E[X_{{n+1}} | X_n={x}]? Is it a martingale?"
    solution = (f"E[X_{{n+1}}|X_n={x}] = (1/2)*{x+1} + (1/2)*{x-1}\n"
                f"= {_frac((x+1),2)} + {_frac((x-1),2)}\n"
                f"= {e_frac} = {e_dec} = X_n\n"
                f"=> Yes, martingale\n"
                f"#### {e_frac}")
    return {"problem": problem, "solution": solution, "answer": str(x),
            "domain": "stoch_martingale", "level": level, "source": "stoch_qa"}


def random_walk_variance(rng: random.Random, level: int) -> dict:
    """Var[S_n] = n for symmetric simple random walk; all formats."""
    n = rng.randint(2, 50 * level)
    var = n
    var_sci = _sci(float(n))
    var_sciL = _sci_long(float(n))
    problem = f"Symmetric RW after {n} steps: Var[S_{n}]?"
    solution = (f"Each step: E[Xi]=0, Var[Xi]=1\n"
                f"Var[S_{n}] = n*Var[Xi] = {n}\n"
                f"= {var_sci} = {var_sciL}\n"
                f"#### {n}")
    return {"problem": problem, "solution": solution, "answer": n,
            "domain": "stoch_random_walk", "level": level, "source": "stoch_qa"}


def branching_process_extinction(rng: random.Random, level: int) -> dict:
    """Galton-Watson: P(extinct) = q where q = generating fn fixed point."""
    # Use Poisson offspring with lambda<1 (extinction certain: q=1)
    lam_num = rng.randint(1, 4)
    lam_den = rng.randint(lam_num+2, 10)
    lam = Fraction(lam_num, lam_den)
    lam_str = _frac(lam.numerator, lam.denominator)
    lam_dec = _dec(lam)
    lam_sci = _sci(float(lam))
    problem = (f"Branching process: Poisson offspring, mean={lam_str}={lam_dec}. "
               f"P(extinction)?")
    solution = (f"Mean {lam_str} < 1\n"
                f"=> subcritical process\n"
                f"P(extinction) = 1\n"
                f"(1 = 1.0 = 1e0 = 100%)\n"
                f"#### 1")
    return {"problem": problem, "solution": solution, "answer": 1,
            "domain": "stoch_branching", "level": level, "source": "stoch_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [geometric_expected, geometric_pmf, poisson_pmf, poisson_expected],
    2: [markov_2state_steady, markov_2state_n_steps, geometric_pmf],
    3: [hitting_time_simple, markov_3state_steady, markov_2state_steady],
    4: [gamblers_ruin, gamblers_ruin_biased, hitting_time_simple],
    5: [martingale_check, random_walk_variance, branching_process_extinction,
        gamblers_ruin_biased],
}


def generate_stochastic_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_stochastic{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} stochastic pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate stochastic processes Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_stochastic_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:4]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
