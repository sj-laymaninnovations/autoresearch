"""
number_formats.py — Number Format Utility for Math QA Generators

Teaches models that the same quantity can appear in many forms:
  - Integer:            42
  - Fraction:           3/4
  - Decimal:            0.75
  - Float (with .0):    0.75
  - Short sci-notation: 7.5e-1
  - Long sci-notation:  7.5 x 10^-1
  - Percentage:         75%
  - Engineering:        750 x 10^-3
  - Mixed fraction:     3 1/4  (integer + fraction)

Usage:
    from number_formats import fmt, fmt_pair, ALL_FORMATS

    fmt(0.75)                  -> "0.75"  or  "3/4"  or  "7.5e-1"  (random)
    fmt(0.75, style="sci")     -> "7.5e-1"
    fmt_pair(3, 4)             -> ("3/4", "0.75")  two different styles

All generators import this module and sprinkle format variety into problems
and solutions so the model sees numbers in every representation.
"""

import random
from fractions import Fraction
from typing import Union

Number = Union[int, float, Fraction]

ALL_FORMATS = [
    "integer",      # 3          (only when value is integral)
    "fraction",     # 3/4
    "decimal",      # 0.75
    "float_dot",    # 0.7500     (explicit trailing zeros)
    "sci_short",    # 7.5e-1
    "sci_long",     # 7.5 x 10^-1
    "sci_E",        # 7.5E-1
    "engineering",  # 750 x 10^-3
    "percentage",   # 75%        (only in [0,1] range)
]

# Formats that make sense for values > 1 with no fractional part needed
INT_FORMATS = ["integer", "sci_short", "sci_long", "sci_E"]
# Formats good for any rational in (0,1)
FRAC_FORMATS = ["fraction", "decimal", "float_dot", "sci_short", "sci_long",
                "sci_E", "percentage"]
# Formats for arbitrary reals
REAL_FORMATS = ["decimal", "float_dot", "sci_short", "sci_long", "sci_E",
                "engineering"]


def _to_fraction(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    # float -> limit denominator to avoid huge fractions
    return Fraction(x).limit_denominator(10000)


def _sci_mantissa_exp(x: float) -> tuple:
    """Return (mantissa, exponent) such that x = mantissa * 10^exponent
    with 1 <= |mantissa| < 10."""
    if x == 0:
        return (0.0, 0)
    import math as _math
    exp = int(_math.floor(_math.log10(abs(x))))
    mantissa = x / (10 ** exp)
    # Round mantissa to avoid floating point noise
    mantissa = round(mantissa, 6)
    return mantissa, exp


def _engineering_exp(x: float) -> tuple:
    """Exponent is multiple of 3."""
    if x == 0:
        return (0.0, 0)
    import math as _math
    exp = int(_math.floor(_math.log10(abs(x))))
    eng_exp = (exp // 3) * 3
    mantissa = x / (10 ** eng_exp)
    mantissa = round(mantissa, 4)
    return mantissa, eng_exp


def fmt(x: Number, style: str = None, rng: random.Random = None,
        precision: int = 4) -> str:
    """
    Format number x in the requested style (or a random suitable style).

    Args:
        x:         The number to format
        style:     One of ALL_FORMATS, or None for random
        rng:       Random instance (uses global random if None)
        precision: Decimal places for decimal/float styles

    Returns:
        String representation of x in the requested format
    """
    if rng is None:
        rng = random.Random()

    frac = _to_fraction(x)
    val = float(frac)
    is_int = frac.denominator == 1

    if style is None:
        # Pick a suitable random style
        candidates = []
        if is_int:
            candidates += ["integer", "sci_short", "sci_long"]
        if 0 < abs(val) < 1:
            candidates += ["fraction", "decimal", "sci_short", "sci_long",
                           "sci_E"]
        if 0 <= val <= 1:
            candidates += ["percentage"]
        if not candidates:
            candidates = ["decimal", "sci_short", "sci_long", "sci_E",
                          "engineering"]
        style = rng.choice(candidates)

    if style == "integer":
        if is_int:
            return str(frac.numerator)
        return str(round(val))

    if style == "fraction":
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"

    if style == "decimal":
        # Show reduced decimal
        s = f"{val:.{precision}f}".rstrip("0").rstrip(".")
        return s if s else "0"

    if style == "float_dot":
        return f"{val:.{precision}f}"

    if style == "sci_short":
        if val == 0:
            return "0e0"
        m, e = _sci_mantissa_exp(val)
        m_s = f"{m:.4f}".rstrip("0").rstrip(".")
        return f"{m_s}e{e}"

    if style == "sci_E":
        if val == 0:
            return "0E0"
        m, e = _sci_mantissa_exp(val)
        m_s = f"{m:.4f}".rstrip("0").rstrip(".")
        return f"{m_s}E{e}"

    if style == "sci_long":
        if val == 0:
            return "0 x 10^0"
        m, e = _sci_mantissa_exp(val)
        m_s = f"{m:.4f}".rstrip("0").rstrip(".")
        return f"{m_s} x 10^{e}"

    if style == "engineering":
        if val == 0:
            return "0 x 10^0"
        m, e = _engineering_exp(val)
        m_s = f"{m:.4f}".rstrip("0").rstrip(".")
        return f"{m_s} x 10^{e}"

    if style == "percentage":
        pct = round(val * 100, 4)
        pct_s = f"{pct:.4f}".rstrip("0").rstrip(".")
        return f"{pct_s}%"

    # Fallback
    return str(val)


def fmt_pair(x: Number, rng: random.Random = None,
             styles: list = None) -> tuple:
    """
    Return (style_a, style_b) two different formatted strings of the same x.
    Useful for teaching: "these are the same number".
    """
    if rng is None:
        rng = random.Random()
    frac = _to_fraction(x)
    val = float(frac)
    is_int = frac.denominator == 1

    pool = []
    if is_int:
        pool.append(("integer", str(frac.numerator)))
    pool.append(("fraction", fmt(x, "fraction")))
    pool.append(("decimal",  fmt(x, "decimal")))
    pool.append(("sci_short", fmt(x, "sci_short")))
    pool.append(("sci_long",  fmt(x, "sci_long")))
    if 0 <= val <= 1:
        pool.append(("percentage", fmt(x, "percentage")))

    # Remove duplicates by value
    seen_vals = set()
    unique = []
    for style, s in pool:
        if s not in seen_vals:
            seen_vals.add(s)
            unique.append((style, s))

    if len(unique) < 2:
        return (str(val), str(val))

    picks = rng.sample(unique, 2)
    return (picks[0][1], picks[1][1])


def equivalence_problem(x: Number, rng: random.Random = None) -> dict:
    """
    Generate a Q/A pair that teaches number format equivalence.
    e.g., "Write 3/4 as a decimal" -> "0.75"
    """
    if rng is None:
        rng = random.Random()

    frac = _to_fraction(x)
    val = float(frac)
    is_int = frac.denominator == 1

    conversions = []
    frac_str  = fmt(x, "fraction")
    dec_str   = fmt(x, "decimal")
    sci_str   = fmt(x, "sci_short")
    sciL_str  = fmt(x, "sci_long")

    if not is_int:
        conversions += [
            (f"Write {frac_str} as a decimal.",
             dec_str,
             f"{frac.numerator} / {frac.denominator} = {dec_str}\n#### {dec_str}"),
            (f"Write {dec_str} as a fraction.",
             frac_str,
             f"{dec_str} = {frac_str}\n#### {frac_str}"),
        ]
    conversions += [
        (f"Write {dec_str} in scientific notation.",
         sci_str,
         f"{dec_str} = {sci_str}\n#### {sci_str}"),
        (f"Write {sci_str} in decimal form.",
         dec_str,
         f"{sci_str} = {dec_str}\n#### {dec_str}"),
        (f"Write {sciL_str} in short scientific notation.",
         sci_str,
         f"{sciL_str} = {sci_str}\n#### {sci_str}"),
    ]

    if 0 < val < 1:
        pct_str = fmt(x, "percentage")
        conversions += [
            (f"Write {dec_str} as a percentage.",
             pct_str,
             f"{dec_str} * 100 = {pct_str}\n#### {pct_str}"),
            (f"Write {pct_str} as a decimal.",
             dec_str,
             f"{pct_str} / 100 = {dec_str}\n#### {dec_str}"),
        ]

    if not conversions:
        conversions = [(f"Write {sci_str} in standard form.",
                        dec_str,
                        f"{sci_str} = {dec_str}\n#### {dec_str}")]

    problem_str, answer, solution = rng.choice(conversions)
    return {
        "problem": problem_str,
        "solution": solution,
        "answer": answer,
        "domain": "number_format",
        "level": 1,
        "source": "number_formats",
    }


def generate_format_pairs(n_pairs: int = 200, seed: int = None) -> list:
    """
    Generate Q/A pairs that teach number format equivalence across
    integers, fractions, decimals, sci-notation, and percentages.
    """
    from fractions import Fraction as F
    rng = random.Random(seed)
    pairs = []
    seen = set()

    # Pool of "interesting" values in various ranges
    def rand_values():
        v = []
        # Small integers
        v += [F(n) for n in rng.choices(range(1, 100), k=30)]
        # Unit fractions 1/n
        v += [F(1, n) for n in rng.choices(range(2, 20), k=20)]
        # Common fractions
        v += [F(a, b) for a in range(1, 10) for b in range(2, 10) if b > a]
        # Powers of 10
        v += [F(10**e) for e in range(-4, 6)]
        # Random decimals
        for _ in range(30):
            num = rng.randint(1, 999)
            den = rng.choice([10, 100, 1000, 4, 8, 16, 5, 25])
            v.append(F(num, den))
        # Large numbers
        v += [F(n * 1000) for n in rng.choices(range(1, 100), k=10)]
        return v

    values = rand_values()
    rng.shuffle(values)

    for x in values * 5:  # repeat pool to hit n_pairs
        if len(pairs) >= n_pairs:
            break
        try:
            p = equivalence_problem(x, rng)
        except Exception:
            continue
        key = p["problem"]
        if key in seen:
            continue
        seen.add(key)
        pairs.append(p)

    rng.shuffle(pairs)
    return pairs[:n_pairs]


# ── CLI (standalone) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, datetime, pathlib
    parser = argparse.ArgumentParser(
        description="Generate number-format equivalence Q/A pairs")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    pairs = generate_format_pairs(args.n, args.seed)
    RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out = RESULTS_DIR / f"harder_qa_numfmt{tag}_{ts}.jsonl"
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} number-format pairs -> {out}")
    print("\nSamples:")
    for p in pairs[:5]:
        print(f"  Q: {p['problem']}")
        print(f"  A: {p['answer']}\n")
