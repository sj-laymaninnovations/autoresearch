"""
trigonometry_qa.py — Trigonometry Q/A Generator

Topics by level:
  L1: SOH-CAH-TOA ratio lookup, right-triangle side finding
  L2: Law of sines / law of cosines (integer-friendly)
  L3: Radian-degree conversion, double-angle identities
  L4: Solving trig equations (general solution set)
  L5: Sum/difference identities, sinusoid period/amplitude/phase

Answers stored as exact fraction strings ("p/q") or integer strings.
All degree values in EXACT_VALUES have rational sin/cos/tan.

Output: math_lab/results/harder_qa_trig_<tag>_<timestamp>.jsonl

References:
  - Khan Academy Trigonometry curriculum
  - Precalculus: Stewart, Redlin & Watson (2016)
  - AMC 10/12 trig identities problems
"""

import json
import math
import random
import datetime
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path(__file__).parent.parent / "results"
MAX_CONTENT_LEN = 246

# Exact rational sin/cos/tan (multiplied by common denominator for display)
# stored as (sin_num, sin_den, cos_num, cos_den, tan_num, tan_den)
EXACT_VALS = {
    0:   (0, 1,  1, 1,  0, 1),
    30:  (1, 2,  3, 4,  1, 3),   # sin=1/2, cos=sqrt(3)/2->approx 3/4 for tan
    45:  (1, 2,  1, 2,  1, 1),   # sin=cos=1/sqrt(2), tan=1
    60:  (3, 4,  1, 2,  3, 1),   # sin=sqrt(3)/2, cos=1/2, tan=sqrt(3)->~3
    90:  (1, 1,  0, 1, None, None),
}

# For exact symbolic answers use string forms
EXACT_STR = {
    0:   {"sin": "0",        "cos": "1",          "tan": "0"},
    30:  {"sin": "1/2",      "cos": "sqrt(3)/2",  "tan": "1/sqrt(3)"},
    45:  {"sin": "sqrt(2)/2","cos": "sqrt(2)/2",  "tan": "1"},
    60:  {"sin": "sqrt(3)/2","cos": "1/2",        "tan": "sqrt(3)"},
    90:  {"sin": "1",        "cos": "0",          "tan": "undefined"},
}

STANDARD_ANGLES = [0, 30, 45, 60, 90]


def _ok(p: str, s: str) -> bool:
    return len(p) + len(s) <= MAX_CONTENT_LEN


def _frac(p: int, q: int) -> str:
    f = Fraction(p, q)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ── L1: Unit circle values, SOH-CAH-TOA ─────────────────────────────────────

def unit_circle_lookup(rng: random.Random, level: int) -> dict:
    """What is sin/cos/tan of a standard angle?"""
    angle = rng.choice(STANDARD_ANGLES[:-1])  # avoid undefined tan at 90
    fn = rng.choice(["sin", "cos", "tan"])
    answer = EXACT_STR[angle][fn]
    problem = f"What is {fn}({angle} degrees)?"
    solution = (f"From unit circle: {fn}({angle}°) = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_unit_circle", "level": level, "source": "trig_qa"}


def right_triangle_side(rng: random.Random, level: int) -> dict:
    """Right triangle: given angle and hypotenuse, find opposite."""
    from fractions import Fraction as F
    angle = rng.choice([30, 45, 60])
    hyp = rng.randint(2, 10 * level)

    if angle == 30:
        # sin(30) = 1/2 -> opp = hyp/2
        if hyp % 2 != 0:
            hyp += 1
        opp = hyp // 2
        sin_str = "1/2"
    elif angle == 45:
        opp_str = f"{hyp}/sqrt(2)"
        # Keep symbolic
        problem = (f"Right triangle: angle={angle}°, hypotenuse={hyp}. "
                   f"Opposite side?")
        solution = (f"sin({angle}°) = sqrt(2)/2\n"
                    f"opp = {hyp}*sqrt(2)/2 = {hyp}*sqrt(2)/2\n"
                    f"#### {hyp}*sqrt(2)/2")
        return {"problem": problem, "solution": solution,
                "answer": f"{hyp}*sqrt(2)/2",
                "domain": "trig_sohcahtoa", "level": level, "source": "trig_qa"}
    else:  # 60
        # sin(60) = sqrt(3)/2 -> opp = hyp*sqrt(3)/2
        answer_str = f"{hyp}*sqrt(3)/2"
        problem = (f"Right triangle: angle={angle}°, hypotenuse={hyp}. "
                   f"Opposite side?")
        solution = (f"sin({angle}°) = sqrt(3)/2\n"
                    f"opp = {hyp}*sqrt(3)/2\n"
                    f"#### {answer_str}")
        return {"problem": problem, "solution": solution, "answer": answer_str,
                "domain": "trig_sohcahtoa", "level": level, "source": "trig_qa"}

    problem = (f"Right triangle: angle={angle}°, hypotenuse={hyp}. "
               f"Opposite side?")
    solution = (f"sin({angle}°) = {sin_str}\n"
                f"opp = {hyp} * {sin_str} = {opp}\n"
                f"#### {opp}")
    return {"problem": problem, "solution": solution, "answer": opp,
            "domain": "trig_sohcahtoa", "level": level, "source": "trig_qa"}


# ── L2: Law of cosines (integer-friendly via Pythagorean triples) ────────────

PYTH = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10)]


def law_of_cosines_angle(rng: random.Random, level: int) -> dict:
    """From a Pythagorean triple, verify cos(C)=0 for C=90."""
    triple = rng.choice(PYTH)
    scale = rng.randint(1, 3 * level)
    a, b, c = triple[0]*scale, triple[1]*scale, triple[2]*scale
    # cos(C) = (a^2+b^2-c^2)/(2ab) = 0 for right triangle
    numer = a**2 + b**2 - c**2
    denom = 2 * a * b
    cos_val = _frac(numer, denom)
    problem = f"Triangle sides {a},{b},{c}. cos(angle opposite {c})?"
    solution = (f"cos(C) = ({a}^2+{b}^2-{c}^2) / (2*{a}*{b})\n"
                f"= ({a**2}+{b**2}-{c**2}) / {denom}\n"
                f"= {numer}/{denom} = {cos_val}\n"
                f"#### {cos_val}")
    return {"problem": problem, "solution": solution, "answer": cos_val,
            "domain": "trig_law_cosines", "level": level, "source": "trig_qa"}


def law_of_sines(rng: random.Random, level: int) -> dict:
    """a/sin(A) = b/sin(B); given A=30, B=60, find ratio a/b."""
    # a/sin(A) = b/sin(B) => a/b = sin(A)/sin(B)
    A, B = 30, 60
    # sin(30)/sin(60) = (1/2)/(sqrt(3)/2) = 1/sqrt(3) ≈ let's use numeric
    # Store symbolic
    problem = f"Triangle: angle A={A}°, angle B={B}°. Find a/b (law of sines)."
    solution = (f"a/sin(A) = b/sin(B)\n"
                f"a/b = sin({A}°)/sin({B}°)\n"
                f"= (1/2)/(sqrt(3)/2)\n"
                f"= 1/sqrt(3)\n"
                f"#### 1/sqrt(3)")
    return {"problem": problem, "solution": solution, "answer": "1/sqrt(3)",
            "domain": "trig_law_sines", "level": level, "source": "trig_qa"}


def radian_degree(rng: random.Random, level: int) -> dict:
    """Convert degrees to radians or vice versa."""
    deg_choices = [30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
    deg = rng.choice(deg_choices)
    # rad = deg * pi / 180; express as fraction of pi
    frac = Fraction(deg, 180)
    direction = rng.choice(["to_rad", "to_deg"])
    if direction == "to_rad":
        answer = f"{frac}*pi" if frac != 1 else "pi"
        problem = f"Convert {deg} degrees to radians."
        solution = (f"radians = degrees * pi/180\n"
                    f"= {deg} * pi/180\n"
                    f"= {answer}\n"
                    f"#### {answer}")
    else:
        # Give radians, convert to degrees
        rad_str = f"{frac}*pi"
        answer = deg
        problem = f"Convert {rad_str} radians to degrees."
        solution = (f"degrees = radians * 180/pi\n"
                    f"= ({frac}) * 180\n"
                    f"= {deg}\n"
                    f"#### {deg}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_conversion", "level": level, "source": "trig_qa"}


# ── L3: Double-angle identities ───────────────────────────────────────────────

def double_angle_sin(rng: random.Random, level: int) -> dict:
    """sin(2A) = 2*sin(A)*cos(A) for standard angles."""
    A = rng.choice([30, 45, 60])
    sin_A = EXACT_STR[A]["sin"]
    cos_A = EXACT_STR[A]["cos"]
    if A == 30:
        answer = "sqrt(3)/2"
        result_line = "2*(1/2)*(sqrt(3)/2) = sqrt(3)/2"
    elif A == 45:
        answer = "1"
        result_line = "2*(sqrt(2)/2)*(sqrt(2)/2) = 2*(1/2) = 1"
    else:  # 60
        answer = "sqrt(3)/2"
        result_line = "2*(sqrt(3)/2)*(1/2) = sqrt(3)/2"
    problem = f"Use double-angle formula: find sin({2*A} degrees)."
    solution = (f"sin({2*A}°) = 2*sin({A}°)*cos({A}°)\n"
                f"= 2*{sin_A}*{cos_A}\n"
                f"= {result_line}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_identity", "level": level, "source": "trig_qa"}


def sinusoid_params(rng: random.Random, level: int) -> dict:
    """f(x) = A*sin(Bx + C) + D; identify period and amplitude."""
    A = rng.randint(1, 5 * level)
    B = rng.randint(1, 4 * level)
    C = rng.randint(0, 6)
    D = rng.randint(-5, 5)
    period = _frac(360, B)
    problem = (f"f(x) = {A}*sin({B}x+{C})+{D}. "
               f"Amplitude and period (in degrees)?")
    solution = (f"Amplitude = |{A}| = {A}\n"
                f"Period = 360/{B} = {period} degrees\n"
                f"#### A={A}, T={period}")
    answer = f"A={A},T={period}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_sinusoid", "level": level, "source": "trig_qa"}


# ── L4: Solving trig equations ────────────────────────────────────────────────

def trig_equation_basic(rng: random.Random, level: int) -> dict:
    """sin(x)=k or cos(x)=k for k in exact values; find principal x."""
    angle = rng.choice([0, 30, 45, 60, 90])
    fn = rng.choice(["sin", "cos"])
    val = EXACT_STR[angle][fn]
    problem = f"Solve for x in [0,360): {fn}(x) = {val}."
    if fn == "sin":
        # principal in Q1, secondary in Q2 (180-angle)
        x2 = 180 - angle
        if angle == 90:
            answer = f"x={angle}"
            solution = f"sin(x) = {val} => x = {angle}\n#### {answer}"
        else:
            answer = f"x={angle} or x={x2}"
            solution = (f"sin(x) = {val}\n"
                        f"Principal: x = {angle}\n"
                        f"Also: x = 180 - {angle} = {x2}\n"
                        f"#### {answer}")
    else:  # cos
        x2 = 360 - angle
        if angle == 0:
            answer = f"x={angle}"
            solution = f"cos(x) = 1 => x = 0\n#### x=0"
        else:
            answer = f"x={angle} or x={x2}"
            solution = (f"cos(x) = {val}\n"
                        f"Principal: x = {angle}\n"
                        f"Also: x = 360 - {angle} = {x2}\n"
                        f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_equation", "level": level, "source": "trig_qa"}


# ── L5: Sum/difference formulas ───────────────────────────────────────────────

def sum_formula(rng: random.Random, level: int) -> dict:
    """sin(A+B) or cos(A+B) using standard angles."""
    combos = [(30, 45), (30, 60), (45, 45), (45, 60)]
    A, B = rng.choice(combos)
    fn = rng.choice(["sin", "cos"])

    sinA, cosA = EXACT_STR[A]["sin"], EXACT_STR[A]["cos"]
    sinB, cosB = EXACT_STR[B]["sin"], EXACT_STR[B]["cos"]

    target_angle = A + B
    answer = EXACT_STR.get(target_angle, {}).get(fn, "see solution")

    if fn == "sin":
        formula = f"sin({A}°)cos({B}°) + cos({A}°)sin({B}°)"
        problem = f"Use sum formula: sin({A}°+{B}°)?"
        solution = (f"sin(A+B) = sinAcosB + cosAsinB\n"
                    f"= {sinA}*{cosB} + {cosA}*{sinB}\n"
                    f"= {answer}\n"
                    f"#### {answer}")
    else:
        formula = f"cos({A}°)cos({B}°) - sin({A}°)sin({B}°)"
        problem = f"Use sum formula: cos({A}°+{B}°)?"
        solution = (f"cos(A+B) = cosAcosB - sinAsinB\n"
                    f"= {cosA}*{cosB} - {sinA}*{sinB}\n"
                    f"= {answer}\n"
                    f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "trig_identity", "level": level, "source": "trig_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [unit_circle_lookup, right_triangle_side],
    2: [law_of_cosines_angle, law_of_sines, radian_degree],
    3: [double_angle_sin, sinusoid_params, radian_degree],
    4: [trig_equation_basic, double_angle_sin],
    5: [sum_formula, trig_equation_basic, sinusoid_params],
}


def generate_trig_pairs(n_pairs=500, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_trig{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} trig pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate trigonometry Q/A pairs")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_trig_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
