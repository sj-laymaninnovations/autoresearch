"""
geometry_qa.py — Geometry Q/A Generator for Math Training

Generates geometry problems with explicit Chain-of-Thought solutions.
Uses integer-friendly values (Pythagorean triples, integer dimensions).

Topics by level:
  L1: Area/perimeter of rectangles & triangles, Pythagorean theorem
  L2: Circles (area, circumference using pi=22/7), rectangular prism volume
  L3: Composite shapes, cylinder/cone/sphere volumes
  L4: Similar triangles, coordinate distance/midpoint
  L5: 3D diagonal in cuboid, sector area, arc length

Output: math_lab/results/harder_qa_geometry_<tag>_<timestamp>.jsonl

References:
  - Common Core Geometry standards
  - AMC 8/10 geometry problems
  - Khan Academy Geometry curriculum
"""

import json
import math
import random
import datetime
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path(__file__).parent.parent / "results"
MAX_CONTENT_LEN = 246

# Pythagorean triples for exact integer right triangles
PYTH_TRIPLES = [
    (3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10),
    (9,12,15),(12,16,20),(15,20,25),(20,21,29),(9,40,41),
]

PI_APPROX = Fraction(22, 7)  # use 22/7 for pi to keep answers rational


def _ok(p: str, s: str) -> bool:
    return len(p) + len(s) <= MAX_CONTENT_LEN


def _frac(p: int, q: int) -> str:
    f = Fraction(p, q)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ── L1: Basic area/perimeter, Pythagorean theorem ────────────────────────────

def rect_area(rng: random.Random, level: int) -> dict:
    l = rng.randint(2, 20 * level)
    w = rng.randint(2, 20 * level)
    area = l * w
    perim = 2 * (l + w)
    problem = f"Rectangle: length {l}, width {w}. Find area and perimeter."
    solution = (f"area = {l} * {w} = {area}\n"
                f"perimeter = 2*({l}+{w}) = 2*{l+w} = {perim}\n"
                f"#### {area}")
    return {"problem": problem, "solution": solution, "answer": area,
            "domain": "geom_rect", "level": level, "source": "geometry_qa"}


def triangle_area(rng: random.Random, level: int) -> dict:
    base = rng.randint(2, 20 * level)
    height = rng.randint(2, 20 * level)
    # ensure integer area
    if (base * height) % 2 != 0:
        base += 1
    area = base * height // 2
    problem = f"Triangle: base {base}, height {height}. Area?"
    solution = (f"area = base*height/2\n"
                f"= {base}*{height}/2\n"
                f"= {area}\n"
                f"#### {area}")
    return {"problem": problem, "solution": solution, "answer": area,
            "domain": "geom_triangle", "level": level, "source": "geometry_qa"}


def pythagorean_find_hyp(rng: random.Random, level: int) -> dict:
    triple = rng.choice(PYTH_TRIPLES)
    scale = rng.randint(1, 3 * level)
    a, b, c = triple[0]*scale, triple[1]*scale, triple[2]*scale
    problem = f"Right triangle legs {a} and {b}. Find the hypotenuse."
    solution = (f"c^2 = {a}^2 + {b}^2\n"
                f"c^2 = {a**2} + {b**2} = {a**2+b**2}\n"
                f"c = sqrt({a**2+b**2}) = {c}\n"
                f"#### {c}")
    return {"problem": problem, "solution": solution, "answer": c,
            "domain": "geom_pythagorean", "level": level, "source": "geometry_qa"}


def pythagorean_find_leg(rng: random.Random, level: int) -> dict:
    triple = rng.choice(PYTH_TRIPLES)
    scale = rng.randint(1, 3 * level)
    a, b, c = triple[0]*scale, triple[1]*scale, triple[2]*scale
    problem = f"Right triangle: hypotenuse {c}, one leg {a}. Find other leg."
    solution = (f"b^2 = {c}^2 - {a}^2\n"
                f"b^2 = {c**2} - {a**2} = {c**2-a**2}\n"
                f"b = {b}\n"
                f"#### {b}")
    return {"problem": problem, "solution": solution, "answer": b,
            "domain": "geom_pythagorean", "level": level, "source": "geometry_qa"}


# ── L2: Circles, 3D rectangular prism ────────────────────────────────────────

def circle_area(rng: random.Random, level: int) -> dict:
    """Use pi=22/7 to get rational area."""
    # r must make 22/7 * r^2 rational => all r are fine, store as fraction
    r = rng.randint(1, 10 * level)
    area_frac = PI_APPROX * r * r
    area_num, area_den = area_frac.numerator, area_frac.denominator
    area_str = _frac(area_num, area_den)
    problem = f"Circle with radius {r}. Area? (use pi=22/7)"
    solution = (f"Area = pi * r^2\n"
                f"= (22/7) * {r}^2\n"
                f"= (22/7) * {r**2}\n"
                f"= {area_str}\n"
                f"#### {area_str}")
    return {"problem": problem, "solution": solution, "answer": area_str,
            "domain": "geom_circle", "level": level, "source": "geometry_qa"}


def circle_circumference(rng: random.Random, level: int) -> dict:
    r = rng.randint(1, 10 * level)
    circ_frac = 2 * PI_APPROX * r
    circ_str = _frac(circ_frac.numerator, circ_frac.denominator)
    problem = f"Circle radius {r}. Circumference? (use pi=22/7)"
    solution = (f"C = 2*pi*r = 2*(22/7)*{r}\n"
                f"= {44*r}/7\n"
                f"= {circ_str}\n"
                f"#### {circ_str}")
    return {"problem": problem, "solution": solution, "answer": circ_str,
            "domain": "geom_circle", "level": level, "source": "geometry_qa"}


def box_volume(rng: random.Random, level: int) -> dict:
    l = rng.randint(2, 15 * level)
    w = rng.randint(2, 15 * level)
    h = rng.randint(2, 15 * level)
    vol = l * w * h
    sa = 2*(l*w + w*h + l*h)
    problem = f"Box: {l}x{w}x{h}. Volume?"
    solution = (f"V = l*w*h\n"
                f"= {l}*{w}*{h}\n"
                f"= {vol}\n"
                f"#### {vol}")
    return {"problem": problem, "solution": solution, "answer": vol,
            "domain": "geom_volume", "level": level, "source": "geometry_qa"}


# ── L3: Cylinder volume, composite shapes ────────────────────────────────────

def cylinder_volume(rng: random.Random, level: int) -> dict:
    """V = pi*r^2*h, use pi=22/7."""
    r = rng.randint(1, 7 * level)
    h = rng.randint(2, 15 * level)
    vol_frac = PI_APPROX * r * r * h
    vol_str = _frac(vol_frac.numerator, vol_frac.denominator)
    problem = f"Cylinder: radius {r}, height {h}. Volume? (pi=22/7)"
    solution = (f"V = pi*r^2*h\n"
                f"= (22/7)*{r**2}*{h}\n"
                f"= {_frac(22*r**2*h, 7)}\n"
                f"#### {vol_str}")
    return {"problem": problem, "solution": solution, "answer": vol_str,
            "domain": "geom_volume", "level": level, "source": "geometry_qa"}


def composite_rect(rng: random.Random, level: int) -> dict:
    """L-shaped figure = big rect minus small rect."""
    bw = rng.randint(5, 20 * level)
    bh = rng.randint(5, 20 * level)
    sw = rng.randint(1, bw - 1)
    sh = rng.randint(1, bh - 1)
    area = bw * bh - sw * sh
    problem = (f"L-shape: big rect {bw}x{bh}, cut out {sw}x{sh} corner. "
               f"Area?")
    solution = (f"Big = {bw}*{bh} = {bw*bh}\n"
                f"Cut = {sw}*{sh} = {sw*sh}\n"
                f"Area = {bw*bh} - {sw*sh} = {area}\n"
                f"#### {area}")
    return {"problem": problem, "solution": solution, "answer": area,
            "domain": "geom_composite", "level": level, "source": "geometry_qa"}


# ── L4: Similar triangles, coordinate geometry ───────────────────────────────

def similar_triangles(rng: random.Random, level: int) -> dict:
    """Triangles with ratio k; find missing side."""
    a = rng.randint(2, 10 * level)
    b = rng.randint(2, 10 * level)
    k = rng.randint(2, 5)
    a2 = a * k
    b2 = b * k
    # Ask for b2
    problem = f"Triangles similar, ratio {k}:1. Side {a} maps to {a2}. Side {b} maps to?"
    answer = b2
    solution = (f"Scale factor = {a2}/{a} = {k}\n"
                f"Missing side = {b} * {k} = {answer}\n"
                f"#### {answer}")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "geom_similar", "level": level, "source": "geometry_qa"}


def coord_distance(rng: random.Random, level: int) -> dict:
    """Distance between two integer-coordinate points."""
    # Use Pythagorean triple so distance is integer
    triple = rng.choice(PYTH_TRIPLES)
    scale = rng.randint(1, 2 * level)
    dx, dy, dist = triple[0]*scale, triple[1]*scale, triple[2]*scale
    x1 = rng.randint(-10, 10)
    y1 = rng.randint(-10, 10)
    x2, y2 = x1 + dx, y1 + dy
    problem = f"Distance between ({x1},{y1}) and ({x2},{y2})?"
    solution = (f"dx = {x2}-{x1} = {dx}\n"
                f"dy = {y2}-{y1} = {dy}\n"
                f"d = sqrt({dx}^2+{dy}^2) = sqrt({dx**2+dy**2}) = {dist}\n"
                f"#### {dist}")
    return {"problem": problem, "solution": solution, "answer": dist,
            "domain": "geom_coordinate", "level": level, "source": "geometry_qa"}


def coord_midpoint(rng: random.Random, level: int) -> dict:
    """Midpoint of two points; ask for midpoint x-coord."""
    x1 = rng.randint(-20, 20) * level
    y1 = rng.randint(-20, 20) * level
    x2 = rng.randint(-20, 20) * level
    y2 = rng.randint(-20, 20) * level
    # ensure integer midpoints
    if (x1 + x2) % 2 != 0:
        x2 += 1
    if (y1 + y2) % 2 != 0:
        y2 += 1
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    problem = f"Midpoint of ({x1},{y1}) and ({x2},{y2})?"
    answer = f"({mx},{my})"
    solution = (f"mx = ({x1}+{x2})/2 = {x1+x2}/2 = {mx}\n"
                f"my = ({y1}+{y2})/2 = {y1+y2}/2 = {my}\n"
                f"#### ({mx},{my})")
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "geom_coordinate", "level": level, "source": "geometry_qa"}


# ── L5: 3D diagonal, sector area ─────────────────────────────────────────────

def cuboid_diagonal(rng: random.Random, level: int) -> dict:
    """Space diagonal = sqrt(l^2+w^2+h^2); use triples for integer result."""
    # l^2+w^2+h^2 must be a perfect square
    # Easy: 1^2+2^2+2^2=9 -> scale
    base_combos = [(1,2,2,3),(2,4,4,6),(2,6,9,11),(6,6,7,11)]
    combo = rng.choice(base_combos)
    scale = rng.randint(1, 3 * level)
    l, w, h, d = combo[0]*scale, combo[1]*scale, combo[2]*scale, combo[3]*scale
    problem = f"Cuboid {l}x{w}x{h}. Space diagonal length?"
    solution = (f"d^2 = {l}^2+{w}^2+{h}^2\n"
                f"= {l**2}+{w**2}+{h**2} = {l**2+w**2+h**2}\n"
                f"d = {d}\n"
                f"#### {d}")
    return {"problem": problem, "solution": solution, "answer": d,
            "domain": "geom_3d", "level": level, "source": "geometry_qa"}


def sector_area(rng: random.Random, level: int) -> dict:
    """Sector area = (theta/360)*pi*r^2. Use theta in {30,45,60,90,120}."""
    r = rng.randint(2, 10 * level)
    theta = rng.choice([30, 45, 60, 90, 120])
    area_frac = Fraction(theta, 360) * PI_APPROX * r * r
    area_str = _frac(area_frac.numerator, area_frac.denominator)
    problem = f"Sector: radius {r}, angle {theta} degrees. Area? (pi=22/7)"
    solution = (f"Area = ({theta}/360)*pi*r^2\n"
                f"= ({theta}/360)*(22/7)*{r**2}\n"
                f"= {area_str}\n"
                f"#### {area_str}")
    return {"problem": problem, "solution": solution, "answer": area_str,
            "domain": "geom_sector", "level": level, "source": "geometry_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [rect_area, triangle_area, pythagorean_find_hyp, pythagorean_find_leg],
    2: [circle_area, circle_circumference, box_volume, rect_area],
    3: [cylinder_volume, composite_rect, box_volume],
    4: [similar_triangles, coord_distance, coord_midpoint],
    5: [cuboid_diagonal, sector_area, similar_triangles],
}


def generate_geometry_pairs(n_pairs=500, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_geometry{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} geometry pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate geometry Q/A pairs")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_geometry_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
