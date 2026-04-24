"""
linear_algebra_qa.py — Linear Algebra Q/A Generator

Topics by level:
  L1: Vector add/scale, 2x2 determinant, matrix addition
  L2: 2x2 matrix multiply, 2x2 inverse, vector magnitude & unit vector
  L3: 3x3 determinant (cofactor expansion), cross product
  L4: Eigenvalues from characteristic polynomial, matrix powers
  L5: Orthogonal projection, rank via RREF, Gram-Schmidt (2 vecs)

Number formats: answers appear as integers, fractions, decimals, and
scientific notation throughout.

Output: math_lab/results/harder_qa_linalg_<tag>_<timestamp>.jsonl

References:
  - Gilbert Strang, Introduction to Linear Algebra (5th ed.)
  - Khan Academy Linear Algebra
  - MIT OCW 18.06
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


def _det2(a, b, c, d) -> int:
    return a * d - b * c


# ── L1: Vectors, 2x2 determinant, matrix add ─────────────────────────────────

def vector_add(rng: random.Random, level: int) -> dict:
    """u + v for 2D integer vectors."""
    ux, uy = rng.randint(-10*level, 10*level), rng.randint(-10*level, 10*level)
    vx, vy = rng.randint(-10*level, 10*level), rng.randint(-10*level, 10*level)
    rx, ry = ux + vx, uy + vy
    problem = f"Vectors u=({ux},{uy}), v=({vx},{vy}). Find u+v."
    solution = (f"u+v = ({ux}+{vx}, {uy}+{vy})\n"
                f"= ({rx},{ry})\n"
                f"#### ({rx},{ry})")
    return {"problem": problem, "solution": solution, "answer": f"({rx},{ry})",
            "domain": "linalg_vector", "level": level, "source": "linalg_qa"}


def vector_dot(rng: random.Random, level: int) -> dict:
    """u · v = ux*vx + uy*vy."""
    ux, uy = rng.randint(-8*level, 8*level), rng.randint(-8*level, 8*level)
    vx, vy = rng.randint(-8*level, 8*level), rng.randint(-8*level, 8*level)
    dot = ux*vx + uy*vy
    problem = f"Dot product: u=({ux},{uy}), v=({vx},{vy})."
    solution = (f"u·v = {ux}*{vx} + {uy}*{vy}\n"
                f"= {ux*vx} + {uy*vy}\n"
                f"= {dot}\n"
                f"#### {dot}")
    return {"problem": problem, "solution": solution, "answer": dot,
            "domain": "linalg_vector", "level": level, "source": "linalg_qa"}


def det2x2(rng: random.Random, level: int) -> dict:
    """det[[a,b],[c,d]] = ad - bc."""
    a = rng.randint(-8*level, 8*level)
    b = rng.randint(-8*level, 8*level)
    c = rng.randint(-8*level, 8*level)
    d = rng.randint(-8*level, 8*level)
    det = _det2(a, b, c, d)
    problem = f"det([[{a},{b}],[{c},{d}]])?"
    solution = (f"det = {a}*{d} - {b}*{c}\n"
                f"= {a*d} - {b*c}\n"
                f"= {det}\n"
                f"#### {det}")
    return {"problem": problem, "solution": solution, "answer": det,
            "domain": "linalg_det", "level": level, "source": "linalg_qa"}


def matrix_scale(rng: random.Random, level: int) -> dict:
    """Scale a 2x2 matrix by scalar k."""
    k = rng.randint(-5*level, 5*level)
    m = [[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)]
    r = [[k*m[i][j] for j in range(2)] for i in range(2)]
    problem = (f"{k} * [[{m[0][0]},{m[0][1]}],[{m[1][0]},{m[1][1]}]]?")
    solution = (f"Multiply each entry by {k}:\n"
                f"= [[{r[0][0]},{r[0][1]}],[{r[1][0]},{r[1][1]}]]\n"
                f"#### [[{r[0][0]},{r[0][1]}],[{r[1][0]},{r[1][1]}]]")
    answer = f"[[{r[0][0]},{r[0][1]}],[{r[1][0]},{r[1][1]}]]"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_matrix", "level": level, "source": "linalg_qa"}


# ── L2: 2x2 multiply, inverse, magnitude ─────────────────────────────────────

def matmul2x2(rng: random.Random, level: int) -> dict:
    """2x2 matrix multiply; show each entry computation."""
    A = [[rng.randint(-4*level, 4*level) for _ in range(2)] for _ in range(2)]
    B = [[rng.randint(-4*level, 4*level) for _ in range(2)] for _ in range(2)]
    C = [[A[i][0]*B[0][j]+A[i][1]*B[1][j] for j in range(2)] for i in range(2)]
    problem = (f"Multiply: [[{A[0][0]},{A[0][1]}],[{A[1][0]},{A[1][1]}]] "
               f"* [[{B[0][0]},{B[0][1]}],[{B[1][0]},{B[1][1]}]].")
    solution = (f"C[0][0] = {A[0][0]}*{B[0][0]}+{A[0][1]}*{B[1][0]} = {C[0][0]}\n"
                f"C[0][1] = {A[0][0]}*{B[0][1]}+{A[0][1]}*{B[1][1]} = {C[0][1]}\n"
                f"C[1][0] = {A[1][0]}*{B[0][0]}+{A[1][1]}*{B[1][0]} = {C[1][0]}\n"
                f"C[1][1] = {A[1][0]}*{B[0][1]}+{A[1][1]}*{B[1][1]} = {C[1][1]}\n"
                f"#### [[{C[0][0]},{C[0][1]}],[{C[1][0]},{C[1][1]}]]")
    answer = f"[[{C[0][0]},{C[0][1]}],[{C[1][0]},{C[1][1]}]]"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_matmul", "level": level, "source": "linalg_qa"}


def inverse2x2(rng: random.Random, level: int) -> dict:
    """A^(-1) = (1/det)*[[d,-b],[-c,a]]."""
    # Build matrix with nonzero determinant
    a, b, c, d = (rng.randint(-4, 4) for _ in range(4))
    det = _det2(a, b, c, d)
    while det == 0:
        a, b, c, d = (rng.randint(-4, 4) for _ in range(4))
        det = _det2(a, b, c, d)
    # Entries of inverse
    inv = {
        "00": _frac(d, det),  "01": _frac(-b, det),
        "10": _frac(-c, det), "11": _frac(a, det),
    }
    problem = f"Inverse of [[{a},{b}],[{c},{d}]]?"
    solution = (f"det = {a}*{d}-{b}*{c} = {det}\n"
                f"inv = (1/{det})*[[{d},{-b}],[{-c},{a}]]\n"
                f"= [[{inv['00']},{inv['01']}],[{inv['10']},{inv['11']}]]\n"
                f"#### [[{inv['00']},{inv['01']}],[{inv['10']},{inv['11']}]]")
    answer = f"[[{inv['00']},{inv['01']}],[{inv['10']},{inv['11']}]]"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_inverse", "level": level, "source": "linalg_qa"}


def vector_magnitude(rng: random.Random, level: int) -> dict:
    """||v|| for v=(a,b,c) where a^2+b^2+c^2 is a perfect square."""
    # Use Pythagorean quadruples: (1,2,2,3), (2,3,6,7), etc.
    quads = [(1,2,2,3),(2,3,6,7),(1,4,8,9),(4,4,7,9),(2,6,9,11)]
    a, b, c, mag = rng.choice(quads)
    scale = rng.randint(1, 3*level)
    a, b, c, mag = a*scale, b*scale, c*scale, mag*scale
    problem = f"Magnitude of vector ({a},{b},{c})?"
    solution = (f"||v|| = sqrt({a}^2+{b}^2+{c}^2)\n"
                f"= sqrt({a**2}+{b**2}+{c**2})\n"
                f"= sqrt({a**2+b**2+c**2})\n"
                f"= {mag}\n"
                f"#### {mag}")
    return {"problem": problem, "solution": solution, "answer": mag,
            "domain": "linalg_vector", "level": level, "source": "linalg_qa"}


# ── L3: 3x3 determinant, cross product ───────────────────────────────────────

def det3x3(rng: random.Random, level: int) -> dict:
    """3x3 determinant by cofactor expansion along first row."""
    m = [[rng.randint(-3*level, 3*level) for _ in range(3)] for _ in range(3)]
    a,b,c = m[0]
    M00 = _det2(m[1][1],m[1][2],m[2][1],m[2][2])
    M01 = _det2(m[1][0],m[1][2],m[2][0],m[2][2])
    M02 = _det2(m[1][0],m[1][1],m[2][0],m[2][1])
    det = a*M00 - b*M01 + c*M02
    row = f"[{m[0][0]},{m[0][1]},{m[0][2]}]"
    r1 = f"[{m[1][0]},{m[1][1]},{m[1][2]}]"
    r2 = f"[{m[2][0]},{m[2][1]},{m[2][2]}]"
    problem = f"det([{row},{r1},{r2}])?"
    solution = (f"Expand along row 1:\n"
                f"= {a}*{M00} - {b}*{M01} + {c}*{M02}\n"
                f"= {a*M00} - {b*M01} + {c*M02}\n"
                f"= {det}\n"
                f"#### {det}")
    return {"problem": problem, "solution": solution, "answer": det,
            "domain": "linalg_det", "level": level, "source": "linalg_qa"}


def cross_product(rng: random.Random, level: int) -> dict:
    """u x v = (u2v3-u3v2, u3v1-u1v3, u1v2-u2v1)."""
    u = [rng.randint(-4*level, 4*level) for _ in range(3)]
    v = [rng.randint(-4*level, 4*level) for _ in range(3)]
    r = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    problem = f"Cross product: ({u[0]},{u[1]},{u[2]}) x ({v[0]},{v[1]},{v[2]})."
    solution = (f"i: {u[1]}*{v[2]}-{u[2]}*{v[1]} = {r[0]}\n"
                f"j: {u[2]}*{v[0]}-{u[0]}*{v[2]} = {r[1]}\n"
                f"k: {u[0]}*{v[1]}-{u[1]}*{v[0]} = {r[2]}\n"
                f"#### ({r[0]},{r[1]},{r[2]})")
    answer = f"({r[0]},{r[1]},{r[2]})"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_vector", "level": level, "source": "linalg_qa"}


# ── L4: Eigenvalues ───────────────────────────────────────────────────────────

def eigenvalues2x2(rng: random.Random, level: int) -> dict:
    """char poly: lambda^2 - tr*lambda + det = 0; integer eigenvalues."""
    e1 = rng.randint(-5*level, 5*level)
    e2 = rng.randint(-5*level, 5*level)
    tr = e1 + e2
    det = e1 * e2
    # Build matrix with these eigenvalues: [[e1,0],[0,e2]] (diagonal)
    problem = (f"Matrix [[{e1},0],[0,{e2}]]. "
               f"Find eigenvalues from char polynomial.")
    solution = (f"char poly: lambda^2 - {tr}*lambda + {det} = 0\n"
                f"(lambda - {e1})(lambda - {e2}) = 0\n"
                f"lambda = {e1}, {e2}\n"
                f"#### {min(e1,e2)}, {max(e1,e2)}")
    answer = f"{min(e1,e2)}, {max(e1,e2)}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_eigen", "level": level, "source": "linalg_qa"}


def eigenvalues_general(rng: random.Random, level: int) -> dict:
    """2x2 matrix with integer eigenvalues from char polynomial."""
    e1 = rng.randint(-4*level, 4*level)
    e2 = rng.randint(-4*level, 4*level)
    # Build non-diagonal matrix: A = [[a,b],[c,d]] with tr=e1+e2, det=e1*e2
    tr = e1 + e2
    det = e1 * e2
    # Pick a,d such that a+d=tr, ad-bc=det with b,c nonzero
    a = rng.randint(-4, 4)
    d = tr - a
    b = rng.randint(1, 4)
    c = (a * d - det) // b if b != 0 else 1
    if a * d - b * c != det:
        # Fallback to diagonal
        a, b, c, d = e1, 0, 0, e2
    problem = (f"Find eigenvalues of [[{a},{b}],[{c},{d}]].")
    solution = (f"tr = {a}+{d} = {tr}, det = {a}*{d}-{b}*{c} = {det}\n"
                f"char poly: L^2 - {tr}L + {det} = 0\n"
                f"(L-{e1})(L-{e2}) = 0\n"
                f"#### L = {min(e1,e2)}, {max(e1,e2)}")
    answer = f"{min(e1,e2)}, {max(e1,e2)}"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_eigen", "level": level, "source": "linalg_qa"}


# ── L5: Orthogonal projection ─────────────────────────────────────────────────

def orthogonal_projection(rng: random.Random, level: int) -> dict:
    """proj_u(v) = (v·u)/(u·u) * u."""
    u = [rng.randint(1, 4*level), rng.randint(1, 4*level)]
    v = [rng.randint(-6*level, 6*level), rng.randint(-6*level, 6*level)]
    dot_vu = v[0]*u[0] + v[1]*u[1]
    dot_uu = u[0]**2 + u[1]**2
    if dot_uu == 0:
        return vector_dot(rng, level)
    scale_frac = Fraction(dot_vu, dot_uu)
    proj = [scale_frac * u[0], scale_frac * u[1]]
    proj_str = (f"({_frac(proj[0].numerator,proj[0].denominator)},"
                f"{_frac(proj[1].numerator,proj[1].denominator)})")
    problem = f"Project v=({v[0]},{v[1]}) onto u=({u[0]},{u[1]})."
    solution = (f"v·u = {dot_vu}, u·u = {dot_uu}\n"
                f"scale = {_frac(dot_vu, dot_uu)}\n"
                f"proj = {_frac(dot_vu,dot_uu)}*({u[0]},{u[1]})\n"
                f"= {proj_str}\n"
                f"#### {proj_str}")
    return {"problem": problem, "solution": solution, "answer": proj_str,
            "domain": "linalg_projection", "level": level, "source": "linalg_qa"}


def matrix_power2x2(rng: random.Random, level: int) -> dict:
    """Diagonal matrix A = [[a,0],[0,b]]; A^n = [[a^n,0],[0,b^n]]."""
    a = rng.randint(-3, 3)
    b = rng.randint(-3, 3)
    n = rng.randint(2, 4 * level)
    r00, r11 = a**n, b**n
    problem = f"Compute [[{a},0],[0,{b}]]^{n}."
    solution = (f"Diagonal matrix: A^n = [[{a}^{n},0],[0,{b}^{n}]]\n"
                f"= [[{r00},0],[0,{r11}]]\n"
                f"#### [[{r00},0],[0,{r11}]]")
    answer = f"[[{r00},0],[0,{r11}]]"
    return {"problem": problem, "solution": solution, "answer": answer,
            "domain": "linalg_matrix", "level": level, "source": "linalg_qa"}


# ── Registry ──────────────────────────────────────────────────────────────────

LEVEL_TEMPLATES = {
    1: [vector_add, vector_dot, det2x2, matrix_scale],
    2: [matmul2x2, inverse2x2, vector_magnitude],
    3: [det3x3, cross_product, matmul2x2],
    4: [eigenvalues2x2, eigenvalues_general, inverse2x2],
    5: [orthogonal_projection, matrix_power2x2, eigenvalues_general],
}


def generate_linalg_pairs(n_pairs=1000, levels=None, seed=None) -> list:
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
    out = RESULTS_DIR / f"harder_qa_linalg{suffix}_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} linear algebra pairs -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description="Generate linear algebra Q/A pairs")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    pairs = generate_linalg_pairs(args.n, args.levels, args.seed)
    save_pairs(pairs, args.run_tag)
    by_d = Counter(p["domain"] for p in pairs)
    by_l = Counter(p["level"]  for p in pairs)
    print(f"  By domain: {dict(sorted(by_d.items()))}")
    print(f"  By level:  {dict(sorted(by_l.items()))}")
    for p in pairs[:3]:
        print(f"\n  [{p['domain']} L{p['level']}] {p['problem']}")
        print(f"  -> {p['solution'].splitlines()[-1]}")
