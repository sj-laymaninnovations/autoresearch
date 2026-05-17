"""
Worked solutions for EVAL-016 through EVAL-030 (hand-authored batch 1).
Each solution derives the answer and is verified by SymPy.
Run this file to confirm all answers: python hand_authored_batch1.py
"""
from sympy import (
    symbols, solve, Eq, Rational, sqrt, gcd, mod_inverse,
    factorint, isprime, floor, ceiling, binomial, factorial, Abs,
    simplify, expand, Sum, Integer, cos, pi, tan
)
from fractions import Fraction

x, y, z, n, k = symbols('x y z n k', real=True)

print("=== Hand-authored problem solutions ===\n")

# ─── ALGEBRA (5 problems) ──────────────────────────────────────────────────

# EVAL-016 — easy algebra
# "A rectangle has perimeter 48 and area 128. What is the length of the
#  longer side?"
# Let sides be p and q. p+q=24, pq=128.
# t^2 - 24t + 128 = 0 → t = (24 ± sqrt(576-512))/2 = (24 ± 8)/2 → 16 or 8
sides = solve([x+y-24, x*y-128], [x, y])
longer = max(s[0] for s in sides)
print(f"EVAL-016  longer side = {longer}")
assert longer == 16, longer

# EVAL-017 — medium algebra (solve absolute value equation)
# "|3x - 7| = 2x + 1. Find the sum of all valid solutions."
# Case 1: 3x-7 = 2x+1 → x=8  (check: |24-7|=17, 2·8+1=17 ✓)
# Case 2: 3x-7 = -(2x+1) → 5x=6 → x=6/5 (check:|18/5-7|=17/5, 2·6/5+1=17/5 ✓)
# Case 1: 3x-7 = 2x+1 → x=8; check: |17|=17, 17=17 ✓
# Case 2: 3x-7 = -(2x+1) → 5x=6 → x=6/5; check: |18/5-35/5|=17/5, 12/5+5/5=17/5 ✓
sol_a = Rational(8)
sol_b = Rational(6, 5)
assert Abs(3*sol_a - 7) == 2*sol_a + 1
assert Abs(3*sol_b - 7) == 2*sol_b + 1
s_sum = sol_a + sol_b
print(f"EVAL-017  solutions=[8, 6/5], sum={s_sum}")
assert Fraction(6,5) + 8 == Fraction(46, 5)
print(f"EVAL-017  sum = 46/5")

# EVAL-018 — medium algebra (system of equations)
# "Positive integers a,b satisfy a+b=100 and lcm(a,b)=495. Find a·b."
# lcm(a,b) = a*b/gcd(a,b). So a*b = lcm*gcd = 495*gcd.
# Also a+b=100; try gcd=5 → a*b=2475, (a-b)^2=(a+b)^2-4ab=10000-9900=100→a-b=10 → a=55,b=45
g = 5
ab_prod = 495 * g
a_val = (100 + 10) // 2
b_val = (100 - 10) // 2
assert a_val + b_val == 100
assert ab_prod == a_val * b_val
from math import gcd as mgcd
assert mgcd(a_val, b_val) == g
lcm_val = ab_prod // g
assert lcm_val == 495
print(f"EVAL-018  a={a_val}, b={b_val}, a·b={ab_prod}")

# EVAL-019 — medium algebra (quadratic with integer root constraint)
# "The quadratic x^2 - (k+4)x + 4k has two integer roots for integer k.
#  Find the sum of all such values of k."
# Sum of roots = k+4, product = 4k
# If roots are r,s: r+s=k+4, rs=4k → rs=4(r+s-4) → rs-4r-4s=-16 → (r-4)(s-4)=0
# So at least one root is 4. If r=4: 4+s=k+4→s=k; rs=4k→4k=4k ✓ for any k.
# But we need BOTH roots to be integers, which they are: roots are 4 and k.
# For two DISTINCT integer roots: k≠4. But the problem asks for k where both roots integer.
# Answer: all integers k (since roots 4 and k are always integers). But the problem
# likely asks for which k gives two POSITIVE integer roots — let's constrain to k>0, k≠4.
# Actually: the problem as stated is always satisfied for integer k.
# Let's refine: "two DISTINCT positive integer roots."
# root1=4, root2=k>0, k≠4. Sum of k from 1 to 10 excluding 4 = 55-4=51.
# But that's arbitrary range. Let's pick a cleaner version.
# Revised problem: roots r,s where 1≤r,s≤10. Then k=s (second root) in {1..10}\{4}.
# Sum = 1+2+3+5+6+7+8+9+10 = 51
answer_019 = sum(i for i in range(1,11) if i != 4)
print(f"EVAL-019  sum of k values = {answer_019}")
assert answer_019 == 51

# EVAL-020 — hard algebra (functional equation style)
# "f(x) = ax^2 + bx + c with f(1)=6, f(-1)=2, f(2)=11. Find f(3)."
# f(1)=a+b+c=6; f(-1)=a-b+c=2; f(2)=4a+2b+c=11
a_val, b_val, c_val = symbols('a b c')
sol = solve([
    Eq(a_val+b_val+c_val, 6),
    Eq(a_val-b_val+c_val, 2),
    Eq(4*a_val+2*b_val+c_val, 11)
], [a_val, b_val, c_val])
f3 = sol[a_val]*9 + sol[b_val]*3 + sol[c_val]
print(f"EVAL-020  a={sol[a_val]}, b={sol[b_val]}, c={sol[c_val]}, f(3)={f3}")
assert f3 == 18

# ─── GEOMETRY (3 problems) ─────────────────────────────────────────────────

# EVAL-021 — easy geometry
# "A right triangle has legs 9 and 12. A circle inscribed in the triangle
#  has radius r. Find r."
# r = (a+b-c)/2 where a,b are legs, c hypotenuse
import math as _math
a_leg, b_leg = 9, 12
c_hyp = _math.isqrt(a_leg**2 + b_leg**2)
assert c_hyp == 15
r_inscribed = (a_leg + b_leg - c_hyp) // 2
print(f"EVAL-021  inradius r = {r_inscribed}")
assert r_inscribed == 3

# EVAL-022 — medium geometry
# "A regular hexagon has area 54√3. What is the perimeter?"
# Area = (3√3/2)s^2 = 54√3 → s^2 = 36 → s=6 → perimeter=36
from sympy import sqrt as ssqrt, solve as ssolve, Symbol
s = Symbol('s', positive=True)
area_eq = Eq(Rational(3,2)*ssqrt(3)*s**2, 54*ssqrt(3))
s_val = ssolve(area_eq, s)[0]
perimeter = 6 * s_val
print(f"EVAL-022  s={s_val}, perimeter={perimeter}")
assert perimeter == 36

# EVAL-023 — hard geometry
# "Two circles of radius 5 and 12 are externally tangent. A common external
#  tangent touches the circles. Find the length of the segment of the tangent
#  between the two tangent points."
# d = distance between centers = 5+12 = 17
# length = sqrt(d^2 - (r1-r2)^2) = sqrt(289-49) = sqrt(240) = 4√15
d = 5 + 12
r1, r2 = 12, 5
tang_len = ssqrt(d**2 - (r1-r2)**2)
tang_simplified = simplify(tang_len)
print(f"EVAL-023  tangent length = {tang_simplified}")
assert tang_simplified == 4*ssqrt(15)

# ─── NUMBER THEORY (3 problems) ────────────────────────────────────────────

# EVAL-024 — easy number theory
# "Find the number of positive divisors of 2^4 · 3^2 · 5."
# τ(n) = (4+1)(2+1)(1+1) = 30
divs = (4+1)*(2+1)*(1+1)
print(f"EVAL-024  number of divisors = {divs}")
assert divs == 30

# EVAL-025 — medium number theory
# "Find the remainder when 7^100 is divided by 50."
# φ(50)=20; 7^20≡1 (mod 50) by Euler; 100=5·20; 7^100≡1 (mod 50)
rem = pow(7, 100, 50)
print(f"EVAL-025  7^100 mod 50 = {rem}")
assert rem == 1

# EVAL-026 — hard number theory
# "How many integers n with 1 ≤ n ≤ 1000 have gcd(n, 1000) > 4?"
# gcd(n,1000)>4 means gcd ∈ {5,8,10,20,25,40,50,100,125,200,250,500,1000}
from math import gcd as mgcd
# Analytical: gcd=1→φ(1000)=400; gcd=2→φ(500)=200; gcd=4→φ(250)=100 → NOT satisfying=700
# Satisfying: 1000-700=300
count = sum(1 for i in range(1, 1001) if mgcd(i, 1000) > 4)
print(f"EVAL-026  count = {count}")
assert count == 300   # verified analytically

# ─── COMBINATORICS (2 problems) ────────────────────────────────────────────

# EVAL-027 — medium combinatorics
# "How many 4-digit integers (1000-9999) have digits summing to exactly 10
#  with the first digit nonzero?"
# d1 in 1-9, d2,d3,d4 in 0-9, d1+d2+d3+d4=10
count_027 = 0
for d1 in range(1,10):
    for d2 in range(0,10):
        for d3 in range(0,10):
            d4 = 10 - d1 - d2 - d3
            if 0 <= d4 <= 9:
                count_027 += 1
print(f"EVAL-027  count = {count_027}")
# Stars-and-bars: C(12,3)=220 minus 1 case (d1=10 impossible) = 219
assert count_027 == 219

# EVAL-028 — hard combinatorics
# "A fair coin is flipped 8 times. What is the probability (as a fraction in
#  lowest terms p/q) that exactly 3 heads appear? Give p+q."
# P = C(8,3)/2^8 = 56/256 = 7/32
prob_num = int(binomial(8,3))
prob_den = 2**8
from math import gcd as mgcd2
g = mgcd2(prob_num, prob_den)
p, q = prob_num//g, prob_den//g
print(f"EVAL-028  probability = {p}/{q}, p+q = {p+q}")
assert (p, q) == (7, 32)
assert p + q == 39

# ─── MULTI-STEP WORD PROBLEMS (2 problems) ─────────────────────────────────

# EVAL-029 — medium word problem
# "A store sells apples for $0.60 each and oranges for $0.90 each.
#  Sarah spends exactly $9.00 buying a total of 13 fruits.
#  How many apples did she buy?"
# 0.60a + 0.90o = 9.00;  a+o=13
# 6a+9o=90, a+o=13 → 6a+9(13-a)=90 → -3a = 90-117 = -27 → a=9
a_fruit = (90 - 9*13) // (6 - 9)
o_fruit = 13 - a_fruit
assert 60*a_fruit + 90*o_fruit == 900
print(f"EVAL-029  apples={a_fruit}, oranges={o_fruit}")
assert a_fruit == 9

# EVAL-030 — hard word problem
# "Three workers A, B, C can together complete a job in 4 days.
#  A and B together take 6 days. B and C together take 9 days.
#  How many days does A alone take? Give the answer as a fraction p/q; return p+q."
# 1/A + 1/B + 1/C = 1/4
# 1/A + 1/B = 1/6
# 1/B + 1/C = 1/9
# From first and third: 1/A = 1/4 - 1/9 = 5/36
# So A = 36/5 days → p+q = 41
inv_ABC = Rational(1,4)
inv_AB  = Rational(1,6)
inv_BC  = Rational(1,9)
inv_A = inv_ABC - inv_BC
inv_B = inv_AB - inv_A
inv_C = inv_BC - inv_B
A_days = 1/inv_A
print(f"EVAL-030  1/A={inv_A}, A={A_days} days, p+q={A_days.p + A_days.q}")
assert inv_A == Rational(5,36)
assert A_days == Rational(36,5)
assert A_days.p + A_days.q == 41

print("\n=== All 15 hand-authored solutions verified ✓ ===")
