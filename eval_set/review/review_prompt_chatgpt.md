# MathGPT Eval Set — LLM Review (Chatgpt)

You are acting as an independent mathematical reviewer for a held-out evaluation set.

Instructions:
1. For each problem below, solve it INDEPENDENTLY — do not look at any answer key.
2. Write your solution and final answer clearly.
3. After solving, note if any problem is:
   (a) Ambiguous or ill-posed (multiple valid interpretations)
   (b) Has an answer you consider incorrect or incomplete
   (c) Appears to be a problem you recognize from a published source or training dataset
      (flag the source if known)
4. Do NOT guess — if you cannot solve a problem, say so explicitly.

Format your response for each problem as:
---
PROBLEM ID: [ID]
MY ANSWER: [your computed answer]
WORKING: [brief step-by-step]
FLAGS: [none | ambiguous | answer-suspect | contamination-risk: {description}]
---

The recorded answers will be compared to yours after you submit. We are looking for
problems where your answer differs from ours — this is a quality-control check.


---

## Problems

PROBLEM EVAL-016 [algebra / easy]
A rectangle has perimeter 48 and area 128. What is the length of the longer side?
(Source: Hand-authored)

PROBLEM EVAL-017 [algebra / medium]
|3x - 7| = 2x + 1. Find the sum of all solutions x satisfying the equation.
(Source: Hand-authored)

PROBLEM EVAL-018 [number_theory / medium]
Positive integers a and b satisfy a + b = 100 and lcm(a, b) = 495. Find the product a·b.
(Source: Hand-authored)

PROBLEM EVAL-019 [algebra / medium]
The quadratic x^2 - (k+4)x + 4k has two distinct positive integer roots for integer k with 1 ≤ k ≤ 10 and k ≠ 4. Find the sum of all such valid values of k.
(Source: Hand-authored)

PROBLEM EVAL-020 [algebra / easy]
A quadratic f(x) = ax^2 + bx + c satisfies f(1) = 6, f(-1) = 2, and f(2) = 11. Find f(3).
(Source: Hand-authored)

PROBLEM EVAL-021 [geometry / easy]
A right triangle has legs of length 9 and 12. What is the radius of the inscribed circle?
(Source: Hand-authored)

PROBLEM EVAL-022 [geometry / medium]
A regular hexagon has area 54√3. What is its perimeter?
(Source: Hand-authored)

PROBLEM EVAL-023 [geometry / hard]
Two circles with radii 12 and 5 are externally tangent to each other. A common external tangent is drawn touching both circles. Find the length of the segment of the tangent between the two tangent points. Express your answer in simplest radical form.
(Source: Hand-authored)

PROBLEM EVAL-024 [number_theory / easy]
Find the number of positive divisors of the integer 2^4 · 3^2 · 5.
(Source: Hand-authored)

PROBLEM EVAL-025 [number_theory / medium]
Find the remainder when 7^100 is divided by 50.
(Source: Hand-authored)

PROBLEM EVAL-026 [number_theory / hard]
How many integers n with 1 ≤ n ≤ 1000 have gcd(n, 1000) > 4?
(Source: Hand-authored)

PROBLEM EVAL-027 [combinatorics / medium]
How many 4-digit positive integers (from 1000 to 9999) have digits that sum to exactly 10?
(Source: Hand-authored)

PROBLEM EVAL-028 [combinatorics / medium]
A fair coin is flipped 8 times. The probability of getting exactly 3 heads is p/q in lowest terms. Find p + q.
(Source: Hand-authored)

PROBLEM EVAL-029 [word_problem / easy]
A store sells apples for $0.60 each and oranges for $0.90 each. Sarah spends exactly $9.00 buying a total of 13 fruits. How many apples did she buy?
(Source: Hand-authored)

PROBLEM EVAL-030 [word_problem / hard]
Workers A, B, and C together complete a job in 4 days. A and B together take 6 days, and B and C together take 9 days. If A working alone takes p/q days (in lowest terms), find p + q.
(Source: Hand-authored)

