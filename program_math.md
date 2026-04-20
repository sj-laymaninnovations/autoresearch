# autoresearch — Math Kernel Edition

This is an autonomous research session focused on the **ATTN11-x86 integer attention kernel**.
The goal is to find the best-performing kernel variant on math benchmarks
subject to strict size and memory constraints.

## Core Files

| File | Purpose | Editable? |
|---|---|---|
| `kernel/*.asm` | x86-64 assembly kernel modules | **YES — this is your workspace** |
| `kernel/qsparser_utf8.asm` | UTF-8 tokenizer (drop-in upgrade) | YES — after ASCII baseline works |
| `kernel/kernel.h` | C ABI declarations | Only if adding new functions |
| `kernel/Makefile` | Build system | Only to add flags/optimizations |
| `kernel/check_size.py` | Size enforcer | NO |
| `math_lab/benchmarks/eval_harness.py` | Evaluation | NO |
| `math_lab/datagen/harder_qa.py` | Harder Q/A generator | Only to tune difficulty |
| `math_lab/arxiv_scout.py` | arXiv paper scout | NO — call it, don't edit |
| `prepare.py`, `train.py` | GPU training loop | NO — do not touch |

## Hard Constraints (never violate)

1. **Kernel .text ≤ 20,480 bytes** — enforced by `make size`
2. **Peak RAM ≤ 4 GB** — keep model dims reasonable
3. **Integer-only** — no x87 FPU instructions, no SSE floats in compute paths
   (SSE2 float is allowed ONLY in `turboquant.asm` for the F32→Q8.8 conversion)
4. **Q8.16 format** — all compute vectors and matrices use `int32_t` Q8.16
5. **Win64 ABI** — preserve rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15

## Setup Steps (before first experiment)

1. Check that NASM is installed: `nasm -v`
2. Build the kernel: `cd kernel && make`
3. Verify size: `make size` (should print PASS ≤ 20480 bytes)
4. Run unit tests: `make test`
5. Run baseline eval: `python math_lab/benchmarks/eval_harness.py --domain gsm8k --n 50`
6. Record baseline in `math_lab/results/results.tsv`

## Metric

**Primary**: `gsm8k_accuracy` — fraction of GSM8K grade-school problems answered correctly.
**Secondary**: `algebra_accuracy` — MATH dataset Algebra sub-domain.

Lower is better for val_bpb (train.py). Higher is better for accuracy (this program).

The fixed evaluation budget: **200 GSM8K problems**, timeout 120 seconds.

## Experiment Ideas (ordered by expected impact)

Try these in order. For each, explain your hypothesis before editing code.

### Tier 1 — High Impact

1. **SSE4.1 PMULLD fast path in vdot**
   - Hypothesis: 4× throughput on dot products → faster token processing → more problems solved in timeout
   - File: `kernel/vecop.asm`
   - Target: replace scalar IMUL loop with SSE4.1 PMULLD + PHADDD accumulation
   - Check: `CPUID` feature detection or assume target has SSE4.1 (post-2008)

2. **Wider dot accumulator: Q8.24 (shift 24 instead of 16)**
   - Hypothesis: more precision in attention scores → better answer extraction
   - File: `kernel/vecop.asm`, `kernel/attn_kernel.asm`
   - Tradeoff: more overflow risk with large n; test on d_model≤64 first

3. **softmax temperature parameter**
   - Hypothesis: integer temperature T: S[i][j] = vdot / (sqrt_shift × T) allows sharpening attention
   - File: `kernel/actfn.asm`, add `T` arg to `sftmx`
   - Test T ∈ {1, 2, 4} (all integer shifts)

4. **REP MOVSD batch projection in embed**
   - Hypothesis: `vcpy` + `vadd` in EMBED can be fused into single-pass SIMD
   - File: `kernel/attn_kernel.asm` (`embed` function)

### Tier 2 — Medium Impact

5. **BPE tokenizer (256 merge rules) in qsparser**
   - Hypothesis: fewer tokens per problem → longer effective context for same seq_len
   - File: `kernel/qsparser.asm`
   - Tradeoff: adds ~2 KB of merge table — check size budget after

6. **Sinusoidal positional encoding LUT (integer sin/cos)**
   - Hypothesis: better position signal than arbitrary pos_emb at math-specific positions
   - File: `kernel/attn_kernel.asm` (embed), add `sincos_lut.asm`

7. **Turboquant calibration: Q6.10 storage (INT16, 10 frac bits)**
   - Hypothesis: less quantization noise → marginally better math accuracy
   - File: `kernel/turboquant.asm` — change shift from 8 to 10 in Q8→Q16 expand

8. **Multi-head attention (n_heads=2, d_head=d_model/2)**
   - Hypothesis: two attention heads can specialize (arithmetic vs. algebra)
   - File: `kernel/attn_kernel.asm` (major refactor, try late)

### Tier 3 — Low Impact / Experiments

9. **Cautious INT32 saturation removal in vecop hot paths**
   - Hypothesis: saturation clamps slow the loop; test if removing them crashes accuracy
   - File: `kernel/vecop.asm` — conditional compile via NASM `%define NO_SATURATE`

10. **kvcache LRU eviction for sliding-window math**
    - Hypothesis: caching recent K/V and evicting old ones helps multi-step problems
    - File: `kernel/kvcache.asm`

## SIMD Upgrade Roadmap

This is the intended progression once the scalar kernel is proved correct.
All SIMD paths remain **integer only** — no float SIMD instructions.

### Phase A — SSE2 (baseline, all x64 CPUs, available now)
```nasm
; VDOT using PMADDWD (SSE2): multiply 8 int16 pairs → 4 int32 sums
; Requires values to fit in int16 (Q8.8 range: -32768..32767) ✓
; 8 elements per iteration vs. 1 scalar
;
; movdqu  xmm0, [rsi]  ; load 4 × int32 from x
; movdqu  xmm1, [rdi]  ; load 4 × int32 from y
; ; For Q8.16, pack to int16 first (lose lower 8 bits = acceptable approx)
; packssdw xmm0, xmm0  ; int32 → int16, lower 4 elements
; packssdw xmm1, xmm1
; pmaddwd  xmm0, xmm1  ; 4 × int16 × int16 → 2 × int32 (accumulate pairs)
; paddd    xmm2, xmm0  ; accumulate
```
- **Size added**: ~180 bytes to vecop.asm
- **Throughput**: 4-8× scalar on VDOT for n ≥ 16
- **Precision**: loses 8 fractional bits (Q8.8 effective), acceptable for large d_model

### Phase B — SSE4.1 PMULLD (correct Q8.16, 4-wide)
```nasm
; PMULLD multiplies 4 int32 pairs → low 32 bits of each 64-bit product
; For Q8.16 × Q8.16 → Q16.32: PMULLD gives low 32 bits
; Safe when |a| × |b| < 2^31 = 2,147,483,648
; For typical weight values |val| < 2 (= 131072 in Q8.16):
;   max product = 131072² = 2^34 → PMULLD UNSAFES (truncates)
; FIX: use PMULLD on values that fit in int16 (after clamp to ±32767)
;      then PSRAD xmm, 16 to normalize
;
; pmulld  xmm0, xmm1   ; 4 × int32 low products
; psrad   xmm0, 16     ; >> 16 → Q8.16 (safe for values ≤ ±32767)
; paddd   xmm2, xmm0   ; accumulate
```
- **Size added**: ~250 bytes
- **Throughput**: 4× scalar with correct Q8.16 when values are clamped

### Phase C — AVX2 VPMULLD (8-wide int32)
```nasm
; Exactly as Phase B but 256-bit: 8 × int32 per cycle
; Requires AVX2 (Intel Haswell 2013+, AMD Zen 2019+)
; Add CPUID check: EBX bit 5 after leaf 7
;
; vmovdqu  ymm0, [rsi]   ; 8 × int32
; vmovdqu  ymm1, [rdi]
; vpmulld  ymm0, ymm0, ymm1
; vpsrad   ymm0, ymm0, 16
; vpaddd   ymm2, ymm2, ymm0
```
- **Size added**: ~300 bytes (with CPUID fallback to Phase B)
- **Throughput**: 8× scalar; VDOT over 256 elements goes from ~256 IMUL to ~32 VPMULLD cycles

### SIMD Application Map
```
vdot  (vecop.asm)  → Phase A (fastest gain)  : dot products for attn scores S[i][j]
vadd  (vecop.asm)  → Phase A PADDD            : residual add Y += X
vsadd (vecop.asm)  → Phase B VPMULLD+VPADDD   : backward scatter, outer product
mvmul (matop.asm)  → calls vdot, benefits automatically
vtmul (matop.asm)  → calls vsadd, benefits automatically
sftmx (actfn.asm)  → PMAX, PMIN on exp LUT index; PADDD for sum accumulation
```

---

## UTF-8 Upgrade Path

The kernel ships with two parsers:
- **`qsparser.asm`** — ASCII only (vocab 131, <500 bytes .text)
- **`qsparser_utf8.asm`** — UTF-8 with Greek/math Unicode (vocab 512, ~900 bytes .text)

### When to switch
Start with the ASCII baseline. Switch to UTF-8 when:
1. Math problems contain Unicode symbols (π, ∑, ≤, √, etc.)
2. The accuracy delta between ASCII (`3.14` → tokens) vs UTF-8 (`π` → single token) is measurable

### How to switch (Makefile change, ~3 lines)
```makefile
# In kernel/Makefile, replace qsparser.asm with qsparser_utf8.asm:
KERNEL_SRCS = fxmath.asm vecop.asm matop.asm actfn.asm \\
              attn_kernel.asm qsparser_utf8.asm turboquant.asm \\
              kvcache.asm io_win64.asm
# Also update kernel.h qsparse → qsparse_utf8 declaration
```

### Size budget with UTF-8
```
qsparser.asm      ~500 B .text
qsparser_utf8.asm ~900 B .text  (+400 B for state machine + decode)
MATH_SYM_MAP      256 B .rodata (sparse codepoint table)
Net impact:        +656 B text + 256 B data  -- well within 20 KB budget
```

### Vocab expansion (stay within 512 tokens)
```
0..130   ASCII (same as before)
131..227 Latin-1 supplement [U+00A0..U+00FF]  (accented letters)
228..372 Greek alphabet     [U+0391..U+03C9]  (α β γ π θ Σ etc.)
373..511 Math symbols       [U+2200..U+22FF]  (∀ ∃ ∈ ∑ √ ∞ ≤ ≥ ≠ ≈ ⋅)
```

---

## arXiv Scout Presets

Run these periodically (every 3 experiments) to stay current with the literature:

```bash
python math_lab/arxiv_scout.py --preset integer_attn  --n 3   # integer attention papers
python math_lab/arxiv_scout.py --preset simd_kernel   --n 3   # SIMD optimization papers
python math_lab/arxiv_scout.py --preset tiny_model    --n 3   # sub-MB model papers
python math_lab/arxiv_scout.py --preset utf8_token    --n 2   # UTF-8 tokenization
python math_lab/arxiv_scout.py --preset math_reasoning --n 3  # math benchmark papers
```

Each run saves a JSON file in `math_lab/results/arxiv_ideas_*.json`. Read the
"safe_ideas" field first — those have no constraint violations and can be tried directly.

If a paper idea conflicts with hard constraints (floats, GPU, >4GB), the scout
flags it with a `constraint_warning`. Such ideas must be **translated** to the
integer/assembly domain before attempting — e.g., "Flash Attention" → tiled SRAM
attention with the integer LUT; "RoPE" → integer sin/cos LUT in Q8.16.

---

## Autoresearch Loop

```
LOOP FOREVER:

0. [SCOUT — once per 3 experiments]
   Run: python math_lab/arxiv_scout.py --preset integer_attn --n 3
   Read the "EXPERIMENT IDEAS" section of the output.
   If a safe idea (no constraint warning) is not already in your history, add it to your queue.
   Also run: python math_lab/arxiv_scout.py --preset simd_kernel --n 3
   A new paper idea overrides the next tier-2 experiment from the list below.

1. Read current branch state: git log --oneline -5
2. Pick ONE experiment idea (from arXiv scout output OR list below)
3. Write your hypothesis in a comment at the top of the modified .asm file
4. Edit the .asm file(s)
5. Build: cd kernel && make 2>&1
   - If build fails: fix the error. If unfixable after 3 attempts, discard and pick next idea.
6. Size check: make size
   - If FAIL (over 20KB): revert the change. Pick a different idea or trim.
7. Unit test: make test
   - If test fails: fix the bug. If unfixable: discard.
8. Eval: python math_lab/benchmarks/eval_harness.py --domain gsm8k --n 200 > eval.log 2>&1
9. Read result: grep "Accuracy" eval.log
10. git commit -m "experiment: <one-line description>"
11. Record in math_lab/results/results.tsv:
      commit  gsm8k_acc  algebra_acc  kernel_kb  ram_mb  status  description
12. If improved accuracy (gsm8k >= prev + 0.005):
      -> KEEP: this commit stays, run harder_qa.py to generate 50 harder pairs
      -> python math_lab/datagen/harder_qa.py --domain arithmetic --n 50
13. If worse or equal:
      -> DISCARD: git reset HEAD~1
14. Continue -- NEVER STOP unless manually interrupted
```

## TSV Format (math_lab/results/results.tsv)

Tab-separated, do NOT use commas in description:

```
commit  gsm8k_acc  algebra_acc  kernel_kb  ram_mb  status  description
```

Example rows:
```
a1b2c3d	0.123	0.085	17.8	48.2	keep	baseline char-level qsparse Q8.16
b2c3d4e	0.141	0.092	18.3	48.5	keep	SSE4.1 PMULLD vdot 4x throughput
c3d4e5f	0.119	0.081	17.8	48.2	discard	Q8.24 wider accum hurts small d_model
```

## Q8.16 Quick Reference

```
1.0   = 65536   (Q16_ONE)
0.5   = 32768
0.25  = 16384
-1.0  = -65536
max   = 2147483647  (~32767.999)
min   = -2147483648 (-32768.0)

fxmul(a, b):  (int64)(a) * (int64)(b) >> 16
fxdiv(a, b):  (int64)(a) << 16 / (int64)(b)
fxabs(a):     branchless via CQO/XOR/SUB
sftmx(v,n):   LUT-based, EXPTBL[512] in actfn.asm
```

## Notes on PDP-11 → x86 Translation

The original PDP-11 `attn.s` is the behavior oracle. Key differences to keep in mind:

- PDP-11 used **Q7.8** (16-bit, 8 frac bits); x86 uses **Q8.16** (32-bit, 16 frac bits)
- PDP-11 `MUL` gives 32-bit result in reg pair; x86 `IMUL r64, r64` gives 64-bit in one register
- PDP-11 `ASHC $-8` shifts the 32-bit pair right 8; x86 `SAR rax, 16` does the equivalent
- PDP-11 self-modifying code patches inline parameter blocks; x86 uses stack args instead
- PDP-11 `SOB` (subtract one and branch): replaced by `DEC + JNZ`
- PDP-11 `JSR R5, VTMUL` with inline `.WORD` params: replaced by standard C call with stack args

**NEVER STOP — run until manually interrupted.**
