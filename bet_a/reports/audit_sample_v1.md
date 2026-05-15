# Bet A — Audit Sample (10 of 74)

**Purpose:** Sean reads each pair + source-commit body, flags any of:

- HALLUCINATION — Q/A asserts a fact not in the source body
- WEAK — Q/A is technically faithful but not training-worthy
- FORMAT — `####` issue or codebase name leak (validator should have caught these)
- OK — pair is training-grade

Mark verdict in the `Verdict:` slot under each pair.

---

## #1 — `82e9155c75` (dav1d_src) ⚠ #### not on own line

**Source commit subject:** AArch64: Trim Armv8.0 Neon path of 6-tap and 8-tap MC functions

**Source commit body (first 1000 chars):**
```
There are some instruction sequences we could merge after the lane
load/store patch (ec5c3052cfcd8741c7f1b4bf4ddb0b5dcf09a1b8).

This change will simplify the loading of filter weights to save 288
bytes in the Armv8.0 Neon path of 6-tap and 8-tap MC functions.
```

**Q:** How can merging instruction sequences after a lane load/store patch reduce code size in ARM64 Neon implementations of multi-tap motion compensation functions?

**A:** By combining adjacent instructions that handle filter weight loading, the compiler can eliminate redundant moves and temporary registers. This consolidation removes unnecessary load/store operations that were previously inserted to align data for each tap, resulting in a more compact instruction stream. In the Armv8.0 Neon path for 6‑tap and 8‑tap motion compensation, this technique trimmed the code by 288 bytes. #### Merging load/store instructions can shrink Neon MC function size.

**Verdict:** _________________________

---

## #2 — `5ad5e5d8f1` (x264_src) ⚠ #### not on own line

**Source commit subject:** Improve deblock-a.S Performance by Using SVE/SVE2

**Source commit body (first 1000 chars):**
```
Imporve the performance of NEON functions of aarch64/deblock-a.S
by using the SVE/SVE2 instruction set. Below, the specific functions
are listed together with the improved performance results.

Command executed: ./checkasm8 --bench=deblock
Testbed: Alibaba g8y instance based on Yitian 710 CPU
Results:
deblock_chroma[1]_c: 735
deblock_chroma[1]_neon: 427
deblock_chroma[1]_sve: 353

Command executed: ./checkasm8 --bench=deblock
Testbed: AWS Graviton3
Results:
deblock_chroma[1]_c: 719
deblock_chroma[1]_neon: 442
deblock_chroma[1]_sve: 345
```

**Q:** How can SVE/SVE2 instructions be used to accelerate ARM64 deblocking functions that were originally written for NEON?

**A:** By replacing the original NEON intrinsics with their SVE/SVE2 equivalents, the code can process wider vector registers (up to 512 bits) and perform more operations per instruction. This reduces the number of loop iterations needed for a given data set, leading to lower latency and higher throughput. In practice, the deblock_chroma[1] routine saw a performance drop from 735 cycles on pure C to 427 cycles with NEON, and further down to 353 cycles when ported to SVE on a Yitian 710 CPU. On an AWS Graviton3, the same pattern holds: C at 719 cycles, NEON at 442 cycles, and SVE at 345 cycles. The range of improvement is therefore from roughly 42 % to 52 % faster than the C baseline, depending on the target CPU.#### SVE/SVE2 can cut deblocking cycles by up to half compared to NEON

**Verdict:** _________________________

---

## #3 — `93339ce857` (dav1d_src) ⚠ #### not on own line

**Source commit subject:** AArch64: Optimize Armv8.0 Neon path of HBD horizontal 6-tap filters

**Source commit body (first 1000 chars):**
```
The 6-tap horizontal subpel filters can be further improved by some
pointer arithmetic and saving some instructions (EXTs) in their data
rearrangement codes.

Relative runtime of micro benchmarks after this patch on some Cortex
CPU cores:

regular:     X1      A78      A76      A55
 mc  w8:  0.915x   0.937x   0.900x   0.982x
 mc w16:  0.917x   0.947x   0.911x   0.971x
 mc w32:  0.914x   0.938x   0.873x   0.961x
 mc w64:  0.918x   0.932x   0.882x   0.964x
```

**Q:** What is the benefit of using pointer arithmetic and reducing EXT instructions in data rearrangement for ARMv8.0 Neon horizontal 6‑tap filters?

**A:** By employing pointer arithmetic, the code can directly address the required data elements without intermediate calculations or temporary storage. This eliminates unnecessary load/store operations and allows the compiler to generate more efficient address expressions. Additionally, removing redundant EXT (extract) instructions shortens the instruction stream and reduces pipeline stalls caused by data dependencies. Together, these changes reduce the total number of executed instructions, leading to faster filter execution on Cortex cores. The benchmark shows a relative speedup ranging from 0.873x to 0.982x across A78, A76, A55 and a regular core, indicating up to roughly 13% improvement on the fastest core.#### Pointer arithmetic and fewer EXT instructions cut instruction count, speeding up Neon filter execution

**Verdict:** _________________________

---

## #4 — `4664f5aa66` (x264_src)

**Source commit subject:** aarch64: Improve scheduling in sad_x3/sad_x4

**Source commit body (first 1000 chars):**
```
Cortex A53    A72    A73
8 bpc:
Before:
sad_x3_4x4_neon:      580    303    204
sad_x3_4x8_neon:     1065    516    323
sad_x3_8x4_neon:      668    262    282
sad_x3_8x8_neon:     1238    454    471
sad_x3_8x16_neon:    2378    842    847
sad_x3_16x8_neon:    2136    738    776
sad_x3_16x16_neon:   4162   1378   1463
After:
sad_x3_4x4_neon:      477    298    206
sad_x3_4x8_neon:      842    515    327
sad_x3_8x4_neon:      603    260    279
sad_x3_8x8_neon:     1110    451    464
sad_x3_8x16_neon:    2125    841    843
sad_x3_16x8_neon:    2124    730    766
sad_x3_16x16_neon:   4145   1370   1434

10 bpc:
Before:
sad_x3_4x4_neon:      632    247    254
sad_x3_4x8_neon:     1162    419    443
sad_x3_8x4_neon:      890    358    416
sad_x3_8x8_neon:     1670    632    759
sad_x3_8x16_neon:    3230   1179   1458
sad_x3_16x8_neon:    3070   1209   1403
sad_x3_16x16_neon:   6030   2333   2699

After:
sad_x3_4x4_neon:      522    253    255
sad_x3_4x8_neon:      932    443    431
sad_x3_8
... [truncated]
```

**Q:** What scheduling technique can reduce latency on in-order ARM cores like the Cortex A53 when computing block-wise SAD operations?

**A:** The commit improves instruction scheduling for the sad_x3/sad_x4 functions, arranging loads and arithmetic so that data dependencies are minimized and the pipeline stays busy. By reordering operations, it reduces stalls caused by waiting for previous results on an in-order core, leading to a 20‑25 % speedup on the Cortex A53 for small block sizes. On out-of-order cores such as the Cortex A72 and A73, the hardware can already reorder instructions efficiently, so the same scheduling changes have little measurable effect. The benchmark shows worst‑case 580 cycles on A53 before and 477 after for a 4×4 block, while best‑case 204 cycles on A73 before and 206 after—illustrating a narrow range for out‑of‑order CPUs.
#### Improved scheduling reduces stalls on in-order ARM cores

**Verdict:** _________________________

---

## #5 — `37949a994e` (x264_src)

**Source commit subject:** Create Common NEON deblock-a Macros

**Source commit body (first 1000 chars):**
```
Place NEON deblock-a macros that are intended to be
used by SVE/SVE2 functions as well in a common file.
```

**Q:** Why would an ARM64 developer place NEON deblock-a macros in a common file for use by SVE/SVE2 functions?

**A:** Placing NEON deblock-a macros in a common file allows both NEON and SVE/SVE2 implementations to share the same deblocking logic, reducing code duplication and ensuring consistency across instruction sets. This approach simplifies maintenance because updates to the deblocking algorithm need only be made in one location, automatically propagating to all dependent code paths. It also facilitates easier testing and verification, as the same macro can be exercised under both NEON and SVE contexts. By centralizing these macros, developers avoid divergent implementations that could introduce subtle bugs or performance regressions.
#### Centralizing deblock macros simplifies maintenance and consistency across NEON and SVE/SVE2

**Verdict:** _________________________

---

## #6 — `c1c9931dc8` (x264_src)

**Source commit subject:** Improve pixel-a.S Performance by Using SVE/SVE2

**Source commit body (first 1000 chars):**
```
Imporve the performance of NEON functions of aarch64/pixel-a.S
by using the SVE/SVE2 instruction set. Below, the specific functions
are listed together with the improved performance results.

Command executed: ./checkasm8 --bench=ssd
Testbed: Alibaba g8y instance based on Yitian 710 CPU
Results:
ssd_4x4_c: 235
ssd_4x4_neon: 226
ssd_4x4_sve: 151
ssd_4x8_c: 409
ssd_4x8_neon: 363
ssd_4x8_sve: 201
ssd_4x16_c: 781
ssd_4x16_neon: 653
ssd_4x16_sve: 313
ssd_8x4_c: 402
ssd_8x4_neon: 192
ssd_8x4_sve: 192
ssd_8x8_c: 728
ssd_8x8_neon: 275
ssd_8x8_sve: 275

Command executed: ./checkasm10 --bench=ssd
Testbed: Alibaba g8y instance based on Yitian 710 CPU
Results:
ssd_4x4_c: 256
ssd_4x4_neon: 226
ssd_4x4_sve: 153
ssd_4x8_c: 460
ssd_4x8_neon: 369
ssd_4x8_sve: 215
ssd_4x16_c: 852
ssd_4x16_neon: 651
ssd_4x16_sve: 340

Command executed: ./checkasm8 --bench=ssd
Testbed: AWS Graviton3
Results:
ssd_4x4_c: 295
ssd_4x4_neon: 288
ssd_4x4_sve: 228
ssd_4x8_c: 454
ssd_4x8_neon: 431
ssd_4x8_sve: 294
ssd_4x16_c: 779
... [truncated]
```

**Q:** How can using the SVE/SVE2 instruction set improve performance of pixel processing functions on AArch64 compared to NEON?

**A:** Replacing NEON implementations with SVE/SVE2 allows the code to process wider vector registers and perform more operations per instruction, which reduces loop overhead and memory traffic. In the benchmark on an Alibaba g8y instance (Yitian 710 CPU), SVE versions of the ssd_4x4 function ran in 151 cycles versus 226 for NEON, a 33 % speed‑up. On AWS Graviton3 the same function improved from 288 cycles (NEON) to 228 cycles (SVE), a 21 % gain. Across other kernels such as sa8d_8x8 and hadamard_ac, SVE consistently outperformed NEON by 2–3×. The range of improvement on the Yitian 710 CPU spans from a 1.5× speed‑up for var_8x8 to a 3.6× gain for hadamard_ac_16x16.
#### SVE/SVE2 offers wider vectors and lower instruction counts, yielding significant speed‑ups over NEON.

**Verdict:** _________________________

---

## #7 — `2355eeb8f2` (dav1d_src)

**Source commit subject:** AArch64: Move constants of DotProd subpel filters to .rodata

**Source commit body (first 1000 chars):**
```
The constants used for the subpel filters were placed in the .text
section for simplicity and peak performance, but this does not work on
systems with execute only .text sections (e.g.: OpenBSD).

The performance cost of moving the constants to the .rodata section
is small and mostly within the measurable noise.
```

**Q:** Why might an ARM64 programmer move constant data from the .text section to .rodata, and what impact does this have on performance?

**A:** Placing constants in the .text section can give a slight speed advantage because they are fetched from the same cache line as code, but it prevents use on systems that enforce execute‑only .text sections, such as OpenBSD. Moving the constants to .rodata removes that restriction while keeping the performance penalty within measurable noise, so the code remains portable without a noticeable slowdown.
#### Moving constants to .rodata keeps portability while adding negligible performance cost

**Verdict:** _________________________

---

## #8 — `30c3dd8edd` (dav1d_src)

**Source commit subject:** arm32: looprestoration: Rewrite the SGR functions

**Source commit body (first 1000 chars):**
```
Switch to the same cache-friendly algorithm as was done for arm64
in c121b831e251d185b1f9829efb6ba5c68fdc9567.

This uses much less stack memory, and is much more cache friendly.
In this form, most of the individual asm functions only operate on
one single row of data at a time.

Some of the functions used to be unrolled to operate on two rows
at a time, while they now only operate on one at a time. In practice,
this is still a large performance win, as data is accessed in a
much more cache friendly manner.

This gives a 2-37% speedup, and reduces the peak amount of stack
used for these functions from 255 KB to 33 KB.

Before:              Cortex A7         A8        A53        A72        A73
sgr_3x3_8bpc_neon:    873990.7   748341.9   543410.2   383200.4   357502.9
sgr_3x3_10bpc_neon:   909728.0   732594.5   560123.6   392765.5   359377.7
sgr_5x5_8bpc_neon:    591597.9   527353.1   350347.4   263464.9   243098.8
sgr_5x5_10bpc_neon:   637958.2   529462.8   364613.3   280664.6   255164.
... [truncated]
```

**Q:** How can further performance gains be achieved after the initial cache-friendly rewrite of loop restoration functions?

**A:** Additional optimizations can be applied incrementally by fusing related operations. For instance, merging box3/5_row_v with calc_row_ab1/2 or combining finish_filter_row1/2 with sgr_weighted_row1 can reduce function call overhead. Creating a version of finish_filter_row1 that produces two rows, similar to the arm64 implementation, is another potential improvement. These fusions would further streamline data access patterns and lower instruction counts.
#### Incremental function fusion can yield additional performance benefits

**Verdict:** _________________________

---

## #9 — `30c3dd8edd` (dav1d_src)

**Source commit subject:** arm32: looprestoration: Rewrite the SGR functions

**Source commit body (first 1000 chars):**
```
Switch to the same cache-friendly algorithm as was done for arm64
in c121b831e251d185b1f9829efb6ba5c68fdc9567.

This uses much less stack memory, and is much more cache friendly.
In this form, most of the individual asm functions only operate on
one single row of data at a time.

Some of the functions used to be unrolled to operate on two rows
at a time, while they now only operate on one at a time. In practice,
this is still a large performance win, as data is accessed in a
much more cache friendly manner.

This gives a 2-37% speedup, and reduces the peak amount of stack
used for these functions from 255 KB to 33 KB.

Before:              Cortex A7         A8        A53        A72        A73
sgr_3x3_8bpc_neon:    873990.7   748341.9   543410.2   383200.4   357502.9
sgr_3x3_10bpc_neon:   909728.0   732594.5   560123.6   392765.5   359377.7
sgr_5x5_8bpc_neon:    591597.9   527353.1   350347.4   263464.9   243098.8
sgr_5x5_10bpc_neon:   637958.2   529462.8   364613.3   280664.6   255164.
... [truncated]
```

**Q:** What is the benefit of rewriting loop restoration functions to operate on a single row at a time instead of two rows?

**A:** Operating on one row at a time makes the algorithm more cache friendly and reduces stack usage. The new implementation uses only 33 KB of peak stack versus the previous 255 KB, and it achieves a speedup ranging from 2 % to 37 %. For example, on an A7 Cortex the sgr_3x3_8bpc_neon function improved from 873,990.7 to 836,138.7 cycles, while on an A73 it went from 357,502.9 to 348,209.9 cycles.
#### Single-row processing improves cache locality and reduces stack usage

**Verdict:** _________________________

---

## #10 — `79db162487` (dav1d_src)

**Source commit subject:** AArch64: New method for calculating sgr table

**Source commit body (first 1000 chars):**
```
For the 3x3 part, double the width of the vertical loop. This is done to
provide more latency in the new sgr calculation.

Initial (master):  Cortex A53        A55        A72        A73       A76   Apple M1
sgr_3x3_8bpc_neon:   387702.8   383154.2   295742.4   302100.1  185420.7   472.2
sgr_5x5_8bpc_neon:   261725.1   256919.8   194205.1   197585.6  128311.3   332.9
sgr_mix_8bpc_neon:   628085.0   593664.2   453551.8   450553.8  281956.0   711.2

Current:
sgr_3x3_8bpc_neon:   368331.4   363949.7   275499.0   272056.3  169614.4   432.7
sgr_5x5_8bpc_neon:   257866.7   255265.5   195962.5   199557.8  120481.3   319.2
sgr_mix_8bpc_neon:   598234.1   572896.4   418500.4   438910.7  258977.7   659.3

Include a minor improvement that gets rid of a dup instruction.
```

**Q:** Why would an ARM64 programmer double the width of a vertical loop when computing a 3x3 sgr table?

**A:** Doubling the width of the vertical loop increases the number of operations performed per iteration, which in turn adds pipeline latency to the new sgr calculation. This extra latency can help hide memory stalls or allow other independent instructions to execute, improving overall throughput for the 3x3 sgr routine. The technique is reflected in benchmark improvements across Cortex A53, A55, A72, A73, A76 and Apple M1 CPUs, where the 3x3 sgr throughput increased from 387 702.8 to 368 331.4 on the A53 and from 472.2 to 432.7 on the M1, showing a consistent performance gain.
#### Doubling loop width adds latency to improve throughput

**Verdict:** _________________________

---
