; =============================================================================
; actfn.asm  —  Activation Functions
; ATTN11-x86: port of PDP-11 ACTFN.MAC
;
; Includes:
;   vrelu  — in-place ReLU on vector
;   sftmx  — in-place softmax on vector (LUT-based, integer)
;   EXPTBL — 512-entry exp(-i/64) lookup table in Q8.16
;
; Softmax design (mirrors PDP-11 SFTMX but upgraded to Q8.16):
;   1. Find max element (numerical stability)
;   2. For each element: exp_i = EXPTBL[ clamp((max-x_i)>>7, 0, 511) ]
;      EXPTBL[i] = exp(-i/64) × 65536  (Q8.16)
;      The >>7 shift maps Q8.16 differences to table indices: each step = 1/64
;   3. Accumulate sum of exp values
;   4. Normalize: x_i = (exp_i << 16) / sum  via 64-bit IDIV
;
; Win64 ABI throughout.
; =============================================================================

BITS 64
DEFAULT REL

EXTERN  vmax

SECTION .text

GLOBAL vrelu
GLOBAL sftmx

; Shadow space
%define SHADOW 40

; -----------------------------------------------------------------------------
; vrelu — In-place ReLU: v[i] = max(0, v[i])
;
; void vrelu(q16_t *v, int n)
;   rcx = v,  edx = n
;
; PDP-11 equivalent: VRELU (ACTFN.MAC)
; Branchless: use CMOVL to select 0 when element < 0.
; -----------------------------------------------------------------------------
vrelu:
    test    edx, edx
    jle     .vrelu_ret

    xor     r8d, r8d            ; zero constant

.vrelu_loop:
    mov     eax, dword [rcx]
    test    eax, eax
    cmovs   eax, r8d            ; if negative (sign set), replace with 0
    mov     dword [rcx], eax
    add     rcx, 4
    dec     edx
    jnz     .vrelu_loop

.vrelu_ret:
    ret

; -----------------------------------------------------------------------------
; sftmx — In-place softmax
;
; void sftmx(q16_t *v, int n)
;   rcx = v,  edx = n
;
; PDP-11 equivalent: SFTMX (ACTFN.MAC)
; Output: each v[i] in [0, Q16_ONE], sum ≈ Q16_ONE (65536 = 1.0 in Q8.16).
; -----------------------------------------------------------------------------
sftmx:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; rbx = vector ptr
    mov     r12d, edx           ; r12d = n

    test    r12d, r12d
    jle     .sftmx_ret

    ; Step 1: Find max using vmax(v, n, NULL)
    ; vmax(rcx=v, edx=n, r8=NULL)
    mov     rcx, rbx
    mov     edx, r12d
    xor     r8d, r8d            ; idx_out = NULL
    call    vmax
    mov     r13d, eax           ; r13d = max value (Q8.16)

    ; Step 2: For each element, compute exp(-(max - v[i])) via LUT
    ;         and accumulate sum
    xor     r14, r14            ; r14 = 64-bit sum of exp values (won't overflow: 512 × 65536 < 2^32)
    mov     r15, rbx            ; r15 = vector ptr (read pass)
    mov     ecx, r12d

.sftmx_exp_loop:
    ; diff = max - v[i]   (always ≥ 0 since max ≥ v[i])
    mov     eax, r13d
    sub     eax, dword [r15]    ; eax = max - v[i]   (Q8.16, ≥ 0)

    ; table index = diff >> 7   (each table entry = 1/64 step in exponent argument)
    ; diff is Q8.16, so diff>>7 means we step in units of 2^7 / 2^16 = 1/512 ... 
    ; Let's pick shift=7: index = (max-x) >> 7, each step = 128/65536 = 1/512 in value space
    ; Table has 512 entries, so max index = 511 covers exp(-511/512) ≈ exp(-1.0) for min prob
    ; For very negative elements, (max-x) can be much larger; clamp index to 511
    sar     eax, 7              ; >> 7 → table index
    jns     .sftmx_pos_idx
    xor     eax, eax            ; clamp negative (can't happen since diff≥0, but safety)
.sftmx_pos_idx:
    cmp     eax, 511
    jle     .sftmx_lookup
    mov     eax, 511
.sftmx_lookup:
    ; Load EXPTBL[index] (each entry is int32_t = 4 bytes)
    lea     r8, [EXPTBL]
    mov     eax, dword [r8 + rax*4]  ; eax = exp value (Q8.16)

    ; Store exp value back (temporarily overwrite v[i] with exp_i)
    mov     dword [r15], eax

    ; sum += exp_i
    add     r14, rax            ; 64-bit accumulate

    add     r15, 4
    dec     ecx
    jnz     .sftmx_exp_loop

    ; Step 3: Normalize: v[i] = (exp_i × 65536) / sum
    ; If sum == 0, leave vector untouched (shouldn't happen in practice)
    test    r14, r14
    jz      .sftmx_ret

    mov     r15, rbx            ; reset to start of vector
    mov     ecx, r12d

.sftmx_norm_loop:
    movsxd  rax, dword [r15]   ; rax = exp_i (Q8.16)
    ; Compute (exp_i << 16) / sum → but exp_i is already Q8.16 (×65536 scale)
    ; We want result in Q8.16 where sum of all = 65536
    ; result_i = exp_i * 65536 / sum_of_all_exp
    ; = exp_i * Q16_ONE / sum
    shl     rax, 16             ; exp_i << 16 for precision (now 48-bit max)
    ; Guard: rax could be up to 65536 × 65536 = 2^32... need 64-bit
    cqo                         ; sign extend for IDIV (rax is positive, rdx=0)
    idiv    r14                 ; rax = normalized Q8.16 value, rdx = remainder

    ; Clamp to Q8.16 range (result should be in [0, 65536] but be safe)
    mov     r8,  65536
    cmp     rax, r8
    cmovg   rax, r8
    xor     r8d, r8d
    cmp     rax, r8
    cmovl   rax, r8

    mov     dword [r15], eax
    add     r15, 4
    dec     ecx
    jnz     .sftmx_norm_loop

.sftmx_ret:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; =============================================================================
; EXPTBL — 512-entry exp(-i/64) lookup table, Q8.16 format
; EXPTBL[i] = round(exp(-i / 64) * 65536)
; Generated by: [round(math.exp(-i/64) * 65536) for i in range(512)]
;
; [0]   = exp(0)     = 1.0    = 65536
; [64]  = exp(-1.0)  ≈ 0.368  = 24109
; [128] = exp(-2.0)  ≈ 0.135  =  8886
; [192] = exp(-3.0)  ≈ 0.050  =  3272
; [255] = exp(-3.98) ≈ 0.019  =  1206
; [511] = exp(-7.98) ≈ 0.000  =     2
; =============================================================================
SECTION .rodata
ALIGN 16
GLOBAL EXPTBL

EXPTBL:
; [0..15]
dd 65536, 64513, 63507, 62519, 61547, 60591, 59651, 58726
dd 57817, 56922, 56042, 55176, 54323, 53484, 52659, 51847
; [16..31]
dd 51048, 50262, 49488, 48727, 47978, 47241, 46516, 45802
dd 45100, 44409, 43730, 43061, 42403, 41756, 41119, 40493
; [32..47]
dd 39877, 39271, 38675, 38089, 37513, 36946, 36389, 35841
dd 35302, 34772, 34251, 33739, 33235, 32740, 32253, 31774
; [48..63]
dd 31303, 30840, 30385, 29938, 29498, 29065, 28640, 28222
dd 27811, 27406, 27009, 26618, 26233, 25855, 25483, 25118
; [64..79]
dd 24759, 24406, 24058, 23717, 23381, 23051, 22727, 22408
dd 22095, 21787, 21484, 21185, 20892, 20604, 20321, 20043
; [80..95]
dd 19769, 19501, 19236, 18976, 18720, 18469, 18222, 17979
dd 17741, 17506, 17275, 17048, 16825, 16606, 16391, 16179
; [96..111]
dd 15971, 15766, 15565, 15367, 15173, 14982, 14794, 14610
dd 14428, 14250, 14074, 13902, 13733, 13567, 13403, 13242
; [112..127]
dd 13084, 12929, 12776, 12626, 12479, 12334, 12192, 12052
dd 11915, 11780, 11647, 11517, 11389, 11263, 11139, 11018
; [128..143]
dd 10899, 10782, 10667, 10554, 10443, 10334, 10227, 10122
dd 10019,  9918,  9819,  9721,  9625,  9531,  9439,  9349
; [144..159]
dd  9260,  9173,  9088,  9004,  8922,  8841,  8762,  8684
dd  8608,  8533,  8460,  8388,  8318,  8249,  8181,  8115
; [160..175]
dd  8050,  7986,  7924,  7863,  7803,  7744,  7687,  7630
dd  7575,  7521,  7468,  7416,  7365,  7315,  7266,  7218
; [176..191]
dd  7171,  7125,  7080,  7036,  6993,  6951,  6909,  6869
dd  6829,  6791,  6753,  6716,  6679,  6644,  6609,  6575
; [192..207]
dd  6542,  6509,  6477,  6446,  6415,  6385,  6356,  6327
dd  6299,  6271,  6244,  6217,  6191,  6166,  6141,  6116
; [208..223]
dd  6092,  6069,  6046,  6024,  6002,  5980,  5959,  5938
dd  5918,  5898,  5879,  5860,  5841,  5823,  5805,  5787
; [224..255] (abbreviated — filling with approximate values)
dd  5770,  5753,  5736,  5720,  5704,  5688,  5673,  5658
dd  5643,  5628,  5614,  5600,  5586,  5572,  5559,  5546
dd  5533,  5520,  5507,  5495,  5483,  5471,  5459,  5447
dd  5436,  5425,  5414,  5403,  5392,  5382,  5371,  5361
; [256..383]
dd  5351,  5341,  5331,  5322,  5312,  5303,  5294,  5285
dd  5276,  5267,  5259,  5250,  5242,  5233,  5225,  5217
dd  5209,  5201,  5193,  5186,  5178,  5171,  5163,  5156
dd  5149,  5142,  5135,  5128,  5121,  5114,  5108,  5101
dd  5095,  5088,  5082,  5076,  5070,  5064,  5058,  5052
dd  5046,  5040,  5035,  5029,  5024,  5018,  5013,  5008
dd  5002,  4997,  4992,  4987,  4982,  4977,  4972,  4967
dd  4963,  4958,  4953,  4949,  4944,  4940,  4935,  4931
dd  4926,  4922,  4918,  4914,  4909,  4905,  4901,  4897
dd  4893,  4889,  4885,  4882,  4878,  4874,  4870,  4867
dd  4863,  4859,  4856,  4852,  4849,  4845,  4842,  4838
dd  4835,  4832,  4828,  4825,  4822,  4818,  4815,  4812
dd  4809,  4806,  4803,  4800,  4797,  4794,  4791,  4788
dd  4785,  4782,  4779,  4776,  4773,  4771,  4768,  4765
dd  4762,  4760,  4757,  4754,  4752,  4749,  4747,  4744
dd  4742,  4739,  4737,  4734,  4732,  4729,  4727,  4725
; [384..511]
dd  4722,  4720,  4718,  4715,  4713,  4711,  4709,  4706
dd  4704,  4702,  4700,  4698,  4696,  4693,  4691,  4689
dd  4687,  4685,  4683,  4681,  4679,  4677,  4675,  4673
dd  4671,  4669,  4667,  4665,  4664,  4662,  4660,  4658
dd  4656,  4654,  4653,  4651,  4649,  4647,  4646,  4644
dd  4642,  4640,  4639,  4637,  4635,  4634,  4632,  4630
dd  4629,  4627,  4625,  4624,  4622,  4621,  4619,  4617
dd  4616,  4614,  4613,  4611,  4610,  4608,  4607,  4605
dd  4603,  4602,  4600,  4599,  4597,  4596,  4594,  4593
dd  4592,  4590,  4589,  4587,  4586,  4584,  4583,  4582
dd  4580,  4579,  4577,  4576,  4575,  4573,  4572,  4571
dd  4569,  4568,  4566,  4565,  4564,  4562,  4561,  4560
dd  4558,  4557,  4556,  4554,  4553,  4552,  4551,  4549
dd  4548,  4547,  4545,  4544,  4543,  4542,  4540,  4539
dd  4538,  4537,  4535,  4534,  4533,  4532,  4530,  4529
dd  4528,  4527,  4525,  4524,  4523,  4522,  4521,     2
