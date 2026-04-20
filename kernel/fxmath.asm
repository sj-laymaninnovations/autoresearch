; =============================================================================
; fxmath.asm  —  Q8.16 Fixed-Point Arithmetic Primitives
; ATTN11-x86: port of PDP-11 FXMATH.MAC
;
; Format: int32_t Q8.16
;   1.0   =  65536  = 0x00010000
;   -1.0  = -65536  = 0xFFFF0000
;   max   =  2147483647  (~32767.99998)
;   min   = -2147483648  (-32768.0)
;   resolution: 1/65536 ≈ 0.0000153
;
; Calling convention: Win64 (rcx, rdx, r8, r9 → rax)
;   Callee-preserved: rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15
;   Scratch:          rax, rcx, rdx, r8-r11, xmm0-xmm5
;
; All functions return q16_t (int32_t) in eax (zero-extended to rax).
; =============================================================================

BITS 64
DEFAULT REL

SECTION .text

; -----------------------------------------------------------------------------
; GLOBAL exports
; -----------------------------------------------------------------------------
GLOBAL fxmul
GLOBAL fxdiv
GLOBAL fxabs
GLOBAL fxclamp
GLOBAL fxone                        ; constant: 1.0 in Q8.16

; -----------------------------------------------------------------------------
; fxmul — Q8.16 multiply
;
; q16_t fxmul(q16_t a, q16_t b)
;   rcx = a (int32_t)
;   rdx = b (int32_t)
;   → eax = a * b in Q8.16, saturated to int32 range
;
; PDP-11 equivalent: MUL Rb, Ra  →  ASHC $-8, Ra
; x86 upgrade:       IMUL r64, r64  →  SAR r64, 16
;                    (wider: Q16.32 product → Q8.16 with 2× more precision)
; -----------------------------------------------------------------------------
fxmul:
    movsxd  rax, ecx            ; sign-extend a → 64-bit
    movsxd  r8,  edx            ; sign-extend b → 64-bit (don't clobber rdx)
    imul    rax, r8             ; 64-bit Q16.32 product
    sar     rax, 16             ; >> 16 → Q8.16 result in 64-bit container
    ; Saturate to int32 range [-2147483648, 2147483647]
    mov     r8,  2147483647
    cmp     rax, r8
    cmovg   rax, r8             ; if > INT32_MAX, clamp high
    mov     r8, -2147483648
    cmp     rax, r8
    cmovl   rax, r8             ; if < INT32_MIN, clamp low
    ; Result now fits in int32; eax is the low 32 bits
    ret

; -----------------------------------------------------------------------------
; fxdiv — Q8.16 division
;
; q16_t fxdiv(q16_t a, q16_t b)
;   rcx = a (dividend)
;   rdx = b (divisor)
;   → eax = a / b in Q8.16, saturated
;
; PDP-11 equivalent: Build 32-bit Q16 dividend then DIV
; x86: shift 64-bit dividend left 16, use IDIV for exact integer divide
; Division by zero → saturate to sign of a.
; -----------------------------------------------------------------------------
fxdiv:
    movsxd  r8,  edx            ; save b early — CQO will clobber rdx
    movsxd  rax, ecx            ; sign-extend a
    shl     rax, 16             ; a << 16 → Q16.32 numerator
    test    r8,  r8
    jz      .div_zero
    cqo                         ; sign-extend rax → rdx:rax for IDIV
    idiv    r8                  ; rax = quotient (Q8.16), rdx = remainder
    ; Saturate
    mov     r9,  2147483647
    cmp     rax, r9
    cmovg   rax, r9
    mov     r9, -2147483648
    cmp     rax, r9
    cmovl   rax, r9
    ret

.div_zero:
    ; Divide by zero: return INT32_MAX if a≥0, INT32_MIN if a<0
    test    ecx, ecx
    js      .neg_inf
    mov     eax, 2147483647
    ret
.neg_inf:
    mov     eax, -2147483648
    ret

; -----------------------------------------------------------------------------
; fxabs — Q8.16 absolute value
;
; q16_t fxabs(q16_t a)
;   rcx = a (int32_t)
;   → eax = |a|  (note: fxabs(INT32_MIN) = INT32_MIN — same as PDP-11)
;
; PDP-11 equivalent: FXABS macro (TST + NEG)
; x86: branchless using arithmetic sign mask
; -----------------------------------------------------------------------------
fxabs:
    movsxd  rax, ecx
    cqo                         ; rdx = 0 if a≥0, -1 if a<0
    xor     rax, rdx            ; if negative: bitwise NOT
    sub     rax, rdx            ; if negative: +1 → two's complement negation
    ret

; -----------------------------------------------------------------------------
; fxclamp — Clamp Q8.16 value to explicit [lo, hi] range
;
; q16_t fxclamp(q16_t val, q16_t lo, q16_t hi)
;   rcx = val
;   rdx = lo
;   r8  = hi
;   → eax = clamp(val, lo, hi)
; -----------------------------------------------------------------------------
fxclamp:
    mov     eax, ecx
    cmp     eax, edx
    cmovl   eax, edx            ; if val < lo, use lo
    cmp     eax, r8d
    cmovg   eax, r8d            ; if val > hi, use hi
    ret

; -----------------------------------------------------------------------------
; fxone — Return 1.0 in Q8.16 format (65536)
;
; q16_t fxone(void)
;   → eax = 65536
; -----------------------------------------------------------------------------
fxone:
    mov     eax, 65536
    ret
