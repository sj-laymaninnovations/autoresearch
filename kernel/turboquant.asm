; =============================================================================
; turboquant.asm  —  Integer Weight Quantization
; ATTN11-x86: NEW component (no PDP-11 equivalent)
;
; Bridges between Python float32 training weights and the integer kernel.
;
; Quantization formats:
;   float32  → int16_t Q8.8  (tq_f32_to_q8)  : storage format for weights/KV cache
;   int16_t  → int32_t Q8.16 (tq_q8_to_q16)  : expand for compute
;   int32_t  → int16_t Q8.8  (tq_q16_to_q8)  : compress back for storage
;   int16_t  → int32_t Q8.16 (tq_q8_to_q16_exact) : lossless expand (<< 8)
;
; Float conversions use SSE2 for throughput. All int paths are scalar.
; Win64 ABI throughout.
; =============================================================================

BITS 64
DEFAULT REL

EXTERN  malloc
EXTERN  fabsf

SECTION .data
ALIGN 16
; SSE2 constants
.q8_scale_255:  dd  256.0, 256.0, 256.0, 256.0    ; x * 256 for Q8.8
.q8_max_f:      dd  32767.0, 32767.0, 32767.0, 32767.0
.q8_min_f:      dd  -32768.0, -32768.0, -32768.0, -32768.0

SECTION .text

GLOBAL  tq_f32_to_q8
GLOBAL  tq_q8_to_q16
GLOBAL  tq_q16_to_q8
GLOBAL  tq_q8_to_q16_exact

; -----------------------------------------------------------------------------
; tq_f32_to_q8 — float32[] → int16_t Q8.8
;
; float tq_f32_to_q8(const float *src, q8_t *dst, int n, float scale)
;   rcx = src,  rdx = dst,  r8d = n,  xmm3 = scale (float)
;   → xmm0 = scale_used (float)
;
; If scale == 0.0: auto-compute as 127.0 / max(|src|).
; Uses SSE2 CVTPS2DQ for bulk conversion.
;
; Q8.8 encoding: float_value * (256 * scale) → round → clamp → int16_t
; -----------------------------------------------------------------------------
tq_f32_to_q8:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 40

    mov     rbx, rcx            ; rbx = src
    mov     r12, rdx            ; r12 = dst
    mov     r13d, r8d           ; r13d = n
    ; xmm3 = scale (passed in XMM3 per Win64 float convention)

    ; --- Compute auto-scale if scale == 0.0 ---
    xorps   xmm0, xmm0
    ucomiss xmm3, xmm0
    jnz     .tq_have_scale      ; scale != 0 → use it

    ; Auto-scale: max |src[i]|
    test    r13d, r13d
    jle     .tq_zero_scale

    movss   xmm2, dword [rbx]   ; xmm2 = running max |v|
    andps   xmm2, [.tq_abs_mask]; abs
    mov     r14d, 1
.tq_max_loop:
    cmp     r14d, r13d
    jge     .tq_max_done
    movss   xmm0, dword [rbx + r14*4]
    andps   xmm0, [.tq_abs_mask]
    maxss   xmm2, xmm0
    inc     r14d
    jmp     .tq_max_loop
.tq_max_done:
    ; scale = 127.0 / max_abs  (use 127 not 128 for headroom)
    movss   xmm3, dword [.tq_127f]
    divss   xmm3, xmm2
    jmp     .tq_have_scale

.tq_zero_scale:
    movss   xmm3, dword [.tq_onef]

.tq_have_scale:
    ; Combine: effective multiplier = scale * 256.0
    movss   xmm4, dword [.tq_256f]
    mulss   xmm4, xmm3          ; xmm4 = scale * 256

    ; Convert each float: dst[i] = clamp(round(src[i] * scale * 256), -32768, 32767)
    test    r13d, r13d
    jle     .tq_f2q_done

    mov     r14d, r13d          ; r14d = counter
    shufps  xmm4, xmm4, 0      ; broadcast xmm4[0] to all lanes

.tq_f2q_loop:
    cmp     r14d, 4
    jl      .tq_f2q_scalar

    ; Process 4 floats at once with SSE2
    movups  xmm0, [rbx]        ; load 4 floats
    mulps   xmm0, xmm4          ; * (scale * 256)
    cvtps2dq xmm0, xmm0         ; round to int32 (SSE2)
    ; Clamp to int16 range via PACKSSDW
    pxor    xmm1, xmm1
    packssdw xmm0, xmm1         ; saturate int32→int16 (lower 4)
    movq    qword [r12], xmm0   ; write 4 × int16_t = 8 bytes
    add     rbx, 16
    add     r12, 8
    sub     r14d, 4
    jmp     .tq_f2q_loop

.tq_f2q_scalar:
    test    r14d, r14d
    jle     .tq_f2q_done
    movss   xmm0, dword [rbx]
    mulss   xmm0, xmm4
    cvtss2si eax, xmm0          ; round float → int32
    ; Clamp to int16
    cmp     eax, 32767
    jle     .tq_clamp_lo
    mov     eax, 32767
    jmp     .tq_clamp_done
.tq_clamp_lo:
    cmp     eax, -32768
    jge     .tq_clamp_done
    mov     eax, -32768
.tq_clamp_done:
    mov     word [r12], ax
    add     rbx, 4
    add     r12, 2
    dec     r14d
    jmp     .tq_f2q_scalar

.tq_f2q_done:
    movss   xmm0, xmm3          ; return scale_used

    add     rsp, 40
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; tq_q8_to_q16 — int16_t Q8.8 → q16_t Q8.16 (with external scale factor)
;
; void tq_q8_to_q16(const q8_t *src, q16_t *dst, int n, float scale)
;   rcx = src,  rdx = dst,  r8d = n,  xmm3 = scale (float)
;
; dst[i] = round(src[i] * (65536 / (scale * 256))) = src[i] * (256 / scale)
; In pure integer: dst[i] = (int32_t)src[i] << 8  if scale == 1.0
; For scale ≠ 1.0, applies the float correction.
; -----------------------------------------------------------------------------
tq_q8_to_q16:
    test    r8d, r8d
    jle     .q8q16_ret

    push    rbx
    mov     rbx, rcx            ; src
    ; rdx = dst, r8d = n, xmm3 = scale

    ; Check if scale == 1.0: use fast integer path (just << 8)
    movss   xmm0, dword [.tq_onef]
    ucomiss xmm3, xmm0
    jz      .q8q16_fast

.q8q16_float:
    ; multiplier = 256.0 / scale  (dequantize: reverse the encoding)
    movss   xmm0, dword [.tq_256f]
    divss   xmm0, xmm3          ; xmm0 = 256 / scale

.q8q16_floop:
    test    r8d, r8d
    jle     .q8q16_fdone
    movsx   eax, word [rbx]     ; sign-extend int16 → int32
    cvtsi2ss xmm1, eax          ; int32 → float
    mulss   xmm1, xmm0          ; * (256 / scale)
    cvtss2si eax, xmm1          ; → int32 Q8.16
    mov     dword [rdx], eax
    add     rbx, 2
    add     rdx, 4
    dec     r8d
    jmp     .q8q16_floop
.q8q16_fdone:
    pop     rbx
    ret

.q8q16_fast:
    ; Fast path: scale==1.0  →  dst[i] = src[i] << 8
.q8q16_fst_loop:
    test    r8d, r8d
    jle     .q8q16_fst_done
    movsx   eax, word [rbx]     ; sign-extend int16_t
    shl     eax, 8              ; Q8.8 → Q8.16
    mov     dword [rdx], eax
    add     rbx, 2
    add     rdx, 4
    dec     r8d
    jmp     .q8q16_fst_loop
.q8q16_fst_done:
    pop     rbx
.q8q16_ret:
    ret

; -----------------------------------------------------------------------------
; tq_q16_to_q8 — q16_t Q8.16 → int16_t Q8.8 (lossily, >> 8 with saturation)
;
; void tq_q16_to_q8(const q16_t *src, q8_t *dst, int n)
;   rcx = src,  rdx = dst,  r8d = n
; -----------------------------------------------------------------------------
tq_q16_to_q8:
    test    r8d, r8d
    jle     .q16q8_ret

.q16q8_loop:
    movsxd  rax, dword [rcx]
    sar     rax, 8              ; Q8.16 → Q8.8
    ; Saturate to int16
    cmp     rax, 32767
    jle     .q16q8_lo
    mov     eax, 32767
    jmp     .q16q8_emit
.q16q8_lo:
    cmp     rax, -32768
    jge     .q16q8_emit
    mov     eax, -32768
.q16q8_emit:
    mov     word [rdx], ax
    add     rcx, 4
    add     rdx, 2
    dec     r8d
    jnz     .q16q8_loop

.q16q8_ret:
    ret

; -----------------------------------------------------------------------------
; tq_q8_to_q16_exact — Lossless expand: int16_t Q8.8 → q16_t Q8.16 (<< 8)
;
; void tq_q8_to_q16_exact(const q8_t *src, q16_t *dst, int n)
;   rcx = src,  rdx = dst,  r8d = n
; -----------------------------------------------------------------------------
tq_q8_to_q16_exact:
    test    r8d, r8d
    jle     .q8exact_ret

.q8exact_loop:
    movsx   eax, word [rcx]     ; sign-extend int16 → int32
    shl     eax, 8              ; << 8: Q8.8 → Q8.16
    mov     dword [rdx], eax
    add     rcx, 2
    add     rdx, 4
    dec     r8d
    jnz     .q8exact_loop

.q8exact_ret:
    ret

; --- Float constants in .rodata ---
SECTION .rodata
ALIGN 16
.tq_abs_mask:   dd  0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
.tq_127f:       dd  127.0
.tq_256f:       dd  256.0
.tq_onef:       dd  1.0
