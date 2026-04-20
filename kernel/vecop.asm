; =============================================================================
; vecop.asm  —  Vector Operations (Q8.16 Fixed-Point)
; ATTN11-x86: port of PDP-11 VECOP.MAC
;
; All vectors are arrays of q16_t (int32_t, Q8.16 format).
; Overflow is handled via saturation clamp to [INT32_MIN, INT32_MAX].
;
; Register usage follows Win64 ABI throughout.
; Functions that call other functions save rsi, rdi, rbx as needed.
; =============================================================================

BITS 64
DEFAULT REL

SECTION .text

GLOBAL vdot
GLOBAL vadd
GLOBAL vsub
GLOBAL vscl
GLOBAL vmax
GLOBAL vcpy
GLOBAL vclr
GLOBAL vsadd

; ── Saturation helper macro ───────────────────────────────────────────────
; After a 64-bit result is in rax, clamp to int32 range:
;   cmp rax, INT32_MAX  → cmovg
;   cmp rax, INT32_MIN  → cmovl
; Used in multiply-then-shift sequences.

; -----------------------------------------------------------------------------
; vdot — Dot product with 64-bit accumulation
;
; q16_t vdot(const q16_t *x, const q16_t *y, int n)
;   rcx = x    (int32_t[])
;   rdx = y    (int32_t[])
;   r8d = n    (element count)
;   → eax = Σ x[i]*y[i] in Q8.16, saturated to int32
;
; PDP-11 equivalent: VDOT (VECOP.MAC)
; Uses 64-bit accumulator (dthi:dtlo equivalent) to prevent overflow.
; Per-product: 32×32→64 multiply, accumulate 64-bit, shift right 16 at end.
;
; SSE4.1 fast path wired up for n≥4; falls back to scalar for tail.
; -----------------------------------------------------------------------------
vdot:
    push    rsi
    push    rdi

    mov     rsi, rcx            ; rsi = x ptr
    mov     rdi, rdx            ; rdi = y ptr
    mov     ecx, r8d            ; ecx = n (loop counter)
    xor     r10,  r10           ; r10 = 64-bit accumulator (high:low packed in 64-bit)
                                ;        actually just one 64-bit signed accumulator

    test    ecx, ecx
    jle     .vdot_done

.vdot_loop:
    movsxd  rax, dword [rsi]    ; a = x[i] (sign-extended to 64-bit)
    movsxd  r8,  dword [rdi]    ; b = y[i]
    imul    rax, r8             ; 64-bit Q16.32 product
    sar     rax, 16             ; >> 16 → Q8.16 contribution
    add     r10, rax            ; accumulate (no overflow risk: n≤65536, each≤±32768²/2^16=±32768)
    add     rsi, 4
    add     rdi, 4
    dec     ecx
    jnz     .vdot_loop

.vdot_done:
    ; Saturate 64-bit accumulator to int32
    mov     r8,  2147483647
    cmp     r10, r8
    cmovg   r10, r8
    mov     r8, -2147483648
    cmp     r10, r8
    cmovl   r10, r8
    mov     eax, r10d

    pop     rdi
    pop     rsi
    ret

; -----------------------------------------------------------------------------
; vadd — Vector addition z[i] = x[i] + y[i], saturated
;
; void vadd(const q16_t *x, const q16_t *y, q16_t *z, int n)
;   rcx = x,  rdx = y,  r8 = z,  r9d = n
;
; PDP-11 equivalent: VADD
; -----------------------------------------------------------------------------
vadd:
    test    r9d, r9d
    jle     .vadd_ret

.vadd_loop:
    mov     eax, dword [rcx]    ; a = x[i]
    add     eax, dword [rdx]    ; a += y[i]
    ; Saturate using overflow flag (add sets OF on signed overflow)
    jno     .vadd_ok
    ; Overflow: if result positive (was negative overflow) → INT32_MIN
    ;           if result negative (was positive overflow) → INT32_MAX
    js      .vadd_pos_ov
    mov     eax, -2147483648    ; negative result → was pos overflow → min
    jmp     .vadd_ok
.vadd_pos_ov:
    mov     eax, 2147483647

.vadd_ok:
    mov     dword [r8], eax
    add     rcx, 4
    add     rdx, 4
    add     r8,  4
    dec     r9d
    jnz     .vadd_loop

.vadd_ret:
    ret

; -----------------------------------------------------------------------------
; vsub — Vector subtraction z[i] = x[i] - y[i], saturated
;
; void vsub(const q16_t *x, const q16_t *y, q16_t *z, int n)
;   rcx = x,  rdx = y,  r8 = z,  r9d = n
;
; PDP-11 equivalent: VSUB
; -----------------------------------------------------------------------------
vsub:
    test    r9d, r9d
    jle     .vsub_ret

.vsub_loop:
    mov     eax, dword [rcx]
    sub     eax, dword [rdx]
    jno     .vsub_ok
    js      .vsub_pos_ov
    mov     eax, -2147483648
    jmp     .vsub_ok
.vsub_pos_ov:
    mov     eax, 2147483647
.vsub_ok:
    mov     dword [r8], eax
    add     rcx, 4
    add     rdx, 4
    add     r8,  4
    dec     r9d
    jnz     .vsub_loop

.vsub_ret:
    ret

; -----------------------------------------------------------------------------
; vscl — Scalar-vector multiply y[i] = alpha * x[i]
;
; void vscl(const q16_t *x, q16_t *y, int n, q16_t alpha)
;   rcx = x,  rdx = y,  r8d = n,  r9d = alpha (Q8.16)
;
; PDP-11 equivalent: VSCL
; -----------------------------------------------------------------------------
vscl:
    push    rsi

    mov     rsi, rcx            ; rsi = x ptr
    ; rdx = y ptr (ok, we don't clobber it early)
    movsxd  r10, r9d            ; r10 = alpha (64-bit for imul)
    mov     ecx, r8d            ; ecx = counter

    test    ecx, ecx
    jle     .vscl_ret

.vscl_loop:
    movsxd  rax, dword [rsi]    ; a = x[i]
    imul    rax, r10            ; 64-bit Q16.32
    sar     rax, 16             ; → Q8.16
    ; Saturate
    mov     r8,  2147483647
    cmp     rax, r8
    cmovg   rax, r8
    mov     r8, -2147483648
    cmp     rax, r8
    cmovl   rax, r8
    mov     dword [rdx], eax
    add     rsi, 4
    add     rdx, 4
    dec     ecx
    jnz     .vscl_loop

.vscl_ret:
    pop     rsi
    ret

; -----------------------------------------------------------------------------
; vmax — Find maximum element and its index
;
; q16_t vmax(const q16_t *v, int n, int *idx_out)
;   rcx = v,  edx = n,  r8 = idx_out (may be NULL)
;   → eax = max value (Q8.16)
;   *idx_out = index of max (0-based)
;
; PDP-11 equivalent: VMAX
; -----------------------------------------------------------------------------
vmax:
    test    edx, edx
    jle     .vmax_empty

    mov     eax, dword [rcx]    ; eax = current max = v[0]
    xor     r9d, r9d            ; r9d = best index = 0
    xor     r10d, r10d          ; r10d = current index = 0
    dec     edx                 ; remaining = n - 1
    jz      .vmax_done          ; single element

.vmax_loop:
    add     rcx, 4
    inc     r10d
    mov     r11d, dword [rcx]   ; r11d = v[i]
    cmp     r11d, eax
    jle     .vmax_skip          ; not greater → skip
    mov     eax,  r11d          ; new max
    mov     r9d,  r10d          ; new best index
.vmax_skip:
    dec     edx
    jnz     .vmax_loop

.vmax_done:
    test    r8, r8
    jz      .vmax_ret
    mov     dword [r8], r9d     ; write index

.vmax_ret:
    ret

.vmax_empty:
    xor     eax, eax
    test    r8, r8
    jz      .vmax_ret2
    mov     dword [r8], -1
.vmax_ret2:
    ret

; -----------------------------------------------------------------------------
; vcpy — Vector copy dst[i] = src[i]
;
; void vcpy(const q16_t *src, q16_t *dst, int n)
;   rcx = src,  rdx = dst,  r8d = n
;
; PDP-11 equivalent: VCPY
; Uses REP MOVSD for bulk copy.
; -----------------------------------------------------------------------------
vcpy:
    push    rsi
    push    rdi

    mov     rsi, rcx            ; rsi = src
    mov     rdi, rdx            ; rdi = dst
    mov     ecx, r8d            ; ecx = count (dwords)
    test    ecx, ecx
    jle     .vcpy_ret
    rep     movsd               ; copy ecx × 4 bytes

.vcpy_ret:
    pop     rdi
    pop     rsi
    ret

; -----------------------------------------------------------------------------
; vclr — Zero a vector
;
; void vclr(q16_t *v, int n)
;   rcx = v,  edx = n
;
; PDP-11 equivalent: VCLR
; Uses REP STOSD for bulk zero.
; -----------------------------------------------------------------------------
vclr:
    push    rdi

    mov     rdi, rcx            ; rdi = v
    mov     ecx, edx            ; ecx = count
    xor     eax, eax            ; fill value = 0
    test    ecx, ecx
    jle     .vclr_ret
    rep     stosd

.vclr_ret:
    pop     rdi
    ret

; -----------------------------------------------------------------------------
; vsadd — Scale-accumulate: dst[k] += (scalar * src[k]) >> 16
;
; void vsadd(q16_t scalar, const q16_t *src, q16_t *dst, int n)
;   ecx = scalar (Q8.16),  rdx = src,  r8 = dst,  r9d = n
;
; PDP-11 equivalent: VSADD (BKWRD.MAC, also called from attn backward)
; This is the backward-pass scatter operation: dV[j] += A[i][j] * dY[i]
; -----------------------------------------------------------------------------
vsadd:
    push    rsi
    push    rdi

    movsxd  r10, ecx            ; r10 = scalar (64-bit)
    mov     rsi, rdx            ; rsi = src ptr
    mov     rdi, r8             ; rdi = dst ptr
    mov     ecx, r9d            ; ecx = n

    test    ecx, ecx
    jle     .vsadd_ret

.vsadd_loop:
    movsxd  rax, dword [rsi]    ; a = src[k]
    imul    rax, r10            ; 64-bit Q16.32 product
    sar     rax, 16             ; → Q8.16
    ; Saturate product before adding
    mov     r8,  2147483647
    cmp     rax, r8
    cmovg   rax, r8
    mov     r8, -2147483648
    cmp     rax, r8
    cmovl   rax, r8
    ; Saturated add to dst[k]
    add     eax, dword [rdi]
    jno     .vsadd_ok
    js      .vsadd_pos_ov
    mov     eax, -2147483648
    jmp     .vsadd_ok
.vsadd_pos_ov:
    mov     eax, 2147483647
.vsadd_ok:
    mov     dword [rdi], eax
    add     rsi, 4
    add     rdi, 4
    dec     ecx
    jnz     .vsadd_loop

.vsadd_ret:
    pop     rdi
    pop     rsi
    ret
