; =============================================================================
; matop.asm  —  Matrix-Vector Operations (Q8.16 Fixed-Point)
; ATTN11-x86: port of PDP-11 MATOP.MAC
;
; All matrices are row-major, elements are q16_t (int32_t, Q8.16).
; Matrix [rows × cols]: element [i][j] at offset (i*cols + j) * 4 bytes.
;
; Function list:
;   mvmul  — mat × vec  → vout          (M rows × N cols)
;   mvadd  — mat × vec  → vout (accum)
;   vtmul  — mat^T × vec → vout
;   outer  — mat += vx ⊗ vy             (outer product accumulate)
;
; Win64 ABI throughout.
; =============================================================================

BITS 64
DEFAULT REL

EXTERN  vdot
EXTERN  vclr
EXTERN  vsadd

SECTION .text

GLOBAL mvmul
GLOBAL mvadd
GLOBAL vtmul
GLOBAL outer

; Subroutine shadow space macro
%define SHADOW 40

; -----------------------------------------------------------------------------
; mvmul — Matrix-vector multiply: vout[i] = Σ_j mat[i][j] * vin[j]
;
; void mvmul(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols)
;   rcx = mat,  rdx = vin,  r8 = vout,  r9d = rows
;   [rsp + SHADOW + 8] = cols (5th arg, stack)
;
; PDP-11 equivalent: MVMUL (MATOP.MAC)
; For each row i: vout[i] = vdot(mat + i*cols, vin, cols)
; -----------------------------------------------------------------------------
mvmul:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; rbx = mat (advances by row)
    mov     r12, rdx            ; r12 = vin
    mov     r13, r8             ; r13 = vout ptr (advances)
    mov     r14d, r9d           ; r14d = rows (loop counter)
    mov     r15d, dword [rsp + SHADOW + 5*8 + 8]  ; r15d = cols (5th arg)
    ; 5th arg offset: SHADOW(40) + 5 pushes × 8(40) + ret_addr(8) = 88 bytes... 
    ; Let me recalculate: after push×5(40) + sub rsp,40(40) = 80 bytes shift
    ; 5th arg was at [original_rsp + 40] (Win64: 4 reg-args home + 5th on stack)
    ; Now at [rsp + 80 + 40] = [rsp + 120]... For Win64:
    ;   original rsp had ret addr at [0], home space [8..39], 5th arg at [40]
    ;   after sub rsp,80 total: 5th arg at [rsp + 80 + 40] = [rsp + 120]
    ; But push ×5 = 40, sub rsp,40 = 40, total = 80. 5th arg at [rsp + 80 + 40] = [rsp + 120].
    ; Wait: original_rsp[40] = 5th arg.  We moved rsp down by 80. So [rsp+80+40]=[rsp+120].
    ; Actually: [original_rsp + 40] = [rsp + 80 + 40 - 80... no.
    ; current_rsp = original_rsp - 80.
    ; 5th arg at original_rsp + 40 = current_rsp + 80 + 40... no.
    ; original_rsp + 40 = (current_rsp + 80) + 40 = current_rsp + 120. YES.

    mov     r15d, dword [rsp + 120]   ; 5th arg = cols

    test    r14d, r14d
    jle     .mvmul_ret

    mov     r10, r15            ; r10 = cols (for ptr advance calc)
    shl     r10, 2              ; bytes per row = cols * 4

.mvmul_row:
    ; vdot(mat_row_ptr, vin, cols)
    mov     rcx, rbx            ; arg1: mat row ptr
    mov     rdx, r12            ; arg2: vin
    mov     r8d, r15d           ; arg3: cols
    call    vdot                ; → eax = dot product Q8.16

    mov     dword [r13], eax    ; vout[i] = result
    add     rbx, r10            ; advance mat by one row
    add     r13, 4              ; advance vout
    dec     r14d
    jnz     .mvmul_row

.mvmul_ret:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; mvadd — Matrix-vector multiply-add: vout[i] += Σ_j mat[i][j] * vin[j]
;
; void mvadd(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols)
;   Same args as mvmul.
;
; PDP-11 equivalent: MVADD (adds to existing vout rather than overwriting)
; -----------------------------------------------------------------------------
mvadd:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx
    mov     r12, rdx
    mov     r13, r8
    mov     r14d, r9d
    mov     r15d, dword [rsp + 120]  ; cols

    test    r14d, r14d
    jle     .mvadd_ret

    mov     r10, r15
    shl     r10, 2              ; row stride in bytes

.mvadd_row:
    mov     rcx, rbx
    mov     rdx, r12
    mov     r8d, r15d
    call    vdot                ; eax = dot product Q8.16

    ; Saturated add to vout[i]
    add     eax, dword [r13]
    jno     .mvadd_ok
    js      .mvadd_pos_ov
    mov     eax, -2147483648
    jmp     .mvadd_ok
.mvadd_pos_ov:
    mov     eax, 2147483647
.mvadd_ok:
    mov     dword [r13], eax
    add     rbx, r10
    add     r13, 4
    dec     r14d
    jnz     .mvadd_row

.mvadd_ret:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; vtmul — Transpose-matrix × vector: vout[j] = Σ_i mat[i][j] * vin[i]
;
; void vtmul(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols)
;   rcx = mat,  rdx = vin,  r8 = vout,  r9d = rows
;   [rsp+120] = cols
;
; PDP-11 equivalent: VTMUL
; Algorithm: zero vout, then for each row i: vout += vin[i] * mat_row_i
; Uses vsadd as the inner scatter: vout[j] += scalar * mat[i][j]
; -----------------------------------------------------------------------------
vtmul:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; mat
    mov     r12, rdx            ; vin
    mov     r13, r8             ; vout
    mov     r14d, r9d           ; rows
    mov     r15d, dword [rsp + 120]  ; cols

    ; Zero vout[0..cols-1]
    mov     rcx, r13
    mov     edx, r15d
    call    vclr                ; vclr(vout, cols)

    test    r14d, r14d
    jle     .vtmul_ret

    mov     r10, r15
    shl     r10, 2              ; row stride bytes

    xor     r11d, r11d          ; row index i = 0

.vtmul_row:
    ; scalar = vin[i] (Q8.16)
    mov     ecx, dword [r12 + r11*4]  ; vin[i]
    ; vsadd(scalar, mat_row, vout, cols)
    mov     rdx, rbx            ; src = mat row i
    mov     r8,  r13            ; dst = vout
    mov     r9d, r15d           ; n = cols
    call    vsadd

    add     rbx, r10            ; advance mat row
    inc     r11d
    dec     r14d
    jnz     .vtmul_row

.vtmul_ret:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; outer — Outer product accumulate: mat[i][j] += vx[i] * vy[j]
;
; void outer(q16_t *mat, const q16_t *vx, const q16_t *vy, int rows, int cols)
;   rcx = mat,  rdx = vx,  r8 = vy,  r9d = rows
;   [rsp+120] = cols
;
; PDP-11 equivalent: OUTER (used in backward pass weight gradient accumulation)
; For each row i: mat[i] += vx[i] * vy  (via vsadd with scalar vx[i])
; -----------------------------------------------------------------------------
outer:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; mat
    mov     r12, rdx            ; vx
    mov     r13, r8             ; vy
    mov     r14d, r9d           ; rows
    mov     r15d, dword [rsp + 120]  ; cols

    test    r14d, r14d
    jle     .outer_ret

    mov     r10, r15
    shl     r10, 2              ; row stride bytes

    xor     r11d, r11d

.outer_row:
    ; scalar = vx[i]
    mov     ecx, dword [r12 + r11*4]
    ; vsadd(scalar, vy, mat_row_i, cols)
    mov     rdx, r13            ; src = vy
    mov     r8,  rbx            ; dst = mat row i
    mov     r9d, r15d
    call    vsadd

    add     rbx, r10
    inc     r11d
    dec     r14d
    jnz     .outer_row

.outer_ret:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
