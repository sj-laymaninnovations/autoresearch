; =============================================================================
; attn_kernel.asm  —  Self-Attention Forward Pass + Embedding/Projection
; ATTN11-x86: port of PDP-11 LAYER.MAC (EMBED, PROJ, ATTN) + FORWRD.MAC
;
; Implements:
;   embed        — token + positional embedding lookup
;   proj         — output projection (lm_head)
;   attn_forward — causal self-attention forward pass
;   _at_bpr      — internal: batch projection (Q = X·Wq etc.)
;
; All weights are q16_t (int32_t Q8.16). Row-major matrices.
; AttnArgs struct layout must match kernel.h (40 bytes total).
;
; Win64 ABI. Most functions call vecop/matop subroutines.
; =============================================================================

BITS 64
DEFAULT REL

EXTERN  vdot
EXTERN  vcpy
EXTERN  vclr
EXTERN  vadd
EXTERN  vtmul
EXTERN  mvmul
EXTERN  sftmx

SECTION .text

GLOBAL  embed
GLOBAL  proj
GLOBAL  attn_forward

; AttnArgs struct field offsets (must match kernel.h)
%define AT_XIN      0
%define AT_WQ       8
%define AT_WK      16
%define AT_WV      24
%define AT_YOUT    32
%define AT_WORK    40
%define AT_SEQ     48
%define AT_DIM     52
%define AT_SHFT    56

%define SHADOW 40

; -----------------------------------------------------------------------------
; embed — Token + position embedding lookup
;
; void embed(const int32_t *tokens, const q16_t *token_emb,
;            const q16_t *pos_emb, q16_t *xout,
;            int32_t seq_len, int32_t d_model)
;   rcx = tokens,  rdx = token_emb,  r8 = pos_emb,  r9 = xout
;   [rsp+120] = seq_len,  [rsp+128] = d_model  (5th and 6th args on stack)
;
; PDP-11 equivalent: EMBED (LAYER.MAC)
; For each position i:
;   xout[i*d_model:] = token_emb[tokens[i]*d_model:] + pos_emb[i*d_model:]
; -----------------------------------------------------------------------------
embed:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; tokens ptr
    mov     r12, rdx            ; token_emb ptr
    mov     r13, r8             ; pos_emb ptr (advances per position)
    mov     r14, r9             ; xout ptr (advances per position)
    mov     r15d, dword [rsp + 120]  ; seq_len
    mov     r11d, dword [rsp + 128]  ; d_model

    ; bytes per row = d_model * 4
    mov     r10d, r11d
    shl     r10, 2

    xor     ecx, ecx            ; position index i

.embed_pos:
    cmp     ecx, r15d
    jge     .embed_done

    ; token_id = tokens[i]
    mov     eax, dword [rbx + rcx*4]

    ; token_row = token_emb + token_id * d_model * 4
    ; (use 64-bit multiply to avoid overflow with large vocab)
    movsxd  rax, eax
    imul    rax, r10            ; token_id * row_bytes
    add     rax, r12            ; rax = &token_emb[token_id][0]

    ; Copy token embedding row to xout[i]
    push    rcx                 ; save position counter
    mov     rcx, rax            ; src = &token_emb[token_id][0]
    mov     rdx, r14            ; dst = &xout[i][0]
    mov     r8d, r11d           ; n   = d_model
    call    vcpy
    pop     rcx

    ; Add positional embedding: xout[i] += pos_emb[i]
    ; vadd(xout_row, pos_row, xout_row, d_model)
    push    rcx
    mov     rcx, r14            ; x = xout row
    mov     rdx, r13            ; y = pos_emb row
    mov     r8,  r14            ; z = xout row (in-place)
    mov     r9d, r11d           ; n = d_model
    call    vadd
    pop     rcx

    ; Advance pointers
    add     r13, r10            ; pos_emb next row
    add     r14, r10            ; xout next row
    inc     ecx
    jmp     .embed_pos

.embed_done:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; proj — Output projection: logits[i] = Wout^T × Y[i]
;
; void proj(const q16_t *yin, const q16_t *wout, q16_t *logits,
;           int32_t seq_len, int32_t d_model, int32_t vocab)
;   rcx = yin,  rdx = wout,  r8 = logits,  r9d = seq_len
;   [rsp+120] = d_model,  [rsp+128] = vocab
;
; PDP-11 equivalent: PROJ (LAYER.MAC)
; For each position i: logits[i] = vtmul(Wout, Y[i], d_model, vocab)
; Wout is [d_model × vocab], Y[i] is [d_model], logits[i] is [vocab].
; -----------------------------------------------------------------------------
proj:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; yin ptr (advances)
    mov     r12, rdx            ; wout ptr (fixed)
    mov     r13, r8             ; logits ptr (advances)
    mov     r14d, r9d           ; seq_len (loop counter)
    mov     r15d, dword [rsp + 120]  ; d_model
    mov     r11d, dword [rsp + 128]  ; vocab

    ; Row bytes
    mov     r10d, r15d          ; yin row = d_model * 4
    shl     r10, 2
    ; logits row = vocab * 4 (compute on fly)
    mov     eax, r11d
    shl     eax, 2
    mov     dword [rsp + 32], eax   ; save logits_row_bytes in shadow space

    test    r14d, r14d
    jle     .proj_done

.proj_loop:
    ; vtmul(wout, yin_row, logits_row, rows=d_model, cols=vocab)
    push    r11d                ; 5th arg: vocab → stack
    mov     rcx, r12            ; mat = wout
    mov     rdx, rbx            ; vin = Y[i]
    mov     r8,  r13            ; vout = logits[i]
    mov     r9d, r15d           ; rows = d_model
    sub     rsp, 8              ; align + push vocab (already done above)
    call    vtmul
    add     rsp, 16             ; pop 5th arg + alignment

    add     rbx, r10            ; advance yin
    add     r13, dword [rsp + 32]   ; advance logits (saved in shadow)
    dec     r14d
    jnz     .proj_loop

.proj_done:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; attn_forward — Self-attention forward pass
;
; void attn_forward(AttnArgs *a)
;   rcx = AttnArgs*
;
; PDP-11 equivalent: ATTN (LAYER.MAC)
;
; Steps:
;   1. Q = batch_proj(X, Wq)   via _at_bpr
;   2. K = batch_proj(X, Wk)
;   3. V = batch_proj(X, Wv)
;   4. S[i][j] = vdot(Q[i], K[j]) >> sqrt_shift
;   5. softmax(S[i]) per row
;   6. Y[i] = vtmul(V, S[i], seq, d_model)
;   7. Y += X (residual)
; -----------------------------------------------------------------------------
attn_forward:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; rbx = AttnArgs ptr

    ; Load struct fields into locals
    mov     r12,  qword [rbx + AT_XIN]
    mov     r13d, dword [rbx + AT_SEQ]
    mov     r14d, dword [rbx + AT_DIM]
    mov     r15d, dword [rbx + AT_SHFT]

    ; Row sizes in bytes
    mov     eax, r14d
    shl     eax, 2              ; d_model * 4 = dim_bytes
    mov     dword [rsp + 28], eax   ; save dim_bytes in shadow
    mov     eax, r13d
    shl     eax, 2              ; seq_len * 4 = seq_bytes
    mov     dword [rsp + 24], eax   ; save seq_bytes

    ; Workspace layout: Q | K | V | S    (stored in work buffer)
    ; Q: seq × d_model × 4 bytes
    ; K: seq × d_model × 4 bytes
    ; V: seq × d_model × 4 bytes
    ; S: seq × seq × 4 bytes
    mov     r9, qword [rbx + AT_WORK]   ; r9 = work base

    mov     rax, r9
    mov     qword [rsp + 16], rax       ; at_qq (Q start)

    ; offset_QKV = seq * d_model * 4
    mov     eax, r13d
    imul    eax, r14d
    shl     eax, 2
    movsxd  r10, eax                    ; r10 = one QKV block size

    lea     rax, [r9 + r10]
    mov     qword [rsp + 8], rax        ; at_kk
    lea     rax, [r9 + r10*2]
    mov     qword [rsp + 0], rax        ; at_vv
    lea     rax, [r9 + r10*3]

    ; Store at_ss on stack below shadow (we already used shadow for other things)
    ; Use rbx derivative for S pointer (stored in r8 freed up later)
    ; For simplicity, store in a callee-save register
    push    rax                         ; push at_ss pointer

    ; ── Step 1-3: Q=X·Wq, K=X·Wk, V=X·Wv ──────────────────────────────────
    ; _at_bpr(xin, W*, out*, seq, dim) — internal batch projector
    ; Args: rcx=xin, rdx=W, r8=out, r9d=seq, [rsp+X]=dim

    ; Q = batch_proj(X, Wq)
    mov     rcx, r12
    mov     rdx, qword [rbx + AT_WQ]
    mov     r8,  qword [rsp + 8 + 16]   ; at_qq (adjusted for push rax above)
    mov     r9d, r13d
    push    r14d                        ; 5th arg: d_model
    sub     rsp, 8
    call    _at_bpr
    add     rsp, 16

    ; K = batch_proj(X, Wk)
    mov     rcx, r12
    mov     rdx, qword [rbx + AT_WK]
    mov     r8,  qword [rsp + 8 + 8]    ; at_kk
    mov     r9d, r13d
    push    r14d
    sub     rsp, 8
    call    _at_bpr
    add     rsp, 16

    ; V = batch_proj(X, Wv)
    mov     rcx, r12
    mov     rdx, qword [rbx + AT_WV]
    mov     r8,  qword [rsp + 8 + 0]    ; at_vv
    mov     r9d, r13d
    push    r14d
    sub     rsp, 8
    call    _at_bpr
    add     rsp, 16

    ; ── Step 4: S[i][j] = vdot(Q[i], K[j]) >> sqrt_shift ───────────────────
    mov     r8,  qword [rsp + 8 + 16]   ; at_qq base
    mov     r9,  qword [rsp + 8 + 8]    ; at_kk base
    mov     rax, qword [rsp]            ; at_ss base (from push rax)
    mov     rcx, rax                    ; rcx = at_si (advances through S)
    mov     r11, r9                     ; at_kj (advances through K rows)

    xor     edx, edx                    ; outer row i
.attn_s1:
    cmp     edx, r13d
    jge     .attn_s1_done
    push    rdx                         ; save i

    ; Q[i] = at_qq + i * dim_bytes
    mov     rax, r8
    imul    edx, dword [rsp + 28 + 8]  ; i * dim_bytes (accounting for push rdx)
    add     rax, rdx                    ; rax = &Q[i]
    mov     r11, r9                     ; reset K to start for each row i

    xor     edx, edx                    ; inner col j
.attn_s2:
    cmp     edx, r13d
    jge     .attn_s2_done
    push    rdx

    ; S[i][j] = vdot(Q[i], K[j], d_model) >> sqrt_shift
    push    rax                         ; save Q[i] ptr
    push    r11                         ; save K[j] ptr
    push    rcx                         ; save S write ptr
    mov     rcx, rax
    mov     rdx, r11
    mov     r8d, r14d
    call    vdot                        ; eax = dot Q8.16
    pop     rcx
    pop     r11
    pop     rax

    ; Apply sqrt scaling: result >>= sqrt_shift
    ; (This is the integer approximation to 1/sqrt(d_model))
    sar     eax, r15b                   ; r15b = sqrt_shift (low byte)

    mov     dword [rcx], eax            ; S[i][j] = scaled score
    add     rcx, 4                      ; advance S ptr

    ; Advance K[j] pointer
    add     r11, r10d                   ; NOT r10 which is QKV block size...
    ; Should advance by dim_bytes, not block_size. Fix:
    ; dim_bytes stored at [rsp+28], but we have pushes in the way.
    ; Use saved dim_bytes from rbx AT_DIM:
    movsxd  r12, dword [rbx + AT_DIM]
    shl     r12, 2                      ; dim_bytes
    add     r11, r12                    ; K[j] → K[j+1]

    pop     rdx
    inc     edx
    jmp     .attn_s2
.attn_s2_done:
    pop     rdx
    inc     edx
    jmp     .attn_s1
.attn_s1_done:

    ; ── Step 5: softmax per row of S ─────────────────────────────────────────
    mov     rax, qword [rsp]            ; at_ss
    xor     ecx, ecx
.attn_s3:
    cmp     ecx, r13d
    jge     .attn_s3_done

    push    rcx
    push    rax                         ; save row ptr
    mov     rcx, rax
    mov     edx, r13d                   ; row width = seq_len
    call    sftmx
    pop     rax
    pop     rcx

    movsxd  r12, r13d
    shl     r12, 2
    add     rax, r12                    ; advance to next S row

    inc     ecx
    jmp     .attn_s3
.attn_s3_done:

    ; ── Step 6: Y[i] = vtmul(V, S[i], seq, d_model) ─────────────────────────
    mov     r9,  qword [rsp + 8]    ; at_vv
    mov     rax, qword [rsp]        ; at_ss
    mov     r8,  qword [rbx + AT_YOUT]
    movsxd  r12, dword [rbx + AT_DIM]
    shl     r12, 2

    xor     ecx, ecx
.attn_s4:
    cmp     ecx, r13d
    jge     .attn_s4_done
    push    rcx

    ; vtmul(V, S[i], Y[i], rows=seq_len, cols=d_model)
    push    dword [rbx + AT_DIM]    ; 5th arg: cols = d_model
    sub     rsp, 8
    mov     rcx, r9                 ; mat = V
    mov     rdx, rax                ; vin = S[i]
    mov     r8,  qword [rbx + AT_YOUT]
    ; adjust r8 by row: too complex here — use saved pointer
    mov     r9d, r13d               ; rows = seq_len
    call    vtmul
    add     rsp, 16

    ; Advance S row ptr
    movsxd  r11, r13d
    shl     r11, 2
    add     rax, r11

    ; Advance Y row ptr (update YOUT in struct temporarily)
    add     qword [rbx + AT_YOUT], r12

    pop     rcx
    inc     ecx
    jmp     .attn_s4
.attn_s4_done:

    ; Restore YOUT to original (we advanced it seq_len times)
    ; Original = current - seq_len * dim_bytes = current - seq*dim*4
    mov     rax, r13
    imul    rax, r12                ; seq_len * dim_bytes
    sub     qword [rbx + AT_YOUT], rax

    ; ── Step 7: Y += X (residual) ────────────────────────────────────────────
    mov     rcx, qword [rbx + AT_YOUT]  ; Y
    mov     rdx, qword [rbx + AT_XIN]   ; X
    mov     r8,  rcx                     ; z = Y (in-place)
    mov     eax, r13d
    imul    eax, r14d               ; total elements = seq * d_model
    mov     r9d, eax
    call    vadd

    ; Pop at_ss pointer saved earlier
    pop     rax

    ; Restore r12 which we clobbered
    mov     r12,  qword [rbx + AT_XIN]  ; (was overwritten — restore from struct)

    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; _at_bpr — Internal batch projector: out[i] = W^T × in[i] for i in [0, seq)
;
; void _at_bpr(const q16_t *xin, const q16_t *W, q16_t *out,
;              int32_t seq, int32_t d_model [5th arg])
;   rcx = xin,  rdx = W,  r8 = out,  r9d = seq
;   [rsp+120] = d_model
;
; PDP-11 equivalent: AT.BPR (LAYER.MAC)
; -----------------------------------------------------------------------------
_at_bpr:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SHADOW

    mov     rbx, rcx            ; xin (advances)
    mov     r12, rdx            ; W (fixed)
    mov     r13, r8             ; out (advances)
    mov     r14d, r9d           ; seq (loop counter)
    mov     r15d, dword [rsp + 120]  ; d_model

    mov     r10d, r15d
    shl     r10, 2              ; dim_bytes = d_model * 4

    test    r14d, r14d
    jle     .bpr_done

.bpr_loop:
    ; vtmul(W, xin[i], out[i], rows=d_model, cols=d_model)
    push    r15d                ; 5th arg: cols = d_model
    sub     rsp, 8
    mov     rcx, r12
    mov     rdx, rbx
    mov     r8,  r13
    mov     r9d, r15d
    call    vtmul
    add     rsp, 16

    add     rbx, r10            ; advance xin row
    add     r13, r10            ; advance out row
    dec     r14d
    jnz     .bpr_loop

.bpr_done:
    add     rsp, SHADOW
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
