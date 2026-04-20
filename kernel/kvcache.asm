; =============================================================================
; kvcache.asm  —  Key-Value Cache for Autoregressive Inference
; ATTN11-x86: NEW component (no PDP-11 equivalent)
;
; The PDP-11 kernel recomputed all K and V from scratch on every step.
; This cache stores K and V quantized to Q8.8 (int16_t) to halve memory.
;
; Layout (flat array, allocated via malloc):
;   For layer l, position t, element k:
;     K[l][t][k] at base + (l * max_seq * d_kv + t * d_kv + k) * sizeof(q8_t)
;     V[l][t][k] at K_base + n_layers * max_seq * d_kv * sizeof(q8_t) [offset]
;
; KVCache struct layout (must match kernel.h / Python ctypes):
;   +0   int32_t  n_layers
;   +4   int32_t  max_seq
;   +8   int32_t  d_kv
;   +12  int32_t  cur_pos      (current fill position, shared across layers)
;   +16  int64_t  slab_size    (bytes allocated for K or V)
;   +24  ptr      k_data       (pointer to K slab: n_layers × max_seq × d_kv × int16)
;   +32  ptr      v_data       (pointer to V slab: same layout)
;   total struct = 40 bytes
;
; Win64 ABI. Uses C stdlib malloc/free via external linkage.
; =============================================================================

BITS 64
DEFAULT REL

SECTION .text

EXTERN  malloc
EXTERN  free
EXTERN  memset

GLOBAL  kvc_create
GLOBAL  kvc_free
GLOBAL  kvc_insert
GLOBAL  kvc_query
GLOBAL  kvc_reset
GLOBAL  kvc_pos

; KVCache struct offsets
%define KVC_NLAYERS   0
%define KVC_MAX_SEQ   4
%define KVC_D_KV      8
%define KVC_CUR_POS   12
%define KVC_SLAB_SZ   16
%define KVC_K_DATA    24
%define KVC_V_DATA    32
%define KVC_STRUCT_SZ 40

; Quantization constants for Q8.16 → Q8.8
%define Q16_TO_Q8_SHIFT  8         ; right-shift 8 bits: Q8.16 → Q8.8

; -----------------------------------------------------------------------------
; kvc_create — Allocate a KV cache
;
; KVCache* kvc_create(int n_layers, int max_seq, int d_kv)
;   ecx = n_layers,  edx = max_seq,  r8d = d_kv
;   → rax = pointer to new KVCache, or NULL on allocation failure
;
; Allocates:
;   1 × KVC_STRUCT_SZ byte  struct
;   2 × n_layers × max_seq × d_kv × 2 bytes  (K slab + V slab, int16_t)
; -----------------------------------------------------------------------------
kvc_create:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 40             ; shadow space

    mov     r12d, ecx           ; save n_layers
    mov     r13d, edx           ; save max_seq
    mov     r14d, r8d           ; save d_kv

    ; Allocate struct
    mov     ecx, KVC_STRUCT_SZ
    call    malloc
    test    rax, rax
    jz      .kvc_create_fail
    mov     rbx, rax            ; rbx = struct ptr

    ; Compute slab size = n_layers * max_seq * d_kv * sizeof(int16_t)
    mov     eax, r12d
    imul    eax, r13d
    imul    eax, r14d
    cdqe                        ; sign-extend to 64-bit
    shl     rax, 1              ; * 2 (sizeof int16_t)
    mov     qword [rbx + KVC_SLAB_SZ], rax

    ; Allocate K slab
    mov     rcx, rax
    push    rax                 ; save slab_size
    call    malloc
    pop     r9                  ; restore slab_size
    test    rax, rax
    jz      .kvc_free_struct
    mov     qword [rbx + KVC_K_DATA], rax

    ; Zero K slab
    mov     rdx, r9             ; size
    xor     r8d,  r8d           ; value = 0
    mov     rcx, rax            ; ptr
    call    memset

    ; Allocate V slab
    mov     rcx, r9             ; slab_size again
    call    malloc
    test    rax, rax
    jz      .kvc_free_k
    mov     qword [rbx + KVC_V_DATA], rax

    ; Zero V slab
    mov     rdx, r9
    xor     r8d,  r8d
    mov     rcx, rax
    call    memset

    ; Fill struct fields
    mov     dword [rbx + KVC_NLAYERS], r12d
    mov     dword [rbx + KVC_MAX_SEQ], r13d
    mov     dword [rbx + KVC_D_KV],    r14d
    mov     dword [rbx + KVC_CUR_POS], 0

    mov     rax, rbx            ; return struct ptr
    add     rsp, 40
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

.kvc_free_k:
    mov     rcx, qword [rbx + KVC_K_DATA]
    call    free
.kvc_free_struct:
    mov     rcx, rbx
    call    free
.kvc_create_fail:
    xor     eax, eax            ; return NULL
    add     rsp, 40
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; kvc_free — Deallocate a KV cache
;
; void kvc_free(KVCache *c)
;   rcx = cache ptr (may be NULL → no-op)
; -----------------------------------------------------------------------------
kvc_free:
    sub     rsp, 40
    test    rcx, rcx
    jz      .kvc_free_ret

    push    rbx
    mov     rbx, rcx

    ; Free K slab
    mov     rcx, qword [rbx + KVC_K_DATA]
    test    rcx, rcx
    jz      .kvc_free_v
    call    free

.kvc_free_v:
    mov     rcx, qword [rbx + KVC_V_DATA]
    test    rcx, rcx
    jz      .kvc_free_s
    call    free

.kvc_free_s:
    mov     rcx, rbx
    call    free

    pop     rbx
.kvc_free_ret:
    add     rsp, 40
    ret

; -----------------------------------------------------------------------------
; kvc_insert — Store K, V for one position (Q8.16 → Q8.8 quantize on write)
;
; void kvc_insert(KVCache *c, int layer, int pos, const q16_t *k, const q16_t *v)
;   rcx = c,  edx = layer,  r8d = pos,  r9 = k,  [rsp+40+32] = v
;
; Stack layout on entry (Win64, after 40-byte shadow):
;   [rsp+40+32+8] = v ptr  (5th argument, pushed by caller)
;   Wait — in Win64, 5th arg goes on stack at [rsp+32] in the CALLER'S frame.
;   After sub rsp,40: v is at [rsp + 40 + 32] = [rsp+72]... 
;   Actually: caller uses [rsp+32] for 5th arg (no shadow needed by callee for stack args).
;   Simplified: we read v from [rsp + 40 + 8] = [rsp+48] (accounting for return addr? No.)
;
; Win64 5th param stack layout:
;   After sub rsp, 40 (shadow + align):
;   return address was at [rsp+40] before sub
;   5th arg (v) is at [rsp + 40 + 8] = [rsp + 48] (above return address in caller frame)
;   Actually more carefully:
;     On entry: rsp points at return address
;     sub rsp, 40: rsp moves down 40 bytes
;     5th arg was placed at [old_rsp + 8 + 32] = [rsp + 40 + 8] = [rsp + 48]
;   Wait, Win64 convention: home space is 32 bytes (4 regs × 8 bytes),
;     placed by caller starting at [rsp+8] (after ret addr at [rsp]).
;     5th arg goes at [rsp + 8 + 32] = [rsp + 40].
;   After our sub rsp, 40: that becomes [rsp + 80].
; -----------------------------------------------------------------------------
kvc_insert:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40             ; shadow space

    mov     rbx, rcx            ; rbx = cache
    mov     r12d, edx           ; r12d = layer
    mov     r13d, r8d           ; r13d = pos
    mov     r14, r9             ; r14 = k ptr
    mov     r15, qword [rsp + 40 + 5*8] ; r15 = v ptr (5th arg on stack)
                                ; (40 shadow + push 5 regs × 8 = 40 + 40 = 80, but...)
    ; Actually let's compute correctly:
    ; We pushed 5 registers (5×8=40 bytes) then sub rsp,40
    ; So total offset from original rsp = 80 bytes
    ; 5th arg in caller frame at [original_rsp + 8 + 32] = [original_rsp + 40]
    ; = [rsp + 80 + 40] ... This is getting complicated.
    ; Let's simplify: just save rcx,rdx,r8,r9 into our shadow space and reread.

    ; Actually, the simplest correct approach on Win64:
    ; The 5th argument is at [rsp + 40] in the CALLER's frame (before our sub rsp).
    ; After push rbx,r12,r13,r14,r15 (5 × 8 = 40) and sub rsp,40 (40):
    ; Total offset = 40 + 40 = 80 bytes from our current rsp.
    ; Plus caller's ret addr (8 bytes): 5th arg at [rsp + 80 + 8] = [rsp + 88]
    ; Hmm, let me re-derive:
    ;   At function entry:  [rsp+0] = ret addr, [rsp+8..39] = home space (rcx,rdx,r8,r9),
    ;                       [rsp+40] = 5th arg (v ptr)
    ;   After push ×5:      rsp -= 40, so 5th arg now at [rsp + 40 + 40] = [rsp+80]
    ;   After sub rsp, 40:  rsp -= 40, so 5th arg now at [rsp + 80 + 40] = [rsp+120]

    ; Reload v ptr from correct stack offset
    ; (overwrite r15 with correct value)
    mov     r15, qword [rsp + 120]  ; 5th argument (v ptr)

    ; Compute index into K slab:
    ;   offset = (layer * max_seq * d_kv + pos * d_kv) * sizeof(int16_t)
    mov     eax,  r12d          ; layer
    imul    eax,  dword [rbx + KVC_MAX_SEQ]
    add     eax,  r13d          ; + pos
    imul    eax,  dword [rbx + KVC_D_KV]
    cdqe
    shl     rax, 1              ; * 2 (sizeof int16_t)

    ; Write K vector: quantize Q8.16 → Q8.8 (>> 8, saturate to int16)
    mov     rcx, qword [rbx + KVC_K_DATA]
    add     rcx, rax            ; rcx = &K[layer][pos][0]
    mov     edx, dword [rbx + KVC_D_KV]
    call    .kvc_write_vec      ; write r14 → rcx, d_kv=edx

    ; Write V vector (same offset)
    mov     rcx, qword [rbx + KVC_V_DATA]
    lea     rcx, [rcx + rax]
    mov     edx, dword [rbx + KVC_D_KV]
    mov     r14, r15            ; src = v ptr
    call    .kvc_write_vec

    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; Internal: quantize q16_t array r14[edx] → int16_t array at rcx
; Clobbers: rax, r8, r9, r10, r11
.kvc_write_vec:
    test    edx, edx
    jle     .kwv_ret
    mov     r10d, edx
.kwv_loop:
    movsxd  rax, dword [r14]   ; load Q8.16 element
    sar     rax, Q16_TO_Q8_SHIFT  ; → Q8.8 in 32-bit sign-extended
    ; Saturate to int16
    mov     r8,  32767
    cmp     rax, r8
    cmovg   rax, r8
    mov     r8, -32768
    cmp     rax, r8
    cmovl   rax, r8
    mov     word [rcx], ax     ; store as int16_t
    add     r14, 4
    add     rcx, 2
    dec     r10d
    jnz     .kwv_loop
.kwv_ret:
    ret

; -----------------------------------------------------------------------------
; kvc_query — Read K and V matrices for all positions [0..max_pos-1]
;
; void kvc_query(KVCache *c, int layer, int max_pos, q16_t *k_out, q16_t *v_out)
;   rcx = c,  edx = layer,  r8d = max_pos,  r9 = k_out
;   [rsp + shadow + 40] = v_out   (5th arg on stack, recalculated below)
;
; Dequantizes Q8.8 → Q8.16 (shift left 8) on output.
; -----------------------------------------------------------------------------
kvc_query:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40

    mov     rbx, rcx                ; cache
    mov     r12d, edx               ; layer
    mov     r13d, r8d               ; max_pos
    mov     r14, r9                 ; k_out
    mov     r15, qword [rsp + 120]  ; v_out (5th arg)

    ; d_kv shorthand
    mov     r10d, dword [rbx + KVC_D_KV]
    mov     r11d, dword [rbx + KVC_MAX_SEQ]

    ; Base offset for this layer into slabs
    mov     eax, r12d
    imul    eax, r11d           ; layer * max_seq
    ; Reading from pos 0..max_pos-1, so start at layer * max_seq * d_kv * 2
    imul    eax, r10d           ; * d_kv
    cdqe
    shl     rax, 1              ; * sizeof(int16_t)

    ; K slab read
    mov     rcx, qword [rbx + KVC_K_DATA]
    add     rcx, rax            ; rcx = &K[layer][0][0]
    mov     edx, r13d
    imul    edx, r10d           ; total_elements = max_pos * d_kv
    call    .kvc_read_vec       ; rcx → r14, count=edx

    ; V slab read
    mov     rcx, qword [rbx + KVC_V_DATA]
    add     rcx, rax
    mov     edx, r13d
    imul    edx, r10d
    mov     r14, r15
    call    .kvc_read_vec

    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; Internal: expand int16_t array rcx[edx] → q16_t array at r14
.kvc_read_vec:
    test    edx, edx
    jle     .krv_ret
    mov     r10d, edx
.krv_loop:
    movsx   eax, word [rcx]     ; load int16_t (Q8.8), sign-extend to 32-bit
    shl     eax, Q16_TO_Q8_SHIFT ; << 8 → Q8.16
    mov     dword [r14], eax
    add     rcx, 2
    add     r14, 4
    dec     r10d
    jnz     .krv_loop
.krv_ret:
    ret

; -----------------------------------------------------------------------------
; kvc_reset — Reset position counter (keeps allocation)
;
; void kvc_reset(KVCache *c)
;   rcx = c
; -----------------------------------------------------------------------------
kvc_reset:
    test    rcx, rcx
    jz      .kreset_ret
    mov     dword [rcx + KVC_CUR_POS], 0
.kreset_ret:
    ret

; -----------------------------------------------------------------------------
; kvc_pos — Return current fill count
;
; int kvc_pos(const KVCache *c)
;   rcx = c
;   → eax = cur_pos
; -----------------------------------------------------------------------------
kvc_pos:
    test    rcx, rcx
    jz      .kpos_null
    mov     eax, dword [rcx + KVC_CUR_POS]
    ret
.kpos_null:
    xor     eax, eax
    ret
