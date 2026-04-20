; =============================================================================
; qsparser.asm  —  ASCII Sequence Parser / Tokenizer
; ATTN11-x86: NEW component (no PDP-11 equivalent)
;
; Converts null-terminated ASCII text to int32_t token IDs using
; character-level tokenization.
;
; Token ID convention:
;   0 = <PAD>      (null byte or padding)
;   1 = <BOS>      (beginning-of-sequence)
;   2 = <EOS>      (end-of-sequence — emitted at null terminator if requested)
;   3 = <UNK>      (non-ASCII byte ≥ 128)
;   4..130         = ASCII chars 0..126  (token_id = ascii + 4)
;   (Raw ASCII space ' '=32 → token 36, '0'=48 → token 52, etc.)
;
; This gives a fixed vocab size of 131 tokens.
; The qsdecode function inverts the mapping exactly.
;
; Win64 calling convention throughout.
; =============================================================================

BITS 64
DEFAULT REL

SECTION .text

GLOBAL qsparse
GLOBAL qsdecode

; Special token IDs
%define TOK_PAD   0
%define TOK_BOS   1
%define TOK_EOS   2
%define TOK_UNK   3
%define TOK_OFFSET 4        ; ascii_byte + TOK_OFFSET = token_id

; -----------------------------------------------------------------------------
; qsparse — Tokenize ASCII string
;
; int qsparse(const char *text, int32_t *token_ids, int out_cap)
;   rcx = text ptr (null-terminated ASCII)
;   rdx = token_ids output array (int32_t[])
;   r8d = out_cap (max tokens to write)
;   → eax = count of tokens written
;
; Returns count of tokens written. Does NOT write EOS token automatically;
; call qsparse_eos() if you want the EOS appended.
; -----------------------------------------------------------------------------
qsparse:
    ; Save callee-preserved registers used as local vars
    push    rsi
    push    rdi

    mov     rsi, rcx            ; rsi = text ptr (advances)
    mov     rdi, rdx            ; rdi = token_ids ptr (advances)
    mov     r10d, r8d           ; r10d = remaining capacity
    xor     eax, eax            ; eax = count = 0

    test    r10d, r10d
    jle     .done               ; capacity 0 → nothing to do

.loop:
    movzx   ecx, byte [rsi]     ; ecx = current byte (zero-extended)
    inc     rsi                 ; advance text ptr

    test    ecx, ecx
    jz      .done               ; null terminator → stop (do NOT emit EOS here)

    ; Map byte to token ID
    cmp     ecx, 128
    jae     .unk                ; byte ≥ 128 → UNK token

    ; printable ASCII (or control char): token_id = byte + TOK_OFFSET
    add     ecx, TOK_OFFSET
    jmp     .emit

.unk:
    mov     ecx, TOK_UNK

.emit:
    mov     dword [rdi], ecx    ; write token ID
    add     rdi, 4              ; advance output ptr (int32_t = 4 bytes)
    inc     eax                 ; count++
    dec     r10d
    jnz     .loop               ; continue if capacity remaining

.done:
    pop     rdi
    pop     rsi
    ret

; -----------------------------------------------------------------------------
; qsdecode — Decode token IDs back to ASCII string
;
; void qsdecode(const int32_t *token_ids, int n, char *buf, int buf_cap)
;   rcx = token_ids (int32_t[])
;   edx = n (number of tokens)
;   r8  = buf output buffer
;   r9d = buf_cap (including space for null terminator)
;
; Non-printable tokens (PAD=0, BOS=1, EOS=2, UNK=3) are written as '?'.
; Always null-terminates buf if buf_cap > 0.
; -----------------------------------------------------------------------------
qsdecode:
    push    rsi
    push    rdi
    push    rbx

    mov     rsi, rcx            ; rsi = token_ids (advances)
    mov     ebx, edx            ; ebx = n
    mov     rdi, r8             ; rdi = buf
    mov     r10d, r9d           ; r10d = buf_cap
    xor     ecx, ecx            ; ecx = chars written = 0

    test    r10d, r10d
    jle     .dec_done           ; no space at all

    dec     r10d                ; reserve 1 byte for null terminator
    jz      .dec_null           ; only 1 byte → just null-terminate

.dec_loop:
    test    ebx, ebx
    jle     .dec_null

    mov     eax, dword [rsi]    ; load token_id
    add     rsi, 4

    ; Map token_id back to char
    cmp     eax, TOK_OFFSET
    jl      .dec_special        ; special token → '?'

    sub     eax, TOK_OFFSET     ; eax = ASCII byte
    cmp     eax, 127
    jae     .dec_special        ; out of printable range → '?'
    jmp     .dec_emit

.dec_special:
    mov     eax, '?'

.dec_emit:
    mov     byte [rdi], al
    inc     rdi
    inc     ecx
    dec     ebx
    cmp     ecx, r10d
    jl      .dec_loop

.dec_null:
    mov     byte [rdi], 0       ; null terminator

.dec_done:
    pop     rbx
    pop     rdi
    pop     rsi
    ret
