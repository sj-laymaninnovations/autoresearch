; =============================================================================
; io_win64.asm  —  Console I/O Helpers (Win64 WriteFile)
; ATTN11-x86: replacement for PDP-11 BSD _write-based I/O routines
;
; Functions:
;   putq16   — print Q8.16 value as decimal string (e.g. "1.234500")
;   putvec16 — print vector: "[1.000, -0.500, ...]"
;
; Uses Win64 WriteFile(STDOUT_HANDLE, buf, len, &written, NULL).
; wsprintf from kernel32 for int→string conversion.
; =============================================================================

BITS 64
DEFAULT REL

EXTERN  GetStdHandle
EXTERN  WriteFile

SECTION .data
ALIGN 8
.stdout_handle: dq 0                ; cached STDOUT handle
.written:       dd 0                ; bytes written (WriteFile output param)

SECTION .text

GLOBAL  putq16
GLOBAL  putvec16

%define STD_OUTPUT_HANDLE -11

; One-time STDOUT handle fetch (idempotent)
_get_stdout:
    mov     rax, qword [.stdout_handle]
    test    rax, rax
    jnz     .gso_ret
    sub     rsp, 40
    mov     ecx, STD_OUTPUT_HANDLE
    call    GetStdHandle
    add     rsp, 40
    mov     qword [.stdout_handle], rax
.gso_ret:
    ret

; _write_bytes(ptr rcx, len edx) — write edx bytes from rcx to STDOUT
_write_bytes:
    push    rbx
    push    r12
    sub     rsp, 40

    mov     rbx, rcx
    mov     r12d, edx

    call    _get_stdout
    ; WriteFile(hFile, lpBuffer, nBytes, lpWritten, lpOverlapped)
    mov     rcx, rax            ; hFile
    mov     rdx, rbx            ; buffer
    mov     r8d, r12d           ; byte count
    lea     r9, [.written]      ; lpWritten
    push    0                   ; lpOverlapped = NULL
    call    WriteFile
    add     rsp, 8

    add     rsp, 40
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; putq16 — Print Q8.16 value as human-readable decimal
;
; void putq16(q16_t v)   (ecx = v)
;
; Format: [-]INT.FRAC6   e.g. "-1.500000" or "0.003906"
; PDP-11 equivalent: PUTQ8 / putq8 (io_bsd section of attn.s)
; -----------------------------------------------------------------------------
putq16:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 48             ; shadow + local buffer (32 bytes)

    mov     rbx, ecx            ; rbx = v (sign-extended to 64-bit)
    movsxd  rbx, ecx

    ; Buffer starts at rsp+40 (after 8 bytes alignment + shadow)
    lea     r12, [rsp + 16]     ; r12 = char buffer (32 bytes)
    xor     r13d, r13d          ; r13d = buffer index

    ; Handle sign
    test    rbx, rbx
    jns     .pq_pos
    ; Write '-'
    mov     byte [r12 + r13], '-'
    inc     r13d
    neg     rbx                 ; |v|
.pq_pos:
    ; Integer part = v >> 16
    mov     rax, rbx
    sar     rax, 16
    ; Write integer digits
    ; Simple: convert to string via repeated div
    mov     ecx, 10
    ; Push digits in reverse
    xor     r14d, r14d          ; digit count
    mov     rdi, rax
    test    rdi, rdi
    jnz     .pq_int_loop
    ; Integer part is 0 → write "0"
    mov     byte [r12 + r13], '0'
    inc     r13d
    jmp     .pq_dot
.pq_int_loop:
    test    rdi, rdi
    jz      .pq_int_reverse
    mov     rax, rdi
    xor     edx, edx
    div     rcx                 ; rax = quot, rdx = remainder digit
    push    rdx
    inc     r14d
    mov     rdi, rax
    jmp     .pq_int_loop
.pq_int_reverse:
    ; Pop digits in correct order
    test    r14d, r14d
    jz      .pq_dot
    pop     rax
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    dec     r14d
    jmp     .pq_int_reverse

.pq_dot:
    mov     byte [r12 + r13], '.'
    inc     r13d

    ; Fractional part: (v & 0xFFFF) * 1000000 >> 16
    ; i.e.: frac_decimal = (low16 * 1000000) / 65536
    mov     rax, rbx
    and     rax, 0xFFFF         ; low 16 bits
    mov     rcx, 1000000
    imul    rax, rcx
    sar     rax, 16             ; → decimal fraction 0..999999
    ; Write 6 digits
    mov     ecx, 10
    mov     r14d, 6
.pq_frac:
    ; Extract from high to low by successive / 100000, /10000, etc.
    ; Simpler: write all 6 digits right-to-left then reverse
    xor     edx, edx
    push    rax
    mov     rax, rax
    ; Just format 6 decimal digits:
    mov     ecx, 10
    mov     eax, eax
    ; We'll do it simply: compute each digit from most significant
    ; Using a trick: multiply fraction by 10 repeatedly
    ; Fraction is 0..999999, print each of 6 digits:
    pop     rax
    ; rax = 0..999999
.pq_frac6:
    mov     rcx, 100000
    xor     edx, edx
    div     rcx
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    mov     rax, rdx
    imul    rcx, rcx, 10        ; next divisor / 10... simpler:
    ; Just hardcode 6 rounds of "digit = val/divisor; val %= divisor"
    ; The loop above only did 1 round. Let's do all 6:
    ; Already wrote hundred-thousands digit. Remaining = rdx (0..99999)
    ; Continue for 10000, 1000, 100, 10, 1:
    mov     rcx, 10000
    xor     edx, edx
    div     rcx
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    mov     rax, rdx

    mov     rcx, 1000
    xor     edx, edx
    div     rcx
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    mov     rax, rdx

    mov     rcx, 100
    xor     edx, edx
    div     rcx
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    mov     rax, rdx

    mov     rcx, 10
    xor     edx, edx
    div     rcx
    add     al, '0'
    mov     byte [r12 + r13], al
    inc     r13d
    add     dl, '0'
    mov     byte [r12 + r13], dl
    inc     r13d

    ; Newline
    mov     byte [r12 + r13], 10
    inc     r13d

    ; Write buffer
    mov     rcx, r12
    mov     edx, r13d
    call    _write_bytes

    add     rsp, 48
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; -----------------------------------------------------------------------------
; putvec16 — Print vector as "[v0, v1, ..., vn-1]"
;
; void putvec16(const q16_t *v, int n)
;   rcx = v,  edx = n
;
; PDP-11 equivalent: PUTVEC (attn.s io section)
; -----------------------------------------------------------------------------
putvec16:
    push    rbx
    push    r12
    push    r13
    sub     rsp, 40

    mov     rbx, rcx            ; vec ptr
    mov     r12d, edx           ; n
    xor     r13d, r13d          ; i

    ; Print '['
    sub     rsp, 8
    lea     rcx, [.lbracket]
    mov     edx, 1
    call    _write_bytes
    add     rsp, 8

.pvec_loop:
    cmp     r13d, r12d
    jge     .pvec_close

    ; print v[i]
    sub     rsp, 8
    mov     ecx, dword [rbx + r13*4]
    call    putq16
    add     rsp, 8

    inc     r13d
    cmp     r13d, r12d
    jge     .pvec_close

    ; print ", "
    sub     rsp, 8
    lea     rcx, [.comma_sp]
    mov     edx, 2
    call    _write_bytes
    add     rsp, 8
    jmp     .pvec_loop

.pvec_close:
    sub     rsp, 8
    lea     rcx, [.rbracket_nl]
    mov     edx, 2
    call    _write_bytes
    add     rsp, 8

    add     rsp, 40
    pop     r13
    pop     r12
    pop     rbx
    ret

SECTION .rodata
.lbracket:    db '['
.comma_sp:    db ',', ' '
.rbracket_nl: db ']', 10
