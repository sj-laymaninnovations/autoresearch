# =============================================================================
# Makefile  —  ATTN11-x86 Kernel Build System
# Targets: Windows (Win64 COFF) and Linux (ELF64)
#
# Requirements:
#   Windows: nasm, cl (MSVC), link.exe — or nasm + gcc (MinGW)
#   Linux:   nasm, gcc, ld
#
# Usage:
#   make                  Build kernel.dll (Win) or kernel.so (Linux)
#   make test             Build + run unit tests
#   make size             Check .text section size (must be ≤ 20480 bytes)
#   make clean            Remove build artifacts
#   make PLATFORM=linux   Force Linux build
# =============================================================================

# --- Detect platform ---
ifeq ($(OS),Windows_NT)
    PLATFORM   ?= windows
else
    UNAME_S    := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM   ?= macos
    else
        PLATFORM   ?= linux
    endif
endif

# --- Source files ---
KERNEL_SRCS = fxmath.asm   \
              vecop.asm     \
              matop.asm     \
              actfn.asm     \
              attn_kernel.asm \
              qsparser.asm  \
              turboquant.asm \
              kvcache.asm   \
              io_win64.asm

SIZE_LIMIT  = 20480         # 20 KB hard limit on kernel .text

# =============================================================================
# Windows (MSVC / MinGW) target
# =============================================================================
ifeq ($(PLATFORM),windows)
    NASM      = nasm
    NASMFLAGS = -f win64 -g -F cv8
    CC        = cl
    CFLAGS    = /nologo /O2 /W3 /wd4996 /DWIN64
    LD        = link
    LDFLAGS   = /DLL /NOLOGO /NODEFAULTLIB:MSVCRT kernel32.lib msvcrt.lib
    OBJEXT    = obj
    LIBOUT    = kernel.dll
    LIBSTUB   = kernel.lib
    TESTCC    = cl /nologo /O2 /W3 /I..
    TESTLINK  = /link /LIBPATH:.. kernel.lib

    OBJS = $(KERNEL_SRCS:.asm=.obj)

$(LIBOUT): $(OBJS)
	$(LD) $(LDFLAGS) /OUT:$(LIBOUT) /IMPLIB:$(LIBSTUB) $^
	@python check_size.py $(LIBOUT) $(SIZE_LIMIT)

%.obj: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

test\test_fxmath.exe: test\test_fxmath.c $(LIBOUT)
	$(TESTCC) /Fe$@ test\test_fxmath.c $(TESTLINK)

test\test_vecop.exe: test\test_vecop.c $(LIBOUT)
	$(TESTCC) /Fe$@ test\test_vecop.c $(TESTLINK)

test\test_attn.exe: test\test_attn.c $(LIBOUT)
	$(TESTCC) /Fe$@ test\test_attn.c $(TESTLINK)

test: test\test_fxmath.exe test\test_vecop.exe test\test_attn.exe
	test\test_fxmath.exe && test\test_vecop.exe && test\test_attn.exe
	@echo All tests passed.

endif

# =============================================================================
# Linux (ELF64) target
# =============================================================================
ifeq ($(PLATFORM),linux)
    NASM      = nasm
    NASMFLAGS = -f elf64 -g -F dwarf
    CC        = gcc
    CFLAGS    = -O2 -Wall -Wextra -fPIC -DLINUX
    LD        = gcc
    LDFLAGS   = -shared -o
    OBJEXT    = o
    LIBOUT    = kernel.so
    SIZE_FLAG = --elf
    TESTCC    = gcc -O2 -Wall -I..
    TESTLINK  = -L.. -lkernel -Wl,-rpath,.

    OBJS = $(KERNEL_SRCS:.asm=.o)

$(LIBOUT): $(OBJS)
	$(LD) $(LDFLAGS) $(LIBOUT) $(OBJS)
	@python3 check_size.py $(LIBOUT) $(SIZE_LIMIT) --elf

%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

test/test_fxmath: test/test_fxmath.c $(LIBOUT)
	$(TESTCC) -o $@ test/test_fxmath.c $(TESTLINK)

test/test_vecop: test/test_vecop.c $(LIBOUT)
	$(TESTCC) -o $@ test/test_vecop.c $(TESTLINK)

test/test_attn: test/test_attn.c $(LIBOUT)
	$(TESTCC) -o $@ test/test_attn.c $(TESTLINK)

test: test/test_fxmath test/test_vecop test/test_attn
	LD_LIBRARY_PATH=. test/test_fxmath && \
	LD_LIBRARY_PATH=. test/test_vecop  && \
	LD_LIBRARY_PATH=. test/test_attn
	@echo "All tests passed."

endif

# =============================================================================
# macOS (Mach-O) target
# Intel x86-64: NASM macho64 + clang → kernel.dylib
# Apple Silicon ARM64: pure-C fallback (kernel_c.c) → kernel.dylib
# =============================================================================
ifeq ($(PLATFORM),macos)
    ARCH       := $(shell uname -m)
    CC         = clang
    LIBOUT     = kernel.dylib
    SIZE_FLAG  = --macho

    ifeq ($(ARCH),arm64)
        # Apple Silicon: x86-64 NASM cannot target ARM64; use C fallback
        CFLAGS   = -O2 -Wall -fPIC -DMACOS -DARM64

$(LIBOUT): kernel_c.c kernel.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ kernel_c.c -lm
	@python3 check_size.py $(LIBOUT) $(SIZE_LIMIT) $(SIZE_FLAG)

    else
        # Intel Mac: compile x86-64 NASM as Mach-O + link with clang
        NASM      = nasm
        NASMFLAGS = -f macho64 -g -F dwarf
        CFLAGS    = -O2 -Wall -fPIC -DMACOS
        OBJS      = $(KERNEL_SRCS:.asm=.o)

$(LIBOUT): $(OBJS)
	$(CC) $(CFLAGS) -dynamiclib -o $@ $(OBJS) -lm
	@python3 check_size.py $(LIBOUT) $(SIZE_LIMIT) $(SIZE_FLAG)

%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

    endif

test/test_fxmath: test/test_fxmath.c $(LIBOUT)
	$(CC) -O2 -Wall -I.. -o $@ test/test_fxmath.c -L. -lkernel -Wl,-rpath,.

test/test_vecop: test/test_vecop.c $(LIBOUT)
	$(CC) -O2 -Wall -I.. -o $@ test/test_vecop.c -L. -lkernel -Wl,-rpath,.

test/test_attn: test/test_attn.c $(LIBOUT)
	$(CC) -O2 -Wall -I.. -o $@ test/test_attn.c -L. -lkernel -Wl,-rpath,.

test: test/test_fxmath test/test_vecop test/test_attn
	DYLD_LIBRARY_PATH=. test/test_fxmath && \
	DYLD_LIBRARY_PATH=. test/test_vecop  && \
	DYLD_LIBRARY_PATH=. test/test_attn
	@echo "All tests passed."

endif

# =============================================================================
# Shared targets
# =============================================================================
size: $(LIBOUT)
	@python$(if $(findstring windows,$(PLATFORM)),,3) check_size.py $(LIBOUT) $(SIZE_LIMIT) $(SIZE_FLAG)

clean:
	$(if $(findstring windows,$(PLATFORM)), \
	    del /Q *.obj *.exp $(LIBOUT) $(LIBSTUB) test\*.exe 2>NUL, \
	    rm -f *.o $(LIBOUT) test/test_fxmath test/test_vecop test/test_attn)

.PHONY: test size clean
