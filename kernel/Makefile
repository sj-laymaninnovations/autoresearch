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
    PLATFORM   ?= linux
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
# Shared targets
# =============================================================================
size: $(LIBOUT)
	@python$(if $(findstring windows,$(PLATFORM)),,3) check_size.py $(LIBOUT) $(SIZE_LIMIT)

clean:
	$(if $(findstring windows,$(PLATFORM)), \
	    del /Q *.obj *.exp $(LIBOUT) $(LIBSTUB) test\*.exe 2>NUL, \
	    rm -f *.o $(LIBOUT) test/test_fxmath test/test_vecop test/test_attn)

.PHONY: test size clean
