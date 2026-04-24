"""
check_size.py — Kernel binary size enforcer
Usage: python check_size.py <binary> <limit_bytes> [--elf] [--macho]

Exits 0 if binary .text section (or total PE .text) is <= limit_bytes.
Exits 1 and prints error if over limit.

For PE/DLL   (Windows): measures the .text section from the PE header.
For ELF .so  (Linux):   measures the .text section via readelf.
For Mach-O   (macOS):   measures the __text section via otool.
"""

import sys
import os
import struct

def read_pe_text_size(path):
    """Extract .text section size from a PE32+ (Win64) DLL or EXE."""
    with open(path, "rb") as f:
        data = f.read()

    # MZ header
    if data[:2] != b"MZ":
        raise ValueError("Not a PE file")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset:pe_offset+4] != b"PE\x00\x00":
        raise ValueError("PE signature not found")

    # Optional header offset = pe_offset + 4 (sig) + 20 (COFF header)
    coff_offset = pe_offset + 4
    num_sections = struct.unpack_from("<H", data, coff_offset + 2)[0]
    size_of_optional = struct.unpack_from("<H", data, coff_offset + 16)[0]
    section_table_offset = coff_offset + 20 + size_of_optional

    text_size = None
    for i in range(num_sections):
        sec = section_table_offset + i * 40
        name = data[sec:sec+8].rstrip(b"\x00").decode("ascii", errors="replace")
        virtual_size  = struct.unpack_from("<I", data, sec + 8)[0]
        raw_size      = struct.unpack_from("<I", data, sec + 16)[0]
        if name == ".text":
            # VirtualSize is the actual content size (RawDataSize may be padded)
            text_size = min(virtual_size, raw_size)
            break

    if text_size is None:
        # Fall back to total file size if no .text section found
        text_size = os.path.getsize(path)
    return text_size


def read_elf_text_size(path):
    """Extract .text section size from an ELF64 shared library."""
    try:
        import subprocess
        result = subprocess.run(
            ["readelf", "-S", "--wide", path],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if ".text" in line and "PROGBITS" in line:
                parts = line.split()
                # Format: [ N] .text PROGBITS addr offset size ...
                # Find the hex size field
                for j, p in enumerate(parts):
                    if p == ".text":
                        # size is typically 2 positions after PROGBITS
                        try:
                            return int(parts[j + 4], 16)
                        except (IndexError, ValueError):
                            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: just use file size
    return os.path.getsize(path)


def read_macho_text_size(path):
    """Extract __text section size from a Mach-O dylib (macOS)."""
    try:
        import subprocess
        result = subprocess.run(
            ["otool", "-l", path],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        in_text = False
        for line in lines:
            s = line.strip()
            if s == "sectname __text":
                in_text = True
            elif in_text and s.startswith("size"):
                parts = s.split()
                if len(parts) >= 2:
                    return int(parts[1], 16)
                in_text = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.path.getsize(path)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python check_size.py <binary> <limit_bytes> [--elf] [--macho]")
        sys.exit(1)

    binary_path = args[0]
    limit = int(args[1])
    is_elf   = "--elf"   in args
    is_macho = "--macho" in args

    if not os.path.exists(binary_path):
        print(f"ERROR: {binary_path} not found")
        sys.exit(1)

    if is_elf:
        text_size = read_elf_text_size(binary_path)
    elif is_macho:
        text_size = read_macho_text_size(binary_path)
    else:
        try:
            text_size = read_pe_text_size(binary_path)
        except Exception:
            # Fallback for non-PE targets
            text_size = os.path.getsize(binary_path)

    pct = 100.0 * text_size / limit
    bar_width = 40
    filled = int(bar_width * text_size / limit)
    bar = "#" * filled + "." * (bar_width - filled)

    print("")
    print("  Kernel .text size check")
    print(f"  Binary : {binary_path}")
    print(f"  .text  : {text_size:,} bytes  ({pct:.1f}% of {limit:,} B limit)")
    print(f"  [{bar}]")

    if text_size <= limit:
        remaining = limit - text_size
        print(f"  [PASS]  {remaining:,} bytes remaining\n")
        sys.exit(0)
    else:
        over = text_size - limit
        print(f"  [FAIL]  {over:,} bytes OVER limit!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
