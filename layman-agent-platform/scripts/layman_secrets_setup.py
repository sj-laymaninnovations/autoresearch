#!/usr/bin/env python3
"""
Layman Secrets Setup Utility
=============================
A Layman Innovations Open Standard Tool

Accepts SLUG=value pairs via secure in-memory paste,
stores each to your OS native credential vault,
and outputs a clean secrets.yaml with slug references only.

Supported OS vaults:
  Windows  → Windows Credential Manager (via keyring)
  macOS    → Keychain (via keyring)
  Linux    → SecretService / libsecret (via keyring)

Usage:
  python layman_secrets_setup.py

Dependencies:
  pip install keyring pyyaml
"""

import sys
import os
import platform
import getpass
from datetime import date, datetime

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import keyring
except ImportError:
    print("\n[ERROR] Missing dependency: keyring")
    print("  Install it with:  pip install keyring pyyaml\n")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("\n[ERROR] Missing dependency: pyyaml")
    print("  Install it with:  pip install keyring pyyaml\n")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────
SERVICE_NAME = "layman-secrets"
BANNER = """
╔══════════════════════════════════════════════════════════╗
║           Layman Secrets Setup Utility                   ║
║           A Layman Innovations Open Standard             ║
╠══════════════════════════════════════════════════════════╣
║  Your secrets will be stored in your OS credential vault ║
║  Nothing is written to disk. Nothing goes into git.      ║
╚══════════════════════════════════════════════════════════╝
"""

PASTE_INSTRUCTIONS = """
────────────────────────────────────────────────────────────
  PASTE MODE
  
  Paste your SLUG=value pairs below, one per line.
  Example:
  
    PROJECT_ALPHA_ANTHROPIC_API_KEY=sk-ant-...
    BOX_CLIENT_ID=abc123
    BOX_CLIENT_SECRET=xyz789

  Rules:
    • One entry per line
    • Format: SLUG=value  (no spaces around =)
    • Lines starting with # are ignored (comments)
    • Blank lines are ignored

  When done, press:
    • Windows/Linux: Enter, then Ctrl+Z, then Enter
    • macOS:         Enter, then Ctrl+D

────────────────────────────────────────────────────────────
"""

DESCRIPTION_PROMPT = """
────────────────────────────────────────────────────────────
  METADATA (optional — press Enter to skip each field)
  This information goes into your secrets.yaml reference file.
  No secret values are included.
────────────────────────────────────────────────────────────
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_os_info():
    os_name = platform.system()
    vault_map = {
        "Windows": "Windows Credential Manager",
        "Darwin":  "macOS Keychain",
        "Linux":   "SecretService / libsecret"
    }
    vault = vault_map.get(os_name, "OS credential vault")
    print(f"  Detected OS : {os_name}")
    print(f"  Vault target: {vault}\n")


def parse_paste(raw_text: str) -> dict:
    """Parse SLUG=value lines from pasted text. Returns {slug: value}."""
    entries = {}
    errors = []
    for i, line in enumerate(raw_text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"  Line {i}: skipped (no '=' found): {line[:40]}")
            continue
        slug, _, value = line.partition("=")
        slug = slug.strip().upper()
        value = value.strip()
        if not slug:
            errors.append(f"  Line {i}: skipped (empty slug)")
            continue
        if not value:
            errors.append(f"  Line {i}: skipped (empty value for {slug})")
            continue
        entries[slug] = value
    return entries, errors


def store_secret(slug: str, value: str) -> bool:
    """Store a single secret in the OS credential vault."""
    try:
        keyring.set_password(SERVICE_NAME, slug, value)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to store {slug}: {e}")
        return False


def verify_secret(slug: str) -> bool:
    """Verify a secret was stored successfully."""
    try:
        val = keyring.get_password(SERVICE_NAME, slug)
        return val is not None
    except Exception:
        return False


def collect_metadata(slug: str) -> dict:
    """Prompt user for optional metadata for a slug."""
    print(f"\n  Slug: {slug}")
    name        = input(f"  Name/label (default: {slug}): ").strip() or slug
    description = input(f"  Description: ").strip() or None
    expires_raw = input(f"  Expires (YYYY-MM-DD, or blank for none): ").strip()

    expires = None
    if expires_raw:
        try:
            datetime.strptime(expires_raw, "%Y-%m-%d")
            expires = expires_raw
        except ValueError:
            print("  [!] Invalid date format — expiry skipped")

    return {
        "slug":         slug,
        "name":         name,
        "description":  description,
        "date":         str(date.today()),
        "last_updated": str(date.today()),
        "expires":      expires,
    }


def write_secrets_yaml(metadata_list: list, output_path: str):
    """Write secrets.yaml with slug references and metadata only."""
    payload = {"schema_version": 1, "secrets": metadata_list}
    with open(output_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ── Main flow ─────────────────────────────────────────────────────────────────

def main():
    clear_screen()
    print(BANNER)
    print_os_info()

    input("  Press Enter to continue to paste mode...")

    # ── Step 1: Collect paste input ───────────────────────────────────────────
    clear_screen()
    print(PASTE_INSTRUCTIONS)
    print("  Paste below (then signal EOF when done):\n")

    try:
        raw_lines = sys.stdin.read()
    except KeyboardInterrupt:
        print("\n\n  Cancelled. No secrets stored.\n")
        sys.exit(0)

    entries, errors = parse_paste(raw_lines)

    if errors:
        print("\n  [!] Some lines were skipped:")
        for e in errors:
            print(e)

    if not entries:
        print("\n  No valid entries found. Nothing stored.\n")
        sys.exit(0)

    # ── Step 2: Preview slugs (never values) ──────────────────────────────────
    clear_screen()
    print("\n  ── Parsed Slugs (values hidden) ──────────────────────────────\n")
    for slug in entries:
        print(f"    ✓  {slug}")
    print(f"\n  Total: {len(entries)} secret(s) ready to store\n")

    confirm = input("  Store all to OS credential vault? [y/N]: ").strip().lower()
    if confirm != "y":
        print("\n  Cancelled. No secrets stored.\n")
        sys.exit(0)

    # ── Step 3: Store to OS vault ─────────────────────────────────────────────
    clear_screen()
    print("\n  ── Storing secrets ───────────────────────────────────────────\n")
    stored = []
    failed = []

    for slug, value in entries.items():
        success = store_secret(slug, value)
        if success and verify_secret(slug):
            print(f"    ✓  {slug}")
            stored.append(slug)
        else:
            print(f"    ✗  {slug}  [FAILED]")
            failed.append(slug)

    print(f"\n  Stored: {len(stored)}   Failed: {len(failed)}\n")

    if failed:
        print("  [!] Failed slugs — please store these manually:")
        for s in failed:
            print(f"      {s}")
        print()

    if not stored:
        print("  Nothing stored successfully. Exiting.\n")
        sys.exit(1)

    # ── Step 4: Collect metadata ───────────────────────────────────────────────
    print(DESCRIPTION_PROMPT)
    add_meta = input("  Add metadata for secrets.yaml? [Y/n]: ").strip().lower()

    metadata_list = []
    if add_meta != "n":
        for slug in stored:
            meta = collect_metadata(slug)
            metadata_list.append(meta)
    else:
        for slug in stored:
            metadata_list.append({
                "slug":         slug,
                "name":         slug,
                "description":  None,
                "date":         str(date.today()),
                "last_updated": str(date.today()),
                "expires":      None,
            })

    # ── Step 5: Write secrets.yaml ────────────────────────────────────────────
    output_path = input("\n  Output path for secrets.yaml [./secrets.yaml]: ").strip()
    if not output_path:
        output_path = "./secrets.yaml"

    write_secrets_yaml(metadata_list, output_path)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    clear_screen()
    print("""
╔══════════════════════════════════════════════════════════╗
║                    Setup Complete                        ║
╚══════════════════════════════════════════════════════════╝
""")
    print(f"  Secrets stored in OS vault : {len(stored)}")
    print(f"  secrets.yaml written to    : {output_path}")
    print(f"\n  The secrets.yaml contains slug references only.")
    print(f"  It is safe to commit to git.\n")
    print(f"  To verify a stored secret without printing its value:")
    print(f"    python -c \"import keyring; print('Found' if keyring.get_password('layman-secrets', 'YOUR_SLUG') else 'Missing')\"\n")

    if failed:
        print(f"  [!] {len(failed)} secret(s) failed — store them manually.\n")

    print("  Done.\n")


if __name__ == "__main__":
    main()
