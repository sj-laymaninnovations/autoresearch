"""
config.py — paths, env, cloud-sync detection.

The "memory follows me" property requires a path that's identical on every
device the user works on. We look for cloud-sync mounts in this order
(first match wins). Override with MEMORY_PROXY_PATH env var.

Search order:
  1. $MEMORY_PROXY_PATH                            (explicit override)
  2. ~/Library/CloudStorage/OneDrive*               (macOS, modern)
  3. ~/Library/CloudStorage/Box-Box                 (macOS)
  4. ~/Library/CloudStorage/GoogleDrive-*           (macOS)
  5. ~/Library/Mobile Documents/com~apple~CloudDocs (macOS iCloud Drive)
  6. ~/OneDrive                                    (Windows / Linux)
  7. ~/Box                                         (Windows / Linux)
  8. ~/Google Drive                                (Windows / Linux)
  9. ~/Dropbox
 10. ~/.autoresearch/memory_proxy                  (local fallback — does NOT sync)

Inside the chosen base, the proxy creates an `autoresearch_memory/` subdir
with the actual data files. Everything is human-readable JSONL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


CLOUD_CANDIDATES = [
    # macOS modern locations
    "~/Library/CloudStorage/OneDrive-Personal",
    "~/Library/CloudStorage/Box-Box",
    "~/Library/Mobile Documents/com~apple~CloudDocs",
    # Glob-style additionals (resolved at detect time)
    "~/Library/CloudStorage/OneDrive-*",
    "~/Library/CloudStorage/GoogleDrive-*",
    # Cross-platform homedir mounts
    "~/OneDrive",
    "~/Box",
    "~/Google Drive",
    "~/Dropbox",
    # iCloud Drive on Windows
    "~/iCloudDrive",
]

LOCAL_FALLBACK = "~/.autoresearch/memory_proxy"

MEMORY_DIR_NAME = "autoresearch_memory"


def detect_cloud_root() -> tuple[Path, str]:
    """Return (path, label). label says how it was chosen for logging."""
    override = os.environ.get("MEMORY_PROXY_PATH")
    if override:
        p = Path(override).expanduser()
        return p, f"env:MEMORY_PROXY_PATH={override}"

    home = Path("~").expanduser()
    for cand in CLOUD_CANDIDATES:
        p_str = os.path.expanduser(cand)
        # Glob support
        if any(ch in p_str for ch in "*?["):
            import glob
            matches = sorted(glob.glob(p_str))
            if matches:
                return Path(matches[0]), f"detected:{matches[0]}"
        else:
            p = Path(p_str)
            if p.exists() and p.is_dir():
                return p, f"detected:{p_str}"

    # Last resort: local-only, no sync
    p = Path(LOCAL_FALLBACK).expanduser()
    return p, "local-fallback (NOT cloud-synced)"


def memory_root() -> Path:
    """Resolved memory directory. Created on first call."""
    base, _ = detect_cloud_root()
    root = base / MEMORY_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def describe_root() -> str:
    """Human-readable summary used at startup."""
    base, label = detect_cloud_root()
    root = base / MEMORY_DIR_NAME
    return f"memory at {root}  ({label})"


# ---------------------------------------------------------------------------
# Backend / provider keys — read from .env at startup
# ---------------------------------------------------------------------------

def env_or_dotenv(*var_names: str) -> Optional[str]:
    """Look up a value in os.environ first, then in math_lab/.env."""
    for n in var_names:
        v = os.environ.get(n)
        if v:
            return v
    # Fallback to math_lab/.env
    here = Path(__file__).parent
    for envp in (here.parent / ".env", here.parent.parent / ".env"):
        if not envp.exists():
            continue
        for line in envp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v and (v.startswith('"') and v.endswith('"') or
                      v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if k.strip() in var_names and v:
                return v
    return None
