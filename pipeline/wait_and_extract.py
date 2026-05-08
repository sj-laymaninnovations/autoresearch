"""
wait_and_extract.py — Post-enrichment watcher.

Polls the enrichment output directory. Once all 99 shards are complete,
runs extract_prehistoric.py to build the training subset, then updates
the Windows node comms.yaml with a pending directive.

Usage:
    python pipeline/wait_and_extract.py
    python pipeline/wait_and_extract.py --expected 99 --poll 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
ENRICHED_DIR  = BASE_DIR / "sources_enriched" / "fineweb-edu"
COMMAND_HUB   = Path.home() / "Documents" / "autoresearch" / "command-hub"
COMMS_PATH    = COMMAND_HUB / "projects" / "autoresearch" / "agents" / "node-windows" / "comms.yaml"
EXTRACT_SCRIPT = Path(__file__).parent / "extract_prehistoric.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def count_enriched_shards() -> int:
    return len(list(ENRICHED_DIR.glob("data-*.parquet")))


def run_extraction() -> bool:
    print(f"\n[wait_and_extract] All shards complete — running extraction …")
    result = subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), "--dataset", "fineweb-edu"],
        check=False,
    )
    return result.returncode == 0


def write_comms_directive() -> None:
    """Append a pending directive to the Windows node comms.yaml."""
    ts = now_iso()
    directive_block = f"""
  - name: train_prehistoric_30k_10k_bce
    status: pending
    priority: high
    task: "Run train_prehistoric_3080.bat on the RTX 3080"
    description: >
      Prehistoric-era (30,000–10,000 BCE) training subset has been
      extracted from the enriched FineWeb-Edu shards and is ready.
      Run the training bat file to kick off Config A on the 3080.
    instructions: |
      1. Confirm sources_prehistoric/fineweb-edu/ is synced to
         F:\\projects\\Data Triage\\sources_prehistoric\\fineweb-edu\\
      2. Open CMD or PowerShell in F:\\projects\\autoresearch\\
      3. Run: train_prehistoric_3080.bat
      4. Monitor: training\\logs\\prehistoric_config_a\\train.stdout.log
      5. Set status: acknowledged below when started.
    created_at: "{ts}"
"""
    if not COMMS_PATH.exists():
        print(f"  WARNING: comms.yaml not found at {COMMS_PATH}")
        print(f"  Directive that would have been written:\n{directive_block}")
        return

    content = COMMS_PATH.read_text(encoding="utf-8")

    # Remove the empty directives: [] placeholder if present
    content = content.replace("directives: []", "directives:")

    # Append directive if not already present
    if "train_prehistoric_30k_10k_bce" not in content:
        if "directives:" not in content:
            content = content.rstrip() + "\ndirectives:\n"
        content = content.rstrip() + "\n" + directive_block + "\n"
        COMMS_PATH.write_text(content, encoding="utf-8")
        print(f"  ✅ Directive written to {COMMS_PATH}")

        # Git commit
        try:
            import subprocess as sp
            sp.run(["git", "add", str(COMMS_PATH.relative_to(COMMAND_HUB))],
                   cwd=COMMAND_HUB, capture_output=True, timeout=10)
            sp.run(["git", "commit", "-m",
                    f"comms: train_prehistoric_30k_10k_bce directive → node-windows [{ts}]"],
                   cwd=COMMAND_HUB, capture_output=True, timeout=10)
        except Exception:
            pass
    else:
        print("  Directive already present in comms.yaml — skipping.")


def main():
    ap = argparse.ArgumentParser(description="Wait for enrichment completion then extract prehistoric data")
    ap.add_argument("--expected", type=int, default=99,
                    help="Number of enriched shards to wait for (default: 99)")
    ap.add_argument("--poll",     type=int, default=60,
                    help="Poll interval in seconds (default: 60)")
    ap.add_argument("--skip-wait", action="store_true",
                    help="Skip polling and run extraction immediately")
    args = ap.parse_args()

    if not args.skip_wait:
        print(f"[wait_and_extract] Watching {ENRICHED_DIR}")
        print(f"  Waiting for {args.expected} shards  (polling every {args.poll}s)")
        print(f"  Ctrl+C to abort\n")

        while True:
            n = count_enriched_shards()
            ts = now_iso()
            print(f"  [{ts}] {n}/{args.expected} shards complete", end="\r", flush=True)
            if n >= args.expected:
                print(f"\n  [{ts}] ✅ All {n} shards found!")
                break
            time.sleep(args.poll)

    ok = run_extraction()
    if not ok:
        print("\n  ❌ Extraction failed — check output above. Comms directive NOT written.")
        sys.exit(1)

    write_comms_directive()
    print(f"\n[wait_and_extract] Done. Windows node will see the directive on next comms poll.")


if __name__ == "__main__":
    main()
