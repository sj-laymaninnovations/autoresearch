"""
train_all.py — Parallel MathGPT Domain Expert Trainer

Discovers data for each math domain, creates per-domain merged data files,
and trains one MathGPT expert per domain in parallel using subprocesses.

Uses n_layer=6 by default (4 layers proved too small for low loss).

Usage:
    python math_lab/train_all.py                         # all domains, 4 parallel
    python math_lab/train_all.py --domains arithmetic,algebra --max-parallel 2
    python math_lab/train_all.py --epochs 500 --n-layer 6 --n-embd 256
    python math_lab/train_all.py --dry-run               # show plan without training

Memory note: Each finetune.py process allocates ~300-600MB of MPS/Metal memory.
Default max-parallel=4 is conservative. Increase if Activity Monitor shows headroom.

Output:
    results/domain_data/{domain}_merged.jsonl   — merged training data per domain
    results/checkpoints/{domain}/best.pt        — best checkpoint per domain
    results/train_all_summary_{timestamp}.json  — run summary
"""

import os
import sys
import json
import time
import signal
import argparse
import datetime
import subprocess
from pathlib import Path
from threading import Thread

ROOT       = Path(__file__).parent.parent
MATH_LAB   = Path(__file__).parent
RESULTS    = MATH_LAB / "results"
DOMAIN_DATA_DIR = RESULTS / "domain_data"
CKPT_BASE  = RESULTS / "checkpoints"
FINETUNE   = MATH_LAB / "finetune.py"

# Canonical domain list — ordered by estimated difficulty/data richness
ALL_DOMAINS = [
    "arithmetic",
    "algebra",
    "calculus",
    "geometry",
    "trig",
    "number_theory",
    "combinatorics",
    "diffeq",
    "linalg",
    "linreg",
    "stats",
    "stochastic",
]

# Source values to accept per domain. Records whose "source" field is NOT in
# this set are excluded from the merge. None means accept all sources.
# This filters out broken harder_qa.py generate_chain_of_thought() records.
DOMAIN_SOURCE_FILTER: dict[str, set | None] = {
    "arithmetic":    None,   # all sources fine (arithmetic CoT is simple enough)
    "algebra":       {"algebra_qa", "claude_qa"},  # exclude broken harder_qa generated
    "calculus":      None,
    "geometry":      None,
    "trig":          None,
    "number_theory": None,
    "combinatorics": None,
    "diffeq":        None,
    "linalg":        None,
    "linreg":        None,
    "stats":         None,
    "stochastic":    None,
}

# File glob patterns for each domain (matches both local and claude-generated data)
DOMAIN_GLOBS = {
    "arithmetic":   ["harder_qa_arithmetic*.jsonl", "harder_qa_arith*.jsonl", "claude_qa_arithmetic*.jsonl"],
    "algebra":      ["harder_qa_algebra*.jsonl", "claude_qa_algebra*.jsonl"],
    "calculus":     ["harder_qa_calculus*.jsonl", "claude_qa_calculus*.jsonl"],
    "geometry":     ["harder_qa_geometry*.jsonl", "claude_qa_geometry*.jsonl"],
    "trig":         ["harder_qa_trig*.jsonl", "claude_qa_trig*.jsonl", "claude_qa_trigonometry*.jsonl"],
    "number_theory":["harder_qa_number_theory*.jsonl", "claude_qa_number_theory*.jsonl"],
    "combinatorics":["harder_qa_combinatorics*.jsonl", "claude_qa_combinatorics*.jsonl"],
    "diffeq":       ["harder_qa_diffeq*.jsonl", "claude_qa_diffeq*.jsonl"],
    "linalg":       ["harder_qa_linalg*.jsonl", "claude_qa_linalg*.jsonl"],
    "linreg":       ["harder_qa_linreg*.jsonl", "claude_qa_linreg*.jsonl"],
    "stats":        ["harder_qa_stats*.jsonl", "claude_qa_stats*.jsonl", "claude_qa_statistics*.jsonl"],
    "stochastic":   ["harder_qa_stochastic*.jsonl", "claude_qa_stochastic*.jsonl"],
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def collect_files(domain: str) -> list[Path]:
    """Return all JSONL data files for this domain (no smoketests)."""
    files = set()
    for pattern in DOMAIN_GLOBS.get(domain, []):
        for f in RESULTS.glob(pattern):
            if "smoketest" not in f.name and "cot_all" not in f.name:
                files.add(f)
    return sorted(files)


def merge_domain_data(domain: str, force: bool = False) -> tuple[Path, int]:
    """Merge all JSONL files for domain into a single file. Returns (path, n_pairs)."""
    DOMAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = DOMAIN_DATA_DIR / f"{domain}_merged.jsonl"

    if merged_path.exists() and not force:
        n = sum(1 for line in merged_path.read_text().splitlines() if line.strip())
        return merged_path, n

    files = collect_files(domain)
    if not files:
        return merged_path, 0

    source_filter = DOMAIN_SOURCE_FILTER.get(domain)
    pairs = []
    seen = set()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Apply source filter if configured for this domain
                if source_filter is not None:
                    src = rec.get("source", "")
                    if src not in source_filter:
                        continue
                # Deduplicate by problem text
                key = rec.get("problem", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    pairs.append(rec)
            except json.JSONDecodeError:
                pass

    with open(merged_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return merged_path, len(pairs)


# ---------------------------------------------------------------------------
# Memory check (macOS vm_stat)
# ---------------------------------------------------------------------------

def free_memory_gb() -> float:
    """Return approximate free+inactive memory in GB using vm_stat."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        page_size = 16384  # Apple Silicon default
        free = inactive = 0
        for line in result.stdout.splitlines():
            if "Pages free" in line:
                free = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive" in line:
                inactive = int(line.split(":")[1].strip().rstrip("."))
        return (free + inactive) * page_size / (1024 ** 3)
    except Exception:
        return 4.0  # assume 4GB free if vm_stat fails


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def build_finetune_cmd(domain: str, data_path: Path, args) -> list[str]:
    ckpt_dir = CKPT_BASE / domain
    python = getattr(args, "python", None) or sys.executable
    cmd = [
        python, str(FINETUNE),
        "--data",      str(data_path),
        "--ckpt-dir",  str(ckpt_dir),
        "--n-layer",   str(args.n_layer),
        "--n-head",    str(args.n_head),
        "--n-embd",    str(args.n_embd),
        "--epochs",    str(args.epochs),
        "--batch",     str(args.batch),
        "--seq-len",   str(args.seq_len),
        "--lr",        str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--dropout",   str(args.dropout),
        "--val-every", "10",
        "--save-every", str(max(50, args.epochs // 10)),
    ]
    if args.compile:
        cmd.append("--compile")
    return cmd


class DomainJob:
    def __init__(self, domain: str, cmd: list[str], log_path: Path):
        self.domain   = domain
        self.cmd      = cmd
        self.log_path = log_path
        self.proc     = None
        self.started  = None
        self.finished = None
        self.returncode = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        self.started = time.time()
        print(f"  [+] {self.domain}: started (pid={self.proc.pid})", flush=True)

    def poll(self) -> bool:
        """Return True if finished."""
        if self.returncode is not None:
            return True
        ret = self.proc.poll()
        if ret is not None:
            self.returncode = ret
            self.finished = time.time()
            self.log_file.close()
            elapsed = self.finished - self.started
            status = "OK" if ret == 0 else f"FAILED (code {ret})"
            print(f"  [-] {self.domain}: {status} in {elapsed:.0f}s", flush=True)
            return True
        return False

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(domains: list[str], args):
    print(f"\nPreparing data for {len(domains)} domains...")
    ready = []
    for domain in domains:
        data_path, n = merge_domain_data(domain, force=args.remerge)
        if n == 0:
            print(f"  SKIP {domain}: no data found (run datagen first)")
            continue
        print(f"  {domain}: {n} pairs -> {data_path.name}")
        ready.append((domain, data_path, n))

    if not ready:
        print("No domains with data. Generate data first with claude_datagen.py")
        return {}

    if args.dry_run:
        print("\nDry run — would train:")
        for domain, data_path, n in ready:
            cmd = build_finetune_cmd(domain, data_path, args)
            print(f"  {domain}: {' '.join(cmd[2:4])} ...")
        return {}

    print(f"\nTraining {len(ready)} domain experts "
          f"(max {args.max_parallel} parallel, n_layer={args.n_layer})...\n")

    jobs: list[DomainJob] = []
    pending = list(ready)
    running: list[DomainJob] = []
    results = {}

    while pending or running:
        # Poll running jobs
        finished = [j for j in running if j.poll()]
        for j in finished:
            running.remove(j)
            results[j.domain] = {
                "returncode": j.returncode,
                "elapsed_s":  round(j.finished - j.started, 1),
                "log":        str(j.log_path),
            }

        # Launch new jobs if slots available and memory permits
        mem_gb = free_memory_gb()
        # Each job needs ~1.5GB headroom; keep 2GB reserve
        slots = min(
            args.max_parallel - len(running),
            max(0, int((mem_gb - 2.0) / 1.5)),
        )

        for _ in range(slots):
            if not pending:
                break
            domain, data_path, n = pending.pop(0)
            cmd = build_finetune_cmd(domain, data_path, args)
            log_path = RESULTS / "logs" / f"finetune_{domain}.log"
            job = DomainJob(domain, cmd, log_path)
            job.start()
            running.append(job)
            jobs.append(job)
            time.sleep(2)  # stagger launches to avoid Metal init races

        if running or pending:
            n_pending = len(pending)
            n_running = len(running)
            mem_str   = f"{mem_gb:.1f}GB free"
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] "
                  f"running={n_running}  queued={n_pending}  mem={mem_str}",
                  flush=True)
            time.sleep(30)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train one MathGPT expert per domain in parallel")
    parser.add_argument("--domains",   default="all",
                        help="Comma-separated domains or 'all'")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max concurrent finetune.py processes")
    parser.add_argument("--epochs",    type=int,   default=1000)
    parser.add_argument("--seq-len",   type=int,   default=512,
                        help="Token sequence length (512 for algebra/longer solutions)")
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--n-layer",   type=int,   default=6,
                        help="Transformer depth (6 is baseline; 4 was too small)")
    parser.add_argument("--n-head",    type=int,   default=6)
    parser.add_argument("--n-embd",    type=int,   default=192,
                        help="Embedding dimension (192 with 6 heads = 32 head dim)")
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--dropout",   type=float, default=0.15)
    parser.add_argument("--compile",   action="store_true",
                        help="torch.compile each subprocess")
    parser.add_argument("--remerge",   action="store_true",
                        help="Force re-merge data even if merged file exists")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print commands without running training")
    parser.add_argument("--python",    default=None,
                        help="Python executable to use (default: sys.executable). "
                             "Set if your torch is in a different env.")
    args = parser.parse_args()

    domains = ALL_DOMAINS if args.domains == "all" else args.domains.split(",")

    print("=" * 60)
    print("  MathGPT Domain Expert Training")
    print("=" * 60)
    print(f"  Domains  : {', '.join(domains)}")
    print(f"  n_layer  : {args.n_layer}  n_head={args.n_head}  n_embd={args.n_embd}")
    print(f"  epochs   : {args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"  parallel : {args.max_parallel}  (free mem: {free_memory_gb():.1f}GB)")
    print(f"  ckpts    : {CKPT_BASE}/<domain>/")
    print()

    # Handle Ctrl+C gracefully
    active_jobs = []
    def _sigint(sig, frame):
        print("\nInterrupted — terminating subprocesses...")
        for j in active_jobs:
            j.kill()
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint)

    t0 = time.time()
    results = run_all(domains, args)

    # Summary
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS / f"train_all_summary_{ts}.json"
    summary = {
        "timestamp": ts,
        "domains":   domains,
        "config":    vars(args),
        "results":   results,
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done. Results summary -> {summary_path}")
    ok  = sum(1 for r in results.values() if r["returncode"] == 0)
    fail = len(results) - ok
    print(f"  {ok} succeeded  {fail} failed  of {len(results)} trained")
    print(f"  Total elapsed: {summary['total_elapsed_s']:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
