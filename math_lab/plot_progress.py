"""
plot_progress.py — Parse finetune logs and render training progress charts.

Parses one or more finetune_<domain>.log files, extracts per-epoch train/val
loss and durations, and renders a PNG chart showing:
  - Train loss over epochs
  - Val loss points overlaid
  - Per-epoch duration (secondary axis)
  - Best-so-far val line

Usage:
    python math_lab/plot_progress.py                           # all domain logs
    python math_lab/plot_progress.py --domain algebra
    python math_lab/plot_progress.py --log path/to/file.log    # explicit file
"""

import re
import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

LOG_DIR  = Path(__file__).parent / "results" / "logs"
PLOT_DIR = Path(__file__).parent / "results" / "plots"

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"epoch\s+(\d+)/\d+\s+"
    r"train=([\d.]+)\s+"
    r"val=([\d.]+|skip)\s+"
    r"lr=([\d.eE+-]+)\s+"
    r"([\d.]+)s"
)


def parse_log(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        ts_str, epoch, train, val, lr, dur = m.groups()
        rows.append({
            "ts":       datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
            "epoch":    int(epoch),
            "train":    float(train),
            "val":      None if val == "skip" else float(val),
            "lr":       float(lr),
            "duration": float(dur),
        })
    return rows


def render(rows, domain: str, out_path: Path):
    if not rows:
        print(f"[{domain}] no rows parsed, skipping")
        return

    epochs = [r["epoch"] for r in rows]
    train  = [r["train"] for r in rows]
    dur    = [r["duration"] for r in rows]
    val_e  = [r["epoch"] for r in rows if r["val"] is not None]
    val_v  = [r["val"]   for r in rows if r["val"] is not None]

    # Best-so-far line over val points
    best_so_far = []
    cur_best = float("inf")
    for v in val_v:
        cur_best = min(cur_best, v)
        best_so_far.append(cur_best)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, train, color="#1f77b4", linewidth=1.2,
             alpha=0.8, label="train loss")
    if val_e:
        ax1.plot(val_e, val_v, "o", color="#d62728",
                 markersize=7, label="val loss")
        ax1.plot(val_e, best_so_far, "--", color="#d62728",
                 alpha=0.5, linewidth=1, label="best val so far")

    # Label-smoothing floor (vocab=131, smoothing=0.1)
    import math as _m
    floor = 0.1 * _m.log(131)
    ax1.axhline(floor, color="gray", linestyle=":", linewidth=1,
                alpha=0.6, label=f"label-smoothing floor ({floor:.2f})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"MathGPT — {domain} training  "
                  f"(current epoch {epochs[-1]}, "
                  f"{len(epochs)} logged, "
                  f"latest train={train[-1]:.4f})")

    ax2 = ax1.twinx()
    ax2.plot(epochs, dur, color="#2ca02c", linewidth=0.8, alpha=0.3)
    ax2.set_ylabel("sec/epoch", color="#2ca02c", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    # Clip duration axis to reasonable range
    med_dur = sorted(dur)[len(dur) // 2]
    ax2.set_ylim(0, med_dur * 3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labs1, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[{domain}] {len(epochs)} epochs parsed, {len(val_e)} val points  "
          f"-> {out_path}")


def print_table(rows, domain: str, n_recent: int = 8):
    """Print a terminal-friendly progress table."""
    if not rows:
        return
    val_rows = [r for r in rows if r["val"] is not None]
    # Interesting rows: first, every 25th, val points, last few
    sample = set()
    if rows:
        sample.add(id(rows[0]))
        sample.add(id(rows[-1]))
    for r in rows:
        if r["epoch"] % 25 == 0 or r["val"] is not None:
            sample.add(id(r))
    for r in rows[-n_recent:]:
        sample.add(id(r))

    selected = [r for r in rows if id(r) in sample]
    # Compute best val and trend markers
    best_val = float("inf")
    prev_train = None
    print(f"\n{'Timestamp':<20} {'Epoch':>5}  {'Val Loss':>8}  "
          f"{'Train':>7}  {'Dur':>6}  {'Best':>5}  {'Trend':<6}  Notes")
    print("-" * 90)
    for r in selected:
        ts = r["ts"].strftime("%m-%d %H:%M:%S")
        val_str = f"{r['val']:.4f}" if r["val"] is not None else "  skip"
        is_best = r["val"] is not None and r["val"] < best_val
        if r["val"] is not None:
            best_val = min(best_val, r["val"])
        best_mark = "  ✓" if is_best else ""
        if prev_train is None:
            trend = "start"
        else:
            delta = r["train"] - prev_train
            trend = "↓" if delta < -0.005 else ("↑" if delta > 0.005 else "→")
        prev_train = r["train"]
        note = ""
        if r["duration"] > 100:
            note = f"HICCUP {r['duration']:.0f}s"
        print(f"{ts:<20} {r['epoch']:>5}  {val_str:>8}  "
              f"{r['train']:>7.4f}  {r['duration']:>5.1f}s  "
              f"{best_mark:>5}  {trend:<6} {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--log",    default=None)
    args = ap.parse_args()

    if args.log:
        logs = [Path(args.log)]
    elif args.domain:
        logs = [LOG_DIR / f"finetune_{args.domain}.log"]
    else:
        logs = sorted(LOG_DIR.glob("finetune_*.log"))

    for log_path in logs:
        if not log_path.exists():
            print(f"skip (missing): {log_path}")
            continue
        domain = log_path.stem.replace("finetune_", "")
        rows = parse_log(log_path)
        print_table(rows, domain)
        render(rows, domain, PLOT_DIR / f"progress_{domain}.png")


if __name__ == "__main__":
    main()
