"""
make_run_plot.py — Generate per-experiment visualization from a run directory.

Reads the run's train.log (loss trajectory + EM probes) and eval.json (final
per-stage breakdown). Produces a two-panel chart:

  Top:    loss curves (train + val) + EM probe points on secondary axis
  Bottom: final per-stage exact-match bars

Filename encodes the run's timestamp so plots for each experiment are
distinguishable and sortable.

Usage:
    python math_lab/make_run_plot.py <run_dir>
    python math_lab/make_run_plot.py math_lab/results/checkpoints/algebra_canonical/20260423_020000_baseline/
"""

import re
import sys
import json
import math
import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"epoch\s+(\d+)/\d+\s+"
    r"train=(-?[\d.e+-]+|nan)\s+"
    r"val=(-?[\d.e+-]+|skip|nan)\s+"
    r"lr=(-?[\d.eE+-]+)\s+"
    r"(-?[\d.]+)s"
)
EM_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"exact_match probe:\s+([\d.]+)%"
)


def parse_train_log(path: Path):
    epochs = []
    em_points = []
    # We also need to correlate EM lines with epochs. Each EM line immediately
    # follows an "epoch" line in the log, so we track last-seen-epoch.
    last_epoch = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ts_str, epoch, train, val, lr, dur = m.groups()
            try:
                tr = float(train)
                vl = None if val == "skip" else float(val)
                lrv = float(lr)
                dr = float(dur)
                epochs.append({
                    "ts":       datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
                    "epoch":    int(epoch),
                    "train":    tr,
                    "val":      vl,
                    "lr":       lrv,
                    "duration": dr,
                })
                last_epoch = int(epoch)
            except ValueError:
                pass
        else:
            m2 = EM_RE.search(line)
            if m2 and last_epoch is not None:
                em_points.append({
                    "epoch": last_epoch,
                    "em":    float(m2.group(2)) / 100.0,
                })
    return epochs, em_points


def read_config(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def read_eval(run_dir: Path):
    ev = run_dir / "eval.json"
    if ev.exists():
        return json.loads(ev.read_text())
    return None


def render(run_dir: Path, out_path: Path):
    log = run_dir / "train.log"
    if not log.exists():
        print(f"no train.log in {run_dir}")
        return
    epochs, em_points = parse_train_log(log)
    cfg = read_config(run_dir)
    ev = read_eval(run_dir)
    if not epochs:
        print(f"no epoch lines parsed in {log}")
        return

    name = cfg.get("name", run_dir.name)
    notes = cfg.get("notes", "")
    n_params = cfg.get("n_layer", "?"), cfg.get("n_embd", "?")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # ---- Top panel: loss + EM ----
    ax1 = fig.add_subplot(gs[0, 0])
    xs = [e["epoch"] for e in epochs]
    tr = [e["train"] for e in epochs]
    ax1.plot(xs, tr, color="#1f77b4", linewidth=1.2, alpha=0.8, label="train loss")

    val_e = [e["epoch"] for e in epochs if e["val"] is not None]
    val_v = [e["val"]   for e in epochs if e["val"] is not None]
    if val_e:
        ax1.plot(val_e, val_v, "o-", color="#d62728", linewidth=1,
                 markersize=4, alpha=0.8, label="val loss")

    # Label-smoothing floor (assume 0.1 + vocab 131)
    ls_floor = 0.1 * math.log(131)   # ~0.486 (that's the lower sanity bound)
    min_ce = -( (1 - 0.1 + 0.1/131) * math.log(1 - 0.1 + 0.1/131)
                + (131-1) * (0.1/131) * math.log(0.1/131) )
    ax1.axhline(min_ce, color="gray", linestyle=":", linewidth=1,
                alpha=0.5, label=f"CE floor ({min_ce:.2f})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (cross-entropy)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # EM probe on secondary axis
    ax2 = ax1.twinx()
    if em_points:
        em_e = [p["epoch"] for p in em_points]
        em_v = [p["em"] * 100 for p in em_points]
        ax2.plot(em_e, em_v, "s-", color="#2ca02c", markersize=6,
                 linewidth=1.5, alpha=0.85, label="exact-match probe (%)")
        # Annotate peak
        peak_i = max(range(len(em_v)), key=lambda i: em_v[i])
        if em_v[peak_i] > 0:
            ax2.annotate(f"peak {em_v[peak_i]:.1f}% @ ep {em_e[peak_i]}",
                         xy=(em_e[peak_i], em_v[peak_i]),
                         xytext=(10, 10), textcoords="offset points",
                         fontsize=9, color="#2ca02c",
                         arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax2.set_ylabel("Exact-match (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, 100)

    # Merged legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
               framealpha=0.9, fontsize=9)

    title = f"{name}"
    subtitle_parts = []
    if cfg:
        subtitle_parts.append(
            f"n_layer={cfg.get('n_layer','?')} "
            f"n_embd={cfg.get('n_embd','?')} "
            f"epochs={cfg.get('epochs','?')} "
            f"lr={cfg.get('lr','?')} "
            f"wd={cfg.get('weight_decay','?')}"
        )
    if ev:
        subtitle_parts.append(
            f"final EM: {ev['exact_match']*100:.1f}% ({ev['pass']}/{ev['total']}) "
            f"@ ep {ev.get('ckpt_epoch','?')}"
        )
    if notes:
        subtitle_parts.append(notes)
    ax1.set_title(title + "\n" + "  |  ".join(subtitle_parts),
                  fontsize=10, loc="left")

    # ---- Bottom panel: per-stage bars ----
    ax3 = fig.add_subplot(gs[1, 0])
    if ev and "by_stage" in ev:
        stages = sorted(ev["by_stage"].keys(), key=lambda x: int(x))
        pct = [100 * ev["by_stage"][s]["pass"] /
               max(1, ev["by_stage"][s]["pass"] + ev["by_stage"][s]["fail"])
               for s in stages]
        labels = [f"Stage {s}\n{ev['by_stage'][s]['pass']}/{ev['by_stage'][s]['pass']+ev['by_stage'][s]['fail']}"
                  for s in stages]
        bars = ax3.bar(range(len(stages)), pct, color="#2ca02c", alpha=0.8)
        ax3.set_xticks(range(len(stages)))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Pass rate (%)")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.set_title("Per-stage exact-match", fontsize=10, loc="left")
        for b, p in zip(bars, pct):
            if p > 0:
                ax3.text(b.get_x() + b.get_width()/2, p + 2, f"{p:.0f}%",
                         ha="center", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "no eval.json found",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="gray", fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to experiment run directory")
    ap.add_argument("--out", default=None, help="Output PNG path (default: plots/<rundir-basename>.png)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    if args.out:
        out = Path(args.out)
    else:
        # run_dir = results/checkpoints/<domain>/<ts_name>/
        # plots live at results/plots/<ts_name>.png
        plots_dir = run_dir.parent.parent.parent / "plots"
        out = plots_dir / f"{run_dir.name}.png"

    render(run_dir, out)


if __name__ == "__main__":
    main()
