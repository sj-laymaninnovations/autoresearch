"""
autoresearch_canonical.py — Experiment runner for canonical curriculum training.

Takes a YAML/JSON experiment config, launches training with those hyperparameters,
runs eval_canonical.py on the best checkpoint, appends a single row to the
global experiment log (`canonical_experiments.tsv`).

Designed for velocity: each experiment is ~15 min, so we can run dozens per day
and compare results across hypotheses (reverse curriculum, answer-only, transfer
from arithmetic, etc.).

Config shape (YAML or JSON, keys are optional except `name`):
    name: reverse_curriculum
    notes: "Train Stage 5 first, then 4..1"
    # Data
    train_data: math_lab/results/canonical_algebra_train_v1_<ts>.jsonl
    val_data:   math_lab/results/canonical_algebra_val_v1_<ts>.jsonl
    # Init (optional — start from another checkpoint)
    init_ckpt: math_lab/results/checkpoints/algebra_canonical/run_001/best.pt
    # Model architecture
    n_layer: 6
    n_head: 8
    n_embd: 256
    seq_len: 256
    # Training
    epochs: 1000
    batch: 16
    lr: 6e-4
    lr_min: 1e-5
    weight_decay: 1.0
    dropout: 0.0
    label_smoothing: 0.0
    warmup: 50
    restart_period: 0
    val_every: 10
    save_every: 100
    num_workers: 0

Usage:
    python math_lab/autoresearch_canonical.py --config exp/baseline.yaml
    python math_lab/autoresearch_canonical.py --config '{"name":"test",...}'  # inline
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

ROOT         = Path(__file__).parent.parent
MATH_LAB     = Path(__file__).parent
RESULTS      = MATH_LAB / "results"
LOGS_DIR     = RESULTS / "logs"
EXP_DIR      = RESULTS / "experiments"
CKPT_BASE    = RESULTS / "checkpoints" / "algebra_canonical"
LOG_TSV      = RESULTS / "canonical_experiments.tsv"
FINETUNE     = MATH_LAB / "finetune.py"
EVAL_SCRIPT  = MATH_LAB / "eval_canonical.py"

DEFAULT_PYTHON = "/Users/seanjosiah/.lmstudio/extensions/backends/vendor/_amphibian/cpython3.11-mac-arm64@10/bin/python3.11"
# System Python has matplotlib (LM Studio Python's is blocked by dylib signing)
PLOT_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
PLOT_SCRIPT = MATH_LAB / "make_run_plot.py"
PLOTS_DIR   = RESULTS / "plots"


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    notes            = "",
    init_ckpt        = None,
    seed             = 42,
    n_layer          = 6,
    n_head           = 8,
    n_embd           = 256,
    seq_len          = 256,
    epochs           = 1000,
    batch            = 16,
    lr               = 6e-4,
    lr_min           = 1e-5,
    weight_decay     = 1.0,
    dropout          = 0.0,
    label_smoothing  = 0.0,
    warmup           = 50,
    restart_period   = 0,
    val_every        = 10,
    save_every       = 100,
    num_workers      = 0,
    em_every         = 100,   # probe exact-match every N epochs, abort on phantom
    em_sample        = 16,
    python           = DEFAULT_PYTHON,
    max_new_tokens   = 120,
)


def load_config(arg: str) -> dict:
    """Load config from YAML/JSON file or inline JSON string."""
    if arg.strip().startswith("{"):
        cfg = json.loads(arg)
    else:
        p = Path(arg)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
                cfg = yaml.safe_load(text)
            except ImportError:
                # Fallback: simple key: value parser
                cfg = parse_simple_yaml(text)
        else:
            cfg = json.loads(text)
    if "name" not in cfg:
        raise ValueError("config must have a 'name' field")
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return merged


def parse_simple_yaml(text: str) -> dict:
    """Tiny YAML-ish parser for configs without PyYAML installed."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v == "" or v.lower() == "null":
            out[k] = None
        elif v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def build_cmd(cfg: dict, ckpt_dir: Path, log_path: Path) -> list[str]:
    cmd = [
        cfg["python"], str(FINETUNE),
        "--data",           cfg["train_data"],
        "--val-data",       cfg["val_data"],
        "--ckpt-dir",       str(ckpt_dir),
        "--n-layer",        str(cfg["n_layer"]),
        "--n-head",         str(cfg["n_head"]),
        "--n-embd",         str(cfg["n_embd"]),
        "--seq-len",        str(cfg["seq_len"]),
        "--epochs",         str(cfg["epochs"]),
        "--batch",          str(cfg["batch"]),
        "--lr",             str(cfg["lr"]),
        "--lr-min",         str(cfg["lr_min"]),
        "--weight-decay",   str(cfg["weight_decay"]),
        "--dropout",        str(cfg["dropout"]),
        "--label-smoothing", str(cfg["label_smoothing"]),
        "--warmup",         str(cfg["warmup"]),
        "--restart-period", str(cfg["restart_period"]),
        "--val-every",      str(cfg["val_every"]),
        "--save-every",     str(cfg["save_every"]),
        "--num-workers",    str(cfg["num_workers"]),
        "--em-every",       str(cfg.get("em_every", 100)),
        "--em-sample",      str(cfg.get("em_sample", 16)),
    ]
    if cfg.get("device"):
        cmd += ["--device", cfg["device"]]
    if cfg.get("init_ckpt"):
        cmd += ["--ckpt", cfg["init_ckpt"]]
        # Default to resetting epoch counter when init_ckpt is set (warm-start semantics).
        # Explicit `reset_epoch_counter: false` in YAML preserves old resume behavior.
        if cfg.get("reset_epoch_counter", True):
            cmd += ["--reset-epoch-counter"]
    if cfg.get("init_embeddings"):
        cmd += ["--init-embeddings", cfg["init_embeddings"]]
        cmd += ["--init-embeddings-labels", cfg["init_embeddings_labels"]]
        if cfg.get("init_embeddings_scale") is not None:
            cmd += ["--init-embeddings-scale", str(cfg["init_embeddings_scale"])]
    return cmd


def run_training(cfg: dict, run_dir: Path, max_retries: int = 5) -> tuple[int, float]:
    """Run finetune.py subprocess with retry-on-phantom.
    finetune.py exits with code 2 on NaN/Inf/out-of-range/phantom losses.
    On rc=2 we retry with a VERY different seed to escape MPS bad-init basins.
    Returns (returncode, wall_seconds_total)."""
    log_path = run_dir / "train.log"
    total = 0.0
    seed = cfg.get("seed", 42)
    # Large, well-spaced seed offsets — small +1 bumps weren't breaking out of
    # recurring MPS bad-init basins.
    retry_offsets = [0, 101, 503, 1009, 2017, 4001, 8009]

    for attempt in range(max_retries):
        cfg_try = dict(cfg, seed=seed + retry_offsets[attempt])
        cmd = build_cmd(cfg_try, run_dir, log_path)
        if "--seed" not in cmd:
            cmd += ["--seed", str(cfg_try["seed"])]
        if attempt > 0:
            print(f"  [train] retry {attempt} with seed={cfg_try['seed']} (previous NaN)")
        else:
            print(f"  [train] launching: {' '.join(cmd[:6])} ...")

        t0 = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        mode = "w" if attempt == 0 else "a"
        with open(log_path, mode) as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        dur = time.time() - t0
        total += dur
        print(f"  [train] attempt {attempt} done in {dur:.0f}s, rc={proc.returncode}")

        if proc.returncode != 2:
            # success or non-NaN failure; stop retrying
            return proc.returncode, total

    # All MPS retries exhausted — fall back to CPU (slow but reliable)
    print(f"  [train] ALL {max_retries} MPS retries hit phantom — falling back to CPU")
    cfg_cpu = dict(cfg, seed=seed, device="cpu")
    cmd = build_cmd(cfg_cpu, run_dir, log_path)
    if "--seed" not in cmd:
        cmd += ["--seed", str(cfg_cpu["seed"])]
    cmd += ["--device", "cpu"]
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "a") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dur = time.time() - t0
    total += dur
    print(f"  [train] CPU fallback done in {dur:.0f}s, rc={proc.returncode}")
    return proc.returncode, total


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(ckpt_path: Path, val_path: Path, max_new: int = 120) -> dict:
    """Run eval_canonical.py programmatically (same process, faster than subprocess)."""
    sys.path.insert(0, str(MATH_LAB))
    from eval_canonical import evaluate
    return evaluate(ckpt_path, val_path, max_new=max_new, verbose=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

TSV_HEADER = (
    "timestamp\tname\ttrain_s\tckpt_epoch\tckpt_val_loss\t"
    "exact_match\tstage1\tstage2\tstage3\tstage4\tstage5\tfailures\tnotes\n"
)


def append_tsv(row: dict):
    if not LOG_TSV.exists():
        LOG_TSV.write_text(TSV_HEADER)
    line = "\t".join([
        row["timestamp"],
        row["name"],
        f"{row['train_s']:.0f}",
        str(row.get("ckpt_epoch", -1)),
        f"{row.get('ckpt_val_loss', 0):.4f}" if row.get("ckpt_val_loss") is not None else "",
        f"{row['exact_match']:.4f}",
        row.get("stage1", ""),
        row.get("stage2", ""),
        row.get("stage3", ""),
        row.get("stage4", ""),
        row.get("stage5", ""),
        ",".join(row.get("failures", []))[:200],
        row.get("notes", "").replace("\t", " "),
    ])
    with open(LOG_TSV, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to YAML/JSON config file, or inline JSON string")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"\n{'='*70}")
    print(f"  Experiment: {cfg['name']}")
    print(f"  Notes: {cfg.get('notes','(none)')}")
    print(f"{'='*70}")

    # Required fields
    for k in ("train_data", "val_data"):
        if not cfg.get(k):
            raise ValueError(f"config missing required field: {k}")

    # Run directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CKPT_BASE / f"{ts}_{cfg['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    if args.dry_run:
        print("DRY RUN:")
        print(" ".join(build_cmd(cfg, run_dir, run_dir / "train.log")))
        return

    # Train
    rc, train_s = run_training(cfg, run_dir)
    if rc != 0:
        print(f"  [TRAIN FAILED] rc={rc} — see {run_dir/'train.log'}")
        append_tsv(dict(timestamp=ts, name=cfg["name"], train_s=train_s,
                        exact_match=0.0, notes=f"TRAIN_FAILED rc={rc}"))
        return

    # Eval preference: best_em.pt (exact-match winner) > best.pt (val-loss winner) > final.pt
    best_em = run_dir / "best_em.pt"
    best    = run_dir / "best.pt"
    final   = run_dir / "final.pt"
    if best_em.exists():
        eval_target = best_em
    elif best.exists():
        eval_target = best
    else:
        eval_target = final
    if not eval_target.exists():
        print(f"  [NO CHECKPOINT] neither best.pt nor final.pt found")
        append_tsv(dict(timestamp=ts, name=cfg["name"], train_s=train_s,
                        exact_match=0.0, notes="NO_CKPT"))
        return

    print(f"\n  [eval] running on {eval_target.name}...")
    res = run_eval(eval_target, Path(cfg["val_data"]),
                   max_new=cfg.get("max_new_tokens", 120))

    # Save full eval JSON
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))

    # Log row
    stage_str = lambda s: f"{res['by_stage'].get(str(s),{}).get('pass',0)}/{res['by_stage'].get(str(s),{}).get('pass',0)+res['by_stage'].get(str(s),{}).get('fail',0)}"
    append_tsv(dict(
        timestamp    = ts,
        name         = cfg["name"],
        train_s      = train_s,
        ckpt_epoch   = res.get("ckpt_epoch", -1),
        ckpt_val_loss = res.get("ckpt_val"),
        exact_match  = res["exact_match"],
        stage1       = stage_str(1),
        stage2       = stage_str(2),
        stage3       = stage_str(3),
        stage4       = stage_str(4),
        stage5       = stage_str(5),
        failures     = res["failures"],
        notes        = cfg.get("notes", ""),
    ))

    # Generate per-run visualization
    try:
        plot_out = PLOTS_DIR / f"{run_dir.name}.png"
        plot_rc = subprocess.run(
            [PLOT_PYTHON, str(PLOT_SCRIPT), str(run_dir), "--out", str(plot_out)],
            capture_output=True, text=True, timeout=60)
        if plot_rc.returncode == 0:
            print(f"  [plot] {plot_out}")
        else:
            print(f"  [plot WARNING] {plot_rc.stderr.strip().splitlines()[-1] if plot_rc.stderr else 'unknown error'}")
    except Exception as e:
        print(f"  [plot WARNING] {e}")

    print(f"\n  [done] logged to {LOG_TSV}")


if __name__ == "__main__":
    main()
