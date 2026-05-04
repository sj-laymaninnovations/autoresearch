"""
train_generalist.py — Train a tiny generalist MathGPT on the curated OSS corpus.

Architecture presets (--arch):
  nano   : 2L-4H-128D   ~0.8M params  — sanity check
  micro  : 4L-4H-256D   ~3.5M params  — close to current skill models
  small  : 6L-8H-384D   ~11M  params  — likely sweet spot
  medium : 8L-8H-512D   ~25M  params  — upper bound

Usage:
    python3 math_lab/datagen/train_generalist.py --arch small
    python3 math_lab/datagen/train_generalist.py --arch nano --epochs 3
    python3 math_lab/datagen/train_generalist.py --arch medium --data math_lab/results/oss/oss_train_*.jsonl
"""
import argparse, json, time, glob, sys
from pathlib import Path

# ── Architecture presets ──────────────────────────────────────────────────────
ARCHS = {
    #         n_layer  n_head  n_embd  dropout  notes
    "nano":   (2,      4,      128,    0.10,    "~0.8M  sanity check"),
    "micro":  (4,      4,      256,    0.10,    "~3.5M  skill-model scale"),
    "small":  (6,      8,      384,    0.05,    "~11M   likely sweet spot"),
    "medium": (8,      8,      512,    0.05,    "~25M   upper bound"),
}

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULTS = dict(
    epochs    = 5,
    batch     = 32,
    lr        = 3e-4,
    seq_len   = 512,
    eval_every= 500,
)


def find_latest_oss_train(raw_dir: str = "math_lab/results/oss") -> Path:
    """Auto-locate the most recent oss_train_*.jsonl"""
    candidates = sorted(Path(raw_dir).glob("oss_train_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No oss_train_*.jsonl found in {raw_dir}. "
            "Run math_lab/datagen/oss_ingest.py first."
        )
    return candidates[-1]


def count_params(n_layer, n_head, n_embd, vocab=131, seq_len=512):
    """Rough param estimate for our GPT architecture."""
    emb = vocab * n_embd + seq_len * n_embd        # token + position embed
    per_layer = (
        3 * n_embd * n_embd +   # QKV projection
        n_embd * n_embd +        # output projection
        4 * n_embd * n_embd * 4  # MLP (4× expansion)
    )
    head = vocab * n_embd                           # LM head (tied)
    return emb + n_layer * per_layer + head


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a tiny generalist MathGPT on curated OSS data.\n\n"
                    "Architecture presets:\n" +
                    "\n".join(f"  {k:<8} {v[4]}" for k, v in ARCHS.items()))
    ap.add_argument("--arch",     choices=list(ARCHS), default="small",
                    help="Model size preset (default: small)")
    ap.add_argument("--data",     help="Path to train JSONL (auto-detected if omitted)")
    ap.add_argument("--out-dir",  default="math_lab/results/checkpoints_gen")
    ap.add_argument("--epochs",   type=int,   default=DEFAULTS["epochs"])
    ap.add_argument("--batch",    type=int,   default=DEFAULTS["batch"])
    ap.add_argument("--lr",       type=float, default=DEFAULTS["lr"])
    ap.add_argument("--seq-len",  type=int,   default=DEFAULTS["seq_len"])
    ap.add_argument("--device",   default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--tag",      default="")
    ap.add_argument("--all",      action="store_true", help="Train all 4 architectures sequentially")
    ap.add_argument("--gate",     default=None,
                    help="Path to gate_config.yaml — filters and weights corpus before training")
    ap.add_argument("--ka-csv",   default="math_lab/knowledge_areas.csv",
                    help="Path to knowledge_areas.csv (default: math_lab/knowledge_areas.csv)")
    args = ap.parse_args()

    # Resolve data path
    if args.data:
        paths = sorted(glob.glob(args.data))
        if not paths:
            print(f"ERROR: no files matched {args.data}"); sys.exit(1)
        data_path = Path(paths[-1])
    else:
        data_path = find_latest_oss_train()

    archs_to_run = list(ARCHS.keys()) if args.all else [args.arch]

    print(f"Data: {data_path}  ({data_path.stat().st_size/1e6:.1f} MB)")
    if args.gate:
        print(f"Gate: {args.gate}")
    print(f"Archs to train: {archs_to_run}\n")

    # ── Apply knowledge gate to corpus (once, shared across all archs) ────────
    train_data_path = data_path
    gate_name       = None
    if args.gate:
        import tempfile
        sys.path.insert(0, ".")
        from math_lab.datagen.knowledge_gate import KnowledgeGate
        kg = KnowledgeGate(args.ka_csv)
        cfg = kg.load_config(args.gate)
        gate_name = cfg.name

        records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
        kept, summary = kg.gate_corpus(records)
        print(f"Gate '{gate_name}': {summary['kept']}/{summary['total_in']} records kept "
              f"({summary['keep_rate']*100:.0f}%)  steps={summary['step_counts']}\n")

        # Write gated corpus to a temp file for finetune.py
        gated_dir = Path(args.out_dir) / "gated"
        gated_dir.mkdir(parents=True, exist_ok=True)
        train_data_path = gated_dir / f"{data_path.stem}_{gate_name}.jsonl"
        train_data_path.write_text(
            "\n".join(json.dumps(r) for r in kept), encoding="utf-8")
        (gated_dir / f"{data_path.stem}_{gate_name}_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(f"Gated corpus → {train_data_path}\n")

    run_log = []

    for arch in archs_to_run:
        n_layer, n_head, n_embd, dropout, note = ARCHS[arch]
        est_params = count_params(n_layer, n_head, n_embd)

        print(f"{'═'*60}")
        print(f"  arch={arch}  ({note})")
        print(f"  layers={n_layer}  heads={n_head}  embd={n_embd}")
        print(f"  est params: {est_params/1e6:.2f}M")
        print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
        if gate_name:
            print(f"  gate: {gate_name}")
        print(f"{'═'*60}\n")

        # Resolve val file
        val_candidates = sorted(data_path.parent.glob("oss_val_*.jsonl"))
        val_path = val_candidates[-1] if val_candidates else None

        # Output dir: checkpoints_gen/{gate_name or 'base'}/{arch}/
        subdir   = gate_name if gate_name else "base"
        ckpt_dir = Path(args.out_dir) / subdir / arch
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        finetune = Path("math_lab/finetune.py")
        cmd = [
            sys.executable, str(finetune),
            "--data",        str(train_data_path),
            "--epochs",      str(args.epochs),
            "--batch",       str(args.batch),
            "--lr",          str(args.lr),
            "--seq-len",     str(args.seq_len),
            "--n-layer",     str(n_layer),
            "--n-head",      str(n_head),
            "--n-embd",      str(n_embd),
            "--dropout",     str(dropout),
            "--ckpt-dir",    str(ckpt_dir),
        ]
        if val_path:
            cmd += ["--val-data", str(val_path)]
        if args.device != "auto":
            cmd += ["--device", args.device]

        import subprocess
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
        run_log.append({
            "arch":    arch,
            "params_est": est_params,
            "note":    note,
            "data":    str(data_path),
            "epochs":  args.epochs,
            "elapsed_s": round(elapsed),
            "status":  status,
        })
        print(f"\n  Done: {arch} — {status}  ({elapsed/60:.1f} min)\n")

    # Write run log
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "generalist_runs.jsonl"
    with open(log_path, "a") as f:
        for r in run_log:
            f.write(json.dumps(r) + "\n")
    print(f"\nRun log → {log_path}")

    # Print summary table
    print(f"\n{'Arch':<8} {'Params':>8}  {'Status':<10}  {'Time':>8}")
    print("─" * 40)
    for r in run_log:
        print(f"{r['arch']:<8} {r['params_est']/1e6:>6.1f}M  {r['status']:<10}  {r['elapsed_s']//60:>5}m {r['elapsed_s']%60:02d}s")


if __name__ == "__main__":
    main()
