"""
skill_runner.py — Train one atomic skill from scratch, classify the result.

Orchestrates: data generation → training (constant LR, no warmup decay,
weight_decay=1.0 per Power et al. 2022) → checkpoint scan → classification
→ append to skill_registry.tsv.

Usage:
    python -m math_lab.skill_runner --skill add_1d_with_carry \\
        --arch 6x8x256 --epochs 10000 --seeds 42 1234 5678
"""

import argparse, json, csv, time, datetime, subprocess, sys
from pathlib import Path

# Layman Agent Platform status reporting (non-blocking — import failure is silent)
try:
    from math_lab.status_writer import write_agent_status as _write_status
except ImportError:
    try:
        from status_writer import write_agent_status as _write_status
    except ImportError:
        def _write_status(**_kw): return False  # graceful no-op if not found

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / 'math_lab' / 'results'
SKILLS_DIR = RESULTS / 'skills'
REGISTRY_PATH = RESULTS / 'skill_registry.tsv'
SKILL_DATAGEN = ROOT / 'math_lab' / 'datagen' / 'skill_data.py'
FINETUNE = ROOT / 'math_lab' / 'finetune.py'
EVAL_CANONICAL = ROOT / 'math_lab' / 'eval_canonical.py'

PYTHON = sys.executable

REGISTRY_FIELDS = [
    'timestamp', 'skill', 'seed', 'arch', 'n_layer', 'n_head', 'n_embd',
    'epochs_planned', 'epochs_actual', 'wall_seconds',
    'classification', 'peak_val_em', 'peak_val_em_pct', 'peak_epoch',
    'train_em_at_peak', 'train_em_at_peak_pct',
    'ckpt_path', 'data_train', 'data_val', 'notes',
]


def parse_arch(arch):
    """E.g. '6x8x256' -> (6, 8, 256)"""
    parts = arch.split('x')
    if len(parts) != 3: raise ValueError(f"arch must be NxHxE, got {arch!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def generate_skill_data(skill, holdout_frac, seed):
    """Run skill_data generator, return (train_path, val_path)."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(SKILL_DATAGEN), '--skill', skill,
           '--out-dir', str(SKILLS_DIR),
           '--holdout-frac', str(holdout_frac), '--seed', str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Find the latest matching file
    candidates = sorted(SKILLS_DIR.glob(f'{skill}_train_*.jsonl'))
    train_p = candidates[-1]
    val_p = SKILLS_DIR / train_p.name.replace('_train_', '_val_')
    return train_p, val_p


def build_finetune_cmd(train_p, val_p, ckpt_dir, n_layer, n_head, n_embd,
                       epochs, seed, lr=1e-3, weight_decay=1.0):
    """Build finetune.py command per the assembly-line training protocol.

    Power et al. 2022 grokking recipe: constant LR via warmup=0 + lr_min=lr,
    weight_decay=1.0, dropout=0, label_smoothing=0.
    """
    return [PYTHON, str(FINETUNE),
            '--data', str(train_p), '--val-data', str(val_p),
            '--ckpt-dir', str(ckpt_dir),
            '--n-layer', str(n_layer), '--n-head', str(n_head),
            '--n-embd', str(n_embd), '--seq-len', '64',
            '--epochs', str(epochs), '--batch', '16',
            '--lr', str(lr), '--lr-min', str(lr),  # constant LR
            '--weight-decay', str(weight_decay),
            '--dropout', '0.0', '--label-smoothing', '0.0',
            '--warmup', '0', '--restart-period', '0',
            '--val-every', '50', '--save-every', '500',
            '--em-every', '0',  # disable phantom abort — we want full grokking window
            '--seed', str(seed)]


def scan_checkpoints(ckpt_dir, val_path, train_path, max_new=20):
    """Eval every saved checkpoint on val + train. Returns list of dicts."""
    sys.path.insert(0, str(ROOT / 'math_lab'))
    from eval_canonical import load_model, greedy_generate, extract_answer, answer_matches
    from finetune import PROMPT_TEMPLATE

    val_pairs = [json.loads(l) for l in Path(val_path).read_text().splitlines() if l.strip()]
    train_pairs = [json.loads(l) for l in Path(train_path).read_text().splitlines() if l.strip()]
    results = []
    for ck in sorted(Path(ckpt_dir).glob('epoch_*.pt')):
        m, meta = load_model(ck, 'mps')
        def em(pairs):
            ok = 0
            for p in pairs:
                prompt = PROMPT_TEMPLATE.format(problem=p['problem'])
                out = greedy_generate(m, prompt, max_new=max_new)
                gen = out[len(prompt):] if out.startswith(prompt) else out
                if answer_matches(extract_answer(gen), p['answer']): ok += 1
            return ok, len(pairs)
        v_ok, v_n = em(val_pairs)
        t_ok, t_n = em(train_pairs[:min(len(train_pairs), 200)])
        results.append({
            'epoch': meta.get('epoch', -1) + 1,
            'val_em': v_ok, 'val_n': v_n,
            'train_em': t_ok, 'train_n': t_n,
            'val_loss': float(meta.get('val_loss', 0)),
            'ckpt': str(ck),
        })
    return results


def classify(scan_results):
    """Return classification + peak metrics."""
    if not scan_results:
        return 'FAILED', None
    peak = max(scan_results, key=lambda r: r['val_em'])
    val_pct = 100 * peak['val_em'] / peak['val_n'] if peak['val_n'] else 0
    train_pct = 100 * peak['train_em'] / peak['train_n'] if peak['train_n'] else 0
    if val_pct >= 80:
        cls = 'GENERALIZED'
    elif val_pct >= 30:
        cls = 'PARTIAL'
    elif train_pct >= 80:
        cls = 'MEMORIZED'
    else:
        cls = 'FAILED'
    return cls, peak


def append_registry(row):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY_PATH.exists()
    with open(REGISTRY_PATH, 'a') as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, delimiter='\t')
        if new: w.writeheader()
        w.writerow(row)


def run_one(skill, arch, epochs, seed, holdout_frac=0.5):
    n_layer, n_head, n_embd = parse_arch(arch)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_p, val_p = generate_skill_data(skill, holdout_frac, seed)
    ckpt_dir = RESULTS / 'checkpoints' / 'skills' / f'{ts}_{skill}_{arch}_seed{seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_finetune_cmd(train_p, val_p, ckpt_dir,
                              n_layer, n_head, n_embd, epochs, seed)
    log = ckpt_dir / 'train.log'
    print(f'[{skill}] training {arch} seed={seed} epochs={epochs}')
    print(f'  log: {log}')

    # ── Status: skill training starting ───────────────────────────────────
    _write_status(
        current_task=f"skill_runner: {skill} arch={arch} seed={seed} ep={epochs} — training",
        status="in_progress",
    )

    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    print(f'  done in {wall:.0f}s rc={rc}')

    # ── Status: training done, scanning checkpoints ──────────────────────
    _write_status(
        current_task=f"skill_runner: {skill} seed={seed} — trained in {wall:.0f}s, scanning checkpoints",
        status="in_progress",
    )

    scan = scan_checkpoints(ckpt_dir, val_p, train_p)
    cls, peak = classify(scan)
    print(f'  classification: {cls}')
    if peak:
        print(f'  peak: val {peak["val_em"]}/{peak["val_n"]} = {100*peak["val_em"]/peak["val_n"]:.1f}% '
              f'@ ep{peak["epoch"]}  train@peak={peak["train_em"]}/{peak["train_n"]}')

    row = {
        'timestamp': ts, 'skill': skill, 'seed': seed, 'arch': arch,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
        'epochs_planned': epochs, 'epochs_actual': scan[-1]['epoch'] if scan else 0,
        'wall_seconds': int(wall),
        'classification': cls,
        'peak_val_em': peak['val_em'] if peak else 0,
        'peak_val_em_pct': round(100 * peak['val_em'] / peak['val_n'], 2) if peak else 0,
        'peak_epoch': peak['epoch'] if peak else 0,
        'train_em_at_peak': peak['train_em'] if peak else 0,
        'train_em_at_peak_pct': round(100 * peak['train_em'] / peak['train_n'], 2) if peak else 0,
        'ckpt_path': peak['ckpt'] if peak else '',
        'data_train': str(train_p), 'data_val': str(val_p),
        'notes': '',
    }
    append_registry(row)

    # ── Status: skill run complete ───────────────────────────────────────
    em_pct = round(100 * peak['val_em'] / peak['val_n'], 1) if peak else 0.0
    _write_status(
        last_completed=f"{skill} arch={arch} seed={seed} → {cls} {em_pct}% EM @ ep{peak['epoch'] if peak else '?'}",
        current_task="skill_runner: idle — awaiting next skill",
        status="in_progress",
        blockers="none",
    )

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skill', required=True)
    p.add_argument('--arch', default='6x8x256')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--holdout-frac', type=float, default=0.5)
    args = p.parse_args()

    for seed in args.seeds:
        run_one(args.skill, args.arch, args.epochs, seed, args.holdout_frac)


if __name__ == '__main__':
    main()
