#!/usr/bin/env python3
"""
monitor_and_quiz.py — Polls Windows for Track A/B checkpoint completion,
pulls each best.pt as it lands, runs the quiz, and prints a live results table.

Usage:
    python3 monitor_and_quiz.py
"""
import subprocess, time, re, sys, os, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'math_lab'))

# ── Config ────────────────────────────────────────────────────────────────────
SSH     = "ssh -i ~/.ssh/autoresearch_macmini -o ConnectTimeout=10 -o StrictHostKeyChecking=no seanj@192.168.1.210"
WIN_BASE = r"F:\projects\autoresearch\math_lab\results\checkpoints"
LOCAL_V4 = Path("math_lab/results/checkpoints_v4")
LOCAL_V4.mkdir(parents=True, exist_ok=True)

POLL_SECS = 300  # 5 minutes

# Order matters: train sequentially on each GPU, so B can only start after A
TARGETS = [
    # name              win_dir                         local_name               quiz_cases
    ("v4_div1d",        "20260503_v4_div1d",            "v4_div1d_best.pt",
     [("div_1d", [("56 / 7","8"),("42 / 6","7"),("36 / 4","9"),("63 / 9","7"),("48 / 8","6"),
                  ("72 / 8","9"),("66 / 6","11"),("84 / 7","12"),("96 / 8","12"),("60 / 5","12")])]),

    ("v4_alg1step",     "20260503_v4_alg1step",         "v4_alg1step_best.pt",
     [("alg_1step", [("7x = 56","8"),("3x = 12","4"),("5x = 35","7"),
                     ("4x = 28","7"),("9x = 63","7"),("15x = 45","3"),
                     ("12x = 60","5"),("8x = 96","12")])]),

    ("v4_alg2step_hard","20260503_v4_alg2step_hard",    "v4_alg2step_hard_best.pt",
     [("alg_2step_hard", [("3x + 5 = 20","5"),("4x - 7 = 13","5"),
                          ("2x + 9 = 1","-4"),("5x - 3 = 17","4"),
                          ("6x + 4 = 28","4"),("3x + 4 = 2x + 9","5")])]),

    ("v4_alg_distribute","20260503_v4_alg_distribute",  "v4_alg_distribute_best.pt",
     [("distribute", [("3(x + 2) = 18","4"),("4(x - 1) = 20","6"),
                      ("2(x + 5) = 16","3"),("5(x + 3) = 35","4")]),
      ("combine",    [("3x + 2x = 20","4"),("5x + 3x = 24","3"),
                      ("7x + 2x = 45","5"),("4x + 6x = 40","4")])]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def win_has_best(win_dir):
    """Return True if best.pt exists on Windows."""
    cmd = f'{SSH} "dir {WIN_BASE}\\{win_dir}\\best.pt 2>nul"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return "best.pt" in result.stdout


def scp_best(win_dir, local_name):
    """Pull best.pt from Windows to local."""
    src = f"seanj@192.168.1.210:{WIN_BASE}\\{win_dir}\\best.pt"
    dst = str(LOCAL_V4 / local_name)
    cmd = f"scp -i ~/.ssh/autoresearch_macmini -o StrictHostKeyChecking=no '{src}' '{dst}'"
    return subprocess.run(cmd, shell=True, capture_output=True).returncode == 0


def gpu_status():
    cmd = f'{SSH} "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader"'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def quiz_model(ckpt_path, skill_cases):
    """Load model and run all quiz cases. Returns {skill: (passes, total, results)}."""
    import torch
    from finetune import MathGPT, GPTConfig, PROMPT_TEMPLATE

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg  = ckpt.get('config', GPTConfig())
    if isinstance(cfg, dict): cfg = GPTConfig(**cfg)
    model = MathGPT(cfg)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    def ask(prob):
        torch.manual_seed(0)
        return model.generate(PROMPT_TEMPLATE.format(problem=prob),
                              max_new=250, temperature=0.3, top_k=40)

    def extract(t):
        m = re.search(r'####\s*(-?\d+)', t)
        return m.group(1) if m else '?'

    results = {}
    for skill, cases in skill_cases:
        passes = 0
        detail = []
        for prob, exp in cases:
            raw  = ask(prob)
            got  = extract(raw)
            ok   = (got == exp)
            passes += ok
            detail.append((prob, exp, got, ok))
        results[skill] = (passes, len(cases), detail)
    return results


def bar(passes, total):
    return '█'*passes + '░'*(total-passes)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    completed   = set()
    all_results = {}
    pending     = [t[0] for t in TARGETS]

    print(f"\n{'='*64}")
    print(f"  MathGPT Track Monitor — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Polling every {POLL_SECS//60} min for {len(TARGETS)} checkpoints")
    print(f"{'='*64}\n")

    while len(completed) < len(TARGETS):
        for name, win_dir, local_name, quiz_cases in TARGETS:
            if name in completed:
                continue
            if win_has_best(win_dir):
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{ts}] ✅ {name} checkpoint found — pulling...")
                if scp_best(win_dir, local_name):
                    print(f"  SCP OK → {LOCAL_V4/local_name}")
                    print(f"  Running quiz...")
                    try:
                        results = quiz_model(str(LOCAL_V4/local_name), quiz_cases)
                        all_results[name] = results
                        for skill, (passes, total, detail) in results.items():
                            pct = 100*passes//total
                            print(f"  {skill:<28} {bar(passes,total)}  {passes}/{total}  ({pct}%)")
                            if passes < total:
                                for prob, exp, got, ok in detail:
                                    if not ok:
                                        print(f"    ✗ {prob:20} expected={exp} got={got}")
                    except Exception as e:
                        print(f"  Quiz error: {e}")
                    completed.add(name)
                else:
                    print(f"  SCP failed, will retry")

        if len(completed) < len(TARGETS):
            ts  = datetime.now().strftime('%H:%M:%S')
            rem = [n for n in pending if n not in completed]
            gpus = gpu_status().replace('\r','')
            print(f"[{ts}] Waiting for: {', '.join(rem)}")
            for line in gpus.split('\n'):
                if line.strip():
                    print(f"  GPU: {line.strip()}")
            time.sleep(POLL_SECS)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  FINAL RESULTS — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*64}")
    for name, win_dir, local_name, quiz_cases in TARGETS:
        if name not in all_results:
            print(f"  {name:<26}  (no results)")
            continue
        for skill, (passes, total, _) in all_results[name].items():
            pct = 100*passes//total
            print(f"  {name:<24}  {skill:<20}  {bar(passes,total)}  {passes}/{total}  ({pct}%)")

    # Save JSON summary
    summary_path = LOCAL_V4 / "quiz_summary.json"
    summary = {}
    for name, res in all_results.items():
        summary[name] = {sk: {'passes':p,'total':t} for sk,(p,t,_) in res.items()}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved → {summary_path}")


if __name__ == '__main__':
    main()
