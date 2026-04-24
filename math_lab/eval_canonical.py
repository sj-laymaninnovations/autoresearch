"""
eval_canonical.py — Per-concept exact-match evaluation for canonical algebra.

Loads a trained checkpoint, runs greedy decoding on every val problem, parses
the "#### N" answer, compares to ground truth. Produces:
  - Aggregate exact-match accuracy
  - Per-stage breakdown (Stages 1-5)
  - Per-concept pass/fail list
  - Failing-concept analysis

Used standalone or by autoresearch_canonical.py after each training run.

Usage:
    python math_lab/eval_canonical.py --ckpt <path> --val <val_jsonl>
    python math_lab/eval_canonical.py --ckpt ckpts/best.pt  # uses latest canonical val

Output:
    - Terminal report with pass/fail table
    - JSON file at math_lab/results/eval/<ckpt_name>_eval.json
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch

# ensure we can import finetune helpers
sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    MathGPT, GPTConfig, char_encode, char_decode,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID, OFFSET, VOCAB_SIZE,
    PROMPT_TEMPLATE, ANS_SUFFIX,
)

# Checkpoints were pickled with GPTConfig in finetune.__main__ context.
# When loading from a different caller, inject the class into the current
# __main__ so torch.load's unpickler can find it.
sys.modules["__main__"].GPTConfig = GPTConfig
sys.modules["__main__"].MathGPT   = MathGPT

RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DIR    = RESULTS_DIR / "eval"


# ---------------------------------------------------------------------------
# Greedy decoding (deterministic, unlike the sampling generate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(model, prompt: str, max_new: int = 120) -> str:
    """Return text generated greedily after `prompt`. Stops at EOS or max_new."""
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = char_encode(prompt)
    # char_encode wraps with BOS/EOS; drop the trailing EOS so generation continues
    if prompt_ids and prompt_ids[-1] == EOS_ID:
        prompt_ids = prompt_ids[:-1]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = ids[:, -model.cfg.seq_len:]
        logits = model(ctx)          # (1, T, V)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        if next_id.item() == EOS_ID:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return char_decode(ids[0].tolist())


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r"####\s*([^\n]+?)\s*$", re.MULTILINE)
NUMERIC_RE = re.compile(r"-?\d+(?:/-?\d+)?(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the final answer from model output. Prefers '#### X' line."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: last numeric token in output
    nums = NUMERIC_RE.findall(text)
    return nums[-1] if nums else None


def normalize(v) -> str:
    """Normalize a string or int answer for comparison."""
    s = str(v).strip()
    # Allow "7" to match "7.0" and similar
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return s.replace(" ", "").lower()


def answer_matches(predicted: str | None, expected) -> bool:
    if predicted is None:
        return False
    return normalize(predicted) == normalize(expected)


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MathGPT(cfg).to(device)
    # strip _orig_mod. prefix if torch.compile was used
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def evaluate(ckpt_path: Path, val_path: Path, max_new: int = 120,
             device: str = None, verbose: bool = True) -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model, ckpt = load_model(ckpt_path, device)
    pairs = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]

    by_stage   = defaultdict(lambda: {"pass": 0, "fail": 0})
    by_concept = {}
    failures = []

    for p in pairs:
        prompt = PROMPT_TEMPLATE.format(problem=p["problem"])
        output = greedy_generate(model, prompt, max_new=max_new)
        # Strip the prompt prefix to get only the generated solution
        generated = output[len(prompt):] if output.startswith(prompt) else output
        pred = extract_answer(generated)
        ok = answer_matches(pred, p["answer"])

        by_stage[p["stage"]]["pass" if ok else "fail"] += 1
        by_concept[p["concept"]] = {
            "pass":       ok,
            "expected":   str(p["answer"]),
            "predicted":  pred,
            "stage":      p["stage"],
            "generated":  generated[:200],
        }
        if not ok:
            failures.append(p["concept"])

    total_pass = sum(v["pass"] for v in by_stage.values())
    total      = len(pairs)
    exact_match = total_pass / total if total else 0.0

    result = {
        "ckpt":      str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch", -1) + 1,
        "ckpt_val":  ckpt.get("val_loss"),
        "val_path":  str(val_path),
        "total":     total,
        "pass":      total_pass,
        "exact_match": exact_match,
        "by_stage":  {str(k): dict(v) for k, v in sorted(by_stage.items())},
        "by_concept": by_concept,
        "failures":  failures,
    }

    if verbose:
        print_report(result)
    return result


def print_report(r: dict):
    print(f"\n{'='*70}")
    print(f"  Checkpoint: {Path(r['ckpt']).name}")
    print(f"  Ckpt epoch: {r['ckpt_epoch']}  val_loss: {r['ckpt_val']}")
    print(f"  Exact match: {r['pass']}/{r['total']}  ({r['exact_match']*100:.1f}%)")
    print(f"{'='*70}")
    print(f"\nBy stage:")
    for s, v in r["by_stage"].items():
        total = v["pass"] + v["fail"]
        pct = 100 * v["pass"] / total if total else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  Stage {s}: {v['pass']:>2}/{total:<2}  {bar}  {pct:.0f}%")

    if r["failures"]:
        print(f"\nFailing concepts ({len(r['failures'])}):")
        for c in r["failures"]:
            bc = r["by_concept"][c]
            exp = bc["expected"]
            got = bc["predicted"]
            print(f"  [S{bc['stage']}] {c:30s}  expected={exp!r}  got={got!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--val",  default=None,
                    help="Val JSONL (defaults to latest canonical_algebra_val_*.jsonl)")
    ap.add_argument("--max-new", type=int, default=120,
                    help="Max new tokens to generate per problem")
    ap.add_argument("--save", action="store_true", help="Save report to JSON")
    args = ap.parse_args()

    if args.val is None:
        candidates = sorted(RESULTS_DIR.glob("canonical_algebra_val_*.jsonl"))
        if not candidates:
            print("ERROR: no canonical val JSONL found; generate one or pass --val")
            sys.exit(1)
        val_path = candidates[-1]
        print(f"Using latest val: {val_path.name}")
    else:
        val_path = Path(args.val)

    result = evaluate(Path(args.ckpt), val_path, max_new=args.max_new)

    if args.save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        out_name = Path(args.ckpt).stem + "_eval.json"
        out_path = EVAL_DIR / out_name
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
