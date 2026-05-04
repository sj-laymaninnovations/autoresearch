"""
cross_skill_quiz.py — evaluate each GENERALIZED atomic-skill checkpoint
against every val set in the registry, producing a cross-skill transfer
matrix.

Output: rows = checkpoints, cols = val sets, cells = EM%.
Diagonals should match the per-skill peak (sanity check).
Off-diagonals reveal structural transfer (or, more often, near-0 — meaning
single-skill training is narrow).

Usage:
  python cross_skill_quiz.py \
    --checkpoints '<ckpt1>:<label1>' '<ckpt2>:<label2>' ... \
    --vals '<val1.jsonl>:<label1>' ... \
    --device cuda \
    --max-new 200 \
    --sample 0

Set --sample N>0 to use only the first N pairs from each val (for quick passes).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_canonical import load_model, greedy_generate, extract_answer, answer_matches
from finetune import PROMPT_TEMPLATE


def evaluate_pair(model, val_path: Path, max_new: int, sample: int = 0):
    pairs = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]
    if sample > 0:
        pairs = pairs[:sample]
    ok = 0
    for p in pairs:
        prompt = PROMPT_TEMPLATE.format(problem=p['problem'])
        out = greedy_generate(model, prompt, max_new=max_new)
        gen = out[len(prompt):] if out.startswith(prompt) else out
        if answer_matches(extract_answer(gen), p['answer']):
            ok += 1
    return ok, len(pairs)


def parse_pair(s: str):
    if ':' not in s:
        raise SystemExit(f'expected path:label, got {s!r}')
    p, label = s.rsplit(':', 1)
    return Path(p), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='one or more <ckpt_path>:<label>')
    ap.add_argument('--vals', nargs='+', required=True,
                    help='one or more <val_jsonl_path>:<label>')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-new', type=int, default=200)
    ap.add_argument('--sample', type=int, default=0,
                    help='0 = full val set; N>0 = take first N')
    ap.add_argument('--out', default='math_lab/results/cross_skill_quiz.json')
    args = ap.parse_args()

    ckpts = [parse_pair(s) for s in args.checkpoints]
    vals = [parse_pair(s) for s in args.vals]

    matrix = {}
    width = max(len(l) for _, l in vals) + 2
    print(f'{"checkpoint":24s} | ' + ' | '.join(f'{l:>{width}s}' for _, l in vals), flush=True)
    print('-' * (24 + 3 + (width + 3) * len(vals)), flush=True)

    for ckpt_path, ckpt_label in ckpts:
        m, _ = load_model(ckpt_path, args.device)
        row = {}
        cells = []
        for val_path, val_label in vals:
            ok, n = evaluate_pair(m, val_path, args.max_new, args.sample)
            pct = 100 * ok / n if n else 0
            row[val_label] = {'ok': ok, 'n': n, 'pct': pct}
            cells.append(f'{ok}/{n} ({pct:.1f}%)'.rjust(width))
        matrix[ckpt_label] = row
        print(f'{ckpt_label:24s} | ' + ' | '.join(cells), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        'checkpoints': [str(p) for p, _ in ckpts],
        'vals':        [str(p) for p, _ in vals],
        'matrix':      matrix,
        'sample':      args.sample,
        'max_new':     args.max_new,
    }, indent=2))
    print(f'\nMatrix written -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
