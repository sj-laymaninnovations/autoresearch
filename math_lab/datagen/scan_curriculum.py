"""scan_curriculum.py — eval a curriculum checkpoint on every prior pile's
val set. Catastrophic-forgetting check.

Usage:
    py -3.12 math_lab/datagen/scan_curriculum.py --ckpt <ckpt.pt> --piles 1,2,3
"""
import argparse, json, sys
sys.path.insert(0, "math_lab")
from pathlib import Path
from collections import defaultdict
from finetune import PROMPT_TEMPLATE
from eval_canonical import load_model, greedy_generate, extract_answer, answer_matches


def scan_pile(model, pile_n: int) -> dict:
    val = Path(f"math_lab/results/skills/curriculum_pile{pile_n}_val.jsonl")
    pairs = [json.loads(l) for l in val.read_text().splitlines() if l.strip()]
    by_concept = defaultdict(lambda: {"ok": 0, "n": 0})
    overall = {"ok": 0, "n": 0}
    for p in pairs:
        prompt = PROMPT_TEMPLATE.format(problem=p["problem"])
        out = greedy_generate(model, prompt, max_new=300)
        gen = out[len(prompt):] if out.startswith(prompt) else out
        ok = answer_matches(extract_answer(gen), p["answer"])
        c = p.get("concept", "?")
        by_concept[c]["ok"] += int(ok)
        by_concept[c]["n"]  += 1
        overall["ok"] += int(ok)
        overall["n"]  += 1
    return {"overall": overall, "by_concept": dict(by_concept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--piles", required=True, type=str,
                    help="comma-separated pile numbers, e.g. '1,2,3'")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"  ckpt: {args.ckpt}")
    m, _ = load_model(args.ckpt, args.device)

    piles = [int(x.strip()) for x in args.piles.split(",")]
    grand_total = {"ok": 0, "n": 0}
    for p in piles:
        print(f"\n  ===== PILE {p} =====")
        r = scan_pile(m, p)
        for c, s in sorted(r["by_concept"].items()):
            pct = 100 * s["ok"] / s["n"] if s["n"] else 0
            print(f"    {c:32s}  {s['ok']:>5d}/{s['n']:<5d}  {pct:6.1f}%")
        ov = r["overall"]
        pct = 100 * ov["ok"] / ov["n"] if ov["n"] else 0
        print(f"    {'PILE OVERALL':32s}  {ov['ok']:>5d}/{ov['n']:<5d}  {pct:6.1f}%")
        grand_total["ok"] += ov["ok"]
        grand_total["n"]  += ov["n"]

    pct = 100 * grand_total["ok"] / grand_total["n"] if grand_total["n"] else 0
    print(f"\n  GRAND TOTAL across piles {piles}: "
          f"{grand_total['ok']}/{grand_total['n']} = {pct:.1f}%")


if __name__ == "__main__":
    main()
