"""
build_training_data.py — convert distilled/qa_records_v1.jsonl to MathGPT
training format with a 90/10 SHA-based train/val split.

MathGPT format (per math_lab/finetune.py MathQADataset):
    {
      "problem":      <question text>,
      "solution":     <answer with #### line>,
      "solution_cot": <same as solution>,
      "answer":       <text after '#### ' marker>,
      "concept":      "asm_<repo>",
      "stage":        1,
      "level":        1,
      "sha":          <commit sha>,
      "repo":         <repo>,
    }

Output:
    bet_a/training_data/bet_a_asm_v1_train.jsonl
    bet_a/training_data/bet_a_asm_v1_val.jsonl
    bet_a/training_data/bet_a_asm_v1_split.json   (manifest)
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "distilled" / "qa_records_v1.jsonl"
OUT_DIR = ROOT / "training_data"

REPO_TO_CONCEPT = {
    "x264_src":   "asm_codec_x264",
    "dav1d_src":  "asm_codec_dav1d",
    "ffmpeg_src": "asm_codec_ffmpeg",
    "linux_src":  "asm_kernel_arm64",
}

_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE | re.DOTALL)


def extract_summary(answer: str) -> str:
    """Return the text after the last `####` marker."""
    m = list(_HASH_RE.finditer(answer))
    if not m:
        return ""
    return m[-1].group(1).strip().splitlines()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed",         type=int,   default=11)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in SRC.open()]
    print(f"loaded {len(records)} distilled pairs from {SRC.name}")

    # SHA-based split — pairs from the same commit must go to the same split
    shas = sorted({r["sha"] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(shas)
    n_val_shas = max(1, int(len(shas) * args.val_fraction))
    val_shas = set(shas[:n_val_shas])
    train_shas = set(shas[n_val_shas:])
    print(f"  unique SHAs: {len(shas)} → train {len(train_shas)} / val {len(val_shas)}")

    def to_mathgpt(r: dict) -> dict:
        return {
            "problem":      r["question"],
            "solution":     r["answer"],
            "solution_cot": r["answer"],
            "answer":       extract_summary(r["answer"]),
            "concept":      REPO_TO_CONCEPT.get(r["repo"], "asm_unknown"),
            "stage":        1,
            "level":        1,
            "sha":          r["sha"],
            "repo":         r["repo"],
        }

    train_path = OUT_DIR / "bet_a_asm_v1_train.jsonl"
    val_path   = OUT_DIR / "bet_a_asm_v1_val.jsonl"
    n_train = n_val = 0
    train_concepts = Counter()
    val_concepts   = Counter()
    with train_path.open("w") as ftrain, val_path.open("w") as fval:
        for r in records:
            mg = to_mathgpt(r)
            if r["sha"] in val_shas:
                fval.write(json.dumps(mg, ensure_ascii=False) + "\n")
                n_val += 1
                val_concepts[mg["concept"]] += 1
            else:
                ftrain.write(json.dumps(mg, ensure_ascii=False) + "\n")
                n_train += 1
                train_concepts[mg["concept"]] += 1

    # Length statistics — for seq_len sizing
    def chars(r: dict) -> int:
        return len("Q: " + r["question"] + "\nA: " + r["answer"] + "\n")
    lens = sorted(chars(r) for r in records)
    p50 = lens[len(lens)//2]
    p95 = lens[int(len(lens)*0.95)]
    p99 = lens[int(len(lens)*0.99)]
    longest = lens[-1]

    manifest = {
        "source_file":   str(SRC),
        "n_records_total": len(records),
        "n_train":       n_train,
        "n_val":         n_val,
        "split_seed":    args.seed,
        "split_val_fraction": args.val_fraction,
        "train_concepts":dict(train_concepts.most_common()),
        "val_concepts":  dict(val_concepts.most_common()),
        "char_length_p50": p50,
        "char_length_p95": p95,
        "char_length_p99": p99,
        "char_length_max": longest,
        "recommend_seq_len": 1024 if p99 <= 1024 else 2048,
    }
    (OUT_DIR / "bet_a_asm_v1_split.json").write_text(json.dumps(manifest, indent=2))

    print(f"  train: {n_train} → {train_path.name}")
    print(f"  val:   {n_val} → {val_path.name}")
    print(f"  manifest → {OUT_DIR/'bet_a_asm_v1_split.json'}")
    print(f"\n  char-length p50/p95/p99/max = {p50}/{p95}/{p99}/{longest}")
    print(f"  recommend seq_len = {manifest['recommend_seq_len']}")
    print(f"\n  train concepts: {dict(train_concepts.most_common())}")


if __name__ == "__main__":
    main()
