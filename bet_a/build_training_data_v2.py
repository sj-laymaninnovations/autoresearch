"""
build_training_data_v2.py — merge Pipeline A (commit-body) + Pipeline E
(multi-granularity in-source) distilled pairs into a unified v2 training
corpus. Per-source 90/10 train/val split.

Sources:
  - distilled/qa_records_v1.jsonl    (Pipeline A, schema: sha/repo/question/answer/subject)
  - distilled/qa_records_e_v1.jsonl  (Pipeline E, schema: unit_id/file_path/repo/granularity/...)

Output:
  - training_data/bet_a_asm_v2_train.jsonl
  - training_data/bet_a_asm_v2_val.jsonl
  - training_data/bet_a_asm_v2_split.json   (manifest)

MathGPT-format records (matches MathQADataset in math_lab/finetune.py):
  {
    "problem":      <question>,
    "solution":     <answer with #### line>,
    "solution_cot": <same>,
    "answer":       <text after final #### marker>,
    "concept":      "asm_<repo or codec_family>",
    "stage":        1,
    "level":        1,
    "source":       "pipeline_a" | "pipeline_e",
    "source_id":    <sha or unit_id>,
    "repo":         <repo>,
    "granularity":  <comment_block|function|block|null>   # null for A
  }
"""
from __future__ import annotations
import argparse, json, random, re
from collections import Counter
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
A_SRC    = ROOT / "distilled" / "qa_records_v1.jsonl"
E_SRC    = ROOT / "distilled" / "qa_records_e_v1.jsonl"
OUT_DIR  = ROOT / "training_data"

# Concept tag derivation — uses repo name. For Pipeline E unit IDs, we
# additionally suffix with the granularity for tracking.
REPO_TO_CONCEPT = {
    "x264_src":        "asm_codec_x264",
    "dav1d_src":       "asm_codec_dav1d",
    "ffmpeg_src":      "asm_codec_ffmpeg",
    "linux_src":       "asm_kernel_arm64",
    "pytorch/pytorch": "asm_ml_qnnpack",
}

_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE | re.DOTALL)

def extract_summary(answer: str) -> str:
    m = list(_HASH_RE.finditer(answer))
    if not m:
        return ""
    return m[-1].group(1).strip().splitlines()[0]


def harmonize_a(r: dict) -> dict:
    """Pipeline A record -> MathGPT record."""
    return {
        "problem":      r["question"],
        "solution":     r["answer"],
        "solution_cot": r["answer"],
        "answer":       extract_summary(r["answer"]),
        "concept":      REPO_TO_CONCEPT.get(r["repo"], "asm_unknown"),
        "stage":        1,
        "level":        1,
        "source":       "pipeline_a",
        "source_id":    r["sha"],
        "repo":         r["repo"],
        "granularity":  None,
    }


def harmonize_e(r: dict) -> dict:
    """Pipeline E record -> MathGPT record."""
    return {
        "problem":      r["question"],
        "solution":     r["answer"],
        "solution_cot": r["answer"],
        "answer":       extract_summary(r["answer"]),
        "concept":      REPO_TO_CONCEPT.get(r["repo"], "asm_unknown"),
        "stage":        1,
        "level":        1,
        "source":       "pipeline_e",
        "source_id":    r["unit_id"],
        "repo":         r["repo"],
        "granularity":  r.get("granularity"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed",         type=int,   default=11)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_recs = [json.loads(l) for l in A_SRC.open()]
    e_recs = [json.loads(l) for l in E_SRC.open()]
    print(f"Pipeline A: {len(a_recs):>4d} pairs from {A_SRC.name}")
    print(f"Pipeline E: {len(e_recs):>4d} pairs from {E_SRC.name}")
    print(f"Combined:   {len(a_recs) + len(e_recs):>4d}")

    # Per-source split keyed by their natural unit id (sha for A, unit_id for E)
    def split_by_key(records: list[dict], key_field: str,
                      val_fraction: float, seed: int) -> tuple[set, set]:
        keys = sorted({r[key_field] for r in records})
        rng = random.Random(seed)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_fraction))
        return set(keys[:n_val]), set(keys[n_val:])

    a_val, a_train = split_by_key(a_recs, "sha",      args.val_fraction, args.seed)
    e_val, e_train = split_by_key(e_recs, "unit_id",  args.val_fraction, args.seed + 1)
    print(f"  A: train {len(a_train)} / val {len(a_val)} unique SHAs")
    print(f"  E: train {len(e_train)} / val {len(e_val)} unique unit-ids")

    train_path = OUT_DIR / "bet_a_asm_v2_train.jsonl"
    val_path   = OUT_DIR / "bet_a_asm_v2_val.jsonl"
    n_train = n_val = 0
    train_concepts: Counter = Counter()
    val_concepts:   Counter = Counter()
    train_sources:  Counter = Counter()
    val_sources:    Counter = Counter()

    with train_path.open("w") as ftrain, val_path.open("w") as fval:
        for r in a_recs:
            mg = harmonize_a(r)
            if r["sha"] in a_val:
                fval.write(json.dumps(mg, ensure_ascii=False) + "\n");  n_val += 1
                val_concepts[mg["concept"]] += 1; val_sources[mg["source"]] += 1
            else:
                ftrain.write(json.dumps(mg, ensure_ascii=False) + "\n"); n_train += 1
                train_concepts[mg["concept"]] += 1; train_sources[mg["source"]] += 1
        for r in e_recs:
            mg = harmonize_e(r)
            if r["unit_id"] in e_val:
                fval.write(json.dumps(mg, ensure_ascii=False) + "\n");  n_val += 1
                val_concepts[mg["concept"]] += 1; val_sources[mg["source"]] += 1
            else:
                ftrain.write(json.dumps(mg, ensure_ascii=False) + "\n"); n_train += 1
                train_concepts[mg["concept"]] += 1; train_sources[mg["source"]] += 1

    def chars(r: dict) -> int:
        return len("Q: " + r["question"] + "\nA: " + r["answer"] + "\n")
    all_lens = sorted([chars(r) for r in a_recs] + [chars(r) for r in e_recs])
    p50 = all_lens[len(all_lens)//2]
    p95 = all_lens[int(len(all_lens)*0.95)]
    p99 = all_lens[int(len(all_lens)*0.99)]
    longest = all_lens[-1]

    manifest = {
        "sources": [
            {"path": str(A_SRC), "n_records": len(a_recs), "type": "pipeline_a"},
            {"path": str(E_SRC), "n_records": len(e_recs), "type": "pipeline_e"},
        ],
        "n_train":      n_train,
        "n_val":        n_val,
        "split_seed":   args.seed,
        "split_val_fraction": args.val_fraction,
        "train_concepts": dict(train_concepts.most_common()),
        "val_concepts":   dict(val_concepts.most_common()),
        "train_sources":  dict(train_sources.most_common()),
        "val_sources":    dict(val_sources.most_common()),
        "char_length_p50": p50,
        "char_length_p95": p95,
        "char_length_p99": p99,
        "char_length_max": longest,
        "recommend_seq_len": 1024 if p99 <= 1024 else (2048 if p99 <= 2048 else 4096),
    }
    (OUT_DIR / "bet_a_asm_v2_split.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  train: {n_train} → {train_path.name}")
    print(f"  val:   {n_val} → {val_path.name}")
    print(f"  char-length p50/p95/p99/max = {p50}/{p95}/{p99}/{longest}")
    print(f"  recommend seq_len = {manifest['recommend_seq_len']}")
    print(f"\n  train sources: {dict(train_sources.most_common())}")
    print(f"  train concepts: {dict(train_concepts.most_common())}")


if __name__ == "__main__":
    main()
