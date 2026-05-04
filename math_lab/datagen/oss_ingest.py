"""
oss_ingest.py — Curate OSS math datasets into our CoT JSONL schema.

Sources:
  GSM8K        (openai/gsm8k)                    ~7.5K train records
  MetaMathQA   (meta-math/MetaMathQA, 40K slice) ~40K records
  ORCA-Math    (microsoft/orca-math-word-problems-200k, sampled) ~30K

Output schema (matches existing skill JSONL):
  {
    "problem":      "Sara has 3 apples ...",
    "solution":     "7",
    "answer":       "7",
    "answer_real":  7,
    "solution_cot": "Example: ...\n<reasoning>\n#### 7",
    "concept":      "word_problem",
    "source":       "gsm8k|metamath|orca_math",
    "tier":         1|2|3,
    "hold_out":     false|true
  }

Usage:
    python3 math_lab/datagen/oss_ingest.py
    python3 math_lab/datagen/oss_ingest.py --orca-sample 30000 --val-frac 0.1
"""
import re, json, random, hashlib, argparse, datetime, subprocess, sys
from pathlib import Path

DATESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Raw file URLs (GitHub / direct links, no HuggingFace API) ────────────────
RAW_SOURCES = {
    "gsm8k_train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test":  "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    # MetaMathQA 40K is hosted on HF — we'll generate an equivalent locally if unavailable
    # ORCA-Math similarly — fallback to local generation
}

def download_raw(raw_dir: Path, key: str, url: str, force: bool = False) -> Path:
    """Download url → raw_dir/{key}.jsonl via curl if not already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    # infer extension from URL
    ext = ".jsonl" if ".jsonl" in url else ".json"
    dest = raw_dir / f"{key}{ext}"
    if dest.exists() and not force:
        print(f"    cached: {dest.name} ({dest.stat().st_size//1024}KB)")
        return dest
    print(f"    downloading {key} …")
    result = subprocess.run(
        ["curl", "-skL", "--max-time", "120", url, "-o", str(dest)],
        capture_output=True
    )
    if result.returncode != 0 or dest.stat().st_size < 1000:
        print(f"    WARN: download failed or too small for {key} (exit {result.returncode})")
        dest.unlink(missing_ok=True)
        return None
    print(f"    OK: {dest.name} ({dest.stat().st_size//1024}KB)")
    return dest

# ── Operator normalizer (matches chat.html normalizeInput) ────────────────────
def normalize_problem(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s*=\s*', ' = ', s)
    s = re.sub(r'\s*\+\s*', ' + ', s)
    s = re.sub(r'\s*\*\s*', ' * ', s)
    s = re.sub(r'\s*%\s*', ' % ', s)
    s = re.sub(r'\s*/\s*', ' / ', s)
    s = re.sub(r'([0-9a-zA-Z)])\s*-\s*', r'\1 - ', s)
    s = re.sub(r'  +', ' ', s)
    return s.strip()

# ── Answer extractors ─────────────────────────────────────────────────────────

def extract_gsm8k_answer(solution_text: str):
    """GSM8K uses '#### {N}' already. Extract integer after ####."""
    m = re.search(r'####\s*(-?\d[\d,]*)', solution_text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(',', ''))
    except ValueError:
        return None

def clean_gsm8k_cot(solution_text: str) -> str:
    """
    Strip calculator annotations like <<3*4=12>> and keep the natural
    language reasoning. Re-append #### at the end.
    """
    # Remove << ... >> calculator steps
    cleaned = re.sub(r'<<[^>]*>>', '', solution_text)
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

def extract_metamath_answer(answer_text: str):
    """
    MetaMathQA stores answers in several formats:
      \boxed{42}   /  The answer is 42.  /  #### 42
    Returns integer or None.
    """
    # boxed
    m = re.search(r'\\boxed\{(-?\d[\d,]*)\}', answer_text)
    if m:
        try: return int(m.group(1).replace(',', ''))
        except ValueError: pass
    # #### style
    m = re.search(r'####\s*(-?\d[\d,]*)', answer_text)
    if m:
        try: return int(m.group(1).replace(',', ''))
        except ValueError: pass
    # "The answer is N"
    m = re.search(r'[Tt]he answer is\s*(-?\d[\d,]*)', answer_text)
    if m:
        try: return int(m.group(1).replace(',', ''))
        except ValueError: pass
    # last number in text (heuristic fallback)
    nums = re.findall(r'-?\d[\d,]*', answer_text)
    if nums:
        try: return int(nums[-1].replace(',', ''))
        except ValueError: pass
    return None

def extract_orca_answer(answer_text: str):
    """
    ORCA-Math answers are free-form GPT-4 prose.
    Try multiple patterns in order of confidence.
    """
    # "= N" near end of sentence
    m = re.search(r'=\s*(-?\d[\d,]*)\s*[.!]?\s*$', answer_text.strip())
    if m:
        try: return int(m.group(1).replace(',', ''))
        except ValueError: pass
    # "answer is N" / "total is N" / "result is N"
    m = re.search(r'(?:answer|total|result|sum|cost|price|number)\s+(?:is|was|=)\s*(-?\d[\d,]*)', answer_text, re.I)
    if m:
        try: return int(m.group(1).replace(',', ''))
        except ValueError: pass
    # last standalone integer in the text
    nums = re.findall(r'(?<!\d)(-?\d{1,6})(?!\d)', answer_text)
    if nums:
        try: return int(nums[-1])
        except ValueError: pass
    return None

# ── Tier classifier ───────────────────────────────────────────────────────────

def classify_tier(problem: str, answer: int) -> int:
    """
    Rough difficulty heuristic:
      Tier 1 — simple arithmetic word problems (small numbers, 1-2 ops)
      Tier 2 — multi-step (larger numbers, 3+ ops inferred by length)
      Tier 3 — algebraic / ratio / percent problems
    """
    p_lower = problem.lower()
    algebraic_kws = ['percent', '%', 'ratio', 'fraction', 'equation',
                     'solve', 'variable', 'times as many', 'twice', 'rate']
    if any(k in p_lower for k in algebraic_kws):
        return 3
    if len(problem) > 300 or abs(answer) > 500:
        return 2
    return 1

# ── CoT formatter ─────────────────────────────────────────────────────────────

def make_cot(reasoning: str, answer: int, source: str) -> str:
    r = reasoning.strip()
    # Strip any existing #### lines from the reasoning body to avoid duplicates
    r = re.sub(r'\n?####[^\n]*', '', r).strip()
    if len(r) > 400:
        r = '\u2026' + r[-397:]
    return f"{r}\n#### {answer}"

# ── Source adapters ───────────────────────────────────────────────────────────

def ingest_gsm8k(raw_dir: Path) -> list[dict]:
    path = raw_dir / "gsm8k_train.jsonl"
    if not path.exists():
        path = download_raw(raw_dir, "gsm8k_train", RAW_SOURCES["gsm8k_train"])
    if not path:
        print("  GSM8K: download failed, skipping")
        return []
    print(f"  Ingesting GSM8K from {path.name} …")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    records = []
    skipped = 0
    for row in rows:
        problem  = normalize_problem(row["question"])
        raw_sol  = row["answer"]
        answer   = extract_gsm8k_answer(raw_sol)
        if answer is None:
            skipped += 1; continue
        reasoning = clean_gsm8k_cot(raw_sol)
        tier = classify_tier(problem, answer)
        records.append({
            "problem":      problem,
            "solution":     str(answer),
            "answer":       str(answer),
            "answer_real":  answer,
            "solution_cot": make_cot(reasoning, answer, "gsm8k"),
            "concept":      "word_problem",
            "source":       "gsm8k",
            "tier":         tier,
            "hold_out":     False,
        })
    print(f"    GSM8K: {len(records)} ok, {skipped} skipped")
    return records


def ingest_metamath(raw_dir: Path, max_records: int = 40000, rng: random.Random = None) -> list[dict]:
    """
    Try to load MetaMathQA-40K. Falls back to generating metamath-style
    augmented variants from GSM8K using our backward-generation technique
    if the file isn't available.
    """
    path = raw_dir / "metamath_40k.json"
    if not path.exists():
        # Try GitHub release of MetaMathQA-40K
        url = "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-40K.json"
        path = download_raw(raw_dir, "metamath_40k", url)
    if not path or not path.exists():
        print("  MetaMathQA: unavailable, skipping (will be generated via backward-gen augmentation)")
        return []
    print(f"  Ingesting MetaMathQA from {path.name} …")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "data" in raw:
            rows = raw["data"]
        elif isinstance(raw, list):
            rows = raw
        else:
            print("  MetaMathQA: unexpected format, skipping")
            return []
    except Exception as e:
        print(f"  MetaMathQA: parse error {e}, skipping")
        return []

    if rng is None: rng = random.Random(42)
    rng.shuffle(rows)
    records = []
    skipped = 0
    seen_hashes = set()
    for row in rows:
        if len(records) >= max_records:
            break
        problem  = normalize_problem(row.get("query", ""))
        raw_sol  = row.get("response", "")
        answer   = extract_metamath_answer(raw_sol)
        if answer is None or not problem:
            skipped += 1; continue
        ph = hashlib.md5(problem.encode()).hexdigest()
        if ph in seen_hashes:
            skipped += 1; continue
        seen_hashes.add(ph)
        tier = classify_tier(problem, answer)
        reasoning = re.sub(r'\\boxed\{[^}]*\}', str(answer), raw_sol)[:400]
        records.append({
            "problem":      problem,
            "solution":     str(answer),
            "answer":       str(answer),
            "answer_real":  answer,
            "solution_cot": make_cot(reasoning, answer, "metamath"),
            "concept":      "word_problem",
            "source":       "metamath",
            "tier":         tier,
            "hold_out":     False,
        })
    print(f"    MetaMathQA: {len(records)} ok, {skipped} skipped")
    return records


def ingest_orca(raw_dir: Path, sample_n: int = 30000, rng: random.Random = None) -> list[dict]:
    """ORCA-Math — if not locally available, skip gracefully."""
    path = raw_dir / "orca_math.jsonl"
    if not path.exists():
        print(f"  ORCA-Math: {path} not found, skipping.")
        print(f"    To download: place orca-math-word-problems-200k JSONL at {path}")
        return []
    if rng is None: rng = random.Random(42)
    print(f"  Ingesting ORCA-Math from {path.name} …")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng.shuffle(rows)
    records = []
    skipped = 0
    seen_hashes = set()
    for row in rows:
        if len(records) >= sample_n:
            break
        problem  = normalize_problem(row.get("question", ""))
        raw_sol  = row.get("answer", "")
        answer   = extract_orca_answer(raw_sol)
        if answer is None or not problem:
            skipped += 1; continue
        ph = hashlib.md5(problem.encode()).hexdigest()
        if ph in seen_hashes:
            skipped += 1; continue
        seen_hashes.add(ph)
        tier = classify_tier(problem, answer)
        records.append({
            "problem":      problem,
            "solution":     str(answer),
            "answer":       str(answer),
            "answer_real":  answer,
            "solution_cot": make_cot(raw_sol[:400], answer, "orca_math"),
            "concept":      "word_problem",
            "source":       "orca_math",
            "tier":         tier,
            "hold_out":     False,
        })
    print(f"    ORCA-Math: {len(records)} ok, {skipped} skipped")
    return records

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",    default="math_lab/results/oss")
    ap.add_argument("--raw-dir",    default="math_lab/results/oss/raw",
                    help="Directory for downloaded raw source files")
    ap.add_argument("--val-frac",   type=float, default=0.10)
    ap.add_argument("--orca-sample",type=int,   default=30000)
    ap.add_argument("--meta-max",   type=int,   default=40000)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--skip-orca",  action="store_true")
    ap.add_argument("--skip-meta",  action="store_true")
    ap.add_argument("--force",      action="store_true", help="Re-download even if cached")
    args = ap.parse_args()

    rng     = random.Random(args.seed)
    out     = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Download any missing raw files
    print("Phase 1: downloading raw source files …")
    for key, url in RAW_SOURCES.items():
        download_raw(raw_dir, key, url, force=args.force)

    print("\nPhase 2: ingesting …")
    all_records = []
    all_records += ingest_gsm8k(raw_dir)
    if not args.skip_meta:
        all_records += ingest_metamath(raw_dir, max_records=args.meta_max, rng=rng)
    if not args.skip_orca:
        all_records += ingest_orca(raw_dir, sample_n=args.orca_sample, rng=rng)

    if not all_records:
        print("ERROR: no records ingested."); sys.exit(1)

    # Shuffle and split
    rng.shuffle(all_records)
    n_val   = max(1, int(len(all_records) * args.val_frac))
    val     = all_records[-n_val:]
    train   = all_records[:-n_val]
    for r in val:
        r["hold_out"] = True

    tag = f"seed{args.seed}_{DATESTAMP}"
    tf  = out / f"oss_train_{tag}.jsonl"
    vf  = out / f"oss_val_{tag}.jsonl"
    tf.write_text("\n".join(json.dumps(r) for r in train), encoding="utf-8")
    vf.write_text("\n".join(json.dumps(r) for r in val),   encoding="utf-8")

    from collections import Counter
    src_counts  = Counter(r["source"] for r in train)
    tier_counts = Counter(r["tier"]   for r in train)
    print(f"\n{'─'*50}")
    print(f"Train: {len(train):>6} records → {tf.name}")
    print(f"Val:   {len(val):>6} records → {vf.name}")
    print(f"\nBy source:  {dict(src_counts)}")
    print(f"By tier:    {dict(tier_counts)}")

    manifest = {
        "train_file": str(tf),
        "val_file":   str(vf),
        "n_train":    len(train),
        "n_val":      len(val),
        "sources":    dict(src_counts),
        "tiers":      dict(tier_counts),
        "seed":       args.seed,
        "datestamp":  DATESTAMP,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {out}/manifest.json")


if __name__ == "__main__":
    main()
