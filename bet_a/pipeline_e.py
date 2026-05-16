"""
pipeline_e.py — Multi-granularity in-source distillation.

Reads units from a curated/*_asm_units.jsonl (per-file output of
extract_asm_units.py). For each unit, picks the appropriate distillation
prompt based on granularity, calls the local gpt-oss-20b teacher, parses,
validates, and writes Q&A pairs to distilled/qa_records_e_v1.jsonl.

Designed to mirror the Pipeline F architecture (--resume, retry-on-parse-fail,
preprocessor skip logging) but with per-granularity prompts.

Run:
    python pipeline_e.py --in curated/qnnpack_asm_units.jsonl
    python pipeline_e.py --in ... --limit 10            # smoke
    python pipeline_e.py --in ... --resume               # skip done unit-ids
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
DISTILLED = ROOT / "distilled"
REPORTS   = ROOT / "reports"
DISTILLED.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

OUT_JSONL  = DISTILLED / "qa_records_e_v1.jsonl"
STATS_PATH = REPORTS / "pipeline_e_run_stats.json"
SKIP_LOG   = REPORTS / "pipeline_e_skipped.jsonl"

LM_LOCAL   = "http://127.0.0.1:1234"
TEACHER    = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Per-granularity prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
STRICT OUTPUT FORMAT (read this first):
  - Return ONE JSON object, no preamble, no markdown fences.
  - Schema: {"pairs": [{"question":"...","answer":"..."}], "notes":"..."}
  - Each answer MUST end with `\\n#### <single-sentence summary>` on its own line.
    Not appended to the last sentence. Not on the same line as prose.
  - The `####` line has no trailing punctuation.

QUESTION rules:
  - Asks about the underlying principle, idiom, or pattern.
  - Phrased as a question a general ARM64 programmer might ask.
  - Does NOT mention specific codebases (pytorch, qnnpack, ATen, etc.) or
    specific source-file paths or function-symbol names.
  - Codebase-agnostic. Re-usable beyond the source unit.

REFUSAL:
  - If the unit is too thin to support a generally-applicable idiom
    (e.g., a 2-line `# r5 := tmp` register naming with no broader concept),
    return {"pairs": [], "notes": "<reason>"}.
"""

PROMPT_COMMENT_BLOCK = _BASE_RULES + """\

TASK (comment_block granularity):
The input is a short comment block from an ARM64 assembly source file,
typically register-purpose maps, stack layouts, calling-convention notes,
or short rationale snippets. Produce 1 Q&A pair about the underlying
ARM64 idiom or convention the comment implies.

ANSWER rules:
  - 3-5 sentences of instructional prose, then the `####` line.
  - Preserves the expert's claim in substance — do NOT add facts the
    comment doesn't support.
  - If the comment is a register-purpose mapping, generalize to the
    idiom (e.g., "ARM64 sparse GEMM packA kernels typically allocate
    row-stride pointers to x5-x11").

INPUT:
  language: {language}
  source_context: {context}
  comment_text: |
    {text}
"""

PROMPT_FUNCTION = _BASE_RULES + """\

TASK (function granularity):
The input is a full ARM64 assembly function, including its surrounding
inline rationale comments. Produce 1-3 Q&A pairs about the key design
choices and techniques in the function. Each pair must add a NEW idea
not present in earlier pairs. If only one insight is genuinely
separable, return ONE pair.

ANSWER rules:
  - 4-7 sentences of instructional prose, then the `####` line.
  - Ground every claim in the function's inline comments OR in the
    instruction patterns directly visible in the code.
  - When citing specific instruction patterns, use the actual mnemonics
    (e.g., "LDP/STP pair loads", "FMLA accumulating multiply") rather
    than fabricated names.
  - Do NOT invent benchmark numbers. If the unit has none, do not
    introduce them.

PAIR-COUNT GUIDANCE:
  - If the function teaches just one technique → 1 pair
  - If the function teaches 2 separable techniques (e.g., "row-pointer
    setup with CSEL" AND "k_loop transposition") → 2 pairs
  - Max 3 pairs even for very long functions

INPUT:
  language: {language}
  function_name: <stripped per question rules; not given to you>
  unit_text: |
    {text}
"""

PROMPT_BLOCK = _BASE_RULES + """\

TASK (block granularity):
The input is a labelled sub-section of an ARM64 assembly function
(typically an inner loop body or a setup block), including its inline
rationale comments. Produce 1 Q&A pair about the technique this block
implements.

ANSWER rules:
  - 3-6 sentences of instructional prose, then the `####` line.
  - Ground every claim in the block's inline comments OR instruction
    pattern.
  - When the block is an inner loop, emphasize the per-iteration work
    and the data-flow pattern (which registers carry accumulation,
    which are reloaded per iteration, etc.).

INPUT:
  language: {language}
  block_label: <stripped per question rules; not given to you>
  enclosing_function: <stripped per question rules; not given to you>
  unit_text: |
    {text}
"""


# ---------------------------------------------------------------------------
# Preprocessing: skip too-small units, license boilerplate residue
# ---------------------------------------------------------------------------

def should_skip(rec: dict) -> tuple[bool, str]:
    text = rec["text"]
    if rec["granularity"] == "comment_block":
        # Strip comment leaders for length check
        body = "\n".join(
            re.sub(r"^\s*(?:#|//|;)\s?", "", l)
            for l in text.split("\n")
        ).strip()
        if len(body) < 50:
            return True, "comment_too_short"
        # Skip @-style attribute markers that aren't rationale
        if all(l.strip().startswith("@") for l in body.split("\n") if l.strip()):
            return True, "attribute_only"
    if rec["granularity"] == "function":
        if rec["n_lines"] < 8:
            return True, "function_too_short"
        if rec["n_lines"] > 600:
            return True, "function_too_long"  # token-budget guard
    if rec["granularity"] == "block":
        if rec["n_lines"] < 4:
            return True, "block_too_short"
        if rec["n_lines"] > 400:
            return True, "block_too_long"
    return False, ""


# ---------------------------------------------------------------------------
# LM Studio call (mirrors pipeline_f patterns)
# ---------------------------------------------------------------------------

def call_lm(system: str, user: str, max_tokens: int = 4000) -> dict:
    body = json.dumps({
        "model": TEACHER,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.3,
        "max_tokens":  max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{LM_LOCAL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode("utf-8", errors="replace")
        return {"ok": True, "latency_ms": int((time.time()-t0)*1000), "raw": raw}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}", "latency_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.time()-t0)*1000)}


_JSON_OBJ_RE = re.compile(r'\{.*\}', re.DOTALL)

def parse_response(raw: str):
    try:
        j = json.loads(raw)
        content = j["choices"][0]["message"]["content"]
        usage   = j.get("usage")
    except Exception as e:
        return None, None, f"outer-json: {e}"
    if not content:
        return None, usage, "empty content"
    try:
        return json.loads(content), usage, None
    except Exception:
        m = _JSON_OBJ_RE.search(content)
        if m:
            try:
                return json.loads(m.group(0)), usage, "had-preamble"
            except Exception as e2:
                return None, usage, f"inner-json (extract failed): {e2}"
        return None, usage, "inner-json"


# ---------------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------------

_LEAK_TERMS = ("pytorch", "qnnpack", "aten", "libavcodec", "ffmpeg", "dav1d",
                "x264")

def validate_pair(p: dict) -> tuple[bool, list[str]]:
    warns: list[str] = []
    q = (p or {}).get("question", "") or ""
    a = (p or {}).get("answer", "") or ""
    if not q or not a:
        return False, ["empty question or answer"]
    if "####" not in a:
        return False, ["missing #### line"]
    if "\n####" not in a:
        warns.append("#### not on own line")
    for term in _LEAK_TERMS:
        if term in q.lower():
            warns.append(f"codebase name leak in question: {term}")
    return True, warns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",  required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N units (smoke testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip unit-ids already in qa_records_e_v1.jsonl")
    args = ap.parse_args()

    units = [json.loads(l) for l in args.inp.open()]
    print(f"loaded {len(units)} units from {args.inp.name}")
    by_g = {}
    for u in units:
        by_g[u["granularity"]] = by_g.get(u["granularity"], 0) + 1
    for g, n in sorted(by_g.items()):
        print(f"  {g:<14s} {n}")

    done_ids: set[str] = set()
    if args.resume and OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["unit_id"])
                except Exception:
                    pass
        print(f"  resume: {len(done_ids)} unit-ids already processed")

    fout  = open(OUT_JSONL, "a")
    fskip = open(SKIP_LOG,  "a")

    stats = {
        "started":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_units_total":    len(units),
        "n_already_done":   len(done_ids),
        "n_skipped_preprocessor": 0,
        "n_called_teacher": 0,
        "n_parse_failed":   0,
        "n_validation_failed": 0,
        "n_pairs_written":  0,
        "n_refused_by_teacher": 0,
        "n_retries_used":   0,
        "n_retries_recovered": 0,
        "n_retries_failed": 0,
        "by_granularity":   {g: 0 for g in by_g},
        "teacher_latency_ms": [],
    }
    t_start = time.time()
    iter_units = units if args.limit is None else units[:args.limit]

    for idx, u in enumerate(iter_units):
        if u["id"] in done_ids:
            continue

        # Preprocessor skip
        skip, reason = should_skip(u)
        if skip:
            stats["n_skipped_preprocessor"] += 1
            fskip.write(json.dumps({"id": u["id"], "granularity": u["granularity"],
                                     "reason": reason}) + "\n")
            fskip.flush()
            continue

        # Choose prompt per granularity. Avoid str.format() because the
        # prompts contain literal `{"pairs": ...}` JSON examples; use
        # plain replacements instead.
        g = u["granularity"]
        if g == "comment_block":
            tpl = PROMPT_COMMENT_BLOCK
        elif g == "function":
            tpl = PROMPT_FUNCTION
        elif g == "block":
            tpl = PROMPT_BLOCK
        else:
            stats["n_skipped_preprocessor"] += 1
            fskip.write(json.dumps({"id": u["id"], "reason": f"unknown granularity: {g}"}) + "\n")
            continue
        system = (
            tpl
            .replace("{language}", u["language"])
            .replace("{context}",  u.get("context") or "(none)")
            .replace("{text}",     u["text"].replace("\n", "\n    "))
        )
        user = "Produce the JSON output now."
        r = call_lm(system, user, max_tokens=4000)
        stats["n_called_teacher"] += 1
        stats["teacher_latency_ms"].append(r["latency_ms"])

        # Retry on HTTP error
        if not r["ok"]:
            time.sleep(1)
            r2 = call_lm(system, user, max_tokens=4000)
            stats["n_called_teacher"] += 1
            stats["teacher_latency_ms"].append(r2["latency_ms"])
            stats["n_retries_used"] += 1
            if not r2["ok"]:
                stats["n_retries_failed"] += 1
                stats["n_parse_failed"] += 1
                fskip.write(json.dumps({"id": u["id"], "reason": "http_error",
                                         "error": r2.get("error")}) + "\n")
                fskip.flush()
                print(f"  [{idx+1}/{len(iter_units)}] {u['id']} {g}  HTTP-ERR {r2.get('error')}")
                continue
            stats["n_retries_recovered"] += 1
            r = r2

        parsed, usage, parse_err = parse_response(r["raw"])

        # Retry on parse failure
        if parsed is None:
            time.sleep(1)
            r2 = call_lm(system, user, max_tokens=4000)
            stats["n_called_teacher"] += 1
            stats["teacher_latency_ms"].append(r2["latency_ms"])
            stats["n_retries_used"] += 1
            if r2["ok"]:
                parsed2, usage2, parse_err2 = parse_response(r2["raw"])
                if parsed2 is not None:
                    stats["n_retries_recovered"] += 1
                    parsed, usage, parse_err = parsed2, usage2, None
                else:
                    stats["n_retries_failed"] += 1
            else:
                stats["n_retries_failed"] += 1

        if parsed is None:
            stats["n_parse_failed"] += 1
            fskip.write(json.dumps({"id": u["id"], "reason": "parse_failed",
                                     "error": parse_err}) + "\n")
            fskip.flush()
            print(f"  [{idx+1}/{len(iter_units)}] {u['id']} {g}  PARSE-FAIL {parse_err}")
            continue

        pairs = parsed.get("pairs", [])
        notes = parsed.get("notes")
        if not pairs:
            stats["n_refused_by_teacher"] += 1
            fskip.write(json.dumps({"id": u["id"], "reason": "teacher_refused",
                                     "teacher_notes": notes}) + "\n")
            fskip.flush()
            print(f"  [{idx+1}/{len(iter_units)}] {u['id']} {g}  REFUSED  notes={notes!r}")
            continue

        n_written_here = 0
        n_validation_fail_here = 0
        for p in pairs:
            ok, warns = validate_pair(p)
            if not ok:
                n_validation_fail_here += 1
                continue
            rec = {
                "unit_id":     u["id"],
                "file_path":   u["file_path"],
                "repo":        u["repo"],
                "language":    u["language"],
                "license":     u["license"],
                "granularity": u["granularity"],
                "name":        u.get("name"),
                "context":     u.get("context"),
                "question":    p["question"],
                "answer":      p["answer"],
                "teacher":     TEACHER,
                "warnings":    warns,
                "notes":       notes,
                "usage":       usage,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written_here += 1
        fout.flush()
        stats["n_pairs_written"]     += n_written_here
        stats["n_validation_failed"] += n_validation_fail_here
        stats["by_granularity"][g]    = stats["by_granularity"].get(g, 0) + n_written_here

        dt = time.time() - t_start
        eta_s = (dt / (idx + 1)) * (len(iter_units) - idx - 1)
        print(f"  [{idx+1}/{len(iter_units)}] {u['id']} {g:<13s} "
              f"{r['latency_ms']/1000:>5.1f}s  +{n_written_here}  total={stats['n_pairs_written']}  "
              f"eta={eta_s/60:.1f}m")

    fout.close()
    fskip.close()
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["wall_time_s"] = round(time.time() - t_start, 1)
    if stats["teacher_latency_ms"]:
        s = sorted(stats["teacher_latency_ms"])
        stats["teacher_latency_median_s"] = round(s[len(s)//2]/1000, 1)
        stats["teacher_latency_p95_s"]    = round(s[int(len(s)*0.95)]/1000, 1) if len(s) >= 20 else None
    del stats["teacher_latency_ms"]

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT_JSONL}")
    print(f"wrote {STATS_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
