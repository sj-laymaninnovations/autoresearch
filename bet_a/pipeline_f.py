"""
pipeline_f.py — Full distillation pass.

For each unique commit in the curated hunks:
  1. git_show the subject + body
  2. preprocessor.classify_body → skip Type-D, skip too_short
  3. send to local gpt-oss-20b via LM Studio with v4 prompt
  4. parse, validate JSON shape, validate `####` line presence
  5. write to bet_a/distilled/qa_records_v1.jsonl (one line per pair)
  6. write bet_a/reports/pipeline_f_run_stats.json with final tallies

Run:
    python pipeline_f.py
    python pipeline_f.py --resume        # skip commits already in output
    python pipeline_f.py --limit 5       # smoke
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessor import classify_body, dedupe_commits

ROOT      = Path(__file__).resolve().parent
RAW_DIR   = ROOT / "raw"
CURATED   = ROOT / "curated"
DISTILLED = ROOT / "distilled"
REPORTS   = ROOT / "reports"
DISTILLED.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

OUT_JSONL    = DISTILLED / "qa_records_v1.jsonl"
STATS_PATH   = REPORTS / "pipeline_f_run_stats.json"
SKIP_LOG     = REPORTS / "pipeline_f_skipped.jsonl"

LM_LOCAL     = "http://127.0.0.1:1234"
TEACHER      = "openai/gpt-oss-20b"

REPOS = {
    "x264_src":   CURATED / "x264_hunks.jsonl",
    "dav1d_src":  CURATED / "dav1d_hunks.jsonl",
    "ffmpeg_src": CURATED / "ffmpeg_hunks.jsonl",
    "linux_src":  CURATED / "linux_arm64_hunks.jsonl",
}

# v4 prompt — locked in distillation_prompt_v4.md
SYSTEM_PROMPT = """\
You are reformulating commit messages from expert assembly programmers
into Q&A training pairs for a small (5M-parameter) language model.

STRICT OUTPUT FORMAT (read this first):
  - Return ONE JSON object, no preamble, no markdown fences.
  - Schema: {"pairs": [{"question":"...","answer":"..."}], "notes":"..."}
  - Each answer MUST end with `\\n#### <single-sentence summary>` on its own line.
    Not appended to the last sentence. Not on the same line as prose.
  - The `####` line has no trailing punctuation.

TASK:
The commit body below describes one or more optimization techniques.
Produce 1-3 Q&A pairs.

PAIR-COUNT RULE (this is the load-bearing rule):
  - Each pair must add a NEW idea not present in earlier pairs.
  - If the commit body describes 2 or 3 separable techniques
    (e.g., "do X" AND "in case Y, also do Z"), produce ONE PAIR
    PER TECHNIQUE. Folding distinct techniques into a single pair is
    INCORRECT and will degrade training quality.
  - If only ONE insight is present, return ONE pair.
  - Do NOT paraphrase the same insight twice with different wording.

QUESTION rules:
  - Asks about the underlying principle, idiom, or pattern.
  - Phrased as a question a general ARM64 programmer might ask.
  - Does NOT mention specific codebases (x264, dav1d, FFmpeg, Linux,
    etc.) or specific function/file names.
  - Codebase-agnostic. Re-usable beyond the source commit.

ANSWER rules:
  - 3-7 sentences of instructional prose, then the `####` line.
  - Preserves the expert's claim verbatim in substance — do NOT add
    facts not in the commit body.
  - When the body contains a benchmark table, ALWAYS include BOTH the
    worst-case and the best-case data point with the specific CPU and
    operation that produced each endpoint. Then summarize the range.
    Do NOT dump the verbatim table.
  - MAY cite the domain context that ties to the measurements
    (e.g., "in H.264 dequantization", "for AV1 motion compensation").

REFUSAL:
  - If the commit body is too vague (under 3 explanatory sentences) to
    extract a generally-applicable idiom, return:
      {"pairs": [], "notes": "body too vague"}
  - Refusal is rare; most commits have substantive rationale.

REMEMBER: every answer ends with `\\n#### <one-line summary>` on its
own line. No exceptions.
"""


def git_show(repo_dir: Path, sha: str) -> tuple[str, str]:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%s%n---SPLIT---%n%b", sha],
        cwd=repo_dir, capture_output=True, text=True,
    ).stdout
    subj, _, body = out.partition("---SPLIT---\n")
    return subj.strip(), body.strip()


def call_lm(model: str, system: str, user: str, max_tokens: int = 4000) -> dict:
    body = json.dumps({
        "model": model,
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
        with urllib.request.urlopen(req, timeout=240) as r:
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


def validate_pair(p: dict) -> tuple[bool, list[str]]:
    """Return (ok, warnings). Hard-fail conditions return ok=False."""
    warns = []
    q = (p or {}).get("question", "") or ""
    a = (p or {}).get("answer", "") or ""
    if not q or not a:
        return False, ["empty question or answer"]
    if "####" not in a:
        return False, ["missing #### line"]
    if "\n####" not in a:
        warns.append("#### not on own line")
    # Codebase-name leak detection in QUESTION
    for forbidden in ("x264", "dav1d", "ffmpeg", "libavcodec", "FFmpeg"):
        if forbidden.lower() in q.lower():
            warns.append(f"codebase name leak in question: {forbidden}")
    return True, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N commits (smoke testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip commits already in qa_records_v1.jsonl")
    args = ap.parse_args()

    # 1. Load all hunk records, group by SHA
    all_hunks = []
    for repo_id, path in REPOS.items():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                rec["repo"] = repo_id
                all_hunks.append(rec)
    commits = dedupe_commits(all_hunks)
    print(f"loaded {len(all_hunks)} hunks → {len(commits)} unique commits")

    # 2. Map sha → repo (use the first hunk's repo)
    sha_to_repo = {h["sha"]: h["repo"] for h in all_hunks}

    # 3. Resume support
    done_shas = set()
    if args.resume and OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done_shas.add(json.loads(line)["sha"])
                except Exception:
                    pass
        print(f"  resume: {len(done_shas)} commits already in {OUT_JSONL.name}")

    # 4. Process
    fout    = open(OUT_JSONL, "a")
    fskip   = open(SKIP_LOG,  "a")
    stats = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_commits_total":      len(commits),
        "n_already_done":       len(done_shas),
        "n_skipped_too_short":  0,
        "n_skipped_type_d":     0,
        "n_called_teacher":     0,
        "n_parse_failed":       0,
        "n_validation_failed":  0,
        "n_pairs_written":      0,
        "n_refused_by_teacher": 0,
        "teacher_latency_ms":   [],
    }
    t_start = time.time()

    iter_commits = commits if args.limit is None else commits[:args.limit]
    for idx, c in enumerate(iter_commits):
        sha = c["sha"]
        if sha in done_shas:
            continue
        repo = sha_to_repo.get(sha)
        if not repo:
            continue
        subj, body = git_show(RAW_DIR / repo, sha)

        cls, body_stats = classify_body(body)
        if cls == "too_short":
            stats["n_skipped_too_short"] += 1
            fskip.write(json.dumps({"sha": sha, "reason": "too_short",
                                     "body_stats": body_stats}) + "\n")
            fskip.flush()
            continue
        if cls == "type_D":
            stats["n_skipped_type_d"] += 1
            fskip.write(json.dumps({"sha": sha, "subject": subj,
                                     "reason": "type_D", "body_stats": body_stats}) + "\n")
            fskip.flush()
            continue

        user_msg = f"commit_subject: {subj}\ncommit_body: |\n  " + body.replace("\n", "\n  ")
        r = call_lm(TEACHER, SYSTEM_PROMPT, user_msg)
        stats["n_called_teacher"] += 1
        stats["teacher_latency_ms"].append(r["latency_ms"])

        if not r["ok"]:
            # One retry on HTTP error
            time.sleep(1)
            r2 = call_lm(TEACHER, SYSTEM_PROMPT, user_msg)
            stats["n_called_teacher"] += 1
            stats["teacher_latency_ms"].append(r2["latency_ms"])
            if not r2["ok"]:
                stats["n_parse_failed"] += 1
                stats["n_retries_used"] = stats.get("n_retries_used", 0) + 1
                stats["n_retries_failed"] = stats.get("n_retries_failed", 0) + 1
                fskip.write(json.dumps({"sha": sha, "reason": "http_error",
                                         "error": r2.get("error"),
                                         "retried": True}) + "\n")
                fskip.flush()
                print(f"  [{idx+1}/{len(iter_commits)}] {sha[:8]}  HTTP ERROR (retry failed)  {r2.get('error')}")
                continue
            r = r2
            stats["n_retries_used"] = stats.get("n_retries_used", 0) + 1
            stats["n_retries_recovered"] = stats.get("n_retries_recovered", 0) + 1

        parsed, usage, parse_err = parse_response(r["raw"])
        if parsed is None:
            # One retry on parse failure — gpt-oss-20b sometimes emits malformed
            # JSON; a second call with same inputs usually succeeds.
            time.sleep(1)
            r2 = call_lm(TEACHER, SYSTEM_PROMPT, user_msg)
            stats["n_called_teacher"] += 1
            stats["teacher_latency_ms"].append(r2["latency_ms"])
            if r2["ok"]:
                parsed2, usage2, parse_err2 = parse_response(r2["raw"])
                if parsed2 is not None:
                    parsed = parsed2
                    usage  = usage2
                    parse_err = None
                    stats["n_retries_used"] = stats.get("n_retries_used", 0) + 1
                    stats["n_retries_recovered"] = stats.get("n_retries_recovered", 0) + 1
                else:
                    stats["n_retries_used"] = stats.get("n_retries_used", 0) + 1
                    stats["n_retries_failed"] = stats.get("n_retries_failed", 0) + 1
            else:
                stats["n_retries_used"] = stats.get("n_retries_used", 0) + 1
                stats["n_retries_failed"] = stats.get("n_retries_failed", 0) + 1

        if parsed is None:
            stats["n_parse_failed"] += 1
            fskip.write(json.dumps({"sha": sha, "reason": "parse_failed",
                                     "error": parse_err,
                                     "retried": True}) + "\n")
            fskip.flush()
            print(f"  [{idx+1}/{len(iter_commits)}] {sha[:8]}  PARSE FAIL (retry failed)  {parse_err}")
            continue

        pairs = parsed.get("pairs", [])
        notes = parsed.get("notes")
        if len(pairs) == 0:
            stats["n_refused_by_teacher"] += 1
            fskip.write(json.dumps({"sha": sha, "reason": "teacher_refused",
                                     "teacher_notes": notes}) + "\n")
            fskip.flush()
            print(f"  [{idx+1}/{len(iter_commits)}] {sha[:8]}  REFUSED  notes={notes}")
            continue

        n_written_here = 0
        n_validation_fail_here = 0
        for p in pairs:
            ok, warns = validate_pair(p)
            if not ok:
                n_validation_fail_here += 1
                continue
            rec = {
                "sha":            sha,
                "repo":           repo,
                "subject":        subj,
                "question":       p["question"],
                "answer":         p["answer"],
                "teacher":        TEACHER,
                "prompt_version": "v4",
                "warnings":       warns,
                "notes":          notes,
                "usage":          usage,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written_here += 1
        fout.flush()
        stats["n_pairs_written"]      += n_written_here
        stats["n_validation_failed"]  += n_validation_fail_here

        dt = time.time() - t_start
        eta_s = (dt / (idx + 1)) * (len(iter_commits) - idx - 1)
        print(f"  [{idx+1}/{len(iter_commits)}] {sha[:8]} ({repo})  "
              f"{r['latency_ms']/1000:>5.1f}s  pairs+{n_written_here}  total={stats['n_pairs_written']}  "
              f"eta={eta_s/60:.1f}m")

    fout.close()
    fskip.close()
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["wall_time_s"] = round(time.time() - t_start, 1)
    if stats["teacher_latency_ms"]:
        s = sorted(stats["teacher_latency_ms"])
        n = len(s)
        stats["teacher_latency_median_s"] = round(s[n//2]/1000, 1)
        stats["teacher_latency_p95_s"]    = round(s[int(n*0.95)]/1000, 1) if n >= 20 else None
    del stats["teacher_latency_ms"]  # don't dump the raw list

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT_JSONL}")
    print(f"wrote {STATS_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
