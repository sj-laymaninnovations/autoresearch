"""
audit_distillation_v1.py — Run the v2 distillation prompt against
real harvested commits, using local LM Studio teachers (no API cost).

Output:
  bet_a/reports/audit_responses_v1.jsonl   one record per (commit, teacher)
"""
from __future__ import annotations
import json, subprocess, time, urllib.request, urllib.error
from pathlib import Path

ROOT       = Path("/Users/seanjosiah/Documents/autoresearch/bet_a")
LM_HOST    = "http://192.168.1.169:1234"
TEACHERS   = ["qwen/qwen3.6-35b-a3b", "openai/gpt-oss-20b"]

# Commits to audit — chosen to cover all pattern types
COMMITS = [
    {"id": "x264_c1c9931d",  "repo": "x264_src",  "sha": "c1c9931dc87289b8aeba78150467f17bdb97d019",
     "expected_type": "B",   "note": "SVE/SVE2 substitution, 2875-char body, heavy benchmark tables"},
    {"id": "x264_4664f5aa",  "repo": "x264_src",  "sha": "4664f5aa66166ee3e11d99bfd6cbc7064abf76cc",
     "expected_type": "A",   "note": "scheduling conditional (in-order vs OoO), 1441-char body — matches Exemplar 1"},
    {"id": "x264_dc755eab",  "repo": "x264_src",  "sha": "dc755eabb9914e29df004243bee95013487753c3",
     "expected_type": "B",   "note": "rounded right shift, 1398-char body — matches Exemplar 2"},
    {"id": "dav1d_ec5c3052", "repo": "dav1d_src", "sha": "ec5c3052cfcd8741c7f1b4bf4ddb0b5dcf09a1b8",
     "expected_type": "C+E", "note": "lane writes + scalar stores (multi-insight), 6993-char body — matches Exemplar 3/4"},
    {"id": "dav1d_01558f3f", "repo": "dav1d_src", "sha": "01558f3f6609b750f6a850960119bec9e8fbe018",
     "expected_type": "B",   "note": "SVE2 4-element SDOT, 4977-char body, very heavy benchmark tables — stress test for decision (d)"},
]

SYSTEM_PROMPT = """\
You are reformulating commit messages from expert assembly programmers
into Q&A training pairs for a small (5M-parameter) language model. The
student model will produce ARM64 assembly with retrieval scaffolding
handling language-specific syntax. The student needs IDIOMS, PATTERNS,
and RATIONALE — not syntax tables.

The commit body below describes one or more optimization techniques.
Produce 1-3 Q&A pairs (one per separable insight) where each pair
satisfies ALL the following:

QUESTION rules:
  - Asks about the underlying principle, idiom, or pattern.
  - Phrased as a question a general ARM64 programmer might ask.
  - Does NOT mention specific codebases (x264, dav1d, FFmpeg, Linux,
    etc.) or specific function/file names. The question must be
    applicable beyond the source commit.

ANSWER rules:
  - Preserves the expert's claim verbatim in substance — DO NOT add
    facts not in the commit body.
  - 3-7 sentences of instructional prose.
  - INCLUDES the expert's measurement numbers as evidence. For
    multi-row benchmark tables, extract the RANGE (e.g.,
    "8-30% speedup across blend/avg/mask functions") plus the
    notable endpoints (worst case, best case). Do NOT dump the
    verbatim table.
  - MAY cite the domain context that ties to the measurements
    (e.g., "in H.264 dequantization kernels", "for AV1 motion
    compensation") because the measurements are tied to it.
  - Ends with a single line `#### <single-sentence summary>`
    capturing the takeaway. No trailing punctuation on the `####`
    line.

REFUSAL rules:
  - If the commit body lacks substantive rationale (e.g., is only
    a benchmark table with no explanatory text, or only repeats the
    subject line), return {"pairs": []} and explain in `notes`.
  - If the commit body is too vague to extract a generally-applicable
    idiom, return {"pairs": []} and explain in `notes`.

OUTPUT (strict JSON, no preamble, no markdown fences):
  {
    "pairs": [
      {"question": "...", "answer": "..."},
      {"question": "...", "answer": "..."}
    ],
    "notes": "optional brief reason if pairs is empty"
  }
"""

def git_show(repo_dir: Path, sha: str) -> tuple[str, str]:
    """Returns (subject, body) for the commit."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%s%n---SPLIT---%n%b", sha],
        cwd=repo_dir, capture_output=True, text=True
    ).stdout
    subj, _, body = out.partition("---SPLIT---\n")
    return subj.strip(), body.strip()

def call_lm(model: str, system: str, user: str, max_tokens: int = 4000) -> dict:
    # LM Studio doesn't accept OpenAI's {"type":"json_object"} for response_format.
    # Also qwen3.6-35b is a reasoning model — needs token headroom for the
    # internal CoT before the answer comes out.
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
        f"{LM_HOST}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode("utf-8", errors="replace")
        latency_ms = int((time.time() - t0) * 1000)
        return {"ok": True, "latency_ms": latency_ms, "raw": raw}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code} {e.reason}",
                "latency_ms": int((time.time() - t0) * 1000),
                "body": e.read().decode(errors='replace')[:400]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.time() - t0) * 1000)}

def main():
    out_path = ROOT / "reports" / "audit_responses_v1.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")
    print(f"Auditing {len(COMMITS)} commits × {len(TEACHERS)} teachers = {len(COMMITS)*len(TEACHERS)} calls")

    for commit in COMMITS:
        repo = ROOT / "raw" / commit["repo"]
        subj, body = git_show(repo, commit["sha"])
        user_msg = f"commit_subject: {subj}\ncommit_body: |\n  " + body.replace("\n", "\n  ")

        for teacher in TEACHERS:
            print(f"  [{commit['id']}] -> {teacher}", end=" ", flush=True)
            r = call_lm(teacher, SYSTEM_PROMPT, user_msg)
            content = None
            usage   = None
            parsed  = None
            parse_err = None
            if r["ok"]:
                try:
                    j = json.loads(r["raw"])
                    content = j["choices"][0]["message"]["content"]
                    usage   = j.get("usage")
                except Exception as e:
                    parse_err = f"outer-json: {e}"
                if content:
                    try:
                        parsed = json.loads(content)
                    except Exception as e:
                        parse_err = f"inner-json: {e}"
            rec = {
                "commit_id":     commit["id"],
                "expected_type": commit["expected_type"],
                "commit_note":   commit["note"],
                "teacher":       teacher,
                "latency_ms":    r["latency_ms"],
                "ok":            r["ok"],
                "error":         r.get("error"),
                "usage":         usage,
                "parse_err":     parse_err,
                "raw_content":   content,
                "parsed":        parsed,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            pair_count = len(parsed.get("pairs", [])) if isinstance(parsed, dict) else "ERR"
            print(f"  {r['latency_ms']}ms  pairs={pair_count}")

    fout.close()
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
