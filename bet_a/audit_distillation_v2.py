"""
audit_distillation_v2.py — Re-run distillation audit with prompt v3.

Two teacher backends:
  - openai/gpt-oss-20b via local LM Studio (http://127.0.0.1:1234)
  - claude-sonnet-4 via Abacus RouteLLM (cost-guarded; smallest commits first)

6 commits = 5 real (same as v1) + 1 synthetic Type-D bare-benchmark.

Output: bet_a/reports/audit_responses_v2.jsonl
"""
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path("/Users/seanjosiah/Documents/autoresearch/bet_a")

LM_LOCAL  = "http://127.0.0.1:1234"
ABACUS    = "https://routellm.abacus.ai/v1/chat/completions"
ABACUS_KEY = "s2_4a6f06a4058e4694bc066b8766f31d52"

LOCAL_TEACHER  = "openai/gpt-oss-20b"
ABACUS_TEACHER = "claude-3-5-sonnet"  # claude-sonnet-4 if Abacus has it; auto-fallback below

# Real commits from audit v1 + 1 synthetic Type-D
COMMITS = [
    {"id": "x264_c1c9931d",  "repo": "x264_src",  "sha": "c1c9931dc87289b8aeba78150467f17bdb97d019",
     "expected_type": "B",   "body_chars": 2875},
    {"id": "x264_4664f5aa",  "repo": "x264_src",  "sha": "4664f5aa66166ee3e11d99bfd6cbc7064abf76cc",
     "expected_type": "A",   "body_chars": 1441},
    {"id": "x264_dc755eab",  "repo": "x264_src",  "sha": "dc755eabb9914e29df004243bee95013487753c3",
     "expected_type": "B",   "body_chars": 1398},
    {"id": "dav1d_ec5c3052", "repo": "dav1d_src", "sha": "ec5c3052cfcd8741c7f1b4bf4ddb0b5dcf09a1b8",
     "expected_type": "C+E", "body_chars": 6993},
    {"id": "dav1d_01558f3f", "repo": "dav1d_src", "sha": "01558f3f6609b750f6a850960119bec9e8fbe018",
     "expected_type": "B",   "body_chars": 4977},
    # Synthetic Type-D — bare benchmark, must trigger refusal
    {"id": "synthetic_typeD", "repo": None, "sha": None,
     "expected_type": "D-refuse", "body_chars": 220,
     "synthetic_subject": "aarch64: Add idct dimension 4 fast path",
     "synthetic_body": (
         "        Cortex A53   A72   A73   X1\n"
         "Before: 250          165   170   140\n"
         "After:  240          158   163   135\n"
     )},
]

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
Produce 1-3 Q&A pairs, one per separable insight.

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

PAIR-DISTINCTNESS RULE:
  - If you produce more than one pair, each must cover a DISTINCT
    insight. Do not paraphrase the same insight in two pairs.
  - If only one insight is present, return ONE pair.

REFUSAL RULES (return {"pairs": [], "notes": "<reason>"} if):
  - The commit body is more than 50% benchmark tables (by character
    count) AND contains fewer than 3 explanatory sentences. Bare
    benchmark dumps are not enough to support a generally-applicable
    Q&A.
  - The commit body is too vague (under 3 sentences) to extract a
    generally-applicable idiom.

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


def call_local(model: str, system: str, user: str, max_tokens: int = 4000) -> dict:
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
        return {"ok": False, "error": f"HTTP {e.code}",
                "body": e.read().decode(errors='replace')[:600],
                "latency_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.time()-t0)*1000)}


def call_abacus(model: str, system: str, user: str, max_tokens: int = 3000) -> dict:
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
        ABACUS,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {ABACUS_KEY}",
            "User-Agent":    "Mozilla/5.0 audit_distillation_v2/1.0 (bet_a)",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            raw = r.read().decode("utf-8", errors="replace")
        return {"ok": True, "latency_ms": int((time.time()-t0)*1000), "raw": raw}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}",
                "body": e.read().decode(errors='replace')[:600],
                "latency_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.time()-t0)*1000)}


def parse_response(r: dict) -> tuple[dict | None, str | None]:
    """Returns (parsed_pairs_dict, error_string)."""
    if not r.get("ok"):
        return None, r.get("error") or "unknown"
    try:
        j = json.loads(r["raw"])
        content = j["choices"][0]["message"]["content"]
        usage   = j.get("usage")
    except Exception as e:
        return None, f"outer-json: {e}"
    if not content:
        # Sometimes content is empty but reasoning_content has the answer (qwen).
        return None, "empty content"
    try:
        parsed = json.loads(content)
        return {"parsed": parsed, "usage": usage, "raw_content": content}, None
    except Exception as e:
        # Try to extract JSON from a fenced block
        import re
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return {"parsed": parsed, "usage": usage, "raw_content": content}, "had-preamble"
            except Exception as e2:
                return None, f"inner-json (extract failed): {e2}"
        return None, f"inner-json: {e}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"  # "local" | "abacus" | "both"
    assert mode in ("local", "abacus", "both")

    out_path = ROOT / "reports" / "audit_responses_v2.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")  # append; allows phased runs
    print(f"mode={mode}; writing to {out_path}")
    total_abacus_tokens = 0

    for commit in COMMITS:
        if commit["repo"]:
            repo = ROOT / "raw" / commit["repo"]
            subj, body = git_show(repo, commit["sha"])
        else:
            subj = commit["synthetic_subject"]
            body = commit["synthetic_body"]
        user_msg = f"commit_subject: {subj}\ncommit_body: |\n  " + body.replace("\n", "\n  ")

        backends = []
        if mode in ("local", "both"):
            backends.append(("local", LOCAL_TEACHER, call_local))
        if mode in ("abacus", "both"):
            # Cost-guard: only send commits with body <= 1500 chars to Abacus.
            # The bigger ones blow the budget; we already have local data for them.
            if commit["body_chars"] <= 1500:
                backends.append(("abacus", ABACUS_TEACHER, call_abacus))
            else:
                print(f"  [{commit['id']}] -> abacus SKIPPED (body_chars={commit['body_chars']} > 1500 cost-cap)")

        for backend_name, teacher, fn in backends:
            print(f"  [{commit['id']}] -> {backend_name}/{teacher}", end=" ", flush=True)
            r = fn(teacher, SYSTEM_PROMPT, user_msg)
            parsed, err = parse_response(r)
            rec = {
                "commit_id":     commit["id"],
                "expected_type": commit["expected_type"],
                "body_chars":    commit["body_chars"],
                "backend":       backend_name,
                "teacher":       teacher,
                "prompt_version":"v3",
                "ok":            r.get("ok"),
                "error":         r.get("error") or err,
                "error_body":    r.get("body"),
                "latency_ms":    r.get("latency_ms"),
                "raw_content":   (parsed or {}).get("raw_content"),
                "parsed":        (parsed or {}).get("parsed"),
                "usage":         (parsed or {}).get("usage"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            u = rec["usage"] or {}
            tt = u.get("total_tokens", 0)
            pc = len((parsed or {}).get("parsed", {}).get("pairs", [])) if parsed else "ERR"
            print(f"  {rec['latency_ms']}ms  pairs={pc}  tokens={tt}")
            if backend_name == "abacus":
                total_abacus_tokens += tt
                if total_abacus_tokens > 18_000:
                    print(f"  ABACUS BUDGET STOP: ran past 18K tokens, stopping audit-v2 abacus run")
                    fout.close()
                    return

    fout.close()
    print(f"\nDONE. abacus tokens consumed this run: {total_abacus_tokens}")

if __name__ == "__main__":
    main()
