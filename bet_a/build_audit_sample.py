"""Build a 10-pair audit sample + full readable corpus view."""
import json, random, subprocess
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
RECS  = [json.loads(l) for l in (ROOT/"distilled"/"qa_records_v1.jsonl").open()]
RAW   = ROOT / "raw"
REPORTS = ROOT / "reports"

# === Sample selection ===
# Strategy:
#   3 pairs WITH warnings (format violations to inspect)
#   3 pairs from x264 (assembly-craftsman heavy)
#   3 pairs from dav1d (modern AV1 work)
#   1 pair from a multi-pair commit (Type-E split validation)

random.seed(11)

with_warnings = [r for r in RECS if r.get("warnings")]
without_warnings = [r for r in RECS if not r.get("warnings")]
x264 = [r for r in without_warnings if r["repo"] == "x264_src"]
dav1d = [r for r in without_warnings if r["repo"] == "dav1d_src"]

# Commits with 2 pairs
from collections import Counter
shas = Counter(r["sha"] for r in RECS)
multi_pair_shas = [sha for sha, n in shas.items() if n >= 2]
multi_pair_recs = [r for r in RECS if r["sha"] in multi_pair_shas]

samples = []
samples += random.sample(with_warnings, min(3, len(with_warnings)))
samples += random.sample(x264, 3)
samples += random.sample(dav1d, 3)
samples += random.sample(multi_pair_recs, 1) if multi_pair_recs else []

# Dedupe by (sha, question[:80]) in case overlap
seen = set()
final_samples = []
for r in samples:
    key = (r["sha"], r["question"][:80])
    if key in seen: continue
    seen.add(key)
    final_samples.append(r)
final_samples = final_samples[:10]

# === Render audit sample ===
def git_show_body(repo: str, sha: str) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%b", sha],
        cwd=RAW/repo, capture_output=True, text=True
    ).stdout.strip()
    # Truncate to first 1000 chars for review
    return out[:1000] + ("\n... [truncated]" if len(out) > 1000 else "")

lines = ["# Bet A — Audit Sample (10 of 74)",
         "",
         "**Purpose:** Sean reads each pair + source-commit body, flags any of:",
         "",
         "- HALLUCINATION — Q/A asserts a fact not in the source body",
         "- WEAK — Q/A is technically faithful but not training-worthy",
         "- FORMAT — `####` issue or codebase name leak (validator should have caught these)",
         "- OK — pair is training-grade",
         "",
         "Mark verdict in the `Verdict:` slot under each pair.",
         "",
         "---", ""]

for i, r in enumerate(final_samples, 1):
    body = git_show_body(r["repo"], r["sha"])
    warns = r.get("warnings") or []
    warn_tag = " ⚠ " + ", ".join(warns) if warns else ""
    lines.append(f"## #{i} — `{r['sha'][:10]}` ({r['repo']}){warn_tag}")
    lines.append("")
    lines.append(f"**Source commit subject:** {r['subject']}")
    lines.append("")
    lines.append("**Source commit body (first 1000 chars):**")
    lines.append("```")
    lines.append(body)
    lines.append("```")
    lines.append("")
    lines.append(f"**Q:** {r['question']}")
    lines.append("")
    lines.append(f"**A:** {r['answer']}")
    lines.append("")
    lines.append("**Verdict:** _________________________")
    lines.append("")
    lines.append("---")
    lines.append("")

(REPORTS / "audit_sample_v1.md").write_text("\n".join(lines))
print(f"wrote {REPORTS/'audit_sample_v1.md'}  ({len(final_samples)} samples)")

# === Render full corpus view ===
full = ["# Bet A — Full Distilled Corpus (74 pairs)",
        "",
        f"Source: `bet_a/distilled/qa_records_v1.jsonl`",
        f"Teacher: openai/gpt-oss-20b (local LM Studio)",
        f"Prompt: v4 (distillation_prompt_v4.md)",
        f"Run: Pipeline F v1.1",
        "", "---", ""]
for i, r in enumerate(RECS, 1):
    warns = r.get("warnings") or []
    warn_tag = " ⚠ " + ", ".join(warns) if warns else ""
    full.append(f"## #{i} — `{r['sha'][:10]}` ({r['repo']}){warn_tag}")
    full.append(f"_{r['subject']}_")
    full.append("")
    full.append(f"**Q:** {r['question']}")
    full.append("")
    full.append(f"**A:** {r['answer']}")
    full.append("")
    full.append("---")
    full.append("")

(REPORTS / "qa_records_v1.readable.md").write_text("\n".join(full))
print(f"wrote {REPORTS/'qa_records_v1.readable.md'}  ({len(RECS)} pairs)")
