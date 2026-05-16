"""
extract_hunks.py — Extract asm hunks + commit messages from a git repo.

Reusable across x264 / dav1d / FFmpeg / Linux ARM64 etc. Path filters are
specified per-repo. Output goes to bet_a/curated/{repo_id}_hunks.jsonl.

Usage:
    python extract_hunks.py \\
        --repo-id x264 \\
        --repo-dir raw/x264_src \\
        --paths 'x86/*.asm' 'arm/*.S' 'aarch64/*.S' 'loongarch/*.S' \\
        --out curated/x264_hunks.jsonl
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path


def hunks_for(repo_dir: Path, path_patterns: list[str]) -> list[dict]:
    # Find matching paths via the find command — pattern is matched as a
    # path-suffix glob (e.g. '*/x86/*.asm').
    paths = []
    for pat in path_patterns:
        r = subprocess.run(
            ["find", ".", "-path", f"*/{pat}"],
            cwd=repo_dir, capture_output=True, text=True,
        )
        paths.extend(r.stdout.strip().split("\n"))
    paths = sorted(set(p for p in paths if p and not p.startswith("./.git")))
    if not paths:
        return []

    out = subprocess.run(
        ["git", "log", "--no-merges", "-p", "--", *paths],
        cwd=repo_dir, capture_output=True, text=True,
    ).stdout

    commits = []
    cur = {}
    cur_diff = []
    for line in out.split("\n"):
        if line.startswith("commit "):
            if cur:
                cur["diff"] = "\n".join(cur_diff)
                commits.append(cur)
            cur = {"sha": line.split()[1], "subject": "", "body": []}
            cur_diff = []
            cur["_in_header"] = True
        elif cur.get("_in_header"):
            if line.startswith("Author:") or line.startswith("Date:"):
                pass
            elif line.startswith("    "):
                if not cur["subject"]:
                    cur["subject"] = line.strip()
                else:
                    cur["body"].append(line.strip())
            elif line.startswith("diff --git"):
                cur["_in_header"] = False
                cur_diff.append(line)
        else:
            cur_diff.append(line)
    if cur:
        cur["diff"] = "\n".join(cur_diff)
        commits.append(cur)

    out_recs = []
    comment_re = re.compile(r"^\+\s*[;#].*\S", re.MULTILINE)
    diff_split_re = re.compile(
        r"diff --git a/(\S+) b/\S+\n.*?(?=(?:^diff --git )|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for c in commits:
        body_text = "\n".join(c.get("body", []))
        substantive = len(body_text.strip()) >= 40
        for m in diff_split_re.finditer(c["diff"]):
            file_path = m.group(1)
            hunk_text = m.group(0)
            added_lines = [
                l for l in hunk_text.split("\n")
                if l.startswith("+") and not l.startswith("+++")
            ]
            comment_lines = sum(1 for l in added_lines if comment_re.match(l))
            n_added = len(added_lines)
            if n_added < 5:
                continue
            out_recs.append({
                "sha":              c["sha"][:12],
                "subject":          c["subject"],
                "body_chars":       len(body_text.strip()),
                "body_substantive": substantive,
                "file_path":        file_path,
                "n_added":          n_added,
                "n_comment_in_added": comment_lines,
                "comment_ratio":    round(comment_lines / max(1, n_added), 2),
                "hunk_len_chars":   len(hunk_text),
            })
    return out_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id",  required=True, help="Used in output JSONL filename")
    ap.add_argument("--repo-dir", required=True, type=Path)
    ap.add_argument("--paths",    required=True, nargs="+",
                    help="Path suffix globs (e.g. 'x86/*.asm' 'arm/64/*.S')")
    ap.add_argument("--out",      required=True, type=Path)
    args = ap.parse_args()

    hunks = hunks_for(args.repo_dir, args.paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for h in hunks:
            f.write(json.dumps(h) + "\n")
    n_substantive = sum(1 for h in hunks if h["body_substantive"])
    n_commits = len({h["sha"] for h in hunks})
    print(f"  {args.repo_id}: wrote {len(hunks)} hunks across {n_commits} commits "
          f"({n_substantive} substantive bodies) -> {args.out}")


if __name__ == "__main__":
    main()
