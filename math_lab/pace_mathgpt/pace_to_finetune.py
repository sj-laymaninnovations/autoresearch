#!/usr/bin/env python3
"""
pace_to_finetune.py — Convert PACE dist/ jsonls to finetune.py schema.

Why this exists:
    math_lab/finetune.py reads {problem, solution, answer} rows with
    PROMPT_TEMPLATE = "Q: {problem}\nA: " and trains the model to emit
    `solution`. PACE outputs {prompt, response, stage, difficulty_score, ...}
    where prompt contains envelope markers like "User:\n..." and response
    contains "Reasoning:\n... Answer: ...".

    This adapter strips PACE's envelope markers from the prompt (so we get
    back the original problem text), uses the PACE response as `solution_cot`
    (which finetune.py prefers over `solution`), and preserves the original
    `answer` from PACE metadata.

Outputs one jsonl per input stage/phase file, written under
<dist_dir>/../finetune_ready/ by default.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PACE envelope markers we strip from the prompt to recover the bare problem.
_PROMPT_LEADERS = ("User:", "System:")
_PROMPT_TAG_RE = re.compile(
    r"</?(user|user_query|system)>", re.IGNORECASE
)


def _strip_prompt_envelope(prompt: str) -> str:
    """Recover the bare problem text from a PACE prompt.

    Handles both envelope styles:
      * xml  : "<system>...</system>\n<user>\nProblem\n</user>"
      * lite : "User:\nProblem"
    """
    if not prompt:
        return ""
    text = prompt.strip()

    # Lite envelope: drop leading "System:..." block, then "User:" line.
    if text.startswith("System:"):
        # Find a User: marker after the system block.
        idx = text.find("User:")
        if idx != -1:
            text = text[idx + len("User:"):]
        else:
            # No user marker — strip the System: prefix and use rest.
            text = text[len("System:"):]
    elif text.startswith("User:"):
        text = text[len("User:"):]

    # XML envelope: strip outer tags, lift user block content.
    if "<" in text and ">" in text:
        # Pull content of <user>...</user> or <user_query>...</user_query>.
        for tag in ("user", "user_query"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1)
                break
        text = _PROMPT_TAG_RE.sub("", text)

    return text.strip()


def _recover_answer_from_metadata(pace_row: Dict[str, Any]) -> Optional[Any]:
    """The PACE compiler does not currently round-trip the original numeric
    answer into the compiled row. The adapter that produced PACE input rows
    (adapt.py) put the answer in the `output` field, which we can't directly
    recover here from the compiled prompt/response. Callers that need the
    typed `answer` should pair this with a stage_N_curriculum.jsonl where
    the response ends with the numeric answer (which is exactly how lite
    envelope is shaped). We extract the trailing number as a fallback.
    """
    response = pace_row.get("response", "")
    # Lite: response ends with the number on its own line, or after "Answer:".
    if "Answer:" in response:
        tail = response.rsplit("Answer:", 1)[-1].strip()
        m = re.search(r"-?\d+(?:\.\d+)?", tail)
        if m:
            try:
                val = float(m.group())
                return int(val) if val.is_integer() else val
            except ValueError:
                pass
    # Fallback: last numeric token in the response.
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if nums:
        try:
            val = float(nums[-1])
            return int(val) if val.is_integer() else val
        except ValueError:
            pass
    return None


def convert_row(pace_row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single PACE-compiled row to finetune.py schema."""
    # Handle both raw and chatml/alpaca formats.
    if "messages" in pace_row:
        msgs = pace_row["messages"]
        prompt_parts = []
        response = ""
        for m in msgs:
            if m.get("role") in ("system", "user"):
                prompt_parts.append(m.get("content", ""))
            elif m.get("role") == "assistant":
                response = m.get("content", "")
        prompt = "\n".join(prompt_parts)
    elif "instruction" in pace_row and "output" in pace_row:
        # Alpaca-style
        prompt = pace_row["instruction"]
        response = pace_row["output"]
    else:
        prompt = pace_row.get("prompt", "")
        response = pace_row.get("response", "")

    problem = _strip_prompt_envelope(prompt)
    answer = _recover_answer_from_metadata({"response": response})

    out: Dict[str, Any] = {
        "problem": problem,
        # `solution_cot` is the key finetune.py prefers (line 96 of finetune.py).
        # We put the full PACE response here so the model trains on the
        # complete scaffolded sequence, including reasoning markers.
        "solution_cot": response,
        # Preserve a plain `solution` as well so older code paths that read
        # `solution` directly still work. Strip the envelope labels.
        "solution": _to_plain_solution(response),
        # Stage/difficulty metadata that finetune.py ignores but downstream
        # curricula schedulers may use.
        "_pace_stage": pace_row.get("stage"),
        "_pace_difficulty": pace_row.get("difficulty_score"),
        "_pace_flaw_category": pace_row.get("flaw_category"),
    }
    if answer is not None:
        out["answer"] = answer
    return out


def _to_plain_solution(response: str) -> str:
    """Strip envelope markers so the bare solution text remains.

    Lite envelope: "Reasoning:\nA + B = C\n\nAnswer: D" -> "A + B = C\nD"
    XML envelope: same idea but with tags.
    """
    if not response:
        return ""
    text = response
    # Drop XML tags.
    text = re.sub(r"</?[a-zA-Z_][\w]*[^>]*>", "", text)
    # Drop lite labels.
    for label in (
        "Reasoning:", "Thinking:", "Answer:", "Context:", "Constraints:",
        "Draft:", "Critique:", "Revised:", "Source:", "Prior:",
        "Observation:",
    ):
        text = text.replace(label, "")
    # Collapse whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def convert_file(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            converted = convert_row(row)
            f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Convert PACE dist/ jsonls to math_lab/finetune.py schema.")
    parser.add_argument("dist_dir", type=str,
                        help="Directory containing stage_N_curriculum.jsonl (and optionally phase_N).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <dist_dir>/../finetune_ready)")
    parser.add_argument("--include-phases", action="store_true",
                        help="Also convert phase_N_curriculum.jsonl files.")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.is_dir():
        sys.exit(f"[FAIL] dist_dir not found: {dist_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else dist_dir.parent / "finetune_ready"

    patterns = ["stage_*_curriculum.jsonl"]
    if args.include_phases:
        patterns.append("phase_*_curriculum.jsonl")

    converted_files: List[Tuple[Path, int]] = []
    for pattern in patterns:
        for src in sorted(dist_dir.glob(pattern)):
            tgt = out_dir / src.name.replace("_curriculum", "")
            n = convert_file(src, tgt)
            converted_files.append((tgt, n))

    if not converted_files:
        sys.exit(f"[FAIL] no PACE files found in {dist_dir}")

    print("\n=== finetune-ready output ===")
    for path, n in converted_files:
        print(f"  {path.name:<30}  rows={n}")
    print(f"\nDirectory: {out_dir}")
    print("\nFeed any of these directly to finetune.py:")
    print(f"  python math_lab/finetune.py --data {converted_files[0][0]}")


if __name__ == "__main__":
    main()
