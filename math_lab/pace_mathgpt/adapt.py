#!/usr/bin/env python3
"""
adapt.py — mathgpt → PACE schema mapper.

Reads the clean.jsonl produced by verify_solution.py and emits two PACE-shaped
jsonls:
  * adapted_with_reasoning.jsonl — rows whose solution has real verified
    arithmetic steps. Safe for PACE Stages 3 and 6 because the reasoning
    field carries genuine intermediate computations.
  * adapted_answer_only.jsonl    — rows whose solution is just the final
    `#### X` marker (no intermediate steps). Suitable for Stages 1, 2 only.
    PACE Stage 3 / 6 on these would either emit sentinels (strict mode) or
    fall back to generic templates (the anti-pattern the canvas bans).

Schema mapping:
    mathgpt                          PACE
    -------                          ----
    problem                       -> instruction
    answer (stringified)          -> output
    solution minus `#### X`       -> reasoning (Stage 3)
    same, with "Let me check: "   -> thinking  (Stage 6)
    level / 10  (or absent)       -> difficulty
    {domain, level, source}       -> metadata
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _strip_final_marker(solution: str) -> str:
    """Remove the trailing `#### X` line(s) from a solution trace."""
    lines = [ln for ln in solution.split("\n") if ln.strip()]
    cleaned = [ln for ln in lines if not ln.strip().startswith("####")]
    return "\n".join(cleaned)


def _stringify_answer(answer: Any) -> str:
    """Render the answer field as a clean numeric string.

    Floats whose representation is `4.0` collapse to `4`; large integers and
    scientific notation are left alone. This matches the canvas guidance that
    raw numerals reinforce Layer 0 arithmetic basins (canvas.md:478-484).
    """
    if isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, int):
        return str(answer)
    if isinstance(answer, float):
        if answer.is_integer() and abs(answer) < 1e15:
            return str(int(answer))
        return repr(answer)
    return str(answer)


def _difficulty_from_level(level: Any) -> Optional[float]:
    """Map mathgpt `level` field (typically 1-10) to PACE 0-1 scale."""
    try:
        return round(float(level) / 10.0, 3)
    except (TypeError, ValueError):
        return None


def adapt_row(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Convert one verified mathgpt row to PACE schema.

    Returns (route, pace_item) where `route` is one of:
        * "with_reasoning" — has multi-line reasoning, safe for Stages 3/6
        * "answer_only"    — has only the final marker, safe for Stages 1/2
    """
    problem = row.get("problem", "").strip()
    solution = row.get("solution", "")
    answer = row.get("answer")

    output_str = _stringify_answer(answer)
    reasoning_text = _strip_final_marker(solution).strip()

    metadata = {
        "domain": row.get("domain"),
        "level": row.get("level"),
        "source": row.get("source"),
        "verify_reason": row.get("_verify_reason"),
    }

    pace_item: Dict[str, Any] = {
        "instruction": problem,
        "output": output_str,
        "metadata": metadata,
    }

    diff = _difficulty_from_level(row.get("level"))
    if diff is not None:
        pace_item["difficulty"] = diff

    verify_reason = row.get("_verify_reason", "")

    if verify_reason == "clean_answer_only" or not reasoning_text:
        return "answer_only", pace_item

    pace_item["reasoning"] = reasoning_text
    # Thinking variant: the canvas distinguishes structured CoT (Stage 3,
    # declarative) from internal thinking (Stage 6, conversational working
    # memory). For tiny-math data the raw arithmetic is identical; we add a
    # minimal first-person framing so the two stages produce visibly distinct
    # outputs after PACE compilation.
    pace_item["thinking"] = (
        f"Let me work through this. "
        f"{reasoning_text} "
        f"Final value: {output_str}."
    )
    return "with_reasoning", pace_item


def adapt_file(in_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with_reasoning_path = out_dir / "adapted_with_reasoning.jsonl"
    answer_only_path = out_dir / "adapted_answer_only.jsonl"
    stats_path = out_dir / "adapt_stats.json"

    counts = {"with_reasoning": 0, "answer_only": 0}
    by_domain: Dict[str, Dict[str, int]] = {}

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(with_reasoning_path, "w", encoding="utf-8") as f_wr, \
         open(answer_only_path, "w", encoding="utf-8") as f_ao:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            route, item = adapt_row(row)
            counts[route] += 1
            domain = row.get("domain", "unknown")
            by_domain.setdefault(domain, {"with_reasoning": 0, "answer_only": 0})
            by_domain[domain][route] += 1
            target = f_wr if route == "with_reasoning" else f_ao
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "input": str(in_path),
        "counts": counts,
        "by_domain": by_domain,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    print("\n=== Adapt Stats ===")
    print(f"Input: {stats['input']}")
    counts = stats["counts"]
    total = counts["with_reasoning"] + counts["answer_only"]
    print(f"Total: {total}")
    print(f"  with reasoning (safe for Stages 3, 6): {counts['with_reasoning']}")
    print(f"  answer only    (route to Stages 1, 2): {counts['answer_only']}")
    print("\nBy domain:")
    for domain, c in sorted(stats["by_domain"].items()):
        print(f"  {domain:<20}  with_reasoning={c['with_reasoning']:>5}  "
              f"answer_only={c['answer_only']:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Map verified mathgpt jsonl to PACE schema.")
    parser.add_argument("input", type=str,
                        help="Path to clean.jsonl from verify_solution.py")
    parser.add_argument("--out-dir", type=str, default="adapted_out",
                        help="Output directory (default: adapted_out)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    stats = adapt_file(in_path, out_dir)
    print_stats(stats)
    print(f"\nWith reasoning: {out_dir / 'adapted_with_reasoning.jsonl'}")
    print(f"Answer only:    {out_dir / 'adapted_answer_only.jsonl'}")
    print(f"Stats:          {out_dir / 'adapt_stats.json'}")


if __name__ == "__main__":
    main()
