#!/usr/bin/env python3
"""
run.py — End-to-end pipeline: mathgpt jsonl -> PACE curriculum.

Stages:
    1. verify_solution.py  -> audit_out/{clean.jsonl, rejected.jsonl, audit_stats.json}
    2. adapt.py            -> adapted_out/{adapted_with_reasoning.jsonl,
                                            adapted_answer_only.jsonl, adapt_stats.json}
    3. pace_compiler.py    -> dist/{stage_1..8_curriculum.jsonl, phase_1..4_curriculum.jsonl}
    4. validate.py         -> validation pass/fail report

Defaults are tuned for the autoresearch mathgpt regime:
    * profile = tiny             (sub-100M params; scaffolding ratios 1.5-2.0:1)
    * envelope-style = lite      (newline labels instead of XML tags)
    * critique-bank = math       (arithmetic-error templates for Stage 5)
    * emit-phases                (canvas curriculum ordering)

Override on the command line if you need a different regime.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUTORESEARCH = ROOT.parent.parent

# Use the vendored copy of PACE in pace_lib/. This keeps the autoresearch
# repo self-contained — the MacBook only needs `git pull` here, not the
# separate F:\projects\promptengineering repo. To resync with upstream PACE,
# overwrite pace_lib/{pace_compiler,validate,canvas}.* from the source repo.
PACE_LIB = ROOT / "pace_lib"
PACE_COMPILER = PACE_LIB / "pace_compiler.py"
PACE_VALIDATOR = PACE_LIB / "validate.py"

VERIFY = ROOT / "verify_solution.py"
ADAPT = ROOT / "adapt.py"
PACE_TO_FINETUNE = ROOT / "pace_to_finetune.py"


def run(cmd, label):
    print(f"\n>>> {label}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(f"[FAIL] {label} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end mathgpt -> PACE curriculum pipeline.")
    parser.add_argument("input", type=str,
                        help="mathgpt source jsonl (e.g. training_data.jsonl)")
    parser.add_argument("--work-dir", type=str, default=str(ROOT / "pipeline_out"),
                        help="Working directory for all pipeline outputs.")
    parser.add_argument("--profile", default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="PACE scaffolding-ratio profile (default: tiny)")
    parser.add_argument("--envelope-style", default="lite",
                        choices=["xml", "lite"],
                        help="PACE envelope style (default: lite)")
    parser.add_argument("--critique-bank", default="math",
                        choices=["auto", "math", "prose", "all"],
                        help="PACE Stage 5 critique bank (default: math)")
    parser.add_argument("--format", default="raw",
                        choices=["raw", "chatml", "alpaca"],
                        help="PACE output format (default: raw)")
    parser.add_argument("--stages", type=str, default="1,2,3,5,6",
                        help="Comma-separated stages to emit. Default 1,2,3,5,6 "
                             "(skips tool-use/RAG/multiturn stages 4, 7, 8 which "
                             "are not part of mathgpt's job).")
    parser.add_argument("--no-emit-phases", action="store_true",
                        help="Skip the phase emission step.")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip the final validation pass.")
    parser.add_argument("--no-finetune-ready", action="store_true",
                        help="Skip the conversion of PACE outputs to math_lab/finetune.py "
                             "schema (default behaviour produces finetune_ready/ alongside dist/).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional row cap for the audit step (preview mode).")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"[FAIL] input not found: {in_path}")

    work_dir = Path(args.work_dir).resolve()
    audit_dir = work_dir / "audit_out"
    adapted_dir = work_dir / "adapted_out"
    dist_dir = work_dir / "dist"

    # Stage 1: audit
    audit_cmd = [sys.executable, str(VERIFY), str(in_path),
                 "--out-dir", str(audit_dir)]
    if args.sample is not None:
        audit_cmd += ["--sample", str(args.sample)]
    run(audit_cmd, "Auditing arithmetic")

    # Stage 2: adapt
    clean_path = audit_dir / "clean.jsonl"
    if not clean_path.exists():
        sys.exit(f"[FAIL] auditor produced no clean.jsonl at {clean_path}")
    adapt_cmd = [sys.executable, str(ADAPT), str(clean_path),
                 "--out-dir", str(adapted_dir)]
    run(adapt_cmd, "Adapting schema to PACE")

    # Stage 3: PACE compile (one stage at a time so we can route appropriately).
    adapted_main = adapted_dir / "adapted_with_reasoning.jsonl"
    if not adapted_main.exists() or adapted_main.stat().st_size == 0:
        sys.exit(f"[FAIL] adapter produced no rows with reasoning at {adapted_main}")

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    for stage in stages:
        pace_cmd = [
            sys.executable, str(PACE_COMPILER),
            "--input", str(adapted_main),
            "--output-dir", str(dist_dir),
            "--stage", str(stage),
            "--profile", args.profile,
            "--envelope-style", args.envelope_style,
            "--critique-bank", args.critique_bank,
            "--format", args.format,
        ]
        run(pace_cmd, f"PACE compile stage {stage}")

    # Stage 3b: phase emission (canvas ingestion order)
    if not args.no_emit_phases:
        # emit_phases reads the already-written stage files; run a no-op compile
        # call against a tiny throwaway input would re-emit stages. Instead,
        # invoke PACE in --demo mode just to import emit_phases? Simpler: spawn
        # PACE on the same input with --emit-phases but no --stage (which would
        # re-emit all stages 1..8). To avoid redundant work, instead we read
        # the existing stage files and concatenate manually.
        _emit_phases(dist_dir, stages)

    # Stage 4: validate
    if not args.skip_validate:
        run([sys.executable, str(PACE_VALIDATOR), str(dist_dir)],
            "Validating curriculum")

    # Stage 5: convert to finetune.py schema (default on).
    finetune_dir = work_dir / "finetune_ready"
    if not args.no_finetune_ready:
        cmd = [sys.executable, str(PACE_TO_FINETUNE), str(dist_dir),
               "--out-dir", str(finetune_dir)]
        if not args.no_emit_phases:
            cmd.append("--include-phases")
        run(cmd, "Converting to finetune.py schema")

    print(f"\n[SUCCESS] Pipeline complete.")
    print(f"  Audit:           {audit_dir}")
    print(f"  Adapted:         {adapted_dir}")
    print(f"  PACE dist:       {dist_dir}")
    if not args.no_finetune_ready:
        print(f"  Finetune-ready:  {finetune_dir}")


def _emit_phases(dist_dir: Path, emitted_stages):
    """Concatenate per-stage files into phase files matching the canvas order.

    Mirrors PaceCompiler.PHASE_MAP. Skips phases whose stages weren't emitted.
    """
    phase_map = {
        1: [1, 4],
        2: [2, 8],
        3: [3, 6, 7],
        4: [5],
    }
    print("\n>>> Emitting phase files (canvas curriculum order)")
    for phase, stages in phase_map.items():
        usable = [s for s in stages if s in emitted_stages]
        if not usable:
            continue
        out_path = dist_dir / f"phase_{phase}_curriculum.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for stage in usable:
                src = dist_dir / f"stage_{stage}_curriculum.jsonl"
                if not src.exists():
                    continue
                with open(src, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line if line.endswith("\n") else line + "\n")
        print(f"    {out_path.name}  (stages {usable})")


if __name__ == "__main__":
    main()
