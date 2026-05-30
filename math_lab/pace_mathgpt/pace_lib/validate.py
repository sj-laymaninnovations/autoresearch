#!/usr/bin/env python3
"""
PACE Dataset Validation Tool v2
Author: Antigravity

Validates compiled SFT curriculum datasets. State is keyed by a hash of the
compiled jsonl files so cumulative pass counts reflect the *current* dataset:
if the data changes, the counter resets. Older PACE versions accumulated
passes across unrelated datasets, which made the "amplitude recovery"
counter meaningless.
"""

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

STATE_FILE = ".pace_validation_state.json"


def hash_dataset(dist_dir: Path) -> str:
    """SHA256 of the concatenated compiled jsonls, sorted by name.

    Used to key validation state to dataset identity. Changes to the data
    (different inputs, regenerated curriculum, swapped seed) reset counters
    so historical pass counts stop being misleading.
    """
    h = hashlib.sha256()
    for fp in sorted(dist_dir.glob("stage_*.jsonl")):
        h.update(fp.name.encode("utf-8"))
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    for fp in sorted(dist_dir.glob("phase_*.jsonl")):
        h.update(fp.name.encode("utf-8"))
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def load_state(dist_dir: Path, current_hash: str) -> dict:
    state_path = dist_dir / STATE_FILE
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("dataset_hash") == current_hash:
            return state
        # Dataset changed — preserve hash-keyed archive but reset live counters.
        archive = state.get("archive", {})
        prev_hash = state.get("dataset_hash")
        if prev_hash:
            archive[prev_hash] = {
                "cumulative_runs": state.get("cumulative_runs", 0),
                "stage_pass_counts": state.get("stage_pass_counts", {}),
                "history": state.get("history", []),
                "archived_at": datetime.now().isoformat(),
            }
        return {
            "dataset_hash": current_hash,
            "cumulative_runs": 0,
            "stage_pass_counts": {},
            "history": [],
            "archive": archive,
        }
    return {
        "dataset_hash": current_hash,
        "cumulative_runs": 0,
        "stage_pass_counts": {},
        "history": [],
        "archive": {},
    }


def save_state(dist_dir: Path, state: dict):
    state_path = dist_dir / STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def validate_datasets(dist_dir: Path) -> bool:
    print(f"\n{BOLD}{CYAN}=== PACE v2 SFT Dataset Validation ==={RESET}\n")

    if not dist_dir.exists():
        print(f"{RED}[FAIL] Directory not found: {dist_dir}{RESET}")
        return False

    current_hash = hash_dataset(dist_dir)
    state = load_state(dist_dir, current_hash)
    state["cumulative_runs"] += 1
    run_number = state["cumulative_runs"]
    archived = len(state.get("archive", {}))
    print(f"{DIM}Run #{run_number} | Dataset hash: {current_hash} | "
          f"Prior datasets archived: {archived}{RESET}\n")

    # Each rule provides both an XML-envelope expectation and a LITE-envelope
    # expectation. The validator picks the matching set by looking for `<`
    # characters at known positions in the prompt of the first sample.
    rules = {
        1: {
            "desc": "XML Syntactic Enclosure (Answer-Last)",
            "xml": {"prompt": ["<user_query>", "</user_query>"],
                    "response": ["<response>", "</response>", "<context>", "</context>"]},
            "lite": {"prompt": [], "response": ["Context:"]},
        },
        2: {
            "desc": "Constraint Adherence (Answer-Last)",
            "xml": {"prompt": ["<system>", "</system>", "<user>", "</user>"],
                    "response": ["<constraints_acknowledged>", "</constraints_acknowledged>"]},
            "lite": {"prompt": ["System:", "User:"], "response": ["Constraints:"]},
        },
        3: {
            "desc": "Pre-Rationalized CoT (Answer-Last)",
            "xml": {"prompt": ["<user>", "</user>"],
                    "response": ["<chain_of_thought>", "</chain_of_thought>",
                                 "<final_answer>", "</final_answer>"]},
            "lite": {"prompt": [], "response": ["Reasoning:", "Answer:"]},
        },
        4: {
            "desc": "Agentic ReAct (Answer-Last)",
            "xml": {"prompt": ["<system>", "<user>", "</user>"],
                    "response": ["<thought>", "</thought>", "<call name=",
                                 "<observation>", "<response>", "</response>"]},
            "lite": {"prompt": ["System:", "User:"],
                     "response": ["Thinking:", "Tool(", "Observation:"]},
        },
        5: {
            "desc": "Contrastive Critique (flaw-matched, Answer-Last)",
            "xml": {"prompt": ["<user>", "</user>"],
                    "response": ["<draft>", "</draft>", "<critique>", "</critique>",
                                 "<revision>", "</revision>"]},
            "lite": {"prompt": [], "response": ["Draft:", "Critique:", "Revised:"]},
        },
        6: {
            "desc": "Latent Reasoning (Answer-Last)",
            "xml": {"prompt": ["<user>", "</user>"],
                    "response": ["<thought>", "</thought>"]},
            "lite": {"prompt": [], "response": ["Thinking:"]},
        },
        7: {
            "desc": "RAG Context Grounding (Answer-Last)",
            "xml": {"prompt": ["<user>", "</user>"],
                    "response": ["<retrieved_context>", "</retrieved_context>",
                                 "<reasoning>", "</reasoning>",
                                 "<answer>", "</answer>"]},
            "lite": {"prompt": [], "response": ["Source:", "Reasoning:", "Answer:"]},
        },
        8: {
            "desc": "Multi-turn State Tracking (Answer-Last)",
            "xml": {"prompt": ["<user>", "</user>"],
                    "response": ["<prior_context>", "</prior_context>"]},
            "lite": {"prompt": [], "response": ["Prior:"]},
        },
    }

    def _pick_envelope(sample: dict) -> str:
        """Detect envelope by inspecting the first sample's response."""
        if sample.get("envelope_style") == "lite":
            return "lite"
        if "messages" in sample:
            ass = " ".join(m.get("content", "") for m in sample["messages"]
                            if m.get("role") == "assistant")
            return "lite" if "<" not in ass[:200] else "xml"
        if "response" in sample:
            return "lite" if "<" not in sample["response"][:200] else "xml"
        return "xml"

    overall_pass_count = 0
    overall_fail_count = 0

    for stage_num, rule in rules.items():
        file_path = dist_dir / f"stage_{stage_num}_curriculum.jsonl"
        stage_key = str(stage_num)

        print(f"{BOLD}Stage {stage_num}: {rule['desc']}{RESET}")

        if not file_path.exists():
            # Check for split difficulty files
            split_files = list(dist_dir.glob(f"stage_{stage_num}_*.jsonl"))
            if split_files:
                file_path = split_files[0]  # Validate the first one
                print(f"  {DIM}(using split file: {file_path.name}){RESET}")
            else:
                # Intentional skips are legitimate — domain-specific curricula
                # (mathgpt skips 4/7/8) don't need every stage. Surface as
                # SKIP not FAIL.
                print(f"  {DIM}[SKIP] stage_{stage_num}_curriculum.jsonl "
                      f"not present (intentionally not emitted){RESET}")
                print("-" * 65)
                continue

        # Peek at first sample to decide envelope style for this stage.
        envelope = "xml"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    envelope = _pick_envelope(json.loads(line.strip()))
                    break
        active_rule = rule[envelope]
        if envelope == "lite":
            print(f"  {DIM}Envelope: lite{RESET}")

        stage_ok = True
        line_count = 0
        critique_texts = set()
        flaw_categories = set()
        below_ratio = 0
        missing_reasoning = 0
        missing_thinking = 0
        missing_context = 0
        missing_prior = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    line_count += 1
                    data = json.loads(line.strip())

                    # Quality-flag tallies (these are informational, not failures).
                    if data.get("_below_ratio"):
                        below_ratio += 1
                    if data.get("_missing_reasoning"):
                        missing_reasoning += 1
                    if data.get("_missing_thinking"):
                        missing_thinking += 1
                    if data.get("_missing_context"):
                        missing_context += 1
                    if data.get("_missing_prior"):
                        missing_prior += 1
                    if data.get("flaw_category"):
                        flaw_categories.add(data["flaw_category"])

                    # Determine where prompt/response live based on format
                    if "messages" in data:
                        prompt = " ".join(m.get("content", "") for m in data["messages"]
                                          if m.get("role") in ("system", "user"))
                        response = " ".join(m.get("content", "") for m in data["messages"]
                                            if m.get("role") == "assistant")
                    elif "prompt" in data:
                        prompt = data["prompt"]
                        response = data["response"]
                    elif "instruction" in data:
                        prompt = data["instruction"]
                        response = data["output"]
                    else:
                        print(f"  {RED}[FAIL] Line {line_idx}: unrecognized format.{RESET}")
                        stage_ok = False
                        continue

                    for tag in active_rule["prompt"]:
                        if tag not in prompt:
                            print(f"  {RED}[FAIL] Line {line_idx}: prompt missing '{tag}'{RESET}")
                            stage_ok = False

                    for tag in active_rule["response"]:
                        if tag not in response:
                            print(f"  {RED}[FAIL] Line {line_idx}: response missing '{tag}'{RESET}")
                            stage_ok = False

                    if stage_num == 5 and ("<critique>" in response or "Critique:" in response):
                        if "<critique>" in response:
                            crit_start = response.index("<critique>") + len("<critique>")
                            crit_end = response.index("</critique>") if "</critique>" in response else len(response)
                        else:
                            crit_start = response.index("Critique:") + len("Critique:")
                            end_marker = "Revised:" if "Revised:" in response else None
                            crit_end = response.index(end_marker) if end_marker else len(response)
                        critique_texts.add(response[crit_start:crit_end].strip())

        except Exception as e:
            print(f"  {RED}[ERROR] {e}{RESET}")
            stage_ok = False

        if stage_ok:
            overall_pass_count += 1
            prev = state["stage_pass_counts"].get(stage_key, 0)
            state["stage_pass_counts"][stage_key] = prev + 1
            pass_count = prev + 1
            print(f"  {GREEN}[OK]{RESET} {line_count} samples verified. "
                  f"Cumulative passes for this stage: {pass_count}")
        else:
            overall_fail_count += 1
            print(f"  {RED}[FAIL]{RESET} Stage {stage_num} has validation errors.")

        # Stage 5 diversity report
        if stage_num == 5 and critique_texts:
            unique_pct = len(critique_texts) / max(line_count, 1) * 100
            color = GREEN if unique_pct > 50 else YELLOW
            print(f"  {color}Critique diversity: {len(critique_texts)}/{line_count} "
                  f"unique ({unique_pct:.0f}%){RESET}")
            if flaw_categories:
                print(f"  {DIM}Flaw categories in use: "
                      f"{', '.join(sorted(flaw_categories))}{RESET}")

        # Quality-flag warnings (informational; do not fail the stage).
        warnings = []
        if below_ratio:
            warnings.append(f"below-ratio: {below_ratio}/{line_count}")
        if missing_reasoning:
            warnings.append(f"missing reasoning: {missing_reasoning}/{line_count}")
        if missing_thinking:
            warnings.append(f"missing thinking: {missing_thinking}/{line_count}")
        if missing_context:
            warnings.append(f"missing context: {missing_context}/{line_count}")
        if missing_prior:
            warnings.append(f"missing prior turn: {missing_prior}/{line_count}")
        if warnings:
            print(f"  {YELLOW}[WARN]{RESET} {' | '.join(warnings)}")

        print("-" * 65)

    # Record this run in history (keyed against current dataset hash).
    state["history"].append({
        "run": run_number,
        "timestamp": datetime.now().isoformat(),
        "dataset_hash": current_hash,
        "passed": overall_pass_count,
        "failed": overall_fail_count,
    })

    save_state(dist_dir, state)

    if overall_fail_count == 0:
        print(f"\n{BOLD}{GREEN}[SUCCESS] Run #{run_number}: All {overall_pass_count} stages passed.{RESET}")
        print(f"{DIM}Cumulative validation history saved to {dist_dir / STATE_FILE}{RESET}\n")
        return True
    else:
        print(f"\n{BOLD}{RED}[FAIL] Run #{run_number}: {overall_fail_count} stage(s) failed, "
              f"{overall_pass_count} passed.{RESET}\n")
        return False


if __name__ == "__main__":
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    ok = validate_datasets(dist)
    sys.exit(0 if ok else 1)
