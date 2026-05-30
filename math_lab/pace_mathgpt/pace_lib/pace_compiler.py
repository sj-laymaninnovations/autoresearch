#!/usr/bin/env python3
"""
Prompt Archeology & Curriculum Engine (PACE) Compiler v2
Author: Antigravity

Transforms raw Q&A datasets into 8 progressive curriculum training stages
for Small Language Models (<3B parameters). Applies the Answer-Last Principle,
real CoT traces, 50-template critique bank, difficulty scoring, and
circuit-aware data design.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Terminal colours (ASCII-safe for Windows CP1252)
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Persona presets (Stage 2)
# ---------------------------------------------------------------------------
PERSONA_PRESETS = [
    "You are an expert software engineer. Be precise, technical, and omit conversational fluff.",
    "You are a patient, empathetic tutor. Explain concepts clearly for a beginner.",
    "You are a concise analytical assistant. Answer in as few words as possible while preserving accuracy.",
    "You are a senior data scientist. Use statistical language and cite methods precisely.",
    "You are a technical writer. Structure your answer with clear headings and logical flow.",
    "You are a security researcher. Be thorough about edge cases and failure modes.",
    "You are a systems architect. Think about scalability, reliability, and trade-offs.",
    "You are a careful fact-checker. Only state what you can verify and flag uncertainty.",
]

# ---------------------------------------------------------------------------
# Negative constraints (Stage 2)
# ---------------------------------------------------------------------------
NEGATIVE_CONSTRAINTS = [
    ("do not use any bullet points", ["- ", "* "]),
    ("do not use markdown formatting", ["#", "**", "__", "`"]),
    ("do not use introductory filler (like 'Sure!', 'Okay', 'Certainly')",
     ["Sure", "Certainly", "Okay", "Of course", "Absolutely"]),
    ("do not use the passive voice", []),
    ("do not exceed three sentences", []),
    ("do not use any technical jargon", []),
    ("do not repeat any word more than twice", []),
    ("do not start any sentence with 'The'", []),
]

# ---------------------------------------------------------------------------
# Critique bank — keyed by flaw category so Stage 5 critiques can be matched
# to the actual draft flaw rather than randomly sampled.
# ---------------------------------------------------------------------------
CRITIQUES_BY_CATEGORY = {
    "sycophantic_opener": [
        "The opening is sycophantic ('Great question!') and adds no information.",
        "The first sentence flatters the user rather than answering the question.",
    ],
    "verbose_padding": [
        "The response is unnecessarily verbose; the core answer is buried in filler.",
        "The response repeats the same point in different words without adding depth.",
        "Important content is wrapped in throat-clearing phrases that should be removed.",
    ],
    "overconfident_hedging": [
        "Hedging language ('might', 'perhaps', 'it is possible') undermines confidence without adding nuance.",
        "The response states an opinion as established fact, then immediately hedges it.",
        "Uncertainty markers are inserted alongside absolute claims, leaving the reader confused.",
    ],
    "constraint_violation_markdown": [
        "Formatting rules specified in the prompt are ignored.",
        "Bullet points are used where a coherent paragraph would be more appropriate.",
        "A constraint from the system prompt is silently violated.",
    ],
    "incomplete_answer": [
        "The response answers only part of the question; the remaining sub-questions are ignored.",
        "The answer is cut off mid-thought and does not reach a conclusion.",
        "Important context or caveats are omitted that the user would need.",
    ],
    "fabricated_citation": [
        "The response cites a plausible-sounding but fabricated reference.",
        "The model generates a URL or DOI that does not exist.",
        "The response fabricates a specific date, name, or statistic.",
    ],
    "circular_reasoning": [
        "The logic is circular: the conclusion restates the premise.",
        "The reasoning provides no new information; it merely paraphrases the question.",
    ],
    "sycophantic_agreement": [
        "The model agrees with the user's incorrect premise instead of correcting it.",
        "The model capitulates to social pressure rather than maintaining its factual position.",
    ],
    "passive_voice_overload": [
        "The response leans on passive constructions that obscure who is doing what.",
        "Active phrasing would make the response clearer and more direct.",
    ],
    "scope_drift": [
        "The answer addresses a topic outside the scope requested.",
        "The response wanders into related-but-unasked territory at the expense of the actual question.",
    ],
    "missing_caveats": [
        "A key edge case is not addressed that would change the answer.",
        "The response presents one approach as the only viable option.",
        "A complex trade-off is presented as having a clear winner.",
    ],
    "jargon_overload": [
        "The response uses jargon that the user likely does not understand.",
        "Pseudo-technical language is used to mask a lack of real understanding.",
        "Technical terms are used without definition.",
    ],
    "wrong_arithmetic": [
        "The numerical calculation contains an arithmetic error.",
        "An intermediate computation step produces the wrong value.",
        "The arithmetic in the worked steps does not match the final answer.",
    ],
    "wrong_operation": [
        "The wrong operation is applied: the problem requires a different operation than was used.",
        "The response treats the problem as addition when it requires subtraction (or vice versa).",
    ],
    "off_by_one": [
        "An off-by-one error appears in the index or count.",
        "The summation bounds are wrong by one term.",
    ],
    "decimal_misplaced": [
        "The decimal point is in the wrong position in the final answer.",
        "A unit conversion shifts the decimal in the wrong direction.",
    ],
    "sign_error": [
        "A sign error appears in the worked steps; the answer should be negated.",
        "Subtraction order is reversed, producing the wrong sign.",
    ],
    "missed_carry_borrow": [
        "A carry was not propagated in the addition.",
        "A borrow was not applied in the subtraction, producing the wrong digit.",
    ],
    "answer_mismatch": [
        "The final answer does not match the answer produced by the worked steps.",
        "The conclusion contradicts the intermediate computation just shown.",
    ],
}

# Backwards-compatible flat list (used by demo previews and any consumer that
# still wants the whole bank). Sourced from the categorised dict above plus
# legacy generic critiques that don't bind to a specific flaw strategy.
_LEGACY_CRITIQUES = [
    "The response contains an unsupported factual claim that requires a citation or derivation.",
    "The reasoning skips a critical intermediate step, creating a logical gap.",
    "The conclusion does not follow from the stated premises.",
    "An assumption is made without being stated explicitly.",
    "The answer conflates correlation with causation.",
    "Two statements in the response contradict each other.",
    "The response lacks a concrete example to illustrate the abstract claim.",
    "Alternative approaches or viewpoints are not acknowledged.",
    "The answer addresses a simplified version of the question, not the actual question asked.",
    "The tone shifts inconsistently between formal and informal.",
    "The response lacks any structural organisation (no paragraphs, headings, or transitions).",
    "Information is presented in a confusing order; the conclusion should come after the evidence.",
    "The response is a wall of text with no visual breaks.",
    "Vague qualifiers ('very', 'a lot', 'significantly') are used without quantification.",
    "The response provides a general overview when a specific, actionable answer was requested.",
    "The response conflates two distinct concepts that should be differentiated.",
    "Exact numbers or measurements are rounded or approximated when precision was required.",
    "The response is overly eager to help with a request that has problematic implications.",
    "The response provides potentially dangerous information without appropriate caveats.",
    "A non-existent API, function, or library is referenced as if real.",
    "The response invents a plausible-sounding historical event.",
    "Uncertainty is not expressed where the correct answer is genuinely unknown.",
    "The model dismisses legitimate alternative interpretations.",
    "The response exceeds the requested length constraint.",
    "The response uses a prohibited word or phrase.",
    "The model provides code when plain language was requested (or vice versa).",
]

CRITIQUE_TEMPLATES = sorted({
    c for bank in CRITIQUES_BY_CATEGORY.values() for c in bank
} | set(_LEGACY_CRITIQUES))

# Math-only critique categories — surfaced separately for callers that want
# to weight the bank toward arithmetic failure modes (mathgpt training).
MATH_FLAW_CATEGORIES = (
    "wrong_arithmetic", "wrong_operation", "off_by_one",
    "decimal_misplaced", "sign_error", "missed_carry_borrow", "answer_mismatch",
)

# ---------------------------------------------------------------------------
# Flawed-draft generation strategies (Stage 5).
#
# Each strategy returns (draft_text, flaw_category). The category is used to
# select critiques from CRITIQUES_BY_CATEGORY so the critique names the real
# defect rather than a random one.
# ---------------------------------------------------------------------------
import re

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_PROSE_FLAW_STRATEGIES = (
    "sycophantic_opener",
    "verbose_padding",
    "overconfident_hedging",
    "constraint_violation_markdown",
    "incomplete_answer",
    "fabricated_citation",
    "circular_reasoning",
    "sycophantic_agreement",
    "passive_voice_overload",
    "scope_drift",
    "missing_caveats",
    "jargon_overload",
)

_MATH_FLAW_STRATEGIES = (
    "wrong_arithmetic",
    "wrong_operation",
    "off_by_one",
    "decimal_misplaced",
    "sign_error",
    "missed_carry_borrow",
    "answer_mismatch",
)


def _looks_numeric(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _perturb_last_number(text: str, transform) -> Optional[str]:
    """Apply `transform(n)` to the last numeric literal in `text`; return None if no number found."""
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    try:
        original = float(m.group())
        if original.is_integer():
            original = int(original)
        new_val = transform(original)
        if isinstance(new_val, float) and new_val.is_integer():
            new_val = int(new_val)
        return text[:m.start()] + str(new_val) + text[m.end():]
    except (ValueError, ZeroDivisionError):
        return None


def _apply_prose_flaw(output: str, instruction: str, strategy: str) -> str:
    if strategy == "sycophantic_opener":
        return f"Great question! I'd be happy to help. {output}"
    if strategy == "verbose_padding":
        return (
            f"This is an interesting and nuanced topic that many people wonder about. "
            f"Let me provide a comprehensive and thorough explanation. {output} "
            f"I hope this explanation was helpful and addressed all aspects of your question."
        )
    if strategy == "overconfident_hedging":
        return f"The answer is definitely, probably: {output} (though I could be wrong about some details)."
    if strategy == "constraint_violation_markdown":
        return f"## Answer\n\n**Key Points:**\n- {output}"
    if strategy == "incomplete_answer":
        words = output.split()
        cutoff = max(3, len(words) // 2)
        return " ".join(words[:cutoff]) + "..."
    if strategy == "fabricated_citation":
        return f"{output} (Source: Johnson et al., 2024, Journal of Advanced Studies, Vol. 47, pp. 112-118)."
    if strategy == "circular_reasoning":
        return f"The answer to '{instruction[:40]}' is what it is because {output.lower()}, which confirms the answer."
    if strategy == "sycophantic_agreement":
        return f"You're absolutely right to ask this! The answer is exactly what you'd expect: {output}"
    if strategy == "passive_voice_overload":
        parts = [s.strip() for s in output.split(".") if s.strip()]
        return ". ".join(f"It can be noted that {p.lower()}" for p in parts)
    if strategy == "scope_drift":
        return (
            f"{output} Additionally, it's worth noting that the broader implications extend "
            f"far beyond this specific question into areas of philosophy and epistemology."
        )
    if strategy == "missing_caveats":
        return f"Always and without exception: {output}"
    if strategy == "jargon_overload":
        return f"From a paradigmatic standpoint, leveraging synergistic frameworks, {output.lower()}"
    return output


def _apply_math_flaw(output: str, instruction: str, strategy: str) -> Optional[str]:
    """Returns a flawed draft for math content, or None if the output has no number to perturb."""
    if strategy == "wrong_arithmetic":
        return _perturb_last_number(output, lambda n: n + random.choice([-3, -2, -1, 1, 2, 3]))
    if strategy == "off_by_one":
        return _perturb_last_number(output, lambda n: n + random.choice([-1, 1]))
    if strategy == "decimal_misplaced":
        return _perturb_last_number(output, lambda n: n * 10 if abs(n) >= 1 else n / 10)
    if strategy == "sign_error":
        return _perturb_last_number(output, lambda n: -n if n != 0 else 1)
    if strategy == "missed_carry_borrow":
        # Drop 10 from the last number to simulate a missed carry.
        return _perturb_last_number(output, lambda n: n - 10 if abs(n) >= 10 else n + 1)
    if strategy == "wrong_operation":
        # Heuristic: keep the number, but reframe the explanation as the opposite operation.
        return (
            f"Applying subtraction to the problem: {output}"
            if "+" in instruction or "sum" in instruction.lower()
            else f"Applying addition to the problem: {output}"
        )
    if strategy == "answer_mismatch":
        # Worked steps look right but final answer is bumped.
        bumped = _perturb_last_number(output, lambda n: n + random.choice([-2, 2]))
        if bumped is None:
            return None
        return f"Worked steps: {output}\nFinal answer: {bumped}"
    return None


def generate_flawed_draft(output: str, instruction: str,
                          prefer_math: Optional[bool] = None
                          ) -> Tuple[str, str]:
    """Selects and applies a flaw strategy.

    Returns (draft_text, flaw_category). The category is the key into
    CRITIQUES_BY_CATEGORY so callers can match critique to defect.

    If prefer_math is None, detect whether the content looks numeric and
    bias toward math flaws when it does.
    """
    if prefer_math is None:
        prefer_math = _looks_numeric(output) and _looks_numeric(instruction)

    if prefer_math:
        # Try a math flaw first; fall back to prose flaws if perturbation
        # fails (e.g. output has no numeric literal we can latch onto).
        for _ in range(3):
            strategy = random.choice(_MATH_FLAW_STRATEGIES)
            draft = _apply_math_flaw(output, instruction, strategy)
            if draft is not None and draft != output:
                return draft, strategy

    strategy = random.choice(_PROSE_FLAW_STRATEGIES)
    return _apply_prose_flaw(output, instruction, strategy), strategy


def critiques_for_flaw(flaw_category: str, num_critiques: int = 2,
                       extras_from_legacy: bool = True) -> List[str]:
    """Pick critiques that name the real defect, optionally seasoned with a generic one."""
    primary = CRITIQUES_BY_CATEGORY.get(flaw_category, [])
    if not primary:
        primary = random.sample(CRITIQUE_TEMPLATES, min(num_critiques, len(CRITIQUE_TEMPLATES)))
    chosen = random.sample(primary, min(num_critiques, len(primary)))
    if extras_from_legacy and num_critiques > len(chosen):
        gap = num_critiques - len(chosen)
        pool = [c for c in CRITIQUE_TEMPLATES if c not in chosen]
        if pool:
            chosen.extend(random.sample(pool, min(gap, len(pool))))
    return chosen


# ---------------------------------------------------------------------------
# Token counting
#
# Tries tiktoken's cl100k_base (closest publicly-available approximation to
# the BPE families used by the small models PACE targets). Falls back to a
# digits-and-structure-aware heuristic so number-heavy math content isn't
# wildly under-counted.
# ---------------------------------------------------------------------------
try:
    import tiktoken  # type: ignore
    _BPE_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover — fallback path
    _BPE_ENC = None

_STRUCTURAL_CHARS = set("<>/{}[]\":,=")
_DIGIT_RE = re.compile(r"\d")


def count_tokens(text: str) -> int:
    """Best-effort BPE-equivalent token count for PACE's ratio decisions."""
    if not text:
        return 0
    if _BPE_ENC is not None:
        return len(_BPE_ENC.encode(text))
    # Heuristic fallback: each whitespace word ≈ 1 token, each digit ≈ 1 token
    # (digits typically tokenise individually in BPE), each structural char
    # ≈ 1 token. This intentionally over-estimates rather than under-estimates
    # so ratio enforcement errs on the side of more scaffolding.
    word_tokens = len(text.split())
    digit_tokens = len(_DIGIT_RE.findall(text))
    structural_tokens = sum(1 for c in text if c in _STRUCTURAL_CHARS)
    return word_tokens + digit_tokens + structural_tokens


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------
COMPLEXITY_KEYWORDS = [
    "prove", "derive", "compare", "contrast", "analyze", "evaluate",
    "multi-step", "trade-off", "optimise", "optimize", "implement",
    "design", "architect", "debug", "synthesize", "critique",
]

_MATH_OP_RE = re.compile(r"[+\-*/^%]|\*\*|//|sqrt|log|sin|cos|tan|sum|product")
_MULTISTEP_HINTS = (
    "then", "next", "after that", "first", "second", "third",
    "step 1", "step 2", "finally",
)


def _max_digit_run(text: str) -> int:
    """Length of the longest contiguous digit run — proxy for numeric magnitude."""
    best = 0
    run = 0
    for ch in text:
        if ch.isdigit():
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _math_difficulty(instruction: str, output: str) -> float:
    """Numeric-aware difficulty in [0, 1]. Returns 0 if no numeric signal."""
    if not (_looks_numeric(instruction) or _looks_numeric(output)):
        return 0.0

    # Magnitude of the largest number in the prompt (proxy for arithmetic load).
    mag = max(_max_digit_run(instruction), _max_digit_run(output))
    f_mag = min(mag / 6.0, 1.0) * 0.4  # 6 digits ≈ saturated

    # Operator count.
    ops = len(_MATH_OP_RE.findall(instruction + " " + output))
    f_ops = min(ops / 5.0, 1.0) * 0.3

    # Multi-step verbal hints.
    instr_lower = instruction.lower()
    step_hits = sum(1 for h in _MULTISTEP_HINTS if h in instr_lower)
    f_steps = min(step_hits / 3.0, 1.0) * 0.2

    # Output digit count (longer worked solutions = harder).
    out_digits = len(_DIGIT_RE.findall(output))
    f_outd = min(out_digits / 20.0, 1.0) * 0.1

    return round(f_mag + f_ops + f_steps + f_outd, 4)


def difficulty_score(item: Dict[str, Any]) -> float:
    """Scores an item 0.0-1.0. Honours explicit `difficulty` field if present.

    Numeric/math items are scored by digit magnitude and operator count;
    prose items fall back to the legacy length+keyword scoring.
    """
    if "difficulty" in item:
        return float(item["difficulty"])

    instruction = item.get("instruction", "")
    output = item.get("output", "")

    math_score = _math_difficulty(instruction, output)
    if math_score > 0:
        # Math item — return numeric score directly.
        return math_score

    # Prose item — legacy scoring.
    instr_words = len(instruction.split())
    f1 = min(instr_words / 50.0, 1.0) * 0.4
    out_tokens = count_tokens(output)
    f2 = min(out_tokens / 200.0, 1.0) * 0.3
    keyword_hits = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in instruction.lower())
    f3 = min(keyword_hits / 3.0, 1.0) * 0.3
    return round(f1 + f2 + f3, 4)


# ---------------------------------------------------------------------------
# Answer-Last enforcement
#
# Stage ratios are tunable per model size. Auto-padding with generic filler
# is intentionally NOT done here — that pattern is exactly the "BAD"
# template the canvas warns against (canvas.md "Real Thinking (Not Jargon)"
# example). Instead, under-ratio items are flagged with `_below_ratio` so
# downstream tooling can either drop them, regenerate with real reasoning,
# or surface a warning. Auto-filler is available behind an explicit opt-in
# flag for the demo only.
# ---------------------------------------------------------------------------

# Default ratios target the canvas's <3B regime. Use `--profile tiny` for
# sub-100M models where context windows are tight and the extra scaffolding
# crowds out the real signal.
STAGE_RATIO_PROFILES: Dict[str, Dict[int, float]] = {
    "small": {1: 1.0, 2: 1.0, 3: 3.0, 4: 2.0, 5: 2.0, 6: 5.0, 7: 2.0, 8: 2.0},
    "tiny":  {1: 1.0, 2: 1.0, 3: 2.0, 4: 1.5, 5: 1.5, 6: 2.0, 7: 1.5, 8: 1.5},
    "medium": {1: 1.0, 2: 1.0, 3: 4.0, 4: 2.5, 5: 2.5, 6: 6.0, 7: 2.5, 8: 2.5},
}
STAGE_RATIOS = dict(STAGE_RATIO_PROFILES["small"])  # mutable default

SCAFFOLDING_MARKERS = {
    1: "</context>",
    2: "</constraints_acknowledged>",
    3: "</chain_of_thought>",
    4: "<response>",
    5: "<revision>",
    6: "</thought>",
    7: "</reasoning>",
    8: "</prior_context>",
}


def _split_scaffold_answer(response: str, stage: int) -> Tuple[str, str]:
    marker = SCAFFOLDING_MARKERS.get(stage)
    if marker and marker in response:
        idx = response.index(marker) + len(marker)
        return response[:idx], response[idx:]
    chars = len(response)
    cut = int(chars * 0.7)
    return response[:cut], response[cut:]


def enforce_answer_last(item: Dict[str, Any], instruction: str,
                        ratios: Optional[Dict[int, float]] = None,
                        allow_filler: bool = False) -> Dict[str, Any]:
    """Check the scaffolding:answer token ratio for the stage.

    Default behaviour (no filler):
        * If the ratio is met, return unchanged.
        * If the ratio is below target, flag `_below_ratio=True` and record
          observed/target ratios. The item is still emitted; the user can
          filter, regenerate, or warn at validation time.

    Legacy behaviour (allow_filler=True, demo only):
        * Pad scaffolding with a problem-restatement preamble so the ratio
          is satisfied. This reproduces the historical PACE behaviour for
          backwards compatibility with the demo, but the canvas explicitly
          rejects this pattern for real training data.
    """
    response = item["response"]
    stage = item["stage"]
    ratios = ratios or STAGE_RATIOS
    target_ratio = ratios.get(stage, 1.0)

    scaffolding_text, answer_text = _split_scaffold_answer(response, stage)
    scaff_tokens = max(count_tokens(scaffolding_text), 1)
    ans_tokens = max(count_tokens(answer_text), 1)
    current_ratio = scaff_tokens / ans_tokens

    if current_ratio >= target_ratio:
        return item

    item["_below_ratio"] = True
    item["_observed_ratio"] = round(current_ratio, 3)
    item["_target_ratio"] = target_ratio

    if not allow_filler:
        return item

    # Opt-in legacy filler path. Used by `--demo --allow-filler-padding`.
    padding = (
        f"Understanding the request: The user asks to '{instruction[:80]}'. "
        f"Key requirements extracted from the query: "
        f"(1) address the core question directly, "
        f"(2) ensure factual accuracy, "
        f"(3) structure the response for clarity. "
        f"Proceeding with analysis.\n"
    )
    if response.startswith("<"):
        tag_end = response.index(">") + 1
        item["response"] = response[:tag_end] + "\n" + padding + response[tag_end:]
    else:
        item["response"] = padding + response
    item["_padded"] = True
    return item


# ---------------------------------------------------------------------------
# Format adapters
# ---------------------------------------------------------------------------
def format_raw(item: Dict[str, Any]) -> Dict[str, Any]:
    """Returns the item as-is (prompt/response keys)."""
    return item

def format_chatml(item: Dict[str, Any]) -> Dict[str, Any]:
    """Converts to ChatML messages array."""
    prompt = item["prompt"]
    response = item["response"]

    messages = []

    # Extract system content if present
    if "<system>" in prompt and "</system>" in prompt:
        sys_start = prompt.index("<system>") + len("<system>")
        sys_end = prompt.index("</system>")
        system_content = prompt[sys_start:sys_end].strip()
        messages.append({"role": "system", "content": system_content})
        # Remove system block from prompt for user content
        prompt_remainder = prompt[sys_end + len("</system>"):].strip()
    else:
        prompt_remainder = prompt

    # Extract user content
    if "<user>" in prompt_remainder:
        u_start = prompt_remainder.index("<user>") + len("<user>")
        u_end = prompt_remainder.index("</user>") if "</user>" in prompt_remainder else len(prompt_remainder)
        user_content = prompt_remainder[u_start:u_end].strip()
    elif "<user_query>" in prompt_remainder:
        u_start = prompt_remainder.index("<user_query>") + len("<user_query>")
        u_end = prompt_remainder.index("</user_query>") if "</user_query>" in prompt_remainder else len(prompt_remainder)
        user_content = prompt_remainder[u_start:u_end].strip()
    else:
        user_content = prompt_remainder.strip()

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})

    result = {"messages": messages, "stage": item.get("stage"), "difficulty": item.get("difficulty_score")}
    if item.get("_padded"):
        result["_padded"] = True
    return result

def format_alpaca(item: Dict[str, Any]) -> Dict[str, Any]:
    """Converts to Alpaca instruction/input/output format."""
    prompt = item["prompt"]
    # Try to extract clean instruction text
    if "<user>" in prompt:
        u_start = prompt.index("<user>") + len("<user>")
        u_end = prompt.index("</user>") if "</user>" in prompt else len(prompt)
        instr = prompt[u_start:u_end].strip()
    elif "<user_query>" in prompt:
        u_start = prompt.index("<user_query>") + len("<user_query>")
        u_end = prompt.index("</user_query>") if "</user_query>" in prompt else len(prompt)
        instr = prompt[u_start:u_end].strip()
    else:
        instr = prompt.strip()

    return {
        "instruction": instr,
        "input": "",
        "output": item["response"],
        "stage": item.get("stage"),
        "difficulty": item.get("difficulty_score"),
    }

FORMAT_ADAPTERS = {"raw": format_raw, "chatml": format_chatml, "alpaca": format_alpaca}


# ---------------------------------------------------------------------------
# Envelope styles
#
# XML envelopes (default) use scaffolding tags like <chain_of_thought>...
# These carry strong structural signal but cost tokens — every tag is several
# BPE units. For sub-100M models with tight context windows and the Layer 0
# arithmetic basins flagged in the canvas (numeric answers want adjacency to
# the gradient signal, not buried under closing tags), the LITE envelope
# replaces XML with newline+label markers that the tokeniser handles cheaply.
# ---------------------------------------------------------------------------

# (xml_open, xml_close, lite_label) per scaffolding section.
_LITE_REPLACEMENTS = [
    ("<user_query>", "</user_query>", None),  # strip wholesale
    ("<system>", "</system>", "System:"),
    ("<user>", "</user>", "User:"),
    ("<context>", "</context>", "Context:"),
    ("<constraints_acknowledged>", "</constraints_acknowledged>", "Constraints:"),
    ("<chain_of_thought>", "</chain_of_thought>", "Reasoning:"),
    ("<final_answer>", "</final_answer>", "Answer:"),
    ("<thought>", "</thought>", "Thinking:"),
    ("<observation>", "</observation>", "Observation:"),
    ("<response>", "</response>", None),
    ("<draft>", "</draft>", "Draft:"),
    ("<critique>", "</critique>", "Critique:"),
    ("<revision>", "</revision>", "Revised:"),
    ("<retrieved_context>", "</retrieved_context>", "Source:"),
    ("<reasoning>", "</reasoning>", "Reasoning:"),
    ("<answer>", "</answer>", "Answer:"),
    ("<prior_context>", "</prior_context>", "Prior:"),
    ("<current_reasoning>", "</current_reasoning>", "Reasoning:"),
]

_CALL_TAG_RE = re.compile(r'<call\s+name="([^"]+)">\s*(.*?)\s*</call>', re.DOTALL)


def _xml_to_lite(text: str) -> str:
    """Convert XML scaffolding tags to compact label markers."""
    if not text:
        return text
    out = text

    # Handle tool-call tags specially (carry an attribute).
    out = _CALL_TAG_RE.sub(lambda m: f"Tool({m.group(1)}): {m.group(2).strip()}", out)

    for open_tag, close_tag, label in _LITE_REPLACEMENTS:
        if label is None:
            out = out.replace(open_tag, "").replace(close_tag, "")
        else:
            # Replace open tag with label (preserving the content that follows).
            out = out.replace(open_tag, label)
            out = out.replace(close_tag, "")

    # Collapse 3+ consecutive blank lines that the replacements can leave behind.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def apply_envelope_style(item: Dict[str, Any], style: str) -> Dict[str, Any]:
    """Mutates `prompt` and `response` in place to use the requested envelope."""
    if style == "lite":
        item["prompt"] = _xml_to_lite(item["prompt"])
        item["response"] = _xml_to_lite(item["response"])
        item["envelope_style"] = "lite"
    return item


# ===========================================================================
# PACE COMPILER CORE
# ===========================================================================
class PaceCompiler:
    def __init__(self, output_dir: Path, seed: int = 42,
                 profile: str = "small",
                 ratios_override: Optional[Dict[int, float]] = None,
                 allow_generic_fallback: bool = False,
                 allow_filler_padding: bool = False,
                 tool_error_rate: float = 0.3,
                 rag_synthesize_context: bool = False,
                 multiturn_synthesize_prior: bool = False,
                 critique_bank: str = "auto",
                 envelope_style: str = "xml"):
        """
        profile: 'tiny' | 'small' | 'medium' — sets default scaffolding ratios.
        ratios_override: per-stage overrides applied on top of the profile.
        allow_generic_fallback: when True, missing `reasoning`/`thinking` fields
            are filled with generic templates. Default False — the canvas
            explicitly rejects this pattern, so items without real reasoning
            are flagged `_missing_reasoning=True` and the stage's expected
            scaffolding is left empty.
        allow_filler_padding: legacy demo behaviour for enforce_answer_last.
        tool_error_rate: probability that a Stage 4 trajectory includes an
            error+recovery cycle (default 0.3). Set 0 for success-only,
            1.0 for the historical always-fail behaviour.
        rag_synthesize_context: when False (default), Stage 7 items without
            a real `context` field are flagged `_missing_context=True` and
            the retrieved_context block is left as a sentinel. When True, a
            distractor-mixed context is synthesised (still leaky — for
            demo/preview only).
        multiturn_synthesize_prior: same idea for Stage 8 prior turns.
        critique_bank: 'auto' (math bank for numeric items, prose otherwise),
            'math', 'prose', or 'all' (legacy random sampling).
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.pad_count = 0
        self.below_ratio_count = 0
        self.missing_reasoning_count = 0
        self.missing_thinking_count = 0
        self.missing_context_count = 0
        self.missing_prior_count = 0

        profile_ratios = STAGE_RATIO_PROFILES.get(profile, STAGE_RATIO_PROFILES["small"])
        self.ratios = dict(profile_ratios)
        if ratios_override:
            self.ratios.update(ratios_override)
        self.profile = profile

        self.allow_generic_fallback = allow_generic_fallback
        self.allow_filler_padding = allow_filler_padding
        self.tool_error_rate = max(0.0, min(1.0, tool_error_rate))
        self.rag_synthesize_context = rag_synthesize_context
        self.multiturn_synthesize_prior = multiturn_synthesize_prior
        self.critique_bank = critique_bank
        if envelope_style not in ("xml", "lite"):
            raise ValueError(f"envelope_style must be 'xml' or 'lite', got {envelope_style!r}")
        self.envelope_style = envelope_style

    # ---- Stage 1: XML Syntactic Anchoring (Answer-Last) -------------------
    def process_stage_1(self, item: Dict[str, Any]) -> Dict[str, Any]:
        instruction = item.get("instruction", "")
        input_data = item.get("input", "")
        output = item.get("output", "")

        formatted_input = "<user_query>\n"
        if input_data:
            formatted_input += f"<instruction>{instruction}</instruction>\n"
            formatted_input += f"<data>{input_data}</data>\n"
        else:
            formatted_input += f"{instruction}\n"
        formatted_input += "</user_query>"

        # Answer-Last: prepend context restatement before the answer
        context_block = (
            f"<context>\n"
            f"Task: {instruction[:120]}\n"
        )
        if input_data:
            context_block += f"Additional input provided: {input_data[:80]}\n"
        context_block += f"</context>"

        formatted_output = f"<response>\n{context_block}\n{output}\n</response>"

        return {
            "prompt": formatted_input,
            "response": formatted_output,
            "stage": 1,
            "description": "XML Syntactic Enclosure (Answer-Last)",
            "seed": self.seed,
        }

    # ---- Stage 2: Instruction Adherence & Negative Constraints ------------
    def process_stage_2(self, item: Dict[str, Any]) -> Dict[str, Any]:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        system_persona = random.choice(PERSONA_PRESETS)
        constraint_text, forbidden_tokens = random.choice(NEGATIVE_CONSTRAINTS)
        augmented_instruction = f"{instruction} [Constraint: {constraint_text}]"

        # Clean the output to satisfy the constraint
        clean_output = output
        for forbidden in forbidden_tokens:
            for word in ["Sure, ", "Certainly, ", "Okay, ", "Of course, ",
                         "Absolutely, ", "Here is the ", "Here is "]:
                if clean_output.startswith(word):
                    clean_output = clean_output[len(word):]
            if forbidden in ["- ", "* "]:
                clean_output = "\n".join(
                    [line.lstrip("- *") for line in clean_output.split("\n")]
                )
            elif forbidden in ["#", "**", "__", "`"]:
                clean_output = clean_output.replace(forbidden, "")

        # Answer-Last: prepend constraint acknowledgment scaffolding
        constraint_ack = (
            f"<constraints_acknowledged>\n"
            f"Active constraint: {constraint_text}.\n"
            f"Verified: response will comply with this constraint throughout.\n"
            f"</constraints_acknowledged>"
        )

        formatted_prompt = (
            f"<system>\n{system_persona}\n</system>\n"
            f"<user>\n{augmented_instruction}\n</user>"
        )

        return {
            "prompt": formatted_prompt,
            "response": f"{constraint_ack}\n{clean_output}",
            "stage": 2,
            "description": "Constraint Adherence (Answer-Last)",
            "seed": self.seed,
        }

    # ---- Stage 3: Pre-Rationalized Chain-of-Thought -----------------------
    def process_stage_3(self, item: Dict[str, Any]) -> Dict[str, Any]:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        reasoning = item.get("reasoning", "")
        missing = False
        if not reasoning:
            missing = True
            self.missing_reasoning_count += 1
            if self.allow_generic_fallback:
                reasoning = self._generate_problem_specific_cot(instruction, output)
                if not hasattr(self, "_cot_warning_shown"):
                    print(f"{YELLOW}[WARNING]{RESET} Generic CoT fallback active "
                          f"(--allow-generic-fallback). The canvas warns this teaches "
                          f"the model to emit procedural filler. Use only for demos.",
                          file=sys.stderr)
                    self._cot_warning_shown = True
            else:
                reasoning = "[REASONING REQUIRED — distill from teacher model]"
                if not hasattr(self, "_cot_skip_warning_shown"):
                    print(f"{YELLOW}[WARNING]{RESET} Items lack 'reasoning' field; "
                          f"Stage 3 entries flagged `_missing_reasoning=True`. Provide "
                          f"distilled traces or pass --allow-generic-fallback to use "
                          f"the legacy template.", file=sys.stderr)
                    self._cot_skip_warning_shown = True

        formatted_output = (
            f"<chain_of_thought>\n{reasoning}\n</chain_of_thought>\n"
            f"<final_answer>\n{output}\n</final_answer>"
        )

        result = {
            "prompt": f"<user>\n{instruction}\n</user>",
            "response": formatted_output,
            "stage": 3,
            "description": "Pre-Rationalized CoT (Answer-Last)",
            "seed": self.seed,
        }
        if missing:
            result["_missing_reasoning"] = True
        return result

    def _generate_problem_specific_cot(self, instruction: str, output: str) -> str:
        """Generates a problem-grounded CoT trace, not a generic template."""
        instr_lower = instruction.lower()

        if any(kw in instr_lower for kw in ["calculate", "sum", "math", "solve", "compute"]):
            return (
                f"The problem asks: {instruction}\n"
                f"I need to identify the mathematical operation required.\n"
                f"Let me extract the key values from the problem statement.\n"
                f"Working through the computation step by step:\n"
                f"Checking: does the result {output[:60]} satisfy the original constraints?\n"
                f"Yes, the computation is verified."
            )
        elif any(kw in instr_lower for kw in ["explain", "difference", "compare", "what is"]):
            return (
                f"The user wants to understand: {instruction}\n"
                f"First, I need to define the core concepts involved.\n"
                f"Then I should identify the key distinctions or properties.\n"
                f"Let me consider how these relate to each other in practice.\n"
                f"Finally, I should synthesize this into a clear explanation."
            )
        elif any(kw in instr_lower for kw in ["write", "create", "generate", "draft"]):
            return (
                f"The user requests: {instruction}\n"
                f"I need to identify the target audience and appropriate tone.\n"
                f"Key elements to include based on the request: "
                f"relevance, specificity, and appropriate style.\n"
                f"Let me draft the content with these constraints in mind."
            )
        elif any(kw in instr_lower for kw in ["code", "function", "implement", "program"]):
            return (
                f"The programming task: {instruction}\n"
                f"I need to identify the input/output specification.\n"
                f"Consider edge cases: empty inputs, boundary values, type mismatches.\n"
                f"Choose an appropriate algorithm and data structure.\n"
                f"Verify the solution handles all identified edge cases."
            )
        else:
            return (
                f"Analyzing the request: {instruction}\n"
                f"Breaking this down into component parts.\n"
                f"Considering the most relevant information to address each part.\n"
                f"Cross-checking my reasoning against the original question.\n"
                f"Assembling the final response."
            )

    # ---- Stage 4: Agentic ReAct Trajectory --------------------------------
    def process_stage_4(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct trajectory. Includes an error+recovery cycle with probability
        `self.tool_error_rate` so the training set contains both success and
        fault-tolerant paths (canvas Epoch 4 recipe). Default 0.3."""
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        tool_name = item.get("metadata", {}).get("tool", "search_web")
        inject_error = random.random() < self.tool_error_rate

        if inject_error:
            error_msgs = (
                "Error: Request timed out. No results returned.",
                "Error 429: Rate limit exceeded. Retry after 60 seconds.",
                "Error 503: Upstream service unavailable.",
                "Error: Malformed query. The 'q' parameter was missing.",
            )
            err = random.choice(error_msgs)
            formatted_output = (
                f"<thought>\nThe user needs: {instruction[:80]}\n"
                f"I should use the '{tool_name}' tool.\n</thought>\n"
                f'<call name="{tool_name}">\n{{"query": "{instruction[:50]}"}}\n</call>\n'
                f"<observation>\n{err}\n</observation>\n"
                f"<thought>\nThe call failed. I will adjust the query and retry; if it "
                f"fails again, I should fall back to my best knowledge with a caveat.\n</thought>\n"
                f'<call name="{tool_name}">\n{{"query": "{instruction[:40]} detailed"}}\n</call>\n'
                f"<observation>\nResult found: {output[:120]}\n</observation>\n"
                f"<thought>\nThe retry succeeded. Compiling the final answer.\n</thought>\n"
                f"<response>\n{output}\n</response>"
            )
            trajectory = "error_recovery"
        else:
            formatted_output = (
                f"<thought>\nThe user needs: {instruction[:80]}\n"
                f"I should use the '{tool_name}' tool to gather grounded data.\n</thought>\n"
                f'<call name="{tool_name}">\n{{"query": "{instruction[:60]}"}}\n</call>\n'
                f"<observation>\nResult: {output[:120]}\n</observation>\n"
                f"<thought>\nThe call returned the needed data. Composing the final "
                f"response from the observation.\n</thought>\n"
                f"<response>\n{output}\n</response>"
            )
            trajectory = "success"

        return {
            "prompt": (
                f"<system>\nYou have access to the '{tool_name}' tool. "
                f'Output tool calls in <call name="tool">...</call> blocks.\n</system>\n'
                f"<user>\n{instruction}\n</user>"
            ),
            "response": formatted_output,
            "stage": 4,
            "description": "Agentic ReAct (Answer-Last)",
            "seed": self.seed,
            "trajectory": trajectory,
        }

    # ---- Stage 5: Contrastive Critique (flaw-matched bank) ----------------
    def process_stage_5(self, item: Dict[str, Any]) -> Dict[str, Any]:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        # Decide whether to bias toward math flaws.
        if self.critique_bank == "math":
            prefer_math = True
        elif self.critique_bank == "prose":
            prefer_math = False
        else:  # auto / all
            prefer_math = None  # let generate_flawed_draft detect

        flawed_draft, flaw_category = generate_flawed_draft(
            output, instruction, prefer_math=prefer_math,
        )

        num_critiques = random.randint(2, 3)
        if self.critique_bank == "all":
            selected = random.sample(
                CRITIQUE_TEMPLATES,
                min(num_critiques, len(CRITIQUE_TEMPLATES)),
            )
        else:
            selected = critiques_for_flaw(flaw_category, num_critiques)
        critique_text = "\n".join(f"- {c}" for c in selected)

        formatted_output = (
            f"<draft>\n{flawed_draft}\n</draft>\n"
            f"<critique>\n{critique_text}\n</critique>\n"
            f"<revision>\n{output}\n</revision>"
        )

        return {
            "prompt": f"<user>\n{instruction}\n</user>",
            "response": formatted_output,
            "stage": 5,
            "description": "Contrastive Critique (flaw-matched, Answer-Last)",
            "seed": self.seed,
            "flaw_category": flaw_category,
        }

    # ---- Stage 6: Encapsulated Latent Reasoning ---------------------------
    def process_stage_6(self, item: Dict[str, Any]) -> Dict[str, Any]:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        thinking = item.get("thinking", "")
        missing = False
        if not thinking:
            missing = True
            self.missing_thinking_count += 1
            if self.allow_generic_fallback:
                thinking = self._generate_problem_specific_thinking(instruction, output)
                if not hasattr(self, "_think_warning_shown"):
                    print(f"{YELLOW}[WARNING]{RESET} Generic thinking fallback active "
                          f"(--allow-generic-fallback). Demos only — real training "
                          f"requires distilled thinking traces.", file=sys.stderr)
                    self._think_warning_shown = True
            else:
                thinking = "[THINKING REQUIRED — distill from teacher model]"
                if not hasattr(self, "_think_skip_warning_shown"):
                    print(f"{YELLOW}[WARNING]{RESET} Items lack 'thinking' field; "
                          f"Stage 6 entries flagged `_missing_thinking=True`.",
                          file=sys.stderr)
                    self._think_skip_warning_shown = True

        formatted_output = f"<thought>\n{thinking}\n</thought>\n{output}"

        result = {
            "prompt": f"<user>\n{instruction}\n</user>",
            "response": formatted_output,
            "stage": 6,
            "description": "Latent Reasoning (Answer-Last)",
            "seed": self.seed,
        }
        if missing:
            result["_missing_thinking"] = True
        return result

    def _generate_problem_specific_thinking(self, instruction: str, output: str) -> str:
        """Generates realistic internal thinking, not jargon."""
        instr_lower = instruction.lower()

        if any(kw in instr_lower for kw in ["calculate", "sum", "math", "solve"]):
            return (
                f"What am I being asked? {instruction}\n"
                f"Let me identify the mathematical relationship here.\n"
                f"I need to recall the relevant formula or method.\n"
                f"Working through it: I'll compute each step and verify.\n"
                f"Let me double-check by approaching from a different angle.\n"
                f"The answer should be: {output[:40]}... let me verify this is consistent."
            )
        elif any(kw in instr_lower for kw in ["explain", "what is", "describe"]):
            return (
                f"The user wants to understand: {instruction}\n"
                f"What are the core concepts I need to convey?\n"
                f"How can I explain this clearly without oversimplifying?\n"
                f"Are there common misconceptions I should address?\n"
                f"Let me structure this from foundational concepts to specifics."
            )
        elif any(kw in instr_lower for kw in ["write", "create", "draft"]):
            return (
                f"I need to produce: {instruction}\n"
                f"Who is the target audience? What tone is appropriate?\n"
                f"What are the essential elements that must be included?\n"
                f"Let me draft something, then mentally review it for quality.\n"
                f"Does my draft actually satisfy the request? Let me check each requirement."
            )
        else:
            return (
                f"Let me carefully consider: {instruction}\n"
                f"What is the core question being asked here?\n"
                f"What information do I need to draw on?\n"
                f"Are there any constraints or edge cases to consider?\n"
                f"Let me reason through this methodically before answering.\n"
                f"Checking my reasoning: does my conclusion follow from my analysis?"
            )

    # ---- Stage 7: RAG Context Grounding -----------------------------------
    def process_stage_7(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """RAG grounding. If `context` is missing, default behaviour flags the
        item rather than synthesising context from the answer (which would
        leak the answer into the retrieval block and make the RAG circuit
        train on a no-op). Set `rag_synthesize_context=True` for a
        distractor-mixed synthetic context — still leaky, demo/preview only.
        """
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        context = item.get("context", "")
        missing = False

        if not context:
            missing = True
            self.missing_context_count += 1
            if self.rag_synthesize_context:
                distractor = (
                    "Related but distinct passage: this document discusses a "
                    "neighbouring topic and does not contain the precise answer "
                    "the user is asking for."
                )
                context = (
                    f"[Doc 1, Para 3] {output}\n"
                    f"Source: Internal knowledge base, last updated 2026-01.\n\n"
                    f"[Doc 2, Para 1] {distractor}"
                )
            else:
                context = "[CONTEXT REQUIRED — provide retrieved passages]"

        formatted_output = (
            f"<retrieved_context>\n{context}\n</retrieved_context>\n"
            f"<reasoning>\n"
            f"The retrieved context is the authoritative source for this answer. "
            f"I will cite from it rather than relying on parametric memory.\n"
            f"Identifying the passage that addresses '{instruction[:60]}'.\n"
            f"</reasoning>\n"
            f"<answer>\n{output}\n</answer>"
        )

        result = {
            "prompt": f"<user>\n{instruction}\n</user>",
            "response": formatted_output,
            "stage": 7,
            "description": "RAG Context Grounding (Answer-Last)",
            "seed": self.seed,
        }
        if missing:
            result["_missing_context"] = True
        return result

    # ---- Stage 8: Multi-turn State Tracking -------------------------------
    def process_stage_8(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-turn state tracking. Items should provide a real `prior_turn`
        dict (`{"user": str, "assistant": str}`). When absent, the default is
        to flag `_missing_prior=True` rather than synthesise a prior turn
        from the same instruction (which produces a no-op training signal).
        Set `multiturn_synthesize_prior=True` to produce a *distinct*
        synthetic prior — still weaker than real conversations, demo only.
        """
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        prior = item.get("prior_turn") or {}
        missing = False

        prior_q = prior.get("user", "")
        prior_a = prior.get("assistant", "")

        if not prior_q or not prior_a:
            missing = True
            self.missing_prior_count += 1
            if self.multiturn_synthesize_prior:
                # Synthesise a *distinct* prior question — a related sub-topic,
                # not a paraphrase of the current one.
                prior_q = (
                    f"Before I ask my main question, can you confirm whether the "
                    f"topic of '{instruction[:50]}' is something you can help with?"
                )
                prior_a = (
                    "Yes, I can help with that. Please ask your specific question "
                    "and I'll address it directly."
                )
            else:
                prior_q = "[PRIOR USER TURN REQUIRED]"
                prior_a = "[PRIOR ASSISTANT TURN REQUIRED]"

        formatted_output = (
            f"<prior_context>\n"
            f"Turn 1 - User asked: '{prior_q}'\n"
            f"Turn 1 - Assistant answered: '{prior_a}'\n"
            f"State carried forward: prior exchange established the scope; the "
            f"current question must be interpreted in light of it.\n"
            f"</prior_context>\n"
            f"<current_reasoning>\n"
            f"Reading the current turn against the prior context. The user is now "
            f"asking: {instruction[:80]}.\n"
            f"I should reference the state established earlier where relevant and "
            f"avoid restating what was already covered.\n"
            f"</current_reasoning>\n"
            f"{output}"
        )

        result = {
            "prompt": (
                f"<user>\n"
                f"[Previous: {prior_q}]\n"
                f"{instruction}\n"
                f"</user>"
            ),
            "response": formatted_output,
            "stage": 8,
            "description": "Multi-turn State Tracking (Answer-Last)",
            "seed": self.seed,
        }
        if missing:
            result["_missing_prior"] = True
        return result

    # ---- Compilation orchestration ----------------------------------------
    STAGE_MAP = {
        1: "process_stage_1", 2: "process_stage_2", 3: "process_stage_3",
        4: "process_stage_4", 5: "process_stage_5", 6: "process_stage_6",
        7: "process_stage_7", 8: "process_stage_8",
    }

    def compile_dataset(self, data: List[Dict[str, Any]], stage: int) -> List[Dict[str, Any]]:
        method_name = self.STAGE_MAP.get(stage)
        if not method_name:
            raise ValueError(f"Invalid curriculum stage: {stage}")
        method = getattr(self, method_name)

        compiled = []
        for item in data:
            result = method(item)
            result["difficulty_score"] = difficulty_score(item)
            result = enforce_answer_last(
                result, item.get("instruction", ""),
                ratios=self.ratios,
                allow_filler=self.allow_filler_padding,
            )
            result = apply_envelope_style(result, self.envelope_style)
            if result.get("_padded"):
                self.pad_count += 1
            if result.get("_below_ratio"):
                self.below_ratio_count += 1
            compiled.append(result)

        compiled.sort(key=lambda x: x.get("difficulty_score", 0))
        return compiled

    def save_stage(self, data: List[Dict[str, Any]], stage: int,
                   fmt: str = "raw",
                   split_difficulty: bool = False) -> List[Path]:
        adapter = FORMAT_ADAPTERS.get(fmt, format_raw)
        formatted = [adapter(item) for item in data]

        saved_paths = []

        if split_difficulty and len(formatted) >= 3:
            # Split into thirds by difficulty
            n = len(formatted)
            third = n // 3
            splits = {
                "easy": formatted[:third],
                "medium": formatted[third:2*third],
                "hard": formatted[2*third:],
            }
            for label, subset in splits.items():
                fp = self.output_dir / f"stage_{stage}_{label}.jsonl"
                self._write_jsonl(fp, subset)
                saved_paths.append(fp)
        else:
            fp = self.output_dir / f"stage_{stage}_curriculum.jsonl"
            self._write_jsonl(fp, formatted)
            saved_paths.append(fp)

        return saved_paths

    @staticmethod
    def _write_jsonl(path: Path, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Phase emission (canvas curriculum ordering) ----------------------
    # The canvas declares a 4-phase ingestion order:
    #   Phase 1 (Format & Syntax)      : Stages 1, 4
    #   Phase 2 (Constraints & State)  : Stages 2, 8
    #   Phase 3 (Reasoning & Grounding): Stages 3, 6, 7
    #   Phase 4 (Self-Correction)      : Stage 5
    # When --emit-phases is used, the per-stage files are concatenated into
    # phase_N_curriculum.jsonl in that order so trainers can ingest the
    # canvas-prescribed sequence directly.
    PHASE_MAP = {
        1: [1, 4],
        2: [2, 8],
        3: [3, 6, 7],
        4: [5],
    }

    def emit_phases(self, fmt: str = "raw") -> List[Path]:
        """Concatenates already-compiled stage files into phase files."""
        adapter_label = fmt
        saved: List[Path] = []
        for phase, stages in self.PHASE_MAP.items():
            out_path = self.output_dir / f"phase_{phase}_curriculum.jsonl"
            with open(out_path, "w", encoding="utf-8") as out_f:
                for stage in stages:
                    src = self.output_dir / f"stage_{stage}_curriculum.jsonl"
                    if not src.exists():
                        # Fall back to any split file (e.g. easy/medium/hard).
                        for variant in sorted(self.output_dir.glob(f"stage_{stage}_*.jsonl")):
                            with open(variant, "r", encoding="utf-8") as in_f:
                                for line in in_f:
                                    if line.strip():
                                        out_f.write(line if line.endswith("\n") else line + "\n")
                        continue
                    with open(src, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line if line.endswith("\n") else line + "\n")
            saved.append(out_path)
        return saved


# ===========================================================================
# DEMO
# ===========================================================================
def generate_mock_dataset() -> List[Dict[str, Any]]:
    """Rich mock dataset with reasoning/thinking fields for some items."""
    return [
        {
            "instruction": "Calculate the sum of integers from 1 to 100.",
            "input": "",
            "output": "The sum is 5050.",
            "reasoning": (
                "The problem asks for 1 + 2 + 3 + ... + 100.\n"
                "This is an arithmetic series with n=100 terms.\n"
                "The formula for the sum of an arithmetic series is n(n+1)/2.\n"
                "Substituting: 100 * 101 / 2 = 10100 / 2 = 5050.\n"
                "Verification: Gauss's method pairs (1+100) + (2+99) + ... = 50 pairs of 101 = 5050."
            ),
            "thinking": (
                "What is being asked? The sum of all integers from 1 to 100.\n"
                "I know the formula n(n+1)/2 applies to consecutive integer sums.\n"
                "Let me compute: n=100, so 100*101/2.\n"
                "100*101 = 10100. Divided by 2 = 5050.\n"
                "Sanity check: 1+2+3+4+5 = 15, formula gives 5*6/2=15. Correct.\n"
                "I'm confident the answer is 5050."
            ),
            "metadata": {"tool": "calculator_api"},
        },
        {
            "instruction": "Explain the difference between synchronous and asynchronous servers.",
            "input": "Keep it technical and dense.",
            "output": (
                "Synchronous servers process each request in a dedicated thread, blocking on I/O. "
                "This creates O(N) thread overhead. Asynchronous servers use a single-threaded event "
                "loop with I/O multiplexing (epoll/kqueue), achieving O(1) active connection overhead."
            ),
            "reasoning": (
                "The user wants a technical comparison of two server architectures.\n"
                "Synchronous: one thread per connection, blocks on I/O, simple but wasteful.\n"
                "Asynchronous: event loop, non-blocking I/O, efficient but complex.\n"
                "Key difference: thread model and how they handle waiting for I/O.\n"
                "Technical terms to include: epoll, kqueue, event loop, I/O multiplexing."
            ),
            "metadata": {"tool": "search_web"},
        },
        {
            "instruction": "Write a product description for a premium mechanical keyboard.",
            "input": "",
            "output": (
                "Experience tactile precision with hot-swappable switches, gasket-mounted dampening, "
                "and a CNC aluminum chassis engineered for acoustic perfection."
            ),
            "metadata": {"tool": "marketing_database"},
        },
        {
            "instruction": "Compare and contrast the trade-offs between microservices and monolithic architecture for a startup.",
            "input": "",
            "output": (
                "Monoliths offer simpler deployment, easier debugging, and lower operational overhead "
                "for small teams. Microservices provide independent scaling and deployment but introduce "
                "network latency, distributed transaction complexity, and require mature DevOps practices. "
                "For most startups, a modular monolith is the pragmatic starting point."
            ),
            "reasoning": (
                "This is a comparison question with trade-offs.\n"
                "Monolith pros: simplicity, single deployment, easy local dev, lower infra cost.\n"
                "Monolith cons: scaling is all-or-nothing, tight coupling grows over time.\n"
                "Microservices pros: independent scaling, team autonomy, technology flexibility.\n"
                "Microservices cons: network overhead, distributed debugging, data consistency.\n"
                "For a startup specifically: team size matters. Small team = monolith wins.\n"
                "The pragmatic middle ground is a modular monolith."
            ),
            "thinking": (
                "The user is asking about architecture for a startup context.\n"
                "I should not just list pros/cons but give a recommendation.\n"
                "Startups have small teams and need to move fast.\n"
                "Microservices add operational overhead that is hard to justify early.\n"
                "But I should acknowledge when microservices become appropriate.\n"
                "My recommendation: start monolithic, modularise, split later."
            ),
            "difficulty": 0.7,
            "metadata": {"tool": "search_web"},
        },
        {
            "instruction": "What is 2 + 2?",
            "input": "",
            "output": "4",
            "reasoning": "2 + 2 = 4. This is basic addition.",
            "thinking": "Simple arithmetic. 2 + 2 = 4.",
            "difficulty": 0.05,
            "metadata": {"tool": "calculator_api"},
        },
    ]


def run_demo(output_dir: Path, seed: int, fmt: str, split_difficulty: bool,
             compiler_kwargs: Optional[Dict[str, Any]] = None,
             emit_phases: bool = False):
    print(f"\n{BOLD}{CYAN}=== PACE v2 Progressive Curriculum Compiler ==={RESET}")
    print(f"{DIM}Seed: {seed} | Format: {fmt} | Split: {split_difficulty}{RESET}\n")

    random.seed(seed)
    raw_data = generate_mock_dataset()
    print(f"{GREEN}[OK]{RESET} Generated mock dataset: {len(raw_data)} samples.\n")

    compiler = PaceCompiler(output_dir, seed=seed, **(compiler_kwargs or {}))
    print(f"{DIM}Profile: {compiler.profile}  Ratios: {compiler.ratios}{RESET}")
    print(f"{DIM}Critique bank: {compiler.critique_bank}  Tool-error rate: "
          f"{compiler.tool_error_rate}{RESET}\n")

    print(f"{BOLD}Compiling 8 Curriculum Stages:{RESET}")
    for stage in range(1, 9):
        compiled = compiler.compile_dataset(raw_data, stage)
        saved_paths = compiler.save_stage(compiled, stage, fmt=fmt,
                                          split_difficulty=split_difficulty)
        filenames = ", ".join(p.name for p in saved_paths)
        below = sum(1 for c in compiled if c.get("_below_ratio"))
        padded = sum(1 for c in compiled if c.get("_padded"))
        desc = compiled[0]["description"]

        diff_range = f"[{compiled[0].get('difficulty_score', 0):.2f} - {compiled[-1].get('difficulty_score', 0):.2f}]"

        print(f"  {YELLOW}Stage {stage}:{RESET} {desc:<48} -> {filenames}")
        print(f"           Difficulty: {diff_range}  |  Below-ratio: {below}/{len(compiled)}"
              f"  |  Padded: {padded}/{len(compiled)}")

        preview = compiled[0]
        resp_preview = preview['response'].replace('\n', '\n           ')[:200]
        print(f"           {DIM}Response[0]: {resp_preview}...{RESET}")
        print("-" * 78)

    if emit_phases:
        phase_paths = compiler.emit_phases(fmt=fmt)
        print(f"\n{BOLD}Phase emission:{RESET}")
        for p in phase_paths:
            print(f"  {GREEN}[OK]{RESET} {p.name}")

    print(f"\n{BOLD}{GREEN}[SUCCESS]{RESET} Compiled all 8 stages.")
    print(f"  Below-ratio items flagged: {compiler.below_ratio_count}")
    print(f"  Filler-padded items: {compiler.pad_count}")
    print(f"  Missing reasoning (Stage 3): {compiler.missing_reasoning_count}")
    print(f"  Missing thinking (Stage 6): {compiler.missing_thinking_count}")
    print(f"  Missing context (Stage 7): {compiler.missing_context_count}")
    print(f"  Missing prior turn (Stage 8): {compiler.missing_prior_count}")
    print(f"  Output: {output_dir.resolve()}\n")


# ===========================================================================
# CLI
# ===========================================================================
def _parse_ratio_overrides(values: List[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for v in values or []:
        if ":" not in v:
            raise SystemExit(f"--ratio expects STAGE:VALUE, got {v!r}")
        stage_str, ratio_str = v.split(":", 1)
        try:
            stage_num = int(stage_str)
            ratio_val = float(ratio_str)
        except ValueError as exc:
            raise SystemExit(f"--ratio parse error in {v!r}: {exc}") from None
        if stage_num not in range(1, 9):
            raise SystemExit(f"--ratio stage out of range: {stage_num}")
        out[stage_num] = ratio_val
    return out


def main():
    parser = argparse.ArgumentParser(
        description="PACE v2 — Progressive Curriculum Compiler for Tiny Models")
    parser.add_argument("--input", type=str, help="Raw JSON/JSONL input dataset")
    parser.add_argument("--output-dir", type=str, default="dist",
                        help="Output directory (default: dist)")
    parser.add_argument("--stage", type=int, choices=range(1, 9),
                        help="Compile a specific stage only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--format", type=str, choices=["raw", "chatml", "alpaca"],
                        default="raw", help="Output format (default: raw)")
    parser.add_argument("--split-difficulty", action="store_true",
                        help="Split each stage into easy/medium/hard files")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with built-in mock data")

    # Tiny-model adaptation flags
    parser.add_argument("--profile", choices=list(STAGE_RATIO_PROFILES),
                        default="small",
                        help="Scaffolding ratio profile by model size "
                             "(tiny=<100M, small=<3B, medium=3-7B). Default: small.")
    parser.add_argument("--ratio", action="append", default=[],
                        metavar="STAGE:VALUE",
                        help="Per-stage ratio override; can be passed multiple times "
                             "(e.g. --ratio 3:2.0 --ratio 6:2.5)")

    # Fallback / synthesis flags (default behaviour is strict)
    parser.add_argument("--allow-generic-fallback", action="store_true",
                        help="Use generic CoT/thinking templates when 'reasoning'/'thinking' "
                             "fields are missing. The canvas explicitly rejects this for "
                             "real training data — demos only.")
    parser.add_argument("--allow-filler-padding", action="store_true",
                        help="Re-enable the legacy enforce_answer_last filler. Demo only.")
    parser.add_argument("--rag-synthesize-context", action="store_true",
                        help="When Stage 7 items lack a 'context' field, synthesise one "
                             "with distractors. Without this flag, items are flagged "
                             "_missing_context=True and not silently leaked.")
    parser.add_argument("--multiturn-synthesize-prior", action="store_true",
                        help="When Stage 8 items lack a real 'prior_turn', synthesise a "
                             "distinct prior question. Without this flag, items are "
                             "flagged _missing_prior=True.")
    parser.add_argument("--tool-error-rate", type=float, default=0.3,
                        help="Fraction of Stage 4 trajectories that include an error+recovery "
                             "cycle (default 0.3; 0=all success, 1=legacy always-fail).")
    parser.add_argument("--critique-bank", choices=["auto", "math", "prose", "all"],
                        default="auto",
                        help="Stage 5 critique bank selection. 'auto' detects numeric items, "
                             "'all' = legacy random sampling.")
    parser.add_argument("--envelope-style", choices=["xml", "lite"], default="xml",
                        help="Scaffolding envelope. 'xml' = full tags (default). 'lite' = "
                             "compact label markers (Reasoning:\\n... Answer: X) — better "
                             "for sub-100M models where tag tokens crowd out the signal.")

    # Canvas curriculum phase emission
    parser.add_argument("--emit-phases", action="store_true",
                        help="After per-stage emission, also write phase_1..phase_4 files "
                             "concatenating stages per the canvas curriculum order.")

    args = parser.parse_args()
    output_path = Path(args.output_dir)
    ratios_override = _parse_ratio_overrides(args.ratio)

    compiler_kwargs: Dict[str, Any] = dict(
        profile=args.profile,
        ratios_override=ratios_override,
        allow_generic_fallback=args.allow_generic_fallback,
        allow_filler_padding=args.allow_filler_padding,
        tool_error_rate=args.tool_error_rate,
        rag_synthesize_context=args.rag_synthesize_context,
        multiturn_synthesize_prior=args.multiturn_synthesize_prior,
        critique_bank=args.critique_bank,
        envelope_style=args.envelope_style,
    )

    if args.demo:
        run_demo(output_path, args.seed, args.format, args.split_difficulty,
                 compiler_kwargs=compiler_kwargs, emit_phases=args.emit_phases)
        sys.exit(0)

    if not args.input:
        parser.print_help()
        print(f"\n{RED}[ERROR] Specify --input <file> or use --demo.{RESET}")
        sys.exit(1)

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"{RED}[ERROR] File not found: {input_file}{RESET}")
        sys.exit(1)

    random.seed(args.seed)

    try:
        if input_file.suffix == ".jsonl":
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
    except Exception as e:
        print(f"{RED}[ERROR] Failed to load: {e}{RESET}")
        sys.exit(1)

    compiler = PaceCompiler(output_path, seed=args.seed, **compiler_kwargs)
    stages_to_run = [args.stage] if args.stage else range(1, 9)

    print(f"\n{BOLD}{CYAN}Processing {input_file.name} ({len(raw_data)} items)...{RESET}")
    print(f"{DIM}Profile: {compiler.profile}  Ratios: {compiler.ratios}{RESET}")
    for stage in stages_to_run:
        compiled = compiler.compile_dataset(raw_data, stage)
        saved = compiler.save_stage(compiled, stage, fmt=args.format,
                                    split_difficulty=args.split_difficulty)
        for p in saved:
            print(f"{GREEN}[OK]{RESET} Stage {stage} -> {p}")

    if args.emit_phases:
        for p in compiler.emit_phases(fmt=args.format):
            print(f"{GREEN}[OK]{RESET} Phase file -> {p}")

    print(f"\n{BOLD}Quality flags:{RESET}")
    print(f"  Below-ratio: {compiler.below_ratio_count}")
    print(f"  Filler-padded: {compiler.pad_count}")
    print(f"  Missing reasoning: {compiler.missing_reasoning_count} "
          f"(Stage 3) | thinking: {compiler.missing_thinking_count} (Stage 6)")
    print(f"  Missing context: {compiler.missing_context_count} "
          f"(Stage 7) | prior turn: {compiler.missing_prior_count} (Stage 8)")


if __name__ == "__main__":
    main()
