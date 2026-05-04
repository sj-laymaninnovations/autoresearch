"""
trace_inference.py — Per-token telemetry for MathGPT generation.

Instruments the character-level generate loop to capture, at every generated
token position:

  1. entropy          — uncertainty of the distribution (high = unsure)
  2. top1_prob        — confidence in the most likely next token
  3. correct_prob     — probability mass on the correct-answer digits,
                        measured at every token (tracks "how close is the
                        model to committing to the right answer")
  4. intermediate_audit — parse any intermediate numeric values as they appear
                          and verify them against ground truth arithmetic

Outputs:
  - Per-token JSON record (structured telemetry)
  - ASCII timeline chart (entropy + correct_prob)
  - Intermediate value audit table
  - Optional: matplotlib PNG saved to results/traces/

Usage:
    python trace_inference.py \\
        --ckpt results/checkpoints_v2/v2_add2d_best.pt \\
        --problem "47 + 58" --answer 105

    python trace_inference.py \\
        --ckpt results/checkpoints_v2/v2_mul2d_best.pt \\
        --problem "23 * 46" --answer 1058 --plot
"""

import sys, re, json, argparse, math
from pathlib import Path

import torch
import torch.nn.functional as F

# ── bring finetune symbols into scope ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    MathGPT, GPTConfig, PROMPT_TEMPLATE, VOCAB_SIZE,
    char_encode, char_decode, OFFSET, EOS_ID, BOS_ID,
)


# ── telemetry capture ─────────────────────────────────────────────────────────

def digit_ids(answer: int) -> set[int]:
    """Return char-vocab IDs for each digit character in the string form of answer."""
    return {ord(d) + OFFSET for d in str(abs(answer))}


def entropy(probs: torch.Tensor) -> float:
    """Shannon entropy in nats of a probability distribution."""
    p = probs.clamp(min=1e-12)
    return float(-(p * p.log()).sum())


def trace_generate(
    model: MathGPT,
    prompt: str,
    expected_answer: int,
    max_new: int = 250,
    temperature: float = 0.3,   # match serve.py default
    top_k: int = 40,
    seed: int = 42,
) -> dict:
    """
    Run greedy (temperature=1, argmax) generation while capturing full logit
    distributions at every step.

    Returns a trace dict with:
      generated      : str — full generated text
      tokens         : list[str] — each generated character
      entropy        : list[float] — per-token entropy (nats)
      top1_prob      : list[float] — prob of most likely token
      correct_prob   : list[float] — probability of the correct answer
                       digits at each position (summed over answer chars)
      commit_pos     : int — character index where #### appears (or -1)
      pre_commit_chars : int — chars generated before ####
      intermediates  : list[dict] — numbers parsed from generated text
                       with ground-truth check
    """
    model.eval()
    device = next(model.parameters()).device
    ans_ids = digit_ids(expected_answer)
    ans_str  = str(expected_answer)

    ids = torch.tensor(
        char_encode(prompt)[:-1],   # strip EOS from prompt
        dtype=torch.long, device=device
    ).unsqueeze(0)

    tokens      = []
    entropies   = []
    top1_probs  = []
    correct_probs = []

    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(max_new):
            ids_cond = ids[:, -model.cfg.seq_len:]
            raw_logits = model(ids_cond)[:, -1, :] / temperature  # (1, vocab)
            if top_k:
                v, _ = raw_logits.topk(top_k)
                raw_logits[raw_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(raw_logits, dim=-1)[0]        # (vocab,)

            # ── metrics — compute on unmasked distribution ──
            full_logits = model(ids_cond)[:, -1, :]         # re-run at T=1 for clean stats
            full_probs  = F.softmax(full_logits, dim=-1)[0]
            ent         = entropy(full_probs)
            top1_p      = float(full_probs.max())
            corr_p      = float(sum(full_probs[i] for i in ans_ids if i < len(full_probs)))

            # ── sample next token ──
            next_id = int(torch.multinomial(probs, 1))
            if next_id == EOS_ID:
                break

            ch = chr(next_id - OFFSET) if next_id >= OFFSET else ''
            tokens.append(ch)
            entropies.append(ent)
            top1_probs.append(top1_p)
            correct_probs.append(corr_p)

            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    generated = "".join(tokens)

    # ── find commit position (####) ────────────────────────────────────────────
    commit_pos = generated.find("####")
    pre_commit  = commit_pos if commit_pos >= 0 else len(generated)

    # ── intermediate value audit ───────────────────────────────────────────────
    intermediates = audit_intermediates(generated, expected_answer)

    # ── extract final answer ───────────────────────────────────────────────────
    m = re.search(r"####\s*(-?\d+)", generated)
    predicted = int(m.group(1)) if m else None

    return {
        "prompt":          prompt,
        "expected":        expected_answer,
        "predicted":       predicted,
        "correct":         predicted == expected_answer,
        "generated":       generated,
        "tokens":          tokens,
        "entropy":         entropies,
        "top1_prob":       top1_probs,
        "correct_prob":    correct_probs,
        "commit_pos":      commit_pos,
        "pre_commit_chars": pre_commit,
        "total_chars":     len(tokens),
        "intermediates":   intermediates,
    }


def audit_intermediates(text: str, expected_answer: int) -> list[dict]:
    """
    Parse intermediate 'A op B = C' expressions from the *main problem* section
    (everything after the Example: line) and verify arithmetically.
    """
    results = []
    # Skip the example line — only audit the main reasoning body
    body = text
    ex_end = text.find("\n", text.find("Example:")) if "Example:" in text else 0
    if ex_end > 0:
        body = text[ex_end:]

    # Also stop at the #### line
    hash_pos = body.find("####")
    if hash_pos >= 0:
        body = body[:hash_pos]

    pattern = re.compile(
        r"(-?\d+)\s*([+\-*×÷])\s*(-?\d+)\s*=\s*(-?\d+)"
    )
    for m in pattern.finditer(body):
        a, op, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
        a, b, c = int(a), int(b), int(c)
        ops = {'+': a+b, '-': a-b, '*': a*b, '×': a*b}
        expected_c = ops.get(op)
        correct = (expected_c is not None and expected_c == c)
        results.append({
            "expr":     m.group(0).strip(),
            "computed": expected_c,
            "written":  c,
            "correct":  correct,
        })
    return results


# ── visualisation helpers ─────────────────────────────────────────────────────

def ascii_timeline(trace: dict, width: int = 72) -> str:
    """
    Render entropy and correct_prob as stacked ASCII sparklines.
    Marks the #### commit position with a vertical bar.
    """
    n        = len(trace["entropy"])
    commit   = trace["commit_pos"]
    ents     = trace["entropy"]
    cprobs   = trace["correct_prob"]

    def sparkline(values, lo, hi, height=5):
        rows = []
        for row in range(height - 1, -1, -1):
            threshold = lo + (hi - lo) * row / (height - 1)
            line = ""
            for i, v in enumerate(values):
                marker = "│" if i == commit else " "
                if v >= threshold:
                    line += "█" + ""
                else:
                    line += marker
            rows.append(line)
        return rows

    max_ent   = max(ents)  if ents  else 1
    max_cprob = max(cprobs) if cprobs else 1

    lines = []
    lines.append(f"\n{'─'*width}")
    lines.append(f"  ENTROPY (nats)    max={max_ent:.2f}  |  low=confident, high=uncertain")
    lines.append(f"{'─'*width}")
    for row in sparkline(ents, 0, max_ent, height=5):
        lines.append("  " + row[:width])

    lines.append(f"\n{'─'*width}")
    lines.append(f"  CORRECT-ANSWER PROBABILITY  max={max_cprob:.3f}")
    lines.append(f"{'─'*width}")
    for row in sparkline(cprobs, 0, max_cprob, height=5):
        lines.append("  " + row[:width])

    if commit >= 0:
        lines.append(f"\n  ▲ commit (####) at char {commit} of {n}")
    lines.append(f"{'─'*width}")
    return "\n".join(lines)


def print_report(trace: dict) -> None:
    ok = "✅" if trace["correct"] else "❌"
    print(f"\n{'='*72}")
    print(f"  {ok}  Q: {trace['prompt']}   expected={trace['expected']}  got={trace['predicted']}")
    print(f"{'='*72}")
    print(f"\n  Generated ({trace['total_chars']} chars, #### at char {trace['commit_pos']}):")
    # pretty-print with position markers
    gen = trace["generated"]
    if trace["commit_pos"] >= 0:
        gen = gen[:trace["commit_pos"]] + "【####】" + gen[trace["commit_pos"]+4:]
    for line in gen.split("\n"):
        print(f"    {line}")

    print(ascii_timeline(trace))

    # ── per-token table (condensed — only key positions) ─────────────────────
    print(f"\n  TOKEN-LEVEL TELEMETRY (showing every 5th + #### neighbourhood):")
    print(f"  {'pos':>4}  {'char':<6}  {'entropy':>8}  {'top1_p':>8}  {'corr_p':>8}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    n = len(trace["tokens"])
    highlight = set(range(0, n, 5))
    if trace["commit_pos"] >= 0:
        for d in range(-3, 4):
            p = trace["commit_pos"] + d
            if 0 <= p < n:
                highlight.add(p)
    for i in sorted(highlight):
        if i >= n: break
        ch  = repr(trace["tokens"][i])
        ent = trace["entropy"][i]
        t1  = trace["top1_prob"][i]
        cp  = trace["correct_prob"][i]
        marker = " ◀ ####" if i == trace["commit_pos"] else ""
        print(f"  {i:>4}  {ch:<6}  {ent:>8.3f}  {t1:>8.3f}  {cp:>8.4f}{marker}")

    # ── intermediate audit ────────────────────────────────────────────────────
    if trace["intermediates"]:
        print(f"\n  INTERMEDIATE VALUE AUDIT:")
        print(f"  {'expression':<30}  {'computed':>10}  {'written':>10}  ok?")
        print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*3}")
        for item in trace["intermediates"]:
            ok_s = "✅" if item["correct"] else "❌"
            print(f"  {item['expr']:<30}  {str(item['computed']):>10}  {str(item['written']):>10}  {ok_s}")

    # ── summary stats ─────────────────────────────────────────────────────────
    ents   = trace["entropy"]
    cprobs = trace["correct_prob"]
    commit = trace["commit_pos"]
    if commit > 0 and ents:
        pre_ent  = ents[:commit]
        post_ent = ents[commit:]
        pre_cp   = cprobs[:commit]
        post_cp  = cprobs[commit:]
        print(f"\n  SUMMARY:")
        print(f"    chars before ####      : {commit}")
        print(f"    avg entropy  pre-####  : {sum(pre_ent)/len(pre_ent):.3f} nats")
        if post_ent:
            print(f"    avg entropy  post-#### : {sum(post_ent)/len(post_ent):.3f} nats")
        print(f"    avg corr_prob pre-####  : {sum(pre_cp)/len(pre_cp):.4f}")
        if post_cp:
            print(f"    avg corr_prob post-#### : {sum(post_cp)/len(post_cp):.4f}")
        # peak correct-prob in the scratch-pad
        peak_cp_pos = max(range(len(cprobs)), key=lambda i: cprobs[i])
        print(f"    peak correct_prob pos  : char {peak_cp_pos} = {cprobs[peak_cp_pos]:.4f}")
    print()


def save_matplotlib(trace: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(
            f"Token Telemetry — Q: \"{trace['prompt']}\"\n"
            f"Expected: {trace['expected']}   Got: {trace['predicted']}   "
            f"{'✅ Correct' if trace['correct'] else '❌ Wrong'}",
            fontsize=12, fontweight="bold"
        )

        xs     = list(range(len(trace["tokens"])))
        commit = trace["commit_pos"]

        # Entropy
        ax1.plot(xs, trace["entropy"], color="#e05c5c", linewidth=1.4, label="Entropy (nats)")
        ax1.set_ylabel("Entropy (nats)", fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.fill_between(xs, trace["entropy"], alpha=0.15, color="#e05c5c")

        # Top-1 probability
        ax2.plot(xs, trace["top1_prob"], color="#5c8ee0", linewidth=1.4, label="Top-1 probability")
        ax2.set_ylabel("Top-1 prob", fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.fill_between(xs, trace["top1_prob"], alpha=0.15, color="#5c8ee0")

        # Correct-answer probability
        ax3.plot(xs, trace["correct_prob"], color="#5ce07a", linewidth=1.4, label="Correct-answer prob")
        ax3.set_ylabel("Correct-answer prob", fontsize=9)
        ax3.set_xlabel("Token (character) position", fontsize=9)
        ax3.set_ylim(bottom=0)
        ax3.fill_between(xs, trace["correct_prob"], alpha=0.2, color="#5ce07a")

        # #### commit line
        if commit >= 0:
            for ax in (ax1, ax2, ax3):
                ax.axvline(commit, color="#ffaa00", linewidth=2, linestyle="--",
                           label=f"#### commit (pos {commit})")
                ax.legend(fontsize=8, loc="upper right")

        # Annotate intermediate audit
        for item in trace["intermediates"]:
            col = "#2ecc71" if item["correct"] else "#e74c3c"
            # find position of this expression in generated text
            pos = trace["generated"].find(str(item["written"]))
            if 0 <= pos < len(xs):
                ax3.annotate(
                    item["expr"][:20],
                    xy=(pos, trace["correct_prob"][pos]),
                    xytext=(pos, trace["correct_prob"][pos] + 0.05),
                    fontsize=6, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5),
                )

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved -> {out_path}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


# ── batch comparison mode ─────────────────────────────────────────────────────

def compare_models(pairs: list[dict], out_dir: Path | None = None) -> None:
    """
    Run trace on multiple (ckpt, problem, answer) pairs and emit a summary table.
    """
    summaries = []
    for spec in pairs:
        print(f"\nLoading {spec['ckpt']} ...")
        model, _ = load_model(spec["ckpt"])
        trace = trace_generate(model, spec["problem"], spec["answer"])
        print_report(trace)
        if out_dir:
            slug = spec["label"].replace(" ", "_").replace("/", "-")
            save_matplotlib(trace, out_dir / f"{slug}.png")
            json_path = out_dir / f"{slug}.json"
            # exclude heavy token arrays from JSON for readability
            slim = {k: v for k, v in trace.items() if k not in ("tokens",)}
            json_path.write_text(json.dumps(slim, indent=2))
            print(f"  💾 JSON -> {json_path}")
        commit = trace["commit_pos"]
        avg_h_pre = (
            (sum(trace["entropy"][:commit]) / commit)
            if commit > 0
            else (sum(trace["entropy"]) / max(1, len(trace["entropy"])))
        )
        summaries.append({
            "label":         spec["label"],
            "problem":       spec["problem"],
            "expected":      spec["answer"],
            "predicted":     trace["predicted"],
            "correct":       trace["correct"],
            "pre_commit":    trace["pre_commit_chars"],
            "total_chars":   trace["total_chars"],
            "avg_entropy_pre": avg_h_pre,
            "intermediate_errors": sum(1 for i in trace["intermediates"]
                                       if not i["correct"]),
        })

    print(f"\n{'='*80}")
    print("  CROSS-MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model / Problem':<35}  {'pre-####':>8}  {'avg H':>6}  {'int_err':>7}  result")
    print(f"  {'─'*35}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*6}")
    for s in summaries:
        ok = "✅" if s["correct"] else "❌"
        print(f"  {s['label']:<35}  {s['pre_commit']:>8}  {s['avg_entropy_pre']:>6.3f}"
              f"  {s['intermediate_errors']:>7}  {ok}")
    print()


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> tuple[MathGPT, GPTConfig]:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", GPTConfig())
    if isinstance(cfg, dict):
        cfg = GPTConfig(**cfg)
    model = MathGPT(cfg)
    # handle both raw state_dict and wrapped checkpoint
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Per-token inference telemetry for MathGPT")
    ap.add_argument("--ckpt",    required=False, default=None, help="Path to .pt checkpoint")
    ap.add_argument("--problem", required=False, default=None, help="Math problem string, e.g. '47 + 58'")
    ap.add_argument("--answer",  required=False, default=None, type=int, help="Ground-truth answer")
    ap.add_argument("--label",   default=None, help="Display label for this run")
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--plot",    action="store_true", help="Save matplotlib PNG")
    ap.add_argument("--out-dir", default="results/traces",
                    help="Directory for plot + JSON output")
    ap.add_argument("--batch",   default=None,
                    help="JSON file with list of {ckpt,problem,answer,label} for comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.plot else None

    if args.batch:
        pairs = json.loads(Path(args.batch).read_text())
        compare_models(pairs, out_dir)
    else:
        label = args.label or f"{Path(args.ckpt).stem} | {args.problem}"
        print(f"\nLoading {args.ckpt} ...")
        model, _ = load_model(args.ckpt)
        prompt = PROMPT_TEMPLATE.format(problem=args.problem)
        trace  = trace_generate(model, prompt, args.answer, max_new=args.max_new)
        print_report(trace)
        if out_dir:
            slug = label.replace(" ", "_").replace("/", "-")
            save_matplotlib(trace, out_dir / f"{slug}.png")
            slim = {k: v for k, v in trace.items() if k not in ("tokens",)}
            (out_dir / f"{slug}.json").write_text(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
