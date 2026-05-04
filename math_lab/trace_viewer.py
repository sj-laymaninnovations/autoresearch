"""
trace_viewer.py — Cinebench-style benchmark dashboard for MathGPT traces.

Reads all JSON files in results/traces/ and generates a self-contained
HTML report with:
  - Phase segmentation: [EXAMPLE] [SCRATCHPAD] [####] [ANSWER]
  - Per-phase score bars (entropy, confidence, intermediate accuracy)
  - Token-by-token sparkline (SVG)
  - Benchmark score per QA pair
  - Cross-model comparison table

Usage:
    python trace_viewer.py --out results/traces/dashboard.html
"""

import json, re, argparse, math
from pathlib import Path


# ── Phase segmentation ────────────────────────────────────────────────────────

def segment(trace: dict) -> dict:
    gen  = trace.get("generated", "")
    toks = trace.get("tokens", [])
    entr = trace.get("entropy", [])
    corp = trace.get("correct_prob", [])
    t1   = trace.get("top1_prob", [])

    # Find phase boundaries in generated text
    ex_start = gen.find("Example:")
    ex_end   = gen.find("\n", ex_start) if ex_start >= 0 else -1
    commit   = trace.get("commit_pos", -1)

    phases = {}
    # EXAMPLE
    if ex_start >= 0 and ex_end > 0:
        phases["example"] = {"start": ex_start, "end": ex_end,
                              "label": "EXAMPLE", "color": "#f5a623"}
    # SCRATCHPAD (after example line, before ####)
    sc_start = (ex_end + 1) if ex_end >= 0 else 0
    sc_end   = commit if commit >= 0 else len(gen)
    if sc_end > sc_start:
        phases["scratchpad"] = {"start": sc_start, "end": sc_end,
                                 "label": "SCRATCHPAD", "color": "#4a9eff"}
    # COMMIT
    if commit >= 0:
        phases["commit"] = {"start": commit, "end": min(commit + 4, len(gen)),
                             "label": "####", "color": "#ff6b35"}
    # ANSWER
    ans_start = commit + 4 if commit >= 0 else -1
    if ans_start >= 0 and ans_start < len(gen):
        phases["answer"] = {"start": ans_start, "end": len(gen),
                             "label": "ANSWER", "color": "#5ce07a"}

    def phase_stats(s, e):
        sl = slice(max(0,s), min(e, len(entr)))
        es = entr[sl]; cs = corp[sl]; ts = t1[sl]
        n  = max(1, len(es))
        return {
            "n_tokens":   e - s,
            "avg_entropy": sum(es)/n if es else 0,
            "avg_top1":    sum(ts)/n if ts else 0,
            "avg_corr_p":  sum(cs)/n if cs else 0,
            "peak_corr_p": max(cs) if cs else 0,
        }

    for k, p in phases.items():
        p["stats"] = phase_stats(p["start"], p["end"])

    # Benchmark score: penalise entropy in scratchpad, reward correct final
    sp = phases.get("scratchpad", {}).get("stats", {})
    commitment_sharpness = 1.0 - min(1.0, sp.get("avg_entropy", 4) / math.log(131))
    correct_bonus        = 100 if trace.get("correct") else 0
    token_efficiency     = max(0, 1.0 - (trace.get("pre_commit_chars", 200) / 200))
    score = int(commitment_sharpness * 60 + correct_bonus * 0.3 + token_efficiency * 10)

    return {"phases": phases, "score": score,
            "entr": entr, "corp": corp, "t1": t1, "gen": gen}


# ── SVG sparkline ─────────────────────────────────────────────────────────────

def sparkline_svg(values, color, width=400, height=60, label="", commit=-1) -> str:
    if not values:
        return f'<svg width="{width}" height="{height}"></svg>'
    mx = max(values) or 1
    pts = []
    for i, v in enumerate(values):
        x = int(i / len(values) * width)
        y = int(height - (v / mx) * (height - 4) - 2)
        pts.append(f"{x},{y}")
    poly = " ".join(pts)
    cl = ""
    if commit >= 0 and commit < len(values):
        cx = int(commit / len(values) * width)
        cl = f'<line x1="{cx}" y1="0" x2="{cx}" y2="{height}" stroke="#ffaa00" stroke-width="2" stroke-dasharray="3,2"/>'
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'{cl}</svg>')


# ── HTML generation ───────────────────────────────────────────────────────────

CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d12; color: #e0e0e0; font-family: 'Inter', 'Segoe UI', monospace; padding: 24px; }
h1 { color: #fff; font-size: 1.5rem; margin-bottom: 4px; }
.subtitle { color: #666; font-size: 0.85rem; margin-bottom: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 20px; }
.card { background: #16161f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.card-title { font-size: 0.95rem; font-weight: 600; color: #ccc; }
.card-sub { font-size: 0.75rem; color: #555; margin-top: 2px; }
.score { font-size: 2rem; font-weight: 800; font-family: monospace; }
.score.ok { color: #5ce07a; }
.score.fail { color: #e05c5c; }
.phases { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 10px 0 16px; gap: 2px; }
.phase-seg { border-radius: 2px; }
.phase-legend { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; font-size: 0.72rem; }
.badge { padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.stats-row { display: flex; gap: 16px; margin-bottom: 10px; }
.stat { flex: 1; }
.stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 1.05rem; font-weight: 700; font-family: monospace; }
.sparkline-wrap { margin: 8px 0 4px; }
.spark-label { font-size: 0.65rem; color: #555; margin-bottom: 2px; text-transform: uppercase; }
.gen-text { background: #0a0a10; border: 1px solid #222; border-radius: 6px; padding: 10px;
            font-size: 0.7rem; line-height: 1.6; white-space: pre-wrap; color: #999; margin-top: 10px; max-height: 120px; overflow: auto; }
.int-audit { margin-top: 10px; font-size: 0.72rem; }
.int-row { display: flex; gap: 8px; padding: 2px 0; border-bottom: 1px solid #1a1a28; }
.int-expr { flex: 1; color: #888; font-family: monospace; }
.ok-tag { color: #5ce07a; }
.fail-tag { color: #e05c5c; }
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 24px; }
.summary-table th { background: #16161f; color: #777; text-align: left; padding: 8px 12px;
                    border-bottom: 1px solid #2a2a3a; font-weight: 500; text-transform: uppercase; font-size: 0.68rem; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #1a1a26; }
.summary-table tr:hover td { background: #1a1a26; }
"""

def phase_color(k):
    return {"example":"#f5a623","scratchpad":"#4a9eff","commit":"#ff6b35","answer":"#5ce07a"}.get(k,"#444")


def render_card(trace: dict, seg: dict) -> str:
    label  = trace.get("label", "?")
    prob   = trace.get("problem","")
    exp    = trace.get("expected","?")
    pred   = trace.get("predicted","?")
    ok     = trace.get("correct", False)
    score  = seg["score"]
    phases = seg["phases"]
    gen    = seg["gen"]
    commit = trace.get("commit_pos", -1)
    tot    = trace.get("total_chars", 1) or 1

    # Phase bar
    phase_bar = '<div class="phases">'
    for k, p in phases.items():
        w = max(2, int((p["end"] - p["start"]) / tot * 100))
        phase_bar += f'<div class="phase-seg" style="width:{w}%;background:{p["color"]}"></div>'
    phase_bar += '</div>'

    # Legend
    legend = '<div class="phase-legend">'
    for k, p in phases.items():
        n = p["stats"]["n_tokens"]
        legend += f'<span class="badge" style="background:{p["color"]}22;color:{p["color"]}">{p["label"]} {n}t</span>'
    legend += '</div>'

    # Stats
    sp  = phases.get("scratchpad", {}).get("stats", {})
    ans = phases.get("answer", {}).get("stats", {})
    ent_pre  = f'{sp.get("avg_entropy", 0):.2f}'
    peak_cp  = f'{max(p["stats"]["peak_corr_p"] for p in phases.values()):.3f}'
    int_errs = sum(1 for i in trace.get("intermediates",[]) if not i["correct"])

    stats = f'''<div class="stats-row">
      <div class="stat"><div class="stat-label">pre-#### tokens</div>
        <div class="stat-val">{trace.get("pre_commit_chars","?")}</div></div>
      <div class="stat"><div class="stat-label">scratchpad H̄</div>
        <div class="stat-val">{ent_pre} nats</div></div>
      <div class="stat"><div class="stat-label">peak corr_p</div>
        <div class="stat-val">{peak_cp}</div></div>
      <div class="stat"><div class="stat-label">arith errors</div>
        <div class="stat-val" style="color:{'#e05c5c' if int_errs else '#5ce07a'}">{int_errs}</div></div>
    </div>'''

    # Sparklines
    w = 480
    entr_svg = sparkline_svg(seg["entr"], "#e05c5c", w, 50, commit=commit)
    corp_svg = sparkline_svg(seg["corp"], "#5ce07a", w, 50, commit=commit)

    # Generated text with phase highlights
    def highlight(text):
        out = text
        out = out.replace("Example:", '<span style="color:#f5a623;font-weight:600">Example:</span>')
        out = out.replace("####", '<span style="color:#ff6b35;font-weight:800">####</span>')
        return out

    # Intermediate audit
    int_html = ""
    if trace.get("intermediates"):
        int_html = '<div class="int-audit">'
        for item in trace["intermediates"][:6]:
            tag = '<span class="ok-tag">✅</span>' if item["correct"] else '<span class="fail-tag">❌</span>'
            int_html += f'<div class="int-row"><span class="int-expr">{item["expr"]}</span>{tag}</div>'
        int_html += '</div>'

    score_class = "ok" if ok else "fail"
    result_tag = f'<span style="color:#5ce07a">✅ {pred}</span>' if ok else f'<span style="color:#e05c5c">❌ got {pred}</span>'

    return f'''<div class="card">
  <div class="card-header">
    <div>
      <div class="card-title">{label}</div>
      <div class="card-sub">Q: {prob} &nbsp;→&nbsp; expected <strong>{exp}</strong> &nbsp; {result_tag}</div>
    </div>
    <div class="score {score_class}">{score}</div>
  </div>
  {phase_bar}
  {legend}
  {stats}
  <div class="sparkline-wrap"><div class="spark-label">Entropy (uncertainty)</div>{entr_svg}</div>
  <div class="sparkline-wrap"><div class="spark-label">Correct-answer probability</div>{corp_svg}</div>
  <div class="gen-text">{highlight(gen[:400])}</div>
  {int_html}
</div>'''


def build_html(traces: list[dict]) -> str:
    segs = [segment(t) for t in traces]

    cards = "\n".join(render_card(t, s) for t, s in zip(traces, segs))

    # Summary table
    rows = ""
    for t, s in zip(traces, segs):
        ok  = t.get("correct", False)
        dot = '<span style="color:#5ce07a">●</span>' if ok else '<span style="color:#e05c5c">●</span>'
        sp  = s["phases"].get("scratchpad",{}).get("stats",{})
        rows += f'''<tr>
          <td>{dot} {t.get("label","?")}</td>
          <td style="font-family:monospace">{t.get("pre_commit_chars","?")}</td>
          <td style="font-family:monospace">{sp.get("avg_entropy",0):.3f}</td>
          <td style="font-family:monospace">{max(p["stats"]["peak_corr_p"] for p in s["phases"].values()):.4f}</td>
          <td style="font-family:monospace">{sum(1 for i in t.get("intermediates",[]) if not i["correct"])}</td>
          <td style="font-family:monospace;font-weight:700">{s["score"]}</td>
        </tr>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MathGPT Token Telemetry — Benchmark Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CARD_CSS}</style>
</head>
<body>
<h1>MathGPT · Token Telemetry Benchmark</h1>
<p class="subtitle">
  Per-token generation diagnostics — phased like Cinebench: 
  <span style="color:#f5a623">■ EXAMPLE</span> &nbsp;
  <span style="color:#4a9eff">■ SCRATCHPAD</span> &nbsp;
  <span style="color:#ff6b35">■ #### COMMIT</span> &nbsp;
  <span style="color:#5ce07a">■ ANSWER</span>
</p>

<div class="grid">
{cards}
</div>

<table class="summary-table" style="margin-top:32px">
  <thead><tr>
    <th>Model / Problem</th><th>Pre-#### tokens</th>
    <th>Scratchpad H̄</th><th>Peak corr_p</th>
    <th>Arith errors</th><th>Score</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="results/traces")
    ap.add_argument("--out",       default="results/traces/dashboard.html")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    json_files = sorted(trace_dir.glob("*.json"))
    if not json_files:
        print("No trace JSON files found in", trace_dir); return

    traces = []
    for f in json_files:
        d = json.loads(f.read_text())
        d.setdefault("label", f.stem)
        traces.append(d)

    html = build_html(traces)
    out  = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {out}  ({len(traces)} traces)")


if __name__ == "__main__":
    main()
