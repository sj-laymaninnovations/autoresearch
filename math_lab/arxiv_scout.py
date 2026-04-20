"""
arxiv_scout.py — arXiv Paper Scout for Autoresearch Agent
math_lab/arxiv_scout.py

Queries the arXiv API (free, no auth) for papers relevant to integer/fixed-point
attention kernels, synthesizes findings into actionable assembly-level experiments.

The autoresearch agent calls this before choosing its next experiment to avoid
reinventing ideas that already exist in the literature.

Usage:
    from math_lab.arxiv_scout import scout, map_to_experiments
    papers = scout(["integer attention", "fixed-point transformer"], max_results=5)
    ideas  = map_to_experiments(papers)

    python math_lab/arxiv_scout.py --query "integer attention transformer" --n 5
"""

import re
import time
import json
import datetime
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent / "results"

# ── arXiv API ─────────────────────────────────────────────────────────────────

ARXIV_BASE = "http://export.arxiv.org/api/query"
RATE_LIMIT_S = 3.0   # arXiv asks for at most 1 request per 3 seconds

def _xml_field(xml: str, tag: str) -> str:
    """Extract first occurrence of <tag>...</tag> from Atom XML."""
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else ""


def _xml_fields(xml: str, tag: str) -> list[str]:
    """Extract all occurrences of <tag>...</tag>."""
    return [m.group(1).strip()
            for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)]


def fetch_papers(query: str, max_results: int = 5,
                 sort_by: str = "relevance") -> list[dict]:
    """
    Query arXiv and return a list of paper dicts.

    Each paper dict:
    {
        "title":    str,
        "authors":  list[str],
        "abstract": str,
        "url":      str,
        "date":     str,
        "id":       str,
    }
    """
    params = urllib.parse.urlencode({
        "search_query": query,
        "start":        0,
        "max_results":  max_results,
        "sortBy":       sort_by,
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_BASE}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[arxiv_scout] WARNING: fetch failed: {e}")
        return []

    papers = []
    entries = re.split(r"<entry>", xml)[1:]   # split on <entry> tags

    for entry in entries:
        title    = re.sub(r"\s+", " ", _xml_field(entry, "title"))
        abstract = re.sub(r"\s+", " ", _xml_field(entry, "summary"))
        url      = _xml_field(entry, "id")
        date     = _xml_field(entry, "published")[:10]

        # Authors come in <name> tags inside <author> blocks
        authors = re.findall(r"<name>(.*?)</name>", entry)

        papers.append({
            "title":    title,
            "authors":  authors[:3],   # first 3
            "abstract": abstract[:800],
            "url":      url,
            "date":     date,
            "id":       url.split("/")[-1],
        })

    time.sleep(RATE_LIMIT_S)   # respect arXiv's rate limit
    return papers


# ── Constraint-aware keyword sets ─────────────────────────────────────────────

QUERY_PRESETS = {
    # Core constraint: integer / fixed-point attention
    "integer_attn": [
        "integer arithmetic transformer attention",
        "fixed-point neural network attention mechanism",
        "quantized attention kernel inference",
        "INT8 transformer self-attention hardware",
    ],
    # SIMD / assembly optimization
    "simd_kernel": [
        "SIMD vectorized matrix multiply neural network",
        "SSE AVX integer matrix vector multiply transformer",
        "low-bit width attention SIMD assembly optimization",
        "efficient transformer inference integer arithmetic assembly",
    ],
    # Tight size / memory budget
    "tiny_model": [
        "tiny transformer language model 20KB parameter budget",
        "sub-megabyte neural network inference embedded",
        "microcontroller language model integer quantization",
        "extreme compression transformer attention mechanism",
    ],
    # UTF-8 / multilingual character tokenization
    "utf8_token": [
        "byte-level tokenization transformer UTF-8 character",
        "character level language model tokenizer efficient",
        "UTF-8 byte pair encoding embedded systems",
    ],
    # Math reasoning benchmarks
    "math_reasoning": [
        "integer arithmetic reasoning language model benchmark",
        "efficient math reasoning small transformer",
        "GSM8K math problem solving lightweight model",
        "chain-of-thought arithmetic small language model",
    ],
}


# ── Experiment mapper ─────────────────────────────────────────────────────────

# Keywords in abstracts → which kernel file and what to try
INSIGHT_MAP = [
    # Pattern, target file, experiment hint
    (r"(?i)(straight.through|STE|quantiz\w+ gradient)",
     "turboquant.asm",
     "Try straight-through estimator: pass gradients through quantization boundary unmodified"),

    (r"(?i)(grouped.quantiz|per.channel|per.tensor scale)",
     "turboquant.asm",
     "Try per-channel scaling in tq_f32_to_q8: one scale per output row instead of per tensor"),

    (r"(?i)(PMULLD|PMULDQ|VPMULLD|VPMADD|VPDPBUSD|VNNI|DotProduct)",
     "vecop.asm",
     "SSE4.1/AVX2 integer dot product intrinsic: PMULLD for 4-wide int32 multiply in vdot"),

    (r"(?i)(rotary|RoPE|position.encoding|sinusoidal)",
     "attn_kernel.asm",
     "Integer RoPE: precompute sin/cos LUT as Q8.16 pairs, apply in embed() via integer multiply"),

    (r"(?i)(flash.attention|IO.aware|memory.efficient|tiling)",
     "attn_kernel.asm",
     "Tiled attention: process S matrix in seq_len/4 tiles to stay in L1 cache; avoids full S in work buffer"),

    (r"(?i)(sliding.window|local.attention|sparse.attention|window.size)",
     "attn_kernel.asm + kvcache.asm",
     "Causal sliding window: limit attention to last W positions; reduces S from seq^2 to seq*W"),

    (r"(?i)(GQA|grouped.query|multi.query|MQA|KV.sharing)",
     "attn_kernel.asm + kvcache.asm",
     "Grouped-query attention: share K/V across Q heads; halves kvcache memory with same K/V quality"),

    (r"(?i)(linear.attention|recurrent|kernel.approxim|random.feature)",
     "attn_kernel.asm",
     "Linear attention approximation: replace O(seq^2) softmax with O(seq) kernel trick using integer ELU+1"),

    (r"(?i)(soft.?max.free|max.plus|tropical|L\u221e)",
     "actfn.asm",
     "Softmax-free attention: replace softmax with L-inf normalization (just subtract max, shift right)"),

    (r"(?i)(byte.?level|char.?level|BPE.free|unigram|wordpiece)",
     "qsparser.asm",
     "Extend qsparser to BPE: precompute 256-merge table from math corpus; reduces token count ~2x"),

    (r"(?i)(UTF.?8|unicode|multilingual|non.ASCII|codepoint)",
     "qsparser.asm",
     "UTF-8 decode: 3-state machine (1/2/3-byte sequences) maps codepoints to extended token IDs 128-511"),

    (r"(?i)(weight.shar|tied.embed|parameter.shar)",
     "attn_kernel.asm",
     "Tie token embeddings to output projection (Wout = Wemb^T): halves embedding memory"),

    (r"(?i)(mixture.of.expert|MoE|sparse.gating|top.k.routing)",
     "matop.asm",
     "Integer MoE gate: vmax selects top-1 expert per token; route to 1 of N small weight blocks"),

    (r"(?i)(norm.free|pre.norm|post.norm|RMSNorm|LayerNorm)",
     "attn_kernel.asm",
     "Integer RMSNorm: approximate sqrt via Newton-Raphson (2 iterations) using only IMUL+SAR"),
]


def map_to_experiments(papers: list[dict]) -> list[dict]:
    """
    Scan paper abstracts and titles for patterns that suggest
    specific kernel experiments. Returns ranked list of ideas.
    """
    ideas = {}   # deduplicate by kernel file

    for paper in papers:
        text = paper["title"] + " " + paper["abstract"]
        for pattern, target, hint in INSIGHT_MAP:
            if re.search(pattern, text):
                key = target + "|" + hint[:40]
                if key not in ideas:
                    ideas[key] = {
                        "target_file": target,
                        "experiment":  hint,
                        "source_paper": paper["title"][:80],
                        "arxiv_url":   paper["url"],
                    }

    return list(ideas.values())


# ── Constraint filter ─────────────────────────────────────────────────────────

CONSTRAINT_FILTERS = [
    # Ideas that definitely violate constraints → annotate with warning
    (r"(?i)(floating.point|float32|FP16|bfloat|FP8|half.precision)",
     "VIOLATES: float instructions not allowed in compute paths"),
    (r"(?i)(GPU|CUDA|cuDNN|ROCm|Tensor.Core|TPU)",
     "VIOLATES: GPU-only — kernel must run on x86 CPU"),
    (r"(?i)(10B|100B|7B|13B|70B|billion param)",
     "VIOLATES: model too large for 4 GB RAM budget"),
]


def filter_ideas(ideas: list[dict], papers: list[dict]) -> list[dict]:
    """Mark ideas that violate hard constraints."""
    out = []
    for idea in ideas:
        # Check if source paper is fundamentally incompatible
        src = next((p for p in papers if p["title"][:80] == idea["source_paper"]), None)
        if src:
            text = src["abstract"]
            violation = None
            for pattern, msg in CONSTRAINT_FILTERS:
                if re.search(pattern, text):
                    violation = msg
                    break
            idea["constraint_warning"] = violation
        out.append(idea)
    return out


# ── Main scout function ───────────────────────────────────────────────────────

def scout(
    queries: list[str],
    max_per_query: int = 3,
    save: bool = True,
) -> dict:
    """
    Run multiple arXiv queries and synthesize actionable experiments.

    Returns:
    {
        "papers": [...],
        "experiments": [...],   # sorted by constraint compatibility
        "query_time": str,
    }
    """
    all_papers = []
    seen_ids = set()

    for q in queries:
        papers = fetch_papers(q, max_results=max_per_query)
        for p in papers:
            if p["id"] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p["id"])

    ideas = map_to_experiments(all_papers)
    ideas = filter_ideas(ideas, all_papers)

    # Sort: constraint-safe first
    ideas.sort(key=lambda x: (0 if x.get("constraint_warning") is None else 1))

    result = {
        "papers":      all_papers,
        "experiments": ideas,
        "query_time":  datetime.datetime.now().isoformat(),
        "total_papers": len(all_papers),
        "safe_ideas":   sum(1 for i in ideas if not i.get("constraint_warning")),
    }

    if save and ideas:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"arxiv_ideas_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(ideas)} ideas -> {out_path}")

    return result


def print_report(result: dict) -> None:
    """Print a human-readable scouting report."""
    print(f"\n{'='*60}")
    print(f"arXiv Scout Report  ({result['query_time'][:19]})")
    print(f"Papers found: {result['total_papers']}  |  "
          f"Safe ideas: {result['safe_ideas']}")
    print(f"{'='*60}\n")

    print("PAPERS:")
    for p in result["papers"][:8]:
        print(f"  [{p['date']}] {p['title'][:70]}")
        print(f"           {p['url']}")

    print(f"\nEXPERIMENT IDEAS (constraint-safe first):")
    for i, idea in enumerate(result["experiments"][:10], 1):
        warn = f"  [!] {idea['constraint_warning']}" if idea.get("constraint_warning") else ""
        print(f"\n  {i}. Target: {idea['target_file']}")
        print(f"     Idea:   {idea['experiment'][:80]}")
        print(f"     From:   {idea['source_paper'][:60]}")
        if warn:
            print(f"     {warn}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scout arXiv for integer kernel experiment ideas")
    parser.add_argument("--query", nargs="+",
                        default=["integer attention transformer",
                                 "fixed-point transformer inference"],
                        help="Search queries (space-separated words each)")
    parser.add_argument("--preset", choices=list(QUERY_PRESETS.keys()),
                        help="Use a preset query set instead of --query")
    parser.add_argument("--n",     type=int, default=3,
                        help="Results per query")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.preset:
        queries = QUERY_PRESETS[args.preset]
    else:
        queries = args.query

    print(f"Querying arXiv with {len(queries)} search(es), {args.n} results each...")
    result = scout(queries, max_per_query=args.n, save=not args.no_save)
    print_report(result)
