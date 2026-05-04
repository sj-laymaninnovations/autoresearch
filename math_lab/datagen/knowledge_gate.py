"""
knowledge_gate.py — Selective domain gating using knowledge_areas.csv

Maps training records to knowledge area slugs, evaluates gate configs,
and assigns curriculum_step + loss_weight per record.

Usage (standalone tagging):
    from math_lab.datagen.knowledge_gate import KnowledgeGate
    gate = KnowledgeGate("math_lab/knowledge_areas.csv")
    gate.load_config("math_lab/gate_configs/arith_word.yaml")
    result = gate.evaluate({"problem": "7 * 8 = ?", "concept": "mul_1d"})
    # → {"include": True, "loss_weight": 1.5, "curriculum_step": 1, "ka_slugs": ["mathematics"]}

CLI tagging of a JSONL corpus:
    python3 math_lab/datagen/knowledge_gate.py \\
        --data math_lab/results/oss/oss_train_*.jsonl \\
        --gate math_lab/gate_configs/math_only.yaml \\
        --out  math_lab/results/oss/gated_train.jsonl
"""

import csv, re, json, argparse, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Skill-name → (slug, sub_area, curriculum_step) keyword map ───────────────
# Covers our existing skill naming convention and common problem text patterns.
# curriculum_step mirrors the math dependency graph: arithmetic → algebra → word
SKILL_MAP: list[tuple[str, str, str, int]] = [
    # (pattern,            slug,          sub_area,         step)
    # ── Our datagen skill names ───────────────────────────────────────────────
    ("add_",              "mathematics", "Arithmetic",      1),
    ("sub_",              "mathematics", "Arithmetic",      1),
    ("mul_",              "mathematics", "Arithmetic",      1),
    ("div_",              "mathematics", "Arithmetic",      1),
    ("mod_",              "mathematics", "Number Theory",   1),
    ("alg_1step",         "mathematics", "Algebra",         2),
    ("alg_2step",         "mathematics", "Algebra",         2),
    ("alg_distribute",    "mathematics", "Algebra",         2),
    ("alg_",              "mathematics", "Algebra",         2),
    ("word_problem",      "mathematics", "Word Problems",   3),
    ("curriculum",        "mathematics", "Arithmetic",      1),
    ("arith",             "mathematics", "Arithmetic",      1),
    # ── OSS sources ──────────────────────────────────────────────────────────
    ("gsm8k",             "mathematics", "Word Problems",   3),
    ("metamath",          "mathematics", "Algebra",         2),
    ("orca_math",         "mathematics", "Word Problems",   3),
    # ── Problem text keywords ─────────────────────────────────────────────────
    ("percent",           "mathematics", "Arithmetic",      2),
    ("fraction",          "mathematics", "Arithmetic",      2),
    ("probability",       "statistics",  "Descriptive Statistics", 4),
    ("statistic",         "statistics",  "Descriptive Statistics", 4),
    ("geometry",          "mathematics", "Geometry",        3),
    ("triangle",          "mathematics", "Geometry",        3),
    ("circle",            "mathematics", "Geometry",        3),
    ("area",              "mathematics", "Geometry",        3),
    ("volume",            "mathematics", "Geometry",        3),
    ("angle",             "mathematics", "Geometry",        3),
    ("calculus",          "mathematics", "Calculus",        4),
    ("derivative",        "mathematics", "Calculus",        4),
    ("integral",          "mathematics", "Calculus",        4),
]

# ── Tier → curriculum step fallback ──────────────────────────────────────────
TIER_TO_STEP = {1: 1, 2: 2, 3: 3}


@dataclass
class GateConfig:
    name:               str              = "default"
    active_slugs:       list[str]        = field(default_factory=lambda: ["mathematics"])
    active_sub_areas:   dict[str, list]  = field(default_factory=dict)  # slug → [sub_areas]
    follow_dependencies:bool             = True
    curriculum_order:   str              = "dep_graph"   # none|dep_graph|difficulty|source
    loss_weight:        dict[str, float] = field(default_factory=dict)  # slug → multiplier
    exclude_slugs:      list[str]        = field(default_factory=list)
    tier_filter:        Optional[list]   = None          # e.g. [1,2] to exclude tier 3


class KnowledgeGate:
    """
    Loads knowledge_areas.csv and a gate config, then evaluates each training
    record returning: include, loss_weight, curriculum_step, ka_slugs, ka_sub_areas.
    """

    def __init__(self, csv_path: str = "math_lab/knowledge_areas.csv"):
        self.csv_path = Path(csv_path)
        self.ka_table: dict[str, dict] = {}   # slug → row
        self.config: Optional[GateConfig] = None
        self._load_csv()

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"knowledge_areas.csv not found at {self.csv_path}\n"
                "Copy it with: cp ~/Downloads/Knowledge_Areas_signal_harmonized.csv "
                "math_lab/knowledge_areas.csv"
            )
        with open(self.csv_path, encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row.get("identifier_slug", "").strip()
                if slug:
                    self.ka_table[slug] = {
                        "slug":         slug,
                        "field":        row.get("field", "").strip(),
                        "domain":       row.get("domain_category", "").strip(),
                        "sub_areas":    [s.strip() for s in
                                         row.get("knowledge_areas", "").split(";") if s.strip()],
                        "dependencies": [s.strip() for s in
                                         row.get("dependencies", "").replace(";", ",").split(",")
                                         if s.strip()],
                    }

    # ── Config loading ────────────────────────────────────────────────────────

    def load_config(self, config_path: str):
        """Load a gate_config.yaml. Falls back to inline dict if path is a dict."""
        if isinstance(config_path, dict):
            cfg = config_path
        else:
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
            except ImportError:
                # yaml not installed — parse minimal subset manually
                cfg = self._parse_yaml_minimal(Path(config_path).read_text())

        self.config = GateConfig(
            name               = cfg.get("name", Path(str(config_path)).stem),
            active_slugs       = cfg.get("active_slugs", ["mathematics"]),
            active_sub_areas   = cfg.get("active_sub_areas", {}),
            follow_dependencies= cfg.get("follow_dependencies", True),
            curriculum_order   = cfg.get("curriculum_order", "dep_graph"),
            loss_weight        = cfg.get("loss_weight", {}),
            exclude_slugs      = cfg.get("exclude_slugs", []),
            tier_filter        = cfg.get("tier_filter", None),
        )

        # Expand dependencies transitively if follow_dependencies=True
        if self.config.follow_dependencies:
            self._expand_deps()

        return self.config

    def load_config_dict(self, cfg: dict):
        return self.load_config(cfg)

    def _expand_deps(self):
        """Transitively add dependency slugs to active_slugs."""
        visited = set(self.config.active_slugs)
        queue   = list(self.config.active_slugs)
        while queue:
            slug = queue.pop()
            row  = self.ka_table.get(slug, {})
            for dep in row.get("dependencies", []):
                # Normalize dep string to slug format
                dep_slug = dep.lower().replace(" ", "-")
                if dep_slug in self.ka_table and dep_slug not in visited:
                    visited.add(dep_slug)
                    queue.append(dep_slug)
        self.config.active_slugs = list(visited)

    def _parse_yaml_minimal(self, text: str) -> dict:
        """Minimal YAML parser for list/str/bool/float values (no PyYAML needed)."""
        result = {}
        current_key = None
        current_list = None
        sub_dict_key = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("- ") and current_list is not None:
                current_list.append(stripped[2:].strip().strip('"\''))
            elif ":" in stripped and indent == 0:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip().strip('"\'')
                if val == "":
                    current_key = key
                    current_list = []
                    result[key] = current_list
                    sub_dict_key = None
                elif val.lower() == "true":
                    result[key] = True; current_list = None
                elif val.lower() == "false":
                    result[key] = False; current_list = None
                else:
                    try: result[key] = float(val) if "." in val else int(val)
                    except ValueError: result[key] = val
                    current_list = None
            elif ":" in stripped and indent == 2 and current_key == "loss_weight":
                key, _, val = stripped.partition(":")
                if "loss_weight" not in result or not isinstance(result["loss_weight"], dict):
                    result["loss_weight"] = {}
                try: result["loss_weight"][key.strip()] = float(val.strip())
                except ValueError: pass
        return result

    # ── Record tagging ────────────────────────────────────────────────────────

    def tag(self, record: dict) -> dict:
        """
        Tag a training record with ka_slugs, ka_sub_areas, curriculum_step.
        Does NOT filter — call evaluate() for that.
        """
        concept = str(record.get("concept", "")).lower()
        source  = str(record.get("source",  "")).lower()
        problem = str(record.get("problem", "")).lower()
        tier    = int(record.get("tier", 1))

        matched_slugs    = []
        matched_sub      = []
        matched_steps    = []

        probe = concept + " " + source + " " + problem[:200]

        for pattern, slug, sub_area, step in SKILL_MAP:
            if pattern in probe:
                if slug not in matched_slugs:
                    matched_slugs.append(slug)
                if sub_area not in matched_sub:
                    matched_sub.append(sub_area)
                matched_steps.append(step)

        # Fallback: tag as mathematics/Word Problems if no match
        if not matched_slugs:
            matched_slugs = ["mathematics"]
            matched_sub   = ["Word Problems"]
            matched_steps = [TIER_TO_STEP.get(tier, 2)]

        curriculum_step = min(matched_steps)   # easiest prerequisite wins

        return {
            **record,
            "ka_slugs":       matched_slugs,
            "ka_sub_areas":   matched_sub,
            "curriculum_step": curriculum_step,
        }

    def evaluate(self, record: dict) -> dict:
        """
        Full gate evaluation. Returns tagged record + include + loss_weight.
        Requires load_config() to have been called first.
        """
        if self.config is None:
            raise RuntimeError("Call load_config() before evaluate()")

        tagged = self.tag(record)
        cfg    = self.config
        tier   = int(record.get("tier", 1))

        # ── Tier filter ───────────────────────────────────────────────────────
        if cfg.tier_filter and tier not in cfg.tier_filter:
            return {**tagged, "include": False, "loss_weight": 0.0}

        # ── Slug exclusion ────────────────────────────────────────────────────
        if any(s in cfg.exclude_slugs for s in tagged["ka_slugs"]):
            return {**tagged, "include": False, "loss_weight": 0.0}

        # ── Slug inclusion ────────────────────────────────────────────────────
        slug_match = any(s in cfg.active_slugs for s in tagged["ka_slugs"])
        if not slug_match:
            return {**tagged, "include": False, "loss_weight": 0.0}

        # ── Sub-area filter (if specified) ────────────────────────────────────
        if cfg.active_sub_areas:
            sub_ok = False
            for slug in tagged["ka_slugs"]:
                allowed = cfg.active_sub_areas.get(slug)
                if allowed is None:   # slug has no sub-area restriction
                    sub_ok = True; break
                if any(sa in allowed for sa in tagged["ka_sub_areas"]):
                    sub_ok = True; break
            if not sub_ok:
                return {**tagged, "include": False, "loss_weight": 0.0}

        # ── Loss weight ───────────────────────────────────────────────────────
        weight = 1.0
        for slug in tagged["ka_slugs"]:
            if slug in cfg.loss_weight:
                weight = max(weight, cfg.loss_weight[slug])

        return {**tagged, "include": True, "loss_weight": round(weight, 4)}

    # ── Batch corpus gating ───────────────────────────────────────────────────

    def gate_corpus(self, records: list[dict]) -> tuple[list[dict], dict]:
        """
        Filter and tag an entire corpus.
        Returns (kept_records, summary_stats).
        """
        kept, dropped = [], []
        for r in records:
            result = self.evaluate(r)
            if result["include"]:
                kept.append(result)
            else:
                dropped.append(result)

        from collections import Counter
        slug_counts = Counter(s for r in kept for s in r["ka_slugs"])
        step_counts = Counter(r["curriculum_step"] for r in kept)
        summary = {
            "gate":         self.config.name,
            "total_in":     len(records),
            "kept":         len(kept),
            "dropped":      len(dropped),
            "keep_rate":    round(len(kept)/max(len(records),1), 3),
            "slug_counts":  dict(slug_counts),
            "step_counts":  {str(k): v for k, v in sorted(step_counts.items())},
        }
        return kept, summary

    # ── Curriculum sort ───────────────────────────────────────────────────────

    @staticmethod
    def sort_curriculum(records: list[dict], anneal_epoch: int = 0,
                        total_epochs: int = 5) -> list[dict]:
        """
        Sort records by curriculum_step (ascending) in early epochs,
        graduating to fully random by total_epochs.

        anneal_epoch=0  → fully ordered (epoch 1 of training)
        anneal_epoch=N  → fully shuffled (epoch N of training)
        """
        import random
        if anneal_epoch >= total_epochs - 1:
            random.shuffle(records)
            return records

        # Fraction of records to inject randomly (grows each epoch)
        chaos_frac = anneal_epoch / max(total_epochs - 1, 1)
        n_chaos = int(len(records) * chaos_frac)

        ordered  = sorted(records, key=lambda r: r.get("curriculum_step", 2))
        chaos    = random.sample(ordered, n_chaos)
        backbone = [r for r in ordered if r not in chaos]

        # Interleave chaos records at random positions
        result = list(backbone)
        for r in chaos:
            pos = random.randint(0, len(result))
            result.insert(pos, r)
        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gate and tag a training corpus using knowledge_areas.csv")
    ap.add_argument("--data",    required=True, help="Input JSONL (glob ok)")
    ap.add_argument("--gate",    required=True, help="Path to gate_config.yaml")
    ap.add_argument("--out",     help="Output JSONL path (default: <data>_gated.jsonl)")
    ap.add_argument("--csv",     default="math_lab/knowledge_areas.csv")
    ap.add_argument("--dry-run", action="store_true", help="Print stats, don't write")
    args = ap.parse_args()

    import glob
    paths = sorted(glob.glob(args.data))
    if not paths:
        print(f"No files matched: {args.data}"); sys.exit(1)
    data_path = Path(paths[-1])

    records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} records from {data_path.name}")

    gate = KnowledgeGate(args.csv)
    gate.load_config(args.gate)
    print(f"Gate: '{gate.config.name}'  active_slugs: {gate.config.active_slugs}")

    kept, summary = gate.gate_corpus(records)
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return

    out = Path(args.out) if args.out else data_path.with_suffix("").with_suffix(".gated.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in kept), encoding="utf-8")
    print(f"\nWrote {len(kept)} records → {out}")

    # Save summary
    (out.parent / f"{out.stem}_gate_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
