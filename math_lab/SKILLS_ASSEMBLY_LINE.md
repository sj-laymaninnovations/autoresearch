# Assembly-line skill decomposition for math model training

A research program to systematically decompose math capability into **atomic
dependency skills**, train a fixed-config model on each in isolation, classify
the result (memorized / generalized / failed), and then study how (a)
compositional curriculum and (b) weight merging produce combined capabilities.

---

## Core principles

1. **Atomic skills, not coarse subjects.** Break capabilities into the
   smallest learnable unit with a defined dependency graph.
2. **Same architecture across skills.** Use one canonical config (e.g.
   6×8×256) so results are directly comparable. Variance comes from data,
   schedule, and optional inits — not from architecture.
3. **Mandatory novel-pair holdout.** Every skill's val set has operand pairs
   never seen in train (and commutative twins also excluded for symmetric ops).
   Without this, "success" is uninterpretable.
4. **Result classification is non-optional:**
   - **GENERALIZED** — passes ≥80% on novel-pair holdout
   - **PARTIAL** — passes 30-80% on novel-pair holdout
   - **MEMORIZED** — passes ≥80% on training-distribution but ≤30% novel
   - **FAILED** — passes ≤30% on either
5. **Persistent skill registry.** Every trained skill model is tagged with
   its classification, confusion matrix, and weight checkpoint pointer.
   That registry is the substrate for compositional and merging experiments.

---

## Phase 1: atomic-skill taxonomy (arithmetic)

Each row is one training run. Same arch, same compute budget per run.

### Format/representation skills (Tier 0)
| skill | task | input/output |
|---|---|---|
| digit_echo | echo a digit | "5" → "5" |
| digit_pair | reproduce two digits in order | "47" → "47" |
| digit_sort | sort two digits ascending | "73" → "37" |
| digit_compare | which is larger | "7,3" → "7" |

### Single-digit operations (Tier 1)
| skill | task | depends on |
|---|---|---|
| add_1d_no_carry | a+b, a+b<10 | digit_echo |
| add_1d_with_carry | a+b, a+b≥10 | digit_echo, place-value |
| sub_1d_no_borrow | a-b, a≥b, both 1-digit | digit_echo |
| mul_1d | a*b, both 1-digit | (atomic — table) |
| div_1d | a/b clean, b≥2 | mul_1d (inverse) |

### Two-digit operations (Tier 2)
| skill | task | depends on |
|---|---|---|
| add_2d_no_carry | column-wise no carry | add_1d_no_carry, place-value |
| add_2d_with_carry | column-wise with carry | add_1d_with_carry |
| sub_2d_no_borrow | column-wise no borrow | sub_1d_no_borrow |
| sub_2d_with_borrow | column-wise with borrow | sub_1d_no_borrow + borrow logic |
| mul_2d_by_1d | a*b, a 2-digit b 1-digit | mul_1d, add_2d_with_carry |
| mul_2d_by_2d | a*b, both 2-digit | mul_2d_by_1d, add (multi-step) |
| div_2d | clean only | mul_2d_by_1d (inverse) |

### Three-digit and beyond (Tier 3+)
Same pattern but extends operand width. Each higher tier formally depends on
the same-op lower tier + place-value generalization.

### Algebra skills (Tier 4+, after arithmetic stable)
- combine_like_terms (depends on: add)
- distribute_positive (depends on: mul_1d, add)
- distribute_negative (depends on: distribute_positive, sub)
- isolate_var_one_step (depends on: sub or div, depending on op)
- solve_x_plus_a (depends on: isolate_var_one_step)
- solve_ax (depends on: div)
- solve_ax_plus_b (depends on: sub then div)
- ...

---

## Phase 2: per-skill training protocol

For each skill in the taxonomy:

1. **Data generation** — full unique pair/triple/etc. enumeration; hold out
   50% as novel-pair val (commutative twins excluded for symmetric ops).
2. **Train fixed-config model from scratch** — no warm-start. Three runs at
   different seeds. Constant-LR regime (lr=1e-3, weight_decay=1.0,
   dropout=0). Train up to 10000 epochs OR until train EM > 0.99 AND val EM
   plateaus for 1000 epochs.
3. **Classify result** by held-out EM:
   - GENERALIZED if val EM ≥ 80%
   - PARTIAL if 30-80%
   - MEMORIZED if train ≥ 80% but val < 30%
   - FAILED otherwise
4. **Record:** classification, peak val EM, peak epoch, trajectory, sample
   correct/incorrect predictions, checkpoint path. One row per (skill, seed)
   in `math_lab/results/skill_registry.tsv`.

---

## Phase 3: compositional curriculum (after Phase 2 has many GENERALIZED)

For each composite skill (e.g. mul_2d_by_2d):
1. **Test parent skills are GENERALIZED** in the registry.
2. **Train from scratch** on composite-skill data.
3. **Train from warm-start** of one parent's checkpoint.
4. **Train from warm-start** of a multi-parent merged checkpoint (Phase 4).
5. Compare peak val EM and convergence speed across the four conditions.
   Result-class for each.

This empirically maps **which dependencies actually help** vs which the model
can shortcut. The hypothesis is: well-formed dependency parents accelerate
the composite; ill-formed ones don't.

---

## Phase 4: weight merging research

Two-skill merge experiment design:
1. Pick two GENERALIZED single-skill checkpoints (e.g. add_2d, mul_1d).
2. Merge weights using each technique:
   - **Linear average** (50/50 simple)
   - **SLERP** (spherical interpolation, preserves norm)
   - **TIES** (Trim, Elect Sign, & Disjoint Merge — drops conflicting params)
   - **DARE** (Drop and Rescale — sparsify deltas)
   - **Task Arithmetic** (W_merged = W_base + α·(W_skill1 - W_base) + β·(W_skill2 - W_base))
3. Eval merged model on each parent's val set + on the *composition* of the
   two skills (e.g. mul_2d_by_1d, which needs both add_2d and mul_1d).
4. Classify each merged model. Compare to the from-scratch + warm-start
   baselines from Phase 3.

The research question: **is there a reliable merging procedure that
preserves both source skills AND enables the composition?** If so, that's
a foundation for parameter-efficient skill assembly. If no merging
preserves both, it tells us merging is task-conflicting at this scale and
we need parameter isolation (e.g. mixture-of-experts, adapter modules).

---

## Phase 5: stress-test the assembly line

Once Phases 2-4 produce a populated skill registry:

1. Pick a high-tier composite skill (e.g. solve_ax_plus_b — depends on
   distribute_positive, combine_like_terms, sub, div).
2. Try multiple assembly paths:
   - From-scratch baseline
   - Warm-start from any single parent
   - Warm-start from merged (parent1 + parent2)
   - Warm-start from merged-of-merged (full ancestry merge)
   - Curriculum from leaf-up vs root-down
3. Map: which assembly path gives the best peak val EM at the same compute
   budget?

This tells us **whether the assembly line works as a research methodology**
or whether emergent capabilities only happen at scale, ignoring the
dependency graph.

---

## Infrastructure needed

- [ ] Skill data generator (`math_lab/datagen/skill_data.py`) — takes a skill
      name from the taxonomy, generates exhaustive train/val with novel-pair
      holdout, returns JSONL.
- [ ] Skill runner (`math_lab/skill_runner.py`) — wraps autoresearch_canonical
      with a fixed config, classifies the result, appends to registry.
- [ ] Registry format (`results/skill_registry.tsv`) — columns: skill, seed,
      arch, classification, peak_val_em, peak_epoch, ckpt_path, train_em_at_peak,
      notes.
- [ ] Merge tools (`math_lab/merge.py`) — load N checkpoints, apply each
      technique, save merged checkpoint, eval against multiple skills.
- [ ] Result viewer — chart of skill registry tagged by classification +
      ancestry graph showing which skills were composed from which.

---

## What this asks empirically

1. **Is there a "magic config" that absorbs single skills below the 5M
   ceiling?** Phase 2 directly tests this — if even tiny configs (2×2×32)
   grok atomic skills like add_1d_no_carry over 10000 epochs, that's a
   floor we can use.

2. **Is the dependency graph real or imagined?** Phase 3 tests if learning
   parent skills actually accelerates child skills. If from-scratch beats
   warm-start consistently, the dependency graph is anthropomorphic — the
   model doesn't learn skills the way humans do.

3. **Can skill weights be merged usefully at this scale?** Phase 4 tests
   if any of the published merging techniques produce reliable composite
   skills. This is the most ambitious hypothesis — published merging works
   on big models, may not transfer to 5M.

4. **What's the cheapest assembly path to high-tier skills?** Phase 5 maps
   the cost-vs-quality landscape across assembly strategies. Might find a
   curriculum order or merging recipe that consistently wins.

---

## Order of operations

The arithmetic CoT result (28.3%) is a Phase 2 / Phase 3 mixture. We've
proven the recipe exists for arithmetic at our scale. Proceeding:

1. **Finish current arithmetic line** (CoT+blend running, then evaluate
   broad tier 1-10 quiz).
2. **Build infrastructure** (taxonomy data generator, skill runner, registry).
3. **Phase 2 minimum viable** — train 6×8×256 from scratch on:
   `add_1d_no_carry`, `add_1d_with_carry`, `mul_1d`, `sub_1d_no_borrow`,
   `div_1d`. Five atomic skills, three seeds each, classify all 15 results.
4. **Phase 3 minimum viable** — test if `add_1d_with_carry` warm-start helps
   `add_2d_with_carry` vs from-scratch.
5. **Phase 4 minimum viable** — try linear average + task arithmetic on
   two generalized skill checkpoints. See if simple merging preserves both.

Each phase informs whether the next is worth the investment.

---

## Why this matters for the project

The PDP-11 vision was always: *what can a small model do with the right
recipe?* The skill assembly line is a systematic way to discover that recipe
— not by guessing at hyperparameters but by mapping the actual
learnability landscape. A populated skill registry would be a **research
artifact in itself** — a calibration of what 5M-scale char-level transformers
can and cannot absorb, with data on which dependency relationships transfer.

If the merging experiments in Phase 4 produce reliable composition, that's
a route to an assembled multi-skill model at small scale — a literal answer
to "can the PDP-11 vision work without the 50M relax?" If merging fails but
curriculum succeeds, the answer is "yes via warm-start chains, no via
parameter merging." Either is useful.
