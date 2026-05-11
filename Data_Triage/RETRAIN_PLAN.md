# EraGPT retrain plan — 2026-05-11

Plan to get from the current "all four checkpoints produce gibberish" state to
one Config A (7M, ~14M params) EraGPT model that actually generates coherent
text on held-out questions. Once that's working, scale to Config B (25M) and
re-attempt BitNet.

**Author's honesty box.** Today's four checkpoints (`prehistoric-a-era`,
`mixed-b-era`, `egypt-a-era`, `bitnet-a-era`) are proofs-of-pipeline, not
useful models. The training infrastructure is sound (we patched the multi-epoch
data loader bug, the val-empty guard, the `--max-steps` override); what was
missing was data volume + diversity + a competent teacher. This plan fixes
those three.

---

## Diagnosis of what's broken

| Checkpoint | Steps trained | Real failure mode |
|---|---|---|
| `prehistoric-a-era` | 2,135 (Config A budget) | only 5 raw FineWeb-Edu shards — corpus too thin for 140M-token Chinchilla budget |
| `mixed-b-era` | 15,257 | Config B is 25M params; needs ~1B tokens. Same thin corpus, just looped many times. |
| `egypt-a-era` | 9,999 | **catastrophic overfit**: only 802 Q&A pairs, every batch was a repeat. Loss 0.0009 = memorized all 802 surface forms. Output collapses to "Question:... Answer:..." n-grams. |
| `bitnet-a-era` | 9,999 | BitNet 1.58 weights need ~4× the tokens of dense models per published recipe; we did ~600M tokens. Far from saturation. |

Root cause is the same in three of four cases: **not enough diverse data**.

---

## What we already have (no new tools needed)

**Patched and working** (on `inference/eragpt-serve` branch + the mini's
`feature/cohort-tooling`):

- `pipeline/teacher_qa.py` — calls any OpenAI-compatible chat endpoint, generates batched Q&A, JSONL output, resumable
- `pipeline/qa_to_parquet.py` — converts JSONL to the `synthetic_*.parquet` schema `data_loader.py` consumes
- `pipeline/assemble.py` — filter by `corrected_era` or `earliest_historical_year` + foundation padding from `sources_mixed`
- `training/train.py` — `--max-steps`, `--warmup-steps`, `--resume`; multi-epoch corpus support; val-empty guard
- `training/train_bitnet.py` — same patches
- LM Studio fleet on mini: **Qwen 3.6-35B-A3B** (MoE, ~3B active), **Gemma 4-31B**, Nemotron-4B, Phi-4-mini-reasoning, BitNet 2B, ~170 others

**Foundation pool**: `sources_mixed/fineweb-edu` on Windows (1.3 GB, 17 shards,
mixed paleo+ancient+general FineWeb). Already enriched with `domain_slug`,
`corrected_era`, `earliest_historical_year` — usable directly.

---

## Goal

One Config A (7M) checkpoint that, on a 100-question held-out set across
multiple domains, hits **ROUGE-L ≥ 0.4** vs the teacher's recorded answers.
That's the floor for "not gibberish." If we hit it, escalate to Config B and
BitNet retraining. If we don't, the bottleneck is either teacher quality or
fundamental 5–14M capacity for fact-heavy domains (the Thrust-1/2/3 decision
in `RESEARCH_NOTES.md` becomes relevant).

---

## Stage 1 — Generate diverse Q&A with a real teacher

**Teacher choice.** Qwen 3.6-35B-A3B is the best fit:
- MoE with ~3B active params — should run at ~30–50 tok/s on M4 Pro 48GB
- Quality far above Nemotron 4B (the 75% factual-accuracy ceiling we hit)
- Already in LM Studio on the mini, federated to Windows for offload

**Fallback** if Qwen 3.6 is slow on the mini: Gemma 4-31B at 4-bit fits the
same memory budget; quality roughly comparable for factual Q&A. Phi-4 14B for
fast iteration if both are too slow.

**Domains.** Pick 8 fact-heavy knowledge_areas entries spanning eras and
disciplines (not just Egypt). Each is a separate `--domain` invocation of
`teacher_qa.py`. Suggested:

| Domain slug | Era | Notes |
|---|---|---|
| `ancient-egypt` | -3000..-30 BCE | reuse the existing seed list |
| `ancient-greece` | -800..-100 BCE | dense recorded history |
| `roman-empire` | -50..+450 CE | dense recorded history |
| `medieval-europe` | 500..1500 CE | fills the current data gap |
| `early-modern-science` | 1500..1800 CE | Newton/Galileo/Boyle era |
| `industrial-revolution` | 1750..1900 CE | tech inflection |
| `computer-science` | 1940..present | mix of fact + procedural |
| `cellular-biology` | timeless | non-historical, tests domain generalization |

Per domain: **5,000 Q&A pairs**. Total: **40,000**. At ~30 tok/s, each domain
takes 30–60 min; whole batch is **~4–6 hr** unattended. Resumable, so can
restart if the mini sleeps.

**CLI** (run on mini, with LM Studio serving Qwen 3.6-35B-A3B):
```bash
# Load teacher once:
~/.lmstudio/bin/lms load qwen/qwen3.6-35b-a3b --gpu max

# Per domain (extend pipeline/teacher_qa.py's SEEDS dict for the new domains first):
cd ~/Documents/autoresearch/Data_Triage
for d in ancient-egypt ancient-greece roman-empire medieval-europe \
         early-modern-science industrial-revolution computer-science cellular-biology; do
    .venv/bin/python pipeline/teacher_qa.py \
        --domain "$d" \
        --teacher-url http://127.0.0.1:1234/v1/chat/completions \
        --teacher-model qwen/qwen3.6-35b-a3b \
        --n 5000 --batch-size 10 --max-tokens 2000 \
        --out qa/${d}_v1.jsonl
done
```

**Quality spot-check.** Before training, eyeball ~30 random Q&A across domains.
If accuracy looks <80%, switch teacher to Gemma 4-31B and regenerate.
**Investment is 4 hr; bad teacher = bad student no matter what we do next.**

---

## Stage 2 — Convert to parquets + assemble foundation-padded corpus

**Convert each domain's JSONL to parquet** (matches `synthetic_*.parquet`
schema):
```bash
for d in ancient-egypt ancient-greece roman-empire medieval-europe \
         early-modern-science industrial-revolution computer-science cellular-biology; do
    .venv/bin/python pipeline/qa_to_parquet.py \
        --jsonl qa/${d}_v1.jsonl \
        --era-label "$d" \
        --domain-slug "$d" \
        --domain-name "${d^}" \
        --age-epoch "Various" \
        --teacher-id "qwen-3.6-35b-a3b" \
        --max-per-shard 1000 \
        --out sources_${d}/fineweb-edu
done
```

Each domain becomes 5 shards × 1,000 rows. With 8 domains, the loader gets
40 train shards + foundation = plenty of variety per pass.

**Assemble training corpus** (combine all 8 + foundation padding from
`sources_mixed`):
```bash
.venv/bin/python pipeline/assemble.py \
    --sources sources_ancient-egypt/fineweb-edu,sources_ancient-greece/fineweb-edu,\
sources_roman-empire/fineweb-edu,sources_medieval-europe/fineweb-edu,\
sources_early-modern-science/fineweb-edu,sources_industrial-revolution/fineweb-edu,\
sources_computer-science/fineweb-edu,sources_cellular-biology/fineweb-edu \
    --foundation 100000 \
    --foundation-source ../sources_mixed/fineweb-edu \
    --out sources_eragpt_v1/fineweb-edu \
    --max-per-shard 5000
```

Result: ~140K records (40K synthetic + 100K foundation), 28 shards, ~70 MB.
One pass through the loader ≈ 12K–15K training steps at batch=32, seq_len=2048.

**Hold-out for eval.** Reserve last 100 of each domain's JSONL (800 total) by
*not* converting them to parquet. The loader never sees these → real eval signal.

---

## Stage 3 — Train Config A with a real schedule

**Schedule rationale.** Last time we did `--max-steps 10000 --warmup-steps 100`
on 802 records. Disaster because (a) every step revisited the same 802, (b)
warmup of 1% of total left no time for cosine decay. This time:

- `--max-steps 15000` — gives full corpus a few passes + tail for fine-tuning
- `--warmup-steps 1500` (10% of total) — proper warmup, lets LR reach peak
- `--save-every 1000` — five checkpoints to choose from for eval
- Config A defaults: lr=3e-4, weight_decay=0.1, bf16 autocast, batch=32, seq=2048

**Launcher.** Use the existing `training/train_run.bat` parametric launcher:
```bash
# On Windows (3080 = GPU 0):
train_run.bat A sources_eragpt_v1\fineweb-edu eragpt_v1_a 0
```
Or on mini directly:
```bash
cd ~/Documents/autoresearch/Data_Triage/training
.venv/bin/python train.py \
    --config A \
    --data-dir ../sources_eragpt_v1/fineweb-edu \
    --out-dir checkpoints/eragpt_v1_a \
    --max-steps 15000 --warmup-steps 1500 \
    --device mps
```

**Expected wall time:** ~3–4 hr on 3080, ~4–6 hr on mini MPS.

---

## Stage 4 — Eval against the held-out 800

Use the existing `pipeline/eval_student_vs_teacher.py`, one invocation per
domain held-out file. Aggregate to a per-domain ROUGE-L table.

```bash
for d in ancient-egypt ancient-greece roman-empire medieval-europe \
         early-modern-science industrial-revolution computer-science cellular-biology; do
    .venv/bin/python pipeline/eval_student_vs_teacher.py \
        --config A \
        --ckpt training/checkpoints/eragpt_v1_a/ckpt_0015000.pt \
        --qa-jsonl qa/${d}_v1.jsonl \
        --n-held-out 100 \
        --max-new 200 --temperature 0.0 \
        --out qa/${d}_eval.tsv
done
```

**Success criteria** (decision rule for what comes next):

| Mean ROUGE-L | Interpretation | Next |
|---|---|---|
| ≥ 0.5 | recipe works | escalate to Config B (Stage 5) |
| 0.3 – 0.5 | partial success | try once more with 10× more Q&A per domain |
| 0.1 – 0.3 | weak — recipe needs a structural change | revisit Thrust 1/2/3 (research notes) before more compute |
| < 0.1 | broken — same failure mode as today | stop, debug the corpus / teacher / loader before training again |

Also do a **manual qualitative grade** on 30 random samples (1–5 scale: factual
accuracy, coherence, in-domain). ROUGE alone isn't enough.

---

## Stage 5 — Conditional: Config B (25M) on the same corpus

**Only run this if Config A hits the ≥0.3 floor.** Otherwise scaling the
arch just amplifies a broken recipe.

Same corpus, same eval set. Config B is 10L/10H/640d/25M params.
Chinchilla optimum for 25M ≈ 500M tokens ≈ 8K steps at batch=32, seq=2048.

```bash
train_run.bat B sources_eragpt_v1\fineweb-edu eragpt_v1_b 1   # 3090
```

Wall time: **~6–8 hr on 3090**. Definitely an overnight run.

---

## Stage 6 — Conditional: BitNet retrain

**Only after dense Config A works.** BitNet's added complexity (ternary
weight quant, more sensitive to LR schedule, much more data-hungry) means
debugging a failed BitNet run when the dense version also fails is wasteful.

If Stage 4 passes:
- Resume from current `bitnet_a/ckpt_0009999.pt`
- Same corpus as eragpt_v1
- `--max-steps 50000` (significantly longer than dense; BitNet's data appetite)
- Same eval methodology, compare BitNet curve vs dense Config A curve at
  matched training-step budgets

---

## What to commit + when

After each stage that succeeds:

1. **Stage 1 output**: commit `qa/*.jsonl` to git? **No — gitignore them.**
   They're large (~50 MB total) and easy to regenerate. Save to USB instead
   as a one-time artifact.
2. **Stage 2 output**: commit `pipeline/teacher_qa.py` updates if SEEDS dict
   grows; gitignore the `sources_*` dirs.
3. **Stage 3 output**: commit `checkpoints/eragpt_v1_a/run_config.json` (small,
   reproducible). Gitignore the `.pt` files; copy them to USB.
4. **Stage 4 output**: commit the per-domain eval TSVs **and** the manual
   qualitative notes as a single `RESULTS_v1.md` summary. This is the
   research artifact.

Branch: stay on `inference/eragpt-serve` for now since this is the same
research thread (serve UI + training that feeds it). Consider rebasing to
`research/eragpt-v1` if it ends up being a longer thread.

---

## Risks / known unknowns

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen 3.6-35B-A3B slower than expected on M4 Pro (MoE overhead) | medium | Pre-bench with 50 Q&A; if <20 tok/s, swap to Gemma 4-31B |
| Qwen still wrong on niche history domains | low–med | Accept ~85% accuracy; cross-verify on 50 spot-checks before training |
| 8 domains × 5K may not be enough variety | medium | Stage 4 will tell us; expand to 16 domains if eval is weak |
| Foundation rows dominate the loss (40K synthetic gets buried in 100K foundation) | medium | Tune the ratio; first cut is 28% domain / 71% foundation, can flip to 50/50 if needed |
| LR schedule still doesn't fit corpus size | low | Math is straightforward; 15K steps + 10% warmup is conservative |
| The mini sleeps mid-overnight training | low | We've established the caffeinate + nohup pattern; same flow as last time |
| BitNet retrain time exceeds available compute | high | Acceptable — punt to Thrust 1/2/3 decision if so |

---

## Estimated total time

| Stage | Wall time | What's running |
|---|---|---|
| 1: Teacher Q&A across 8 domains | 4–6 hr | mini (LM Studio Qwen) |
| 2: Parquet conversion + assemble | <10 min | mini (pyarrow) |
| 3: Config A training | 3–4 hr | Windows 3080 OR mini MPS |
| 4: Eval + qualitative grade | 1–2 hr | mini or this MacBook |
| **MVP cycle** | **~10–13 hr** | mostly overnight |
| 5: Config B (conditional) | 6–8 hr | Windows 3090 |
| 6: BitNet retrain (conditional) | 8–24 hr | mini MPS |

Stage 1 + 2 + 3 fits into one overnight on the mini if Q&A starts when you
get home tonight (say 19:00); Q&A finishes ~01:00, training finishes ~05:00,
ready for eval by morning. Tight but feasible.

---

## What this plan deliberately does *not* do

- **No inference optimization** (`--dtype bf16`, KV cache port to Data_Triage,
  `--compile`). That work is real but applies to a *working* model. Defer
  until Stage 4 says we have one. Tracked in `inference/eragpt-serve` branch
  comments as a follow-up.
- **No speculative decoding pair config.** Same reason — need a quality
  gradient between lead and draft worth speculating around, which we don't
  have yet.
- **No new chat.html features.** UI is fine for now.
- **No new training-recipe research** (RLHF, DPO, contrastive losses).
  Standard next-token cross-entropy is fine for v1. If Config A fails at
  Stage 4 with mean ROUGE-L < 0.3 across all domains, *then* it's worth
  thinking about why the recipe itself might be wrong.

---

## TL;DR for the bus ride

1. Tonight: kick off Qwen 3.6-35B Q&A across 8 domains on the mini. ~5 hr.
2. Tomorrow AM: assemble + train Config A. ~4 hr.
3. Tomorrow afternoon: eval; if ROUGE-L ≥ 0.3 we have a real model; if not,
   diagnose before doing anything else.
4. Inference optimization waits until we have something worth optimizing.
