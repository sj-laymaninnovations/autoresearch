# Bet A — Corpus Plan
## Meta-Programming Coder / ARM64 first target

**Status:** Draft v1
**Date:** 2026-05-15
**Owner:** thread-meta-programming-coder

This document defines *what training material we harvest, from whom, in what
shape, and to what end* for Bet A. Per CLAUDE.md and the agent status file,
this is the corpus-collection PLANNING task that sits between the
kernel-on-code throughput sanity check (deferred pending this plan's outcome)
and the dataset-construction work itself.

The premise we're committing to: **a 5M-parameter model trained on
language-agnostic programming patterns + retrieval scaffolding over
language-specific reference material can produce useful code generation
across any target language at 100+ tok/s on consumer hardware.**

That premise has a load-bearing assumption about *what we train on*: the
model holds idiom recognition + control-flow + abstraction patterns; the
retrieval scaffolding holds the syntax/API specifics. Where the corpus is
split between these two destinations is the single largest design decision
in this plan.

---

## 1. Goals and non-goals

### Goal
Specify a corpus that, after construction, can train a 5M-parameter model
to produce useful ARM64 assembly given:
1. A natural-language problem statement.
2. A retrieval-pulled context block of ARM64 syntax/instructions/API surface.

The model is expected to learn **idioms** ("when you see a tight inner loop
over a strided buffer, prefer `ldp`/`stp` pairs and unroll by 4 to saturate
the pipeline"), **patterns** ("the standard ARM64 prologue saves the FP/LR
on entry"), and **rationale** ("you use `madd` rather than `mul` + `add`
because madd is single-cycle on the load/store unit"). It is not expected
to memorize the entire ISA.

### Non-goals
- We are **not** training a general code-generation model. ARM64 first; other
  languages later by swapping the retrieval corpus.
- We are **not** trying to match GPT-class general-purpose performance.
  We're targeting *correctness on bounded code-shaped inputs at 100+ tok/s on
  consumer GPU*.
- We are **not** including general programming material (textbook intros,
  CS101 content). Only material from people whose expertise was earned on
  real, performance-critical asm.
- We are **not** including auto-generated documentation, only human-written
  rationale.

---

## 2. Architecture split: what feeds the model vs retrieval

CLAUDE.md's "model holds patterns, retrieval holds syntax" maps cleanly onto
two corpus tracks. Every harvested artifact gets tagged with its destination.

### Track M — Model training corpus
**Goal:** Teach the model *idioms, rationale, and decision patterns*. Long-form
prose, Q&A-style explanations, and *annotated* code where the annotation is
load-bearing.

**Shape of entries:**
- **Commentary records.** Blog post / mailing-list message / commit message /
  talk transcript whose substance is "WHY this technique, not that one."
  These become natural-language Q&A pairs through a distillation step
  (see §4).
- **Annotated code records.** Asm snippets *with* embedded commentary —
  e.g. an x264 inner loop where each chunk has a comment explaining the
  reasoning. The annotation is what trains the model; the asm is the
  illustration.
- **Decision narratives.** A debugging/optimization story: "I had to figure
  out why my SIMD path was slower than the C path. Turned out…" These are
  rare and gold.

**What it explicitly is NOT:** raw asm listings without rationale, ISA
reference tables, API surface dumps. Those go to Track R.

### Track R — Retrieval corpus
**Goal:** Be the lookup table the model consults at inference time for
language-specific specifics it doesn't need to memorize.

**Shape of entries:**
- ARM64 ISA reference (Arm ARM, post-processed) — instruction → encoding,
  operands, latency notes.
- ABI / calling-convention quick-reference (AAPCS64).
- Standard asm idioms catalogued by purpose ("how to do a 64×64→128
  multiply", "how to do a saturated add").
- API surfaces for the layered toolchains: NEON intrinsics, SVE intrinsics,
  syscall numbers, ELF format.

**What it explicitly is NOT:** prose discussing tradeoffs. That's Track M.

---

## 3. Source list

Every source below is annotated with: track destination (M, R, or M+R),
harvest cost (cheap / medium / expensive), and an honest credibility/relevance
note. Sources are grouped by epistemic provenance, not by topic.

### 3.1 Video codec hand-tuners — primary commentary source

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 1 | **Jason Garrett-Glaser ("Dark Shikari")** — x264 deblocker, SIMD lead 2007-2014 | M | medium | "Diary of an x264 Developer" blog. Original site dead; Wayback Machine has most posts. Highest signal-to-noise asm-craftsman commentary in existence. |
| 2 | **Loren Merritt** — x264 lead, AVX2 work | M | cheap | Inline asm with comments. Git history of `x264/common/x86/*.asm` + commit messages. |
| 3 | **Henrik Gramner** — x264 asm specialist | M+R | cheap | Author of `x264asm` macros (`x264/common/x86/x86inc.asm`) — the *file is itself the document*. Also ML threads. |
| 4 | **Fiona Glaser** — x264 successor | M | cheap | Recent x264 git history. |
| 5 | **Michael Niedermayer** — FFmpeg founder | M | cheap | ffmpeg-devel ML threads. Strong opinions on hand-asm vs compiler. |
| 6 | **Diego Biurrun** — FFmpeg asm style maintainer | M | cheap | A decade of ML threads enforcing asm style/idiom. Style commentary is gold. |
| 7 | **Ronald Bultje** — FFmpeg ARM / ARM64 codec author | M+R | cheap | `libavcodec/aarch64/*.S` git history. |
| 8 | **Martin Storsjö** — FFmpeg ARM64 + dav1d ARM port | M+R | cheap | Two repos of evidence. |
| 9 | **dav1d project** (collective; Storsjö, Henry de Valence, others) — modern AV1 decoder | M+R | cheap | `dav1d/src/arm/64/*.S` — the modern reference for hand-tuned ARM64 codec asm. |

### 3.2 Game engine performance engineers — long-form rationale

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 10 | **John Carmack** — Doom/Quake era | M | medium | `.plan` files (archive exists). Quake source (now public). Multiple long-form interviews. Famous for fast-inverse-sqrt commentary. |
| 11 | **Mike Acton** — Insomniac Games, Data-Oriented Design | M | medium | GDC + CppCon talk transcripts. Cache-line + asm-adjacent thinking. |
| 12 | **Tim Sweeney** — Unreal Engine | M | medium | HN posts + interviews. Tradeoff thinking at the engine level. |
| 13 | **Christian Gyrling, Travis Cady (Naughty Dog)** — PS3/PS4 engine, fibers, SIMD | M | expensive | GDC talk videos. Transcripts mostly auto-captioned, noisy. |
| 14 | **Andrew Glassner** — Graphics Gems era | M | medium | Books are not freely available; reviews + summaries are. Likely skip unless we license. |

### 3.3 Compiler/asm interface — the meta-level

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 15 | **Agner Fog** — "Optimizing assembly" reference manuals | M+R | cheap | PDFs directly downloadable from agner.org. x86-centric but methodology applies. Authoritative. |
| 16 | **Chris Lattner** — LLVM/Swift; compiler→asm boundary | M | medium | Talks (transcripts exist). "How to think about asm vs compiler" theme. |
| 17 | **Andrei Alexandrescu** — D / Facebook | M | medium | CppCon talks. Less directly asm, but compile-time-to-runtime tradeoffs. |

### 3.4 Linux ARM64 kernel maintainers — production rationale

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 18 | **Catalin Marinas, Will Deacon, Mark Rutland** — Linux ARM64 arch maintainers | M+R | cheap | `linux-arm-kernel` ML archive + kernel commit messages on `arch/arm64/*`. Per-patch rationale is uniquely good here. |

### 3.5 Apple / Accelerate framework — less public

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 19 | Apple Accelerate framework authors | R | expensive | Mostly closed-source. Documentation only. Likely skip for Track M; may pull docs for Track R. |

### 3.6 PDP-11 / RT-11 / DEC era — the philosophical anchor

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 20 | **Dennis Ritchie, Brian Kernighan, Ken Thompson** — PDP-11 era at Bell Labs | M | medium | Long-form essays. Aligns with the program's PDP-11 attention kernel naming/spirit. "The C Programming Language" intro chapters. K&R papers archive. |
| 21 | **DEC's PDP-11 MACRO-11 manuals** | R | cheap | Archived. Direct source for the asm primitives the kernel is named after. |

### 3.7 Demoscene — extreme size constraints

| # | Person / project | Track | Cost | Notes |
|---|---|---|---|---|
| 22 | **scene.org** Amiga/Atari/x86 demoscene archive | M | medium | "How I fit X into 4KB" writeups. Direct relevance: model size constraint mirrors demoscene size constraint. |
| 23 | **pouet.net** comments threads | M | expensive | Lots of noise. Selective harvest only. |

### 3.8 Coverage gaps (acknowledged)

- **No mobile-game asm specialists** — Andre LaMothe is dated; modern mobile asm is mostly behind NDA.
- **No FPGA/RTL practitioners** — different mental model from CPU asm. Out of scope.
- **No GPU shader asm** — different ISA family. If we later target shaders, this gap is large.
- **Wojciech Mula's blog (SIMD)** — should be added. x86-centric but methodology transfers.
- **Daniel Lemire's blog** — performance + algorithms, occasional asm. Add.

---

## 4. Training-data schema

All harvested material lands in one of two on-disk shapes. Both are JSONL.

### 4.1 `commentary_records.jsonl` (Track M, raw)

```json
{
  "id": "darkshikari_2008_03_15_paragraph_4",
  "source_type": "blog",
  "source_person": "Jason Garrett-Glaser",
  "source_artifact": "x264dev.multimedia.cx/diary/2008-03-15-on-h264-deblocking",
  "source_archive": "web.archive.org/web/2010*/...",
  "date": "2008-03-15",
  "era": "x264-development",
  "track": "M",
  "topic_tags": ["deblock", "x86_sse2", "loop_unrolling"],
  "instruction_set": "x86",     // primary ISA discussed; not necessarily ARM64
  "text": "The naive deblocking loop walks each row...",
  "license": "blog-author-copyright",
  "license_posture": "fair_use_training",
  "tokens_est": 412,
  "quality_signal": "expert_primary_source"
}
```

### 4.2 `code_records.jsonl` (Track M annotated, OR Track R lookup)

```json
{
  "id": "x264_x86inc_macro_HADAMARD4_2D",
  "source_type": "code",
  "source_repo": "https://code.videolan.org/videolan/x264",
  "source_path": "common/x86/x86inc.asm",
  "source_commit": "abc123",
  "source_lineno_start": 1834,
  "source_lineno_end": 1872,
  "track": "M",
  "instruction_set": "x86_64",
  "function_name": "HADAMARD4_2D",
  "language": "yasm",
  "code": "%macro HADAMARD4_2D 5\n    HADAMARD4_V %1, %2, %3, %4, %5\n    ...",
  "embedded_comments_count": 6,
  "annotation_quality": "high",   // high = comments explain WHY; low = mechanical labels
  "license": "GPL-2.0-or-later",
  "license_posture": "fair_use_training",
  "tokens_est": 280
}
```

### 4.3 Distilled Q&A — the actual training input

The model is trained on Q&A pairs derived from the two raw stores above.
Distillation happens via the same teacher-distillation pipeline EraGPT
used (`pipeline/teacher_qa.py` adapted), where the "teacher" reformulates
commentary into a Q&A pair that preserves the expert claim. The expert is
the truth source; the LLM is a reformatter only.

```json
{
  "id": "qa_from_darkshikari_2008_03_15_p4",
  "question": "When implementing an in-loop deblocking filter for H.264, why is loop unrolling by 4 specifically preferred over 2 or 8 on x86 SSE2?",
  "answer": "Because the deblock kernel processes 4×4 blocks naturally, and a 4× unroll keeps the working set in 8 XMM registers (Q, R, S, T plus 4 scratch), avoiding spill. An 8× unroll exceeds 16 XMM registers and spills to L1. A 2× unroll under-utilizes the dual-issue port-0/port-5 pairing on SSE2 pipes.\n#### Unroll factor must match the natural block size of the filter and the architectural register file size.",
  "source_record_id": "darkshikari_2008_03_15_paragraph_4",
  "teacher_id": "claude-sonnet-4",   // for honesty: reformulator, not source
  "track": "M"
}
```

The `#### ...` trailer is the same convention as the math GSM8K format —
keeps tooling consistent.

### 4.4 Retrieval payloads — Track R

Different format, since this is consumed at inference not training.

```json
{
  "id": "arm64_isa_ldp_signed_offset",
  "instruction": "LDP",
  "variant": "signed offset",
  "syntax": "LDP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]",
  "encoding": "1010100101...",
  "description": "Load Pair of Registers...",
  "latency_cycles_cortex_a76": 4,
  "source": "Arm ARM v8-A DDI0487 §C6.2.131",
  "license": "Arm reference manual reuse — see Arm public docs license"
}
```

---

## 5. Harvest pipelines

One pipeline per source format. All run on the MacBook (network-bound, not
compute-bound). All write to `bet_a/raw/{source_id}/` for raw fetch and
`bet_a/curated/{commentary,code,retrieval}_records.jsonl` for normalized
output.

### 5.1 Pipeline A — git history harvest (cheapest, do first)

For x264, FFmpeg, dav1d, Linux kernel ARM64:

```
clone repo
git log --no-merges -p -- '<asm-paths>' \
  | parse into (commit_message, hunks_added, hunks_removed)
  | for each hunk: extract asm code + commit msg as commentary
  | dedupe by hunk content hash
  | write to bet_a/raw/{repo}/git_hunks.jsonl
```

Estimated yield per repo:
- x264: ~3K hunks
- FFmpeg: ~15K hunks (multi-codec, multi-arch)
- dav1d: ~2K hunks
- Linux arch/arm64: ~8K hunks

Total ~28K hunks. Most have commit messages; many have inline `;` or `//`
comments inside the hunk. Hunks with substantive commit messages OR
substantive inline comments become commentary records; bare hunks become
code records.

### 5.2 Pipeline B — blog archive (Wayback Machine)

For Dark Shikari, Wojciech Mula, Daniel Lemire:

```
seed with known post URLs (manually curated, ~50 URLs each)
for each: try live, fallback to Wayback Machine
extract body via readability (no nav/sidebar)
chunk into paragraphs
write to bet_a/raw/{blog}/paragraphs.jsonl
```

Estimated yield: Dark Shikari archive ~120 posts × ~12 paragraphs ≈ 1400 records.

### 5.3 Pipeline C — mailing list archives

For ffmpeg-devel, linux-arm-kernel, vlc-devel:

```
fetch mbox archive (most lists offer one)
parse messages
filter to senders in our target person list
chunk into messages (one record per message)
write to bet_a/raw/{list}/messages.jsonl
```

Estimated yield: 10K-50K messages across all three lists, filtered down to
~3K from our target persons.

### 5.4 Pipeline D — GDC / conference talks

Most expensive. Only for the highest-priority talks.

```
manually identify ~10 highest-priority talks (Mike Acton DoD, Carmack
Quakecon, Gyrling fibers, etc.)
fetch YouTube transcripts (yt-dlp --write-auto-subs)
clean caption artifacts (timestamps, repetition)
chunk into segments (one record per ~30s segment)
write to bet_a/raw/{talk_id}/segments.jsonl
```

Estimated yield: 10 talks × ~60 segments ≈ 600 records.

### 5.5 Pipeline E — PDF reference manuals

For Agner Fog, Arm ARM, DEC MACRO-11:

```
download PDF
text extraction via pdftotext (we have it; same one used elsewhere)
chunk by section
write to bet_a/raw/{manual}/sections.jsonl
```

Arm ARM goes mostly to Track R (lookup tables). Agner Fog goes mostly to
Track M (commentary).

### 5.6 Pipeline F — distillation to Q&A

After raw harvest, run the EraGPT-style teacher distillation:

```
input:  commentary_records.jsonl
output: qa_records.jsonl
teacher round-robin: gpt-4o, claude-sonnet-4, grok-3, gemini-2.5-flash
                     (when Abacus credits are restored)
prompt: "Reformulate this expert commentary into a Q&A pair that preserves
         the expert's claim. Cite the source. Add a `#### <one-line summary>`
         line at the end."
```

**Blocker note:** Abacus credits are currently zero. Either Sean tops up,
or we use a different teacher path (local LM Studio, direct OpenAI/Anthropic
key). Without working teacher API, distillation is blocked but raw harvest
can proceed.

---

## 6. Disk organization

```
bet_a/
  corpus_plan.md                       (this file)
  raw/
    x264/
      git_hunks.jsonl
      readme_excerpts.jsonl
    ffmpeg/
      git_hunks.jsonl
    dav1d/
      git_hunks.jsonl
    linux_arm64/
      git_hunks.jsonl
    darkshikari_blog/
      paragraphs.jsonl
    wojciech_mula_blog/
      paragraphs.jsonl
    ffmpeg_devel_ml/
      messages.jsonl
    linux_arm_kernel_ml/
      messages.jsonl
    gdc_talks/
      acton_dod_2014/
        segments.jsonl
      carmack_quakecon_2013/
        segments.jsonl
    agner_fog/
      sections.jsonl
    arm_arm_v8a/
      sections.jsonl
  curated/
    commentary_records.jsonl    # Track M raw
    code_records.jsonl          # Track M+R raw
    retrieval_records.jsonl     # Track R lookup
  distilled/
    qa_records.jsonl            # Q&A pairs, ready for training
  reports/
    harvest_stats.json
    licensing_audit.md
```

---

## 7. Dedupe and curation policy

- **Dedupe on raw text hash before distillation.** A code snippet copied between
  blog post and commit message should not count twice.
- **Drop records under 50 tokens** unless they're explicit instruction-level
  notes from a reference manual (Track R).
- **Tag low-quality annotation as `annotation_quality: "low"`** but don't
  drop — useful as code records, not commentary records.
- **Per-source quotas.** Cap any single source at 20% of the final curated
  corpus. We want the diversity of opinion, not one person's voice trained
  in.
- **Hold out a per-source eval slice.** 5% of each source goes to held-out
  eval (we score later runs against unseen records from each expert's
  archive).

---

## 8. Licensing posture

This is recorded so a future reviewer (or Sean) can audit.

| Source | License | Posture |
|---|---|---|
| x264 / FFmpeg / dav1d / Linux kernel — code | GPL-2.0-or-later | **Fair-use training** (US precedent for LLM training on GPL is broadly accepted). Distributing the trained model is a separate question. |
| Personal blog posts | author copyright | **Fair-use training.** Distributing the model is the same question as for any LLM trained on web text. |
| Mailing list archives | mixed (per author) | **Fair-use training.** |
| GDC / conference talk transcripts | conference + speaker copyright | **Fair-use training.** Transcripts derived from public videos. |
| PDF reference manuals (Agner Fog) | author terms (varies) | Agner Fog: explicitly permits non-commercial training-like use. Re-check. |
| PDF reference manuals (Arm ARM) | Arm public reference terms | Limited reuse; quote sparingly into Track R. |
| DEC MACRO-11 manuals | public domain (DEC is defunct) | Free use. |

Sean's call: are we comfortable training-only on these, with the
"distributing model = same question as any LLM trained on web" answer? If
not, scope the corpus down to GPL/permissive code + public-domain manuals
only.

---

## 9. Milestones

### Day 1 (today)
- This plan, committed.
- Pipeline A feasibility harvest on x264 + dav1d: clone, run a one-liner
  `git log -p -- common/x86 src/arm/64`, count hunks, peek at format.
  **Deliverable:** `bet_a/reports/harvest_feasibility_day1.md` —
  "got X hunks from x264, Y from dav1d, sample 5 hunks pasted here".

### Week 1
- Pipeline A full run on all 4 repos (x264, FFmpeg, dav1d, Linux ARM64).
- Pipeline B for Dark Shikari + Wojciech Mula + Daniel Lemire (~3 blogs).
- Pipeline E for Agner Fog PDFs + Arm ARM v8-A.
- Curated records first cut.
- **Deliverable:** ~30K raw records, ~12K curated, harvest_stats.json.

### Week 2-3
- Pipeline C (mailing lists). This is the biggest single yield.
- Pipeline D (GDC talks) — only the top 5 talks.
- **Deliverable:** ~3K ML messages, ~300 talk segments.

### Week 4
- Resolve Abacus credit situation (or pivot teacher API).
- Pipeline F (distillation) on a 1K-record subsample.
- **Deliverable:** 1K Q&A pairs, manual audit of 50 of them.

### Decision point — end of Week 4
- Sample audit of distilled Q&A.
- Go/no-go on the FULL distillation run.
- If go: full distillation runs over Week 5-6.

### Week 6-7
- Training-side preparation: pick arch config (4×6, 6×8, or 8×8), refresh
  the kernel-on-code throughput sanity check with real corpus tokens (not
  synthetic ARM64 templates), then queue first training run.

This puts FIRST TRAINING RUN at **Week 7**, not Week 1. That is honest.
If Sean wants to compress, the variable to relax is the distillation audit
(Week 4) — pre-audit launches Week 5.

---

## 10. Open questions for Sean — RESOLVED 2026-05-15

1. **Abacus credits**: $20 = ~20K tokens at current Abacus pricing.
   Decision: **spendable when necessary**, not artificially gated.
   Recommended first spend is still a ~50 Q&A audit sample to qualify
   distillation prompt design before larger runs. Top-up or pivot to a
   direct API key is on the table when needed; no pre-commitment.

2. **License posture**: **fair-use training**. Distribution of the
   trained model is the standard "LLM trained on web text" question and
   is not pre-litigated here. License audit document (`bet_a/reports/
   licensing_audit.md`) will be maintained per-source as we harvest.

3. **PDP-11 / DEC era**: **include**. K&R essays, MACRO-11 manuals, and
   the original DEC literature ship as part of Track M (commentary) and
   Track R (manuals) respectively. Spiritually anchored to the program's
   ATTN11 kernel naming.

4. **GDC transcripts**: **top 10**. Carmack Quakecon, Acton DoD,
   Gyrling fibers, plus 7 others to be selected during Week 2-3 harvest.

5. **First trained arch config**: **defer**. Locked decision postponed
   until corpus is curated and we can size embedding tables / context
   length against real data distribution.

6. **Cross-language / training timing**: **determine as we build**.
   No pre-commitment on when a second-language (x86 / RISC-V) sanity
   check fires. Decision lives with the corpus regimen as it matures.

---

---

## 11. What this plan is NOT committing to (yet)

- A specific architectural config for the trained model.
- A specific training schedule or LR/scheduler choices.
- A specific eval suite (HumanEval-asm? Custom?).
- The kernel-on-code throughput sanity check still needs to happen
  before training. This plan does not subsume that gate.

---

*End of corpus plan v1.*
