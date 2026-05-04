# QuiPu-KV

_A sovereign, file-based memory and coordination substrate for agentic work._

Author: Sean Layman / Layman Innovations LLC
Status: Pre-1.0. This document supersedes prior architecture notes dated 2026-03-31 through 2026-04-06.

---

## What QuiPu-KV is

QuiPu-KV is a **substrate** — a small, stable layer that other systems build on. It does two things that most AI infrastructure does not do together:

1. **Persists structured memory across sessions, models, and platforms.** A memory artifact written from Claude can be read from Gemini, GPT, or an open-source local model. The file is the interoperability guarantee, not any specific application that reads it.

2. **Coordinates work across participants.** Agents, humans, orchestrators, and rule engines all read from and write to the same log. Decisions requested by one participant can be resolved by any other qualified participant. The substrate mediates; no participant owns the relationship.

The name comes from _quipu_ — the Andean knotted-cord recording system — because both encode structured information in a portable physical medium that doesn't depend on any specific reader to remain meaningful.

## What QuiPu-KV is not

- Not a vector database. No embeddings are required for its core function.
- Not a chat log. The event stream is typed and structured, not free-form messages.
- Not a cloud service. Local-first by design. Cloud sync is optional and user-controlled.
- Not tied to any model or vendor. Any LLM, any framework, any language runtime can be a participant.
- Not a replacement for the agent's internal memory. It is the layer where internal state crosses participant boundaries.

## The shape of the substrate

QuiPu-KV is organized as a stack of layers, each one a natural extension of the one below. You can use any layer without requiring the layers above it. This is deliberate: the lower layers ship in weeks; the higher layers earn their complexity by being pulled from below.

### Layer 0 — Wire format

An append-only log of typed events serialized as one JSON object per line (JSONL). Every event has an id, a timestamp, an author (the participant that produced it), a session identifier, and a typed payload. Payload types in the current schema: `decision_pending`, `decision_resolved`, `progress_update`, `artifact_produced`, `anomaly`, `annotation`.

This is the source of truth. Every other layer is a derived view that can be rebuilt from the log alone.

### Layer 1 — Vocabulary registry

A lightweight lookup table, one row per term. Tracks occurrence counts, inquiry counts, class (stopword, suffix, content word, domain term, named entity), and whether a term has graduated to its own dedicated partition.

The registry is the control plane. All promotion, archival, and indexing decisions read from it. Batch analysis jobs update it. No complex orchestration is needed — one table with a counter and a boolean flag drives the whole system.

### Layer 2 — Resolution-partitioned parquet

Content is stored at multiple granularities simultaneously: word, sentence, concept, paragraph, chapter, page. Each resolution is a separate parquet partition. A query at any resolution returns results without loading the others. Finer detail is reachable through parent/child pointers when needed.

This enables _telescoping retrieval_: the model reads at the coarsest resolution sufficient for the current task and drills down only if needed. The multi-resolution layout also surfaces association and meaning as a side effect of Markov transitions at each level.

### Layer 3 — Typed nodes and provenance

Every concept, speaker, and agent becomes a first-class node with a biography: where it first appeared, who produced it, which other nodes it co-occurs with, how its associations have drifted over time. Recurrences append edges to an existing node rather than creating duplicates.

This is where the substrate becomes a _cognitive actor model_ — the memory tracks not just content but the provenance and evolution of every idea.

### Layer 4 — Durability and sovereignty

Write path is append-only. Every write is a ledger entry — immutable, crash-safe, auditable. Optional sync to user-owned cloud drives (Box, Google Drive, OneDrive) provides off-site replication without ceding control to a vendor.

### Layer 5 — Progressive materialization

Batch analysis runs periodically, reads the vocabulary registry, and promotes high-demand terms to dedicated parquet partitions. Cold terms migrate to archived partitions (still queryable, just slower). The schema evolves with actual usage — the storage structure becomes a continuous hypothesis about how knowledge is organized, updated from evidence rather than designed in advance.

## Key architectural commitments

**The format is the product.** Implementations are commodity. What matters is that a `.quipu` file written today remains readable in five years by any participant that speaks the format. This is the same philosophy that made Git, JSON, and Markdown durable.

**Code is dumb. Data is smart.** The complexity lives in the data model (vocabulary registry, typed nodes, resolution partitions). Application code is minimal routing over that data. This is the inversion that makes the system extensible without invasive refactors.

**Read-optimized denormalization.** Storage is cheap, retrieval is expensive. Events fan out to multiple partitions at write time (by resolution, domain, project, producer) so each read hits exactly one file. This is the same design pattern that makes inverted indexes, columnar stores, and CPU caches fast.

**Wire files are always the source of truth.** Parquet partitions, vocabulary registries, and derived indexes are all rebuildable from the JSONL log. If any derived layer is lost or corrupted, nothing essential is lost. This is the architectural invariant that makes all the redundancy safe.

**No single interface is canonical.** The dashboard is one surface over the substrate. The CLI is another. An HTTP API, an MCP tool, a library binding — all are peers. Each participant uses the interface that matches its native affordances. The substrate coordinates without dictating interaction.

## What makes this differentiable

Most memory systems for AI agents choose one of three approaches: vector similarity, explicit knowledge graphs, or session-scoped chat logs. Each has known limitations. Vector similarity is opaque and lossy. Knowledge graphs are expensive to build and maintain. Chat logs have no structure.

QuiPu-KV's multi-resolution Markov stack gives you _structural relationship discovery as a side effect of storage_. You don't build the knowledge graph; the transition probabilities are the knowledge graph, at every resolution simultaneously. Association, meaning, and relationship surface passively as memory accumulates.

The substrate also gives you something none of the alternatives give you cleanly: **symmetric participation**. A human can play any role an agent can play. An agent can play any role a human can play. Decisions route to whichever participant is eligible and available. This is labor-market architecture for cognitive work — and it's the precondition for any serious deployment of multi-agent systems in regulated domains, where auditability of every decision is non-negotiable.

## Near-term focus

The immediate near-term work is the **QuiPu-KV Dashboard**, a Tauri-based HUD that renders a compressed projection of the event stream for a human participant. The dashboard is both a standalone product (solving the real problem of losing overnight agent cycles to unanswered decisions) and the forcing function that validates the substrate architecture through actual use.

The dashboard development generates the first real corpus that the substrate operates on. This is deliberate — the development process of the application is the first test payload for the substrate, which means the substrate has to be correct about the kind of work that designing it involves. Self-referential in a way that's actually useful rather than cute.

## Relationship to other work

QuiPu-KV sits alongside and under several adjacent efforts in the Layman Innovations stack:

- **Markov_t / MDO.** The .5-boundary transition mathematics provides the principled basis for resolution selection across the Markov stack. QuiPu-KV is the storage and retrieval side; Markov_t is the representation mathematics.

- **KFW (Knowledge Framework Wiki).** Provides seed vocabulary for the domain-aware token classifier. Over time, the substrate's observed transitions update what the KFW treats as canonical.

- **CodeStrata / Technical Depth.** Teaches humans to recognize recombinatorial patterns across software history. The substrate operationalizes the same cognitive move as infrastructure.

- **forge_shim_docs.** A 69K-record academic paper metadata corpus that provides a realistic, large-scale bootstrap dataset for validating the multi-resolution Markov stack.

- **Requirements Pass methodology.** The process discipline for decomposing tasks. The substrate records Requirements Pass outputs as first-class events, making them queryable and composable over time.

These projects are not independent. They are different expressions of a single underlying thesis: _meaningful work happens through recombinatorial pattern recognition across layered structure, and most of the friction in knowledge work comes from treating each instance as new when it is actually a recurrence._

QuiPu-KV is the substrate where that thesis becomes infrastructure.
