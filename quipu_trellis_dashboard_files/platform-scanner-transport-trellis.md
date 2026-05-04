# Implementation Spec — Platform · Scanner · Transport · Trellis

_Reference document for the Rust backend of the QuiPu-KV Dashboard, v0.0.2._

Scope: this document specifies the four Rust modules that together implement the platform-backed substrate layer for the dashboard. It does not specify the frontend (covered separately) or the higher QuiPu-KV layers (parquet partitioning, vocabulary registry, batch analysis) which are deferred.

Status: draft. Implemented as scaffolding in `src-tauri/src/`.

This revision (v0.0.2) replaces the prior JSONL-event-log substrate (v0.0.1) with the **Layman Agent Awareness Platform** as the source of truth. The dashboard is now a viewer for the YAML files agents write under a shared `command-hub/` folder. See `docs/quipu-kv.md` for the conceptual rationale.

---

## Module boundaries

Four modules. Each owns one concern.

- `platform` — Layman Agent Platform v1 schema as Rust types
- `scanner` — walks command-hub structure, parses YAML files into a snapshot
- `transport` — watches the command-hub root, emits trellis updates on change
- `trellis` — folds a snapshot into projection tiles for the HUD

The modules communicate via shared types. None of them imports Tauri except `lib.rs`, which is pure wiring.

The substrate is **read-only**. The dashboard never writes to the command-hub. All writes are done by agents (per the platform spec) or by humans editing YAML directly.

---

## The `platform` module

Defines Rust types matching the platform's published v1 schema. Every YAML file the platform produces deserializes into one of these types:

- `AgentStatus` — `agents/{node}/status.yaml`
- `RollupStatus` — `projects/{project}/rollup/status.yaml`
- `CommsFile` containing one or more `Directive` — `comms.yaml` (single or multiple)
- `SecretsFile` containing `SecretRef` entries — `secrets.yaml`

### Schema versioning

Every type carries `schema_version: u32`. The constant `SUPPORTED_SCHEMA_VERSION` declares which version this codebase parses. Files with a different version are surfaced as warnings rather than silently misparsed.

### Stall classification

The `classify_stall` function implements the platform's per-agent stall rule:

- Healthy: `now - last_updated <= expected_cadence_s`
- Over threshold: `expected_cadence_s < now - last_updated <= expected_cadence_s * 3`
- Stalled: `now - last_updated > expected_cadence_s * 3`
- Unknown: `expected_cadence_s` is missing or zero

This is the platform spec's exact rule. No global thresholds.

### Compatibility

- New variant additions require a schema version bump.
- New optional fields can be added without bumping schema version.
- Renaming or removing existing required fields is a breaking change.

---

## The `scanner` module

Walks the command-hub folder structure and produces an in-memory `Snapshot`.

### Folder layout

```
{command_hub_root}/
  projects/{project}/
    rollup/
      status.yaml      → RollupStatus
      status.mermaid   → String
      comms.yaml       → CommsFile (project-wide directives)
      refs.yaml        → (parsed but not yet surfaced)
    agents/{node-id}/
      status.yaml      → AgentStatus
      status.mermaid   → String
      comms.yaml       → CommsFile (agent-scoped directives)
      refs.yaml
      secrets.yaml     → SecretsFile
  standalone-agents/{node-id}/
    (same files as above, no project context)
```

### Scan semantics

- A missing file is not an error; the corresponding `Option` is `None`.
- A malformed file produces a warning entry in `Snapshot::warnings` and that field becomes `None`.
- Schema version mismatches produce a warning but the file is still parsed and accepted (to permit incremental rollouts).
- Subdirectories that are not part of the spec (e.g. `archive/`, `.git/`) are ignored.

### Default root

The default command-hub root is `~/command-hub`. Override with the `QUIPU_COMMAND_HUB` environment variable. This matches the platform spec's "any shared filesystem" deployment model — the user mounts Box Sync, a network share, or any other transport at `~/command-hub` and the dashboard sees it through the same code path.

### Warnings, not errors

Scanner failures never abort the dashboard. A command-hub that doesn't exist yet produces a single warning and an empty snapshot. A folder with a malformed `status.yaml` produces a warning and continues. This matches the operational reality that command-hubs are populated incrementally and may have transient inconsistent states during sync.

### Non-goals

- No git integration. The platform spec mentions git as the version-history layer; the dashboard does not call git directly. Git history is a property of the filesystem we observe, not something we read or write.
- No write path. All writes are out-of-band (agents, the user, or the platform's setup utilities).
- No cross-host federation. The dashboard sees one command-hub root.

---

## The `transport` module

Watches the command-hub root for filesystem changes and emits updated trellises to the Tauri frontend.

### Watcher

Uses `notify` with `RecursiveMode::Recursive` to subscribe to the entire command-hub tree. Runs on a dedicated OS thread.

If the command-hub root does not exist at startup, the watcher logs a warning and parks. The user must restart the dashboard after creating the folder. This is an explicit deferral; future versions can poll for existence.

### Debouncing

Filesystem events arrive in bursts (Box Sync flushes, editor save-on-change, multi-file commits). The transport coalesces events that arrive within `DEBOUNCE_MS` (default 250ms) into a single re-scan. This matches the platform's polling cadence — agents check their comms folder every 15-20 seconds, so 250ms debounce is well below the meaningful change rate.

### Hydration protocol

On startup, the transport synchronously runs one scan, builds a trellis, and emits it as `quipu://trellis/update` immediately. A second event `quipu://trellis/hydrated` is emitted after to signal initial readiness. The frontend uses these to render before subscribing to live updates without race conditions.

### Tauri event names

- `quipu://trellis/update` — payload is the full `Trellis`.
- `quipu://trellis/hydrated` — empty payload, fired once after initial scan.

The URI namespace signals these are substrate events, not application events.

### Why full re-scan rather than incremental updates

Filesystem events from `notify` are fine-grained but not always reliable across platforms (especially with networked filesystems). A full re-scan after debounce is simple, correct, and fast enough for command-hubs of reasonable size (tens of agents, hundreds of files). Optimizing to incremental scans is a future change once we have evidence about real folder sizes.

---

## The `trellis` module

Projection of a `Snapshot` into HUD tiles.

### Output structure

The `Trellis` carries:

- `pending_directives: Vec<DirectiveTile>` — flattened from all `comms.yaml` files, sorted (pending first, then by priority, then by date)
- `agents: Vec<AgentTile>` — one per agent, sorted (stalled first, then over_threshold, then unknown, then healthy)
- `projects: Vec<ProjectTile>` — one per project rollup, with aggregated counts
- `warnings: Vec<String>` — schema and parse warnings from the scanner
- `computed_at: DateTime<Utc>` — server-side reference time used for stall calculations

### Stall classification at projection time

Stall state is computed when the trellis is built, not stored on disk. This means the same snapshot folded at two different times produces different stall classifications — correct behavior, since a stall is a function of "now."

### Sort order rationale

Tiles are pre-sorted by the trellis so the frontend renders in the order most useful for situational awareness:

- Stalled agents first because they need attention now.
- Pending high-priority directives first because they may unblock other agents.
- Project tiles in stable insertion order (BTreeMap iteration), which is alphabetical by project name — predictable across sessions.

### Project rollup aggregation

For each project, the trellis computes:

- `stalled_count` — agents in `Stalled` state
- `over_threshold_count` — agents in `OverThreshold` state
- `pending_directive_count` — directives with `status: pending` across the rollup and all agents in the project

These let the project tile surface health at a glance without the user drilling into individual agents.

### Mermaid passthrough

The trellis carries the raw Mermaid source from `status.mermaid` files into both agent tiles and project tiles. The frontend currently displays it as collapsible text; a future version can render it via mermaid.js.

### Non-goals

- No query language. The trellis exposes structured fields; consumers read them directly.
- No incremental projection updates. Each scan produces a fresh full trellis.
- No per-session sub-trellises yet. All projects appear in the same trellis.

---

## Tauri command surface

Two commands exposed to the frontend via `invoke`:

- `get_trellis()` → `Trellis` — synchronously builds and returns a fresh trellis. Used for initial hydration and explicit refresh.
- `get_command_hub_path()` → `String` — returns the resolved root path for display in the header.

Read-only commands only. No way for the frontend to mutate the command-hub through the dashboard. By design — directives flow through the platform's normal channels (Claude or human writing `comms.yaml`), not through the HUD.

---

## Seed tooling

A companion binary, `seed`, writes a realistic command-hub fixture mirroring the autoresearch project rollup as of 2026-04-30T19:12Z. Four agents (`node-macbook` lead, `node-windows`, `node-macmini`, `node-nas`) with their actual cadences and stall states. Run with `cargo run --bin seed`.

This provides realistic data for development without requiring real agents to be running. The fixture content is hardcoded in the binary so future spec changes can be validated against a known shape.

---

## Testing strategy

Unit tests live alongside each module:

- `platform::classify_stall` — verifies the cadence rule across boundary cases.
- `trellis::build_agent_tile` — verifies stall classification flows correctly into the tile.

Integration tests against a running Tauri app are deferred until the frontend stabilizes. The modules are testable in isolation — none of them imports Tauri.

---

## Deferred items

- **Git integration.** The platform spec mentions git as the version history layer. A future version can read `git log` for the command-hub to surface "what changed since I last looked" tiles.
- **Mermaid rendering.** Currently raw text in collapsible details. Render via mermaid.js when bandwidth allows.
- **Refs and secrets surfaces.** `refs.yaml` and `secrets.yaml` are parsed but only secrets-expiry is in the planned tile set.
- **Subscription cross-project surfacing.** When a project's `subscriptions:` block points at another project, the trellis should surface that other project's blockers.
- **Cold-archive directive surfacing.** Per the platform spec, completed directives move to `archive/comms-{date}.yaml` after 24h. A future "recent activity" tile can read these.
- **Heartbeat-only update awareness.** Heartbeat writes update only `last_updated`/`timestamp`. The trellis treats them as normal updates today; in future versions, distinguishing heartbeat-only vs. task-progressing writes could feed a "live but idle" indicator.
- **Multi-host federation.** The dashboard sees one command-hub. A future federated mode could read multiple hubs and surface them together.

---

## Decision log

- **YAML files as substrate over JSONL events** — the agent platform already specifies the wire format. Re-implementing it as events would have produced two formats requiring translation. Reading the platform's files directly inherits months of design work.
- **Read-only HUD** — writes belong to agents and humans editing YAML. The HUD is observation; mutation flows through normal platform channels. Keeps the dashboard's responsibilities clean.
- **Full re-scan on change** — simpler than incremental update tracking. Fast enough for command-hubs at expected scale. Optimize when evidence justifies it.
- **Tauri 2 over Electron** — smaller binary, native integration, Rust backend matches sovereign-infrastructure thesis.
- **Default to `~/command-hub`** — matches platform spec's deployment model. Box Sync, network shares, and local-only deployments all mount or symlink at this path.
- **Per-agent cadence over global threshold** — platform spec is explicit on this. Fast agents (60s cadence) and slow agents (1800s cadence) cannot share a global stall threshold without false positives.
- **No frontend framework** — plain TypeScript keeps dependencies minimal. Adopt React/Svelte if component complexity justifies the weight.
