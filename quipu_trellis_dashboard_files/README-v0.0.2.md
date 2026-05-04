# quipu-kv-dashboard

A sovereign, local-first HUD for the Layman Agent Awareness Platform.

Built on Tauri 2 (Rust backend, web frontend). Reads the YAML files agents write under a shared `command-hub/` folder and renders a compressed, real-time projection — one tile per project rollup, one tile per agent, one tile per pending directive.

## What this is

The visualization layer for the Layman Agent Awareness Platform. The platform spec defines how agents coordinate via shared YAML files (status, comms, refs, secrets) on a common filesystem (Box Sync, network share, or local). This dashboard reads those files live and surfaces:

- **Per-agent stall classification** computed against each agent's declared `expected_cadence_s` — no global thresholds, no false positives on fast or slow agents.
- **Pending directives** awaiting acknowledgment, sorted by priority and recency.
- **Project rollups** with lead agent, overall status, task summaries, and pipeline mermaid graphs.
- **Schema warnings** for any YAML file that mismatches the platform's v1 schema.

It does not write to the command-hub. It is read-only by design — agents and the user produce; the dashboard observes.

## Folder layout it expects

The dashboard reads the platform's standard layout:

```
command-hub/
  projects/{project-name}/
    rollup/   status.yaml, status.mermaid, comms.yaml, refs.yaml
    agents/{node-id}/   status.yaml, status.mermaid, comms.yaml, refs.yaml, secrets.yaml
  standalone-agents/{node-id}/   ...
```

By default it looks at `~/command-hub`. Override with the `QUIPU_COMMAND_HUB` environment variable.

## Quick start

```bash
# Clone and install
git clone https://github.com/sj-laymaninnovations/quipu-kv-dashboard
cd quipu-kv-dashboard
npm install

# Seed a fixture command-hub (mirrors the autoresearch project at 2026-04-30)
cargo run --manifest-path src-tauri/Cargo.toml --bin seed

# Run the dashboard
npm run tauri dev
```

## Architecture

- `src-tauri/src/platform.rs` — Layman Agent Platform v1 schema as Rust types
- `src-tauri/src/scanner.rs` — walks command-hub structure, parses YAML
- `src-tauri/src/transport.rs` — debounced filesystem watcher, emits trellis updates
- `src-tauri/src/trellis.rs` — folds the snapshot into HUD-ready tiles
- `src-tauri/src/lib.rs` — Tauri commands and wiring
- `src-tauri/src/bin/seed.rs` — fixture writer for development

See `docs/quipu-kv.md` for the substrate explanation and `docs/store-transport-trellis.md` for the module-level spec.

## Status

v0.0.2. Platform-backed scaffolding complete. Not yet validated against a live multi-agent deployment.

## License

TBD. Authored by Sean Layman / Layman Innovations LLC.
