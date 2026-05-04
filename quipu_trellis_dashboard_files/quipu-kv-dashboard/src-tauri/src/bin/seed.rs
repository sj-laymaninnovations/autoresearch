//! Seed: write a realistic command-hub fixture.
//!
//! Mirrors the autoresearch rollup as of 2026-04-30T19:12Z so the dashboard
//! has live-shaped data to render. Safe to run multiple times — it overwrites
//! files in place.
//!
//! Run with:
//!   cargo run --bin seed
//!   QUIPU_COMMAND_HUB=/tmp/command-hub cargo run --bin seed   # custom root

use chrono::{TimeZone, Utc};
use quipu_kv_dashboard_lib::default_command_hub;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let root = default_command_hub();
    println!("seeding command-hub at {}", root.display());

    let proj = root.join("projects").join("autoresearch");
    let rollup = proj.join("rollup");
    fs::create_dir_all(&rollup)?;

    write(
        &rollup.join("status.yaml"),
        ROLLUP_STATUS,
    )?;
    write(&rollup.join("status.mermaid"), ROLLUP_MERMAID)?;
    write(&rollup.join("comms.yaml"), ROLLUP_COMMS)?;

    let agents = [
        ("node-macbook", MACBOOK_STATUS, MACBOOK_MERMAID, ""),
        ("node-windows", WINDOWS_STATUS, WINDOWS_MERMAID, WINDOWS_COMMS),
        ("node-macmini", MACMINI_STATUS, "", ""),
        ("node-nas", NAS_STATUS, "", ""),
    ];

    for (node, status, mermaid, comms) in agents {
        let dir = proj.join("agents").join(node);
        fs::create_dir_all(&dir)?;
        write(&dir.join("status.yaml"), status)?;
        if !mermaid.is_empty() {
            write(&dir.join("status.mermaid"), mermaid)?;
        }
        if !comms.is_empty() {
            write(&dir.join("comms.yaml"), comms)?;
        }
    }

    println!("seeded autoresearch project with 4 agents");
    println!("computed at: {}", Utc.with_ymd_and_hms(2026, 4, 30, 19, 12, 50).unwrap());
    Ok(())
}

fn write(path: &Path, content: &str) -> anyhow::Result<()> {
    fs::write(path, content)?;
    println!("  wrote {}", path.display());
    Ok(())
}

// ─── Fixture content ──────────────────────────────────────────────────────────

const ROLLUP_STATUS: &str = r#"schema_version: 1
node: rollup
project: autoresearch
timestamp: 2026-04-30T19:12:00Z
lead_agent: node-macbook
overall_status: in_progress
agents:
  node-macbook:
    status: in_progress
    current_task: Recipe repeatability — Exp A
    last_updated: 2026-04-30T18:52:57Z
    expected_cadence_s: 300
    blockers: none
  node-windows:
    status: in_progress
    current_task: Generalist 5M test — GPU run
    last_updated: 2026-04-30T11:40:00Z
    expected_cadence_s: 60
    blockers: heartbeat loop not implemented
  node-macmini:
    status: stalled
    current_task: dormant
    last_updated: 2026-04-29T05:00:00Z
    expected_cadence_s: 1800
    blockers: physical wake required
  node-nas:
    status: pending
    current_task: passive standby
    last_updated: 2026-04-30T11:40:00Z
    expected_cadence_s: 3600
    blockers: docker group access needed
tasks_summary:
  done: 8
  in_progress: 2
  pending: 4
  blocked: 0
notes: |
  enforce_heartbeat directive committed; node-windows has not acknowledged.
  node-macmini exempt until physical wake.
last_updated: 2026-04-30T19:12:00Z
"#;

const ROLLUP_MERMAID: &str = r#"graph LR
  M1[Atomic skills] -->|done| M2[Generalist 5M test]
  M2 -->|done| M3[Capacity verdict]
  M3 -->|in_progress| M4[Recipe repeatability]
  M4 -->|pending| M5[Capacity scaling 10M-30M]
  M4 -->|pending| M6[Tier-3 skills]
  M6 -->|pending| M7[Layman secrets migration]
  M5 -->|pending| M8[BitNet i2s comparison]
"#;

const ROLLUP_COMMS: &str = r#"schema_version: 1
agent: all
project: autoresearch
name: Enforce Heartbeat
description: All agents must implement heartbeat write loop per platform spec
date: 2026-04-30T17:55:00Z
task: enforce_heartbeat
status: pending
priority: high
instructions: |
  Implement a heartbeat write loop that touches last_updated and timestamp
  every expected_cadence_s seconds, even when no task state has changed.
  Stall detection cannot work without this. Acknowledge by setting status
  to acknowledged and beginning heartbeats within one cadence interval.
"#;

const MACBOOK_STATUS: &str = r#"schema_version: 1
node: node-macbook
project: autoresearch
status: in_progress
last_completed: Capacity verdict synthesis
current_task: Recipe repeatability — Exp A setup
next_task: Recipe repeatability — Exp B
blockers: none
notes: Lead agent. Heartbeat at 18:52:57Z; next due ~18:57:57Z.
last_updated: 2026-04-30T18:52:57Z
expected_cadence_s: 300
timestamp: 2026-04-30T18:52:57Z
"#;

const MACBOOK_MERMAID: &str = r#"graph LR
  A[Capacity verdict] -->|done| B[Exp A setup]
  B -->|in_progress| C[Exp A run]
  C -->|pending| D[Exp B setup]
  D -->|pending| E[Exp B run]
  E -->|pending| F[Repeatability writeup]
"#;

const WINDOWS_STATUS: &str = r#"schema_version: 1
node: node-windows
project: autoresearch
status: in_progress
last_completed: Atomic skills GPU validation
current_task: Generalist 5M test — Exp A and B
next_task: Capacity scaling 10M-30M
blockers: heartbeat loop not yet implemented in agent code
notes: |
  Experiments running on GPUs; last status write was at scaffold time.
  enforce_heartbeat directive landed but agent has not picked it up yet.
last_updated: 2026-04-30T11:40:00Z
expected_cadence_s: 60
timestamp: 2026-04-30T11:40:00Z
"#;

const WINDOWS_MERMAID: &str = r#"graph LR
  A[Atomic skills] -->|done| B[Generalist 5M Exp A]
  A -->|done| C[Generalist 5M Exp B]
  B -->|in_progress| D[10M scale]
  C -->|in_progress| D
  D -->|pending| E[30M scale]
  E -->|pending| F[BitNet i2s]
"#;

const WINDOWS_COMMS: &str = r#"schema_version: 1
agent: node-windows
project: autoresearch
name: Enforce Heartbeat
description: Implement heartbeat write loop per platform spec
date: 2026-04-30T17:55:00Z
task: enforce_heartbeat
status: pending
priority: high
instructions: |
  Implement a 60s heartbeat write loop. Update last_updated and timestamp
  every cadence interval even when no task state has changed. Acknowledge
  by setting status to acknowledged.
"#;

const MACMINI_STATUS: &str = r#"schema_version: 1
node: node-macmini
project: autoresearch
status: stalled
current_task: dormant
blockers: physical wake required - sshd not responding
notes: Asleep since 2026-04-29T05:00Z. Exempt from heartbeat directive.
last_updated: 2026-04-29T05:00:00Z
expected_cadence_s: 1800
timestamp: 2026-04-29T05:00:00Z
"#;

const NAS_STATUS: &str = r#"schema_version: 1
node: node-nas
project: autoresearch
status: pending
current_task: passive standby
next_task: self-hosted MLflow + Gitea once docker access granted
blockers: claudeli not in docker group
notes: Passive node. Has not written since scaffold.
last_updated: 2026-04-30T11:40:00Z
expected_cadence_s: 3600
timestamp: 2026-04-30T11:40:00Z
"#;
