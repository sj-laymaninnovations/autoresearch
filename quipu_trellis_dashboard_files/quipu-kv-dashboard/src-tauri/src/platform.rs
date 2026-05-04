//! Layman Agent Platform schema — Rust types matching the published v1 spec.
//!
//! These structs deserialize directly from the YAML files agents write under
//! `command-hub/projects/{project}/agents/{node-id}/` and
//! `command-hub/projects/{project}/rollup/`. We track schema_version explicitly
//! so we can refuse to parse files from a future schema rather than silently
//! misinterpret them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// One agent's `status.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub schema_version: u32,
    pub node: String,
    pub project: Option<String>,
    pub status: AgentStatusKind,
    #[serde(default)]
    pub last_completed: Option<String>,
    #[serde(default)]
    pub current_task: Option<String>,
    #[serde(default)]
    pub next_task: Option<String>,
    #[serde(default)]
    pub blockers: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    pub last_updated: DateTime<Utc>,
    /// How often this agent is expected to write a heartbeat. Drives stall
    /// detection: stalled if `now - last_updated > expected_cadence_s * 3`.
    #[serde(default)]
    pub expected_cadence_s: Option<u64>,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatusKind {
    Done,
    InProgress,
    Pending,
    Blocked,
    Stalled,
}

/// Project-level `rollup/status.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupStatus {
    pub schema_version: u32,
    pub node: String, // typically "rollup"
    pub project: String,
    pub timestamp: DateTime<Utc>,
    pub lead_agent: Option<String>,
    pub overall_status: AgentStatusKind,
    #[serde(default)]
    pub agents: BTreeMap<String, RollupAgentEntry>,
    #[serde(default)]
    pub tasks_summary: TasksSummary,
    #[serde(default)]
    pub subscriptions: Vec<Subscription>,
    #[serde(default)]
    pub notes: Option<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupAgentEntry {
    pub status: AgentStatusKind,
    #[serde(default)]
    pub current_task: Option<String>,
    pub last_updated: DateTime<Utc>,
    #[serde(default)]
    pub expected_cadence_s: Option<u64>,
    #[serde(default)]
    pub blockers: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TasksSummary {
    #[serde(default)]
    pub done: u32,
    #[serde(default)]
    pub in_progress: u32,
    #[serde(default)]
    pub pending: u32,
    #[serde(default)]
    pub blocked: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub project: String,
    pub rollup_path: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// `comms.yaml` — agent-scoped or project-wide directive bus.
/// Spec example shows a single directive per file; we tolerate either a
/// single-directive document or a list under `directives:`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommsFile {
    Single(Directive),
    Multiple { directives: Vec<Directive> },
    Empty {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directive {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub agent: String,
    #[serde(default)]
    pub project: Option<String>,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub date: DateTime<Utc>,
    pub task: String,
    pub status: DirectiveStatus,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub instructions: Option<String>,
}

fn default_schema() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectiveStatus {
    Pending,
    Acknowledged,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low,
    #[default]
    Medium,
    High,
}

/// `secrets.yaml` — slug references only, never values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretsFile {
    pub schema_version: u32,
    #[serde(default)]
    pub secrets: Vec<SecretRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretRef {
    pub slug: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub date: chrono::NaiveDate,
    pub last_updated: chrono::NaiveDate,
    #[serde(default)]
    pub expires: Option<chrono::NaiveDate>,
}

/// Stall classification for a single agent against its declared cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StallState {
    Healthy,
    OverThreshold,
    Stalled,
    Unknown,
}

/// Compute stall state from the platform's per-agent cadence rule:
/// stalled if `now - last_updated > expected_cadence_s * 3`.
/// Over-threshold if past 1× cadence but within 3×.
pub fn classify_stall(
    last_updated: DateTime<Utc>,
    expected_cadence_s: Option<u64>,
    now: DateTime<Utc>,
) -> StallState {
    let cadence = match expected_cadence_s {
        Some(s) if s > 0 => s as i64,
        _ => return StallState::Unknown,
    };
    let age_s = (now - last_updated).num_seconds();
    if age_s > cadence * 3 {
        StallState::Stalled
    } else if age_s > cadence {
        StallState::OverThreshold
    } else {
        StallState::Healthy
    }
}
