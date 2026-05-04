//! Trellis: projection of the agent platform snapshot into HUD tiles.
//!
//! The trellis is not the HUD; it is the structured state the HUD renders
//! from. Inputs come from `scanner::Snapshot` (parsed YAML files). Outputs
//! are tiles organized for at-a-glance situational awareness.
//!
//! Three tile families in the default trellis:
//! - Pending directives (any agent's `comms.yaml` with `pending` status)
//! - Agent health (one tile per agent, with stall classification)
//! - Project rollups (one tile per project, with lead, overall status, summary)

use crate::platform::{
    classify_stall, AgentStatusKind, Directive, DirectiveStatus, Priority, StallState,
    TasksSummary,
};
use crate::scanner::Snapshot;
use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Trellis {
    pub pending_directives: Vec<DirectiveTile>,
    pub agents: Vec<AgentTile>,
    pub projects: Vec<ProjectTile>,
    pub warnings: Vec<String>,
    /// Server-side "now" used for stall calculations. Surfaced so the
    /// frontend can label tiles with a consistent reference time.
    pub computed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectiveTile {
    pub agent: String,
    pub project: Option<String>,
    pub name: String,
    pub task: String,
    pub priority: Priority,
    pub status: DirectiveStatus,
    pub date: DateTime<Utc>,
    pub instructions_excerpt: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTile {
    pub node_id: String,
    pub project: Option<String>,
    pub status: Option<AgentStatusKind>,
    pub stall: StallState,
    pub last_updated: Option<DateTime<Utc>>,
    /// Seconds since last update, computed against `computed_at`.
    pub age_seconds: Option<i64>,
    pub expected_cadence_s: Option<u64>,
    pub current_task: Option<String>,
    pub blockers: Option<String>,
    pub notes: Option<String>,
    pub mermaid: Option<String>,
    /// Number of pending directives waiting on this agent.
    pub pending_directives: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectTile {
    pub project_name: String,
    pub lead_agent: Option<String>,
    pub overall_status: Option<AgentStatusKind>,
    pub timestamp: Option<DateTime<Utc>>,
    pub agent_count: usize,
    pub stalled_count: usize,
    pub over_threshold_count: usize,
    pub tasks_summary: TasksSummary,
    pub notes: Option<String>,
    pub rollup_mermaid: Option<String>,
    pub pending_directive_count: u32,
}

impl Trellis {
    /// Build the trellis by folding a scanner snapshot at a given clock time.
    pub fn from_snapshot(snap: &Snapshot, now: DateTime<Utc>) -> Self {
        let mut t = Trellis {
            warnings: snap.warnings.clone(),
            computed_at: now,
            ..Default::default()
        };

        // Project rollups + their agents
        for (proj_name, proj) in &snap.projects {
            let mut stalled = 0usize;
            let mut over = 0usize;
            for (node_id, agent) in &proj.agents {
                let tile = build_agent_tile(node_id, Some(proj_name), agent, now);
                match tile.stall {
                    StallState::Stalled => stalled += 1,
                    StallState::OverThreshold => over += 1,
                    _ => {}
                }
                t.agents.push(tile);
            }

            let mut proj_pending = 0u32;
            for d in &proj.rollup_directives {
                if d.status == DirectiveStatus::Pending {
                    proj_pending += 1;
                }
                t.pending_directives.push(tile_from_directive(d));
            }
            for agent in proj.agents.values() {
                for d in &agent.directives {
                    if d.status == DirectiveStatus::Pending {
                        proj_pending += 1;
                    }
                    t.pending_directives.push(tile_from_directive(d));
                }
            }

            t.projects.push(ProjectTile {
                project_name: proj_name.clone(),
                lead_agent: proj.rollup.as_ref().and_then(|r| r.lead_agent.clone()),
                overall_status: proj.rollup.as_ref().map(|r| r.overall_status),
                timestamp: proj.rollup.as_ref().map(|r| r.timestamp),
                agent_count: proj.agents.len(),
                stalled_count: stalled,
                over_threshold_count: over,
                tasks_summary: proj
                    .rollup
                    .as_ref()
                    .map(|r| r.tasks_summary.clone())
                    .unwrap_or_default(),
                notes: proj.rollup.as_ref().and_then(|r| r.notes.clone()),
                rollup_mermaid: proj.rollup_mermaid.clone(),
                pending_directive_count: proj_pending,
            });
        }

        // Standalone agents
        for (node_id, agent) in &snap.standalone_agents {
            let tile = build_agent_tile(node_id, None, agent, now);
            t.agents.push(tile);
            for d in &agent.directives {
                t.pending_directives.push(tile_from_directive(d));
            }
        }

        // Surface order: stalled agents first, then over-threshold, then healthy.
        t.agents.sort_by_key(|a| match a.stall {
            StallState::Stalled => 0,
            StallState::OverThreshold => 1,
            StallState::Unknown => 2,
            StallState::Healthy => 3,
        });

        // Directives: pending first, then by priority, then by date (most recent first).
        t.pending_directives.sort_by(|a, b| {
            let ord = directive_status_rank(a.status).cmp(&directive_status_rank(b.status));
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
            let pord = priority_rank(a.priority).cmp(&priority_rank(b.priority));
            if pord != std::cmp::Ordering::Equal {
                return pord;
            }
            b.date.cmp(&a.date)
        });

        t
    }
}

fn build_agent_tile(
    node_id: &str,
    project: Option<&str>,
    agent: &crate::scanner::AgentSnapshot,
    now: DateTime<Utc>,
) -> AgentTile {
    let status = agent.status.as_ref();
    let last_updated = status.map(|s| s.last_updated);
    let cadence = status.and_then(|s| s.expected_cadence_s);
    let stall = match last_updated {
        Some(lu) => classify_stall(lu, cadence, now),
        None => StallState::Unknown,
    };
    let age_seconds = last_updated.map(|lu| (now - lu).num_seconds());

    let pending = agent
        .directives
        .iter()
        .filter(|d| d.status == DirectiveStatus::Pending)
        .count() as u32;

    AgentTile {
        node_id: node_id.to_string(),
        project: project.map(|p| p.to_string()),
        status: status.map(|s| s.status),
        stall,
        last_updated,
        age_seconds,
        expected_cadence_s: cadence,
        current_task: status.and_then(|s| s.current_task.clone()),
        blockers: status.and_then(|s| s.blockers.clone()),
        notes: status.and_then(|s| s.notes.clone()),
        mermaid: agent.mermaid.clone(),
        pending_directives: pending,
    }
}

fn tile_from_directive(d: &Directive) -> DirectiveTile {
    DirectiveTile {
        agent: d.agent.clone(),
        project: d.project.clone(),
        name: d.name.clone(),
        task: d.task.clone(),
        priority: d.priority,
        status: d.status,
        date: d.date,
        instructions_excerpt: d.instructions.as_ref().map(|s| {
            let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
            if line.len() > 140 {
                format!("{}…", &line[..140])
            } else {
                line.to_string()
            }
        }),
    }
}

fn directive_status_rank(s: DirectiveStatus) -> u8 {
    match s {
        DirectiveStatus::Pending => 0,
        DirectiveStatus::Acknowledged => 1,
        DirectiveStatus::Failed => 2,
        DirectiveStatus::Complete => 3,
    }
}

fn priority_rank(p: Priority) -> u8 {
    match p {
        Priority::High => 0,
        Priority::Medium => 1,
        Priority::Low => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::AgentStatus;
    use crate::scanner::AgentSnapshot;
    use chrono::Duration;

    #[test]
    fn stall_classification_basics() {
        let now = Utc::now();
        let cadence = Some(60u64);

        assert_eq!(classify_stall(now, cadence, now), StallState::Healthy);
        assert_eq!(
            classify_stall(now - Duration::seconds(90), cadence, now),
            StallState::OverThreshold
        );
        assert_eq!(
            classify_stall(now - Duration::seconds(300), cadence, now),
            StallState::Stalled
        );
        assert_eq!(classify_stall(now, None, now), StallState::Unknown);
    }

    #[test]
    fn agent_tile_marks_stall_correctly() {
        let now = Utc::now();
        let agent = AgentSnapshot {
            node_id: "node-x".into(),
            project: None,
            status: Some(AgentStatus {
                schema_version: 1,
                node: "node-x".into(),
                project: None,
                status: AgentStatusKind::InProgress,
                last_completed: None,
                current_task: Some("benchmark".into()),
                next_task: None,
                blockers: None,
                notes: None,
                last_updated: now - Duration::hours(2),
                expected_cadence_s: Some(60),
                timestamp: None,
            }),
            mermaid: None,
            directives: vec![],
            secrets: None,
        };
        let tile = build_agent_tile("node-x", None, &agent, now);
        assert_eq!(tile.stall, StallState::Stalled);
        assert!(tile.age_seconds.unwrap() > 60 * 3);
    }
}
