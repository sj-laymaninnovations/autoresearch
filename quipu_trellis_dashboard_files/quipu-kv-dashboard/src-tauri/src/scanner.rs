//! Scanner: walks the command-hub folder structure and parses YAML files.
//!
//! This replaces what was previously a JSONL event store. The platform's
//! YAML files on the shared filesystem (Box-synced or local-only) are now
//! the substrate. The scanner produces an in-memory snapshot that the
//! trellis folds into projection tiles.
//!
//! Folder structure (per the platform spec):
//!
//!   command-hub/
//!     projects/{project}/
//!       rollup/   status.yaml, status.mermaid, comms.yaml, refs.yaml
//!       agents/{node-id}/   status.yaml, status.mermaid, comms.yaml, ...
//!     standalone-agents/{node-id}/   ...
//!
//! Schema mismatches are surfaced, not swallowed: a file with an unknown
//! `schema_version` is reported as a warning rather than silently misparsed.

use crate::platform::{
    AgentStatus, CommsFile, Directive, RollupStatus, SecretsFile, SUPPORTED_SCHEMA_VERSION,
};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// In-memory snapshot of everything under a command-hub root.
#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    pub projects: BTreeMap<String, ProjectSnapshot>,
    pub standalone_agents: BTreeMap<String, AgentSnapshot>,
    /// Soft warnings: schema mismatches, parse errors, etc. Collected so the
    /// HUD can surface them instead of failing silently.
    pub warnings: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ProjectSnapshot {
    pub project_name: String,
    pub rollup: Option<RollupStatus>,
    pub rollup_mermaid: Option<String>,
    pub rollup_directives: Vec<Directive>,
    pub agents: BTreeMap<String, AgentSnapshot>,
}

#[derive(Debug, Default, Clone)]
pub struct AgentSnapshot {
    pub node_id: String,
    pub project: Option<String>,
    pub status: Option<AgentStatus>,
    pub mermaid: Option<String>,
    pub directives: Vec<Directive>,
    pub secrets: Option<SecretsFile>,
}

/// Read everything under `root`. Missing files are tolerated; malformed
/// files become warnings.
pub fn scan(root: &Path) -> Result<Snapshot> {
    let mut snap = Snapshot::default();
    if !root.exists() {
        snap.warnings
            .push(format!("command-hub not found: {}", root.display()));
        return Ok(snap);
    }

    let projects_dir = root.join("projects");
    if projects_dir.is_dir() {
        for entry in fs::read_dir(&projects_dir)
            .with_context(|| format!("reading {}", projects_dir.display()))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    snap.warnings.push(format!("projects entry: {e}"));
                    continue;
                }
            };
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let proj_name = entry.file_name().to_string_lossy().into_owned();
            let proj_path = entry.path();
            let proj = scan_project(&proj_name, &proj_path, &mut snap.warnings);
            snap.projects.insert(proj_name, proj);
        }
    }

    let standalone_dir = root.join("standalone-agents");
    if standalone_dir.is_dir() {
        for entry in fs::read_dir(&standalone_dir)
            .with_context(|| format!("reading {}", standalone_dir.display()))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    snap.warnings.push(format!("standalone entry: {e}"));
                    continue;
                }
            };
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let node_id = entry.file_name().to_string_lossy().into_owned();
            let agent = scan_agent(&node_id, None, &entry.path(), &mut snap.warnings);
            snap.standalone_agents.insert(node_id, agent);
        }
    }

    Ok(snap)
}

fn scan_project(name: &str, path: &Path, warnings: &mut Vec<String>) -> ProjectSnapshot {
    let mut proj = ProjectSnapshot {
        project_name: name.to_string(),
        ..Default::default()
    };

    let rollup_dir = path.join("rollup");
    if rollup_dir.is_dir() {
        proj.rollup = read_yaml::<RollupStatus>(&rollup_dir.join("status.yaml"), warnings);
        proj.rollup_mermaid = read_text_optional(&rollup_dir.join("status.mermaid"));
        proj.rollup_directives = read_directives(&rollup_dir.join("comms.yaml"), warnings);
    }

    let agents_dir = path.join("agents");
    if agents_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&agents_dir) {
            for entry in entries.flatten() {
                if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let node_id = entry.file_name().to_string_lossy().into_owned();
                let agent = scan_agent(&node_id, Some(name), &entry.path(), warnings);
                proj.agents.insert(node_id, agent);
            }
        }
    }

    proj
}

fn scan_agent(
    node_id: &str,
    project: Option<&str>,
    path: &Path,
    warnings: &mut Vec<String>,
) -> AgentSnapshot {
    AgentSnapshot {
        node_id: node_id.to_string(),
        project: project.map(|p| p.to_string()),
        status: read_yaml::<AgentStatus>(&path.join("status.yaml"), warnings),
        mermaid: read_text_optional(&path.join("status.mermaid")),
        directives: read_directives(&path.join("comms.yaml"), warnings),
        secrets: read_yaml::<SecretsFile>(&path.join("secrets.yaml"), warnings),
    }
}

fn read_text_optional(path: &Path) -> Option<String> {
    if path.is_file() {
        fs::read_to_string(path).ok()
    } else {
        None
    }
}

fn read_yaml<T: serde::de::DeserializeOwned + HasSchemaVersion>(
    path: &Path,
    warnings: &mut Vec<String>,
) -> Option<T> {
    if !path.is_file() {
        return None;
    }
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            warnings.push(format!("read failed {}: {}", path.display(), e));
            return None;
        }
    };
    match serde_yaml::from_str::<T>(&text) {
        Ok(v) => {
            let sv = v.schema_version();
            if sv != SUPPORTED_SCHEMA_VERSION {
                warnings.push(format!(
                    "{}: unexpected schema_version {} (supported: {})",
                    path.display(),
                    sv,
                    SUPPORTED_SCHEMA_VERSION
                ));
            }
            Some(v)
        }
        Err(e) => {
            warnings.push(format!("yaml parse {}: {}", path.display(), e));
            None
        }
    }
}

fn read_directives(path: &Path, warnings: &mut Vec<String>) -> Vec<Directive> {
    if !path.is_file() {
        return Vec::new();
    }
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            warnings.push(format!("read failed {}: {}", path.display(), e));
            return Vec::new();
        }
    };
    match serde_yaml::from_str::<CommsFile>(&text) {
        Ok(CommsFile::Single(d)) => vec![d],
        Ok(CommsFile::Multiple { directives }) => directives,
        Ok(CommsFile::Empty {}) => Vec::new(),
        Err(e) => {
            warnings.push(format!("comms parse {}: {}", path.display(), e));
            Vec::new()
        }
    }
}

/// Helper trait so the generic `read_yaml` can validate schema_version
/// without naming each concrete type.
pub trait HasSchemaVersion {
    fn schema_version(&self) -> u32;
}

impl HasSchemaVersion for AgentStatus {
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}
impl HasSchemaVersion for RollupStatus {
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}
impl HasSchemaVersion for SecretsFile {
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// Resolve the default command-hub root: `~/command-hub`.
/// Override with the `QUIPU_COMMAND_HUB` environment variable.
pub fn default_command_hub() -> PathBuf {
    if let Ok(p) = std::env::var("QUIPU_COMMAND_HUB") {
        return PathBuf::from(p);
    }
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("command-hub")
}
