// Mirrors src-tauri/src/platform.rs and trellis.rs

export type AgentStatusKind =
  | "done"
  | "in_progress"
  | "pending"
  | "blocked"
  | "stalled";

export type StallState = "healthy" | "over_threshold" | "stalled" | "unknown";

export type DirectiveStatus =
  | "pending"
  | "acknowledged"
  | "complete"
  | "failed";

export type Priority = "low" | "medium" | "high";

export interface DirectiveTile {
  agent: string;
  project: string | null;
  name: string;
  task: string;
  priority: Priority;
  status: DirectiveStatus;
  date: string;
  instructions_excerpt: string | null;
}

export interface AgentTile {
  node_id: string;
  project: string | null;
  status: AgentStatusKind | null;
  stall: StallState;
  last_updated: string | null;
  age_seconds: number | null;
  expected_cadence_s: number | null;
  current_task: string | null;
  blockers: string | null;
  notes: string | null;
  mermaid: string | null;
  pending_directives: number;
}

export interface TasksSummary {
  done: number;
  in_progress: number;
  pending: number;
  blocked: number;
}

export interface ProjectTile {
  project_name: string;
  lead_agent: string | null;
  overall_status: AgentStatusKind | null;
  timestamp: string | null;
  agent_count: number;
  stalled_count: number;
  over_threshold_count: number;
  tasks_summary: TasksSummary;
  notes: string | null;
  rollup_mermaid: string | null;
  pending_directive_count: number;
}

export interface Trellis {
  pending_directives: DirectiveTile[];
  agents: AgentTile[];
  projects: ProjectTile[];
  warnings: string[];
  computed_at: string;
}
