import type { ProjectTile } from "../lib/types";
import { formatTimestamp } from "../lib/format";

export function renderProjectTile(p: ProjectTile): HTMLElement {
  const tile = document.createElement("div");
  tile.className = `tile project overall-${p.overall_status ?? "unknown"}`;

  const header = document.createElement("div");
  header.className = "tile-header";
  const label = document.createElement("p");
  label.className = "tile-label";
  label.textContent = "project";
  const overall = document.createElement("span");
  overall.className = `dstatus-badge overall-${p.overall_status ?? "unknown"}`;
  overall.textContent = p.overall_status ?? "—";
  header.appendChild(label);
  header.appendChild(overall);
  tile.appendChild(header);

  const title = document.createElement("p");
  title.className = "tile-title";
  title.textContent = p.project_name;
  tile.appendChild(title);

  const lead = document.createElement("p");
  lead.className = "tile-meta";
  lead.textContent = p.lead_agent
    ? `lead: ${p.lead_agent} · ${p.agent_count} agent${p.agent_count === 1 ? "" : "s"}`
    : `${p.agent_count} agent${p.agent_count === 1 ? "" : "s"}`;
  tile.appendChild(lead);

  const ts = p.tasks_summary;
  const summary = document.createElement("p");
  summary.className = "tile-meta";
  summary.textContent = `tasks: ${ts.done} done · ${ts.in_progress} in-progress · ${ts.pending} pending · ${ts.blocked} blocked`;
  tile.appendChild(summary);

  if (p.stalled_count > 0 || p.over_threshold_count > 0) {
    const health = document.createElement("p");
    health.className = "tile-meta tile-meta-warn";
    const parts: string[] = [];
    if (p.stalled_count > 0) parts.push(`${p.stalled_count} stalled`);
    if (p.over_threshold_count > 0)
      parts.push(`${p.over_threshold_count} over threshold`);
    health.textContent = parts.join(" · ");
    tile.appendChild(health);
  }

  if (p.pending_directive_count > 0) {
    const dir = document.createElement("p");
    dir.className = "tile-meta";
    dir.textContent = `${p.pending_directive_count} pending directive${p.pending_directive_count === 1 ? "" : "s"}`;
    tile.appendChild(dir);
  }

  if (p.notes) {
    const notes = document.createElement("p");
    notes.className = "tile-notes";
    notes.textContent = p.notes;
    tile.appendChild(notes);
  }

  const rollupTs = document.createElement("p");
  rollupTs.className = "tile-meta";
  rollupTs.textContent = `rollup ${formatTimestamp(p.timestamp)}`;
  tile.appendChild(rollupTs);

  if (p.rollup_mermaid) {
    const details = document.createElement("details");
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = "pipeline";
    summaryEl.className = "tile-summary";
    details.appendChild(summaryEl);
    const pre = document.createElement("pre");
    pre.className = "mermaid-source";
    pre.textContent = p.rollup_mermaid;
    details.appendChild(pre);
    tile.appendChild(details);
  }

  return tile;
}
