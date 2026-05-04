import type { AgentTile } from "../lib/types";
import { formatAge, formatCadence, escapeHtml } from "../lib/format";

export function renderAgentTile(a: AgentTile): HTMLElement {
  const tile = document.createElement("div");
  tile.className = `tile agent stall-${a.stall} status-${a.status ?? "unknown"}`;

  const header = document.createElement("div");
  header.className = "tile-header";
  const ident = document.createElement("p");
  ident.className = "tile-label";
  ident.textContent = a.project ? `${a.project} · ${a.node_id}` : a.node_id;
  const stallBadge = document.createElement("span");
  stallBadge.className = `stall-badge stall-${a.stall}`;
  stallBadge.textContent = stallLabel(a.stall);
  header.appendChild(ident);
  header.appendChild(stallBadge);
  tile.appendChild(header);

  const title = document.createElement("p");
  title.className = "tile-title";
  title.textContent = a.current_task ?? a.status ?? "no task";
  tile.appendChild(title);

  if (a.blockers && a.blockers !== "none") {
    const blockers = document.createElement("p");
    blockers.className = "blockers";
    blockers.textContent = `blocker: ${a.blockers}`;
    tile.appendChild(blockers);
  }

  const meta = document.createElement("p");
  meta.className = "tile-meta";
  meta.innerHTML = `last update ${formatAge(a.age_seconds)} ago · cadence ${formatCadence(a.expected_cadence_s)}`;
  tile.appendChild(meta);

  if (a.pending_directives > 0) {
    const dir = document.createElement("p");
    dir.className = "tile-meta tile-meta-warn";
    dir.textContent = `${a.pending_directives} pending directive${a.pending_directives === 1 ? "" : "s"}`;
    tile.appendChild(dir);
  }

  if (a.notes) {
    const notes = document.createElement("p");
    notes.className = "tile-notes";
    notes.textContent = a.notes;
    tile.appendChild(notes);
  }

  if (a.mermaid) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "task graph";
    summary.className = "tile-summary";
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.className = "mermaid-source";
    pre.textContent = a.mermaid;
    details.appendChild(pre);
    tile.appendChild(details);
  }

  return tile;
}

function stallLabel(s: AgentTile["stall"]): string {
  switch (s) {
    case "healthy":
      return "healthy";
    case "over_threshold":
      return "over threshold";
    case "stalled":
      return "stalled";
    case "unknown":
      return "unknown";
  }
}

// re-export to satisfy noUnusedLocals when the consumer doesn't import escapeHtml
export { escapeHtml };
