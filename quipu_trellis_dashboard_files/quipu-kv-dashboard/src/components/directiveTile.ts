import type { DirectiveTile } from "../lib/types";
import { formatTimestamp } from "../lib/format";

export function renderDirectiveTile(d: DirectiveTile): HTMLElement {
  const tile = document.createElement("div");
  tile.className = `tile directive priority-${d.priority} dstatus-${d.status}`;

  const header = document.createElement("div");
  header.className = "tile-header";
  const label = document.createElement("p");
  label.className = "tile-label";
  label.textContent = d.project
    ? `${d.project} · → ${d.agent}`
    : `→ ${d.agent}`;
  const status = document.createElement("span");
  status.className = `dstatus-badge dstatus-${d.status}`;
  status.textContent = d.status;
  header.appendChild(label);
  header.appendChild(status);
  tile.appendChild(header);

  const title = document.createElement("p");
  title.className = "tile-title";
  title.textContent = d.name;
  tile.appendChild(title);

  const task = document.createElement("p");
  task.className = "tile-meta";
  task.textContent = `task: ${d.task} · priority: ${d.priority}`;
  tile.appendChild(task);

  if (d.instructions_excerpt) {
    const inst = document.createElement("p");
    inst.className = "tile-notes";
    inst.textContent = d.instructions_excerpt;
    tile.appendChild(inst);
  }

  const ts = document.createElement("p");
  ts.className = "tile-meta";
  ts.textContent = `posted ${formatTimestamp(d.date)}`;
  tile.appendChild(ts);

  return tile;
}
