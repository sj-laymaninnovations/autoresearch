import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { Trellis } from "./lib/types";
import { renderAgentTile } from "./components/agentTile";
import { renderDirectiveTile } from "./components/directiveTile";
import { renderProjectTile } from "./components/projectTile";
import "./styles.css";

async function loadInitial(): Promise<Trellis> {
  return invoke<Trellis>("get_trellis");
}

async function getCommandHubPath(): Promise<string> {
  return invoke<string>("get_command_hub_path");
}

function render(root: HTMLElement, trellis: Trellis, hubPath: string) {
  root.innerHTML = "";

  const header = document.createElement("header");
  header.className = "hud-header";
  const stats = trellisStats(trellis);
  header.innerHTML = `
    <div class="hud-title">
      <h1>QuiPu-KV</h1>
      <span class="hud-subtitle">${escape(hubPath)}</span>
    </div>
    <div class="hud-stats">
      <span>${stats.projects} projects</span>
      <span>${stats.agents} agents</span>
      <span class="${stats.stalled > 0 ? "stat-warn" : ""}">${stats.stalled} stalled</span>
      <span>${stats.pending} pending directives</span>
    </div>
  `;
  root.appendChild(header);

  if (trellis.warnings.length > 0) {
    const warnBar = document.createElement("div");
    warnBar.className = "warning-bar";
    warnBar.textContent = `${trellis.warnings.length} parser warning${trellis.warnings.length === 1 ? "" : "s"} — open devtools for details`;
    root.appendChild(warnBar);
    console.warn("Trellis warnings:", trellis.warnings);
  }

  const main = document.createElement("main");
  main.className = "hud-main";

  // Projects column
  const projectsCol = column("projects", `projects · ${trellis.projects.length}`);
  if (trellis.projects.length === 0) {
    projectsCol.appendChild(emptyState("no projects in command-hub yet"));
  } else {
    for (const p of trellis.projects) {
      projectsCol.appendChild(renderProjectTile(p));
    }
  }

  // Agents column
  const agentsCol = column("agents", `agents · ${trellis.agents.length}`);
  if (trellis.agents.length === 0) {
    agentsCol.appendChild(emptyState("no agents reporting"));
  } else {
    for (const a of trellis.agents) {
      agentsCol.appendChild(renderAgentTile(a));
    }
  }

  // Directives column
  const directivesCol = column("directives", `directives · ${trellis.pending_directives.length}`);
  if (trellis.pending_directives.length === 0) {
    directivesCol.appendChild(emptyState("no directives"));
  } else {
    for (const d of trellis.pending_directives) {
      directivesCol.appendChild(renderDirectiveTile(d));
    }
  }

  main.appendChild(projectsCol);
  main.appendChild(agentsCol);
  main.appendChild(directivesCol);
  root.appendChild(main);

  const footer = document.createElement("footer");
  footer.className = "hud-footer";
  footer.textContent = `computed ${new Date(trellis.computed_at).toLocaleString()}`;
  root.appendChild(footer);
}

function trellisStats(t: Trellis) {
  const stalled = t.agents.filter(
    (a) => a.stall === "stalled" || a.stall === "over_threshold",
  ).length;
  const pending = t.pending_directives.filter((d) => d.status === "pending").length;
  return {
    projects: t.projects.length,
    agents: t.agents.length,
    stalled,
    pending,
  };
}

function column(cls: string, title: string): HTMLElement {
  const col = document.createElement("section");
  col.className = `hud-col col-${cls}`;
  const h = document.createElement("h2");
  h.className = "col-title";
  h.textContent = title;
  col.appendChild(h);
  return col;
}

function emptyState(text: string): HTMLElement {
  const p = document.createElement("p");
  p.className = "empty-state";
  p.textContent = text;
  return p;
}

function escape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function main() {
  const root = document.getElementById("app");
  if (!root) return;

  const hubPath = await getCommandHubPath();
  let trellis = await loadInitial();
  render(root, trellis, hubPath);

  await listen<Trellis>("quipu://trellis/update", (event) => {
    trellis = event.payload;
    render(root, trellis, hubPath);
  });
}

main().catch((e) => {
  console.error(e);
  const root = document.getElementById("app");
  if (root) {
    root.innerHTML = `<pre style="padding: 1rem; color: #e24b4a;">Failed to load:\n${escape(String(e))}</pre>`;
  }
});

(window as unknown as { quipu: unknown }).quipu = { invoke, listen };
