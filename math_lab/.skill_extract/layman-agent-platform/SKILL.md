---
name: layman-agent-platform
description: >
  Use this skill whenever the user wants to set up, operate, or interact with the Layman Agent
  Awareness Platform — a distributed command-and-control system using a shared filesystem (Box,
  SharePoint, network share, local drive) and git as the backbone for multi-agent coordination.
  Trigger this skill when the user mentions: agent status, project rollup, node reporting, Layman
  platform, command bus, comms.yaml, secrets.yaml setup, secrets vault storage, slug references,
  agent Kanban, Mermaid status files, YAML detail reports, lead agent, rollup merge, polling agents,
  writing secrets to OS vault, or setting up the layman_secrets_setup.py utility. Also trigger when
  the user asks Claude to read agent status, summarize project health, write a directive to an agent,
  or check if any nodes are stalled or blocked. This skill covers both the secrets management pattern
  (Layman Secrets) and the full agent awareness platform architecture.
---

# Layman Agent Awareness Platform
## Claude Code Operating Guide

This skill gives Claude Code full understanding of the Layman Agent Awareness Platform — a
lightweight, transport-agnostic distributed command-and-control system built on shared filesystems
and git. It also covers the Layman Secrets pattern for OS-native credential management.

---

## Quick Reference

| Need | Go to |
|---|---|
| Platform architecture overview | [Architecture](#architecture) |
| File types and formats | [Six File Types](#six-file-types) |
| Folder structure | [Folder Structure](#folder-structure) |
| Agent modes | [Agent Modes](#agent-modes) |
| Rollup and merge | [Rollup](#rollup--merge) |
| Secrets management | [Layman Secrets](#layman-secrets) |
| Reading agent status | [Reading Status](#reading-status) |
| Writing directives | [Writing Directives](#writing-directives) |
| Setup utilities | [references/setup.md](references/setup.md) |
| Full format specs | [references/formats.md](references/formats.md) |

---

## Architecture

The platform uses a **shared filesystem as a command bus**. Agents read commands from and write
status to a folder structure synced across machines. Git provides version history, diff-based
change detection, and collision resolution.

### Transport Tiers

**Tier 1 — Shared Filesystem** (syncs to local working directory):
- Box + Box Sync ← recommended default
- Local Drive, Network Share, SharePoint, DFS/NAS/SAN, Storage Container

**Tier 2 — Hosted Git Remote** (optional scale-up from local git repo):
- GitHub, GitLab, Azure DevOps, Gitea

**Full sync chain:**
```
Any Shared Filesystem → Local Drive → Git Repo (local) → Hosted Remote (optional)
```

### Core Principles
- **Transport-agnostic** — swap backends without changing file protocol
- **Human-readable** — any teammate can open a folder and understand state
- **Secrets-clean** — no credential values ever touch git or markdown files
- **Git-native** — audit trail, diff, collision resolution all built in
- **Polling-based** — agents check their comms folder every 15–20 seconds

---

## Folder Structure

```
/command-hub/
  projects/
    {project-name}/
      rollup/
        status.mermaid       ← aggregate Kanban across all agents
        status.yaml          ← project-level summary
        comms.yaml           ← project-wide directives
        refs.yaml            ← shared references/supplemental
      agents/
        {node-id}/
          status.mermaid     ← agent Kanban (seeded from rollup on join)
          status.yaml        ← agent detail report
          comms.yaml         ← agent-specific directives (agent polls here)
          refs.yaml          ← agent references/supplemental
          secrets.yaml       ← slug references only
  standalone-agents/
    {node-id}/               ← agents not assigned to a project
      status.mermaid
      status.yaml
      comms.yaml
      refs.yaml
      secrets.yaml
```

---

## Six File Types

| Layer | Format | Purpose |
|---|---|---|
| **High Level Kanban** | MD Mermaid | Quick visual situational awareness |
| **Details** | MD YAML/YAML | Full project/task status report |
| **References / Supplemental** | MD YAML/YAML | Supporting context, links, notes |
| **Secrets** | YAML | Slug references + lifecycle metadata only |
| **Comms** | YAML | Async command bus — directives and instructions |

See [references/formats.md](references/formats.md) for full format specifications and examples.

---

## Agent Modes

### Standalone Agent
- Creates its own full set of files from scratch
- Independent operation, self-contained reporting
- Lives under `standalone-agents/{node-id}/`

### Project-Joined Agent
- **Seeds** local files from the project rollup on join
- Maintains its own copy, writes status updates independently
- Contributions flow back up to rollup on cadence or trigger
- Lives under `projects/{project-name}/agents/{node-id}/`

**Seeding procedure on project join:**
1. Read `projects/{project}/rollup/status.yaml` and `status.mermaid`
2. Copy rollup files to own agent folder as starting state
3. Update `node` field in `status.yaml` to own node ID
4. Begin normal polling and reporting cycle

---

## Rollup & Merge

The project rollup aggregates all agent status files into a single project-level view.

### Rollup Authority (priority order)

| Priority | Authority | Trigger |
|---|---|---|
| 1 | **User Request** | Always honored immediately |
| 2 | **Claude** | Proactive — blocker detected, stalled agent, conflict |
| 3 | **Lead Agent** | Autonomous — cadence-driven |

### Lead Agent Designation
Stored in `projects/{project}/rollup/status.yaml` under `lead_agent: {node-id}`.

**Lead agent responsibilities:**
- Read all sibling agent `status.yaml` and `status.mermaid` files
- Merge into project rollup files
- Write rollup on cadence (recommended: every 5 minutes or after own task completion)
- Never assign rollup writes to other agents

**Failover:** If lead agent goes offline or stalls, Claude or user can:
1. Trigger rollup directly by reading all agent files and synthesizing
2. Reassign lead by writing new `lead_agent` value to rollup `status.yaml` via `comms.yaml`

### Merge Logic
- **Mermaid rollup:** Union of all agent task graphs, collapsed to project-level milestones only
- **YAML rollup:** Aggregate — tasks done/in-progress/blocked per agent, overall health, latest timestamp

### Collision Handling
If two agents write the same file simultaneously:
- Append timestamp suffix: `status-20260429T091500.yaml`
- Lead agent resolves on next rollup cycle
- Git merge handles conflicts automatically on commit

---

## Reading Status

When the user asks "what's the status?" or "how is project X doing?":

1. Locate the project rollup folder: `projects/{project}/rollup/`
2. Read `status.mermaid` — summarize the visual task flow
3. Read `status.yaml` — extract health, blockers, last updated, lead agent
4. If more detail needed, read individual agent `status.yaml` files
5. Synthesize into a plain-language summary for the user
6. Flag any agents that haven't updated in >30 minutes as potentially stalled

**When reading for Claude's own awareness (not user-facing):**
- Focus on `current_task`, `blockers`, and `status` fields
- Check `last_updated` timestamps across all agents
- Note any `comms.yaml` entries with `status: pending` — these may need attention

---

## Writing Directives

When the user asks Claude to send a command to an agent:

1. Identify target: specific node, all nodes in a project, or project-wide
2. Write to `agents/{node-id}/comms.yaml` (node-specific) or `rollup/comms.yaml` (project-wide)
3. Use the Comms YAML format (see [references/formats.md](references/formats.md))
4. Commit the file: `git add . && git commit -m "directive: {task-name} → {node-id}"`
5. Confirm to user: "Directive written. Agent will pick it up within 15–20 seconds."

**Never write secret values into comms.yaml.** Reference slugs only if credentials are needed.

---

## Layman Secrets

The Layman Secrets pattern keeps all credential values out of git and off disk.

### Core Rule
`secrets.yaml` contains **slug references and metadata only** — never actual values.
Actual values live in the OS native credential vault, keyed by slug name.

### OS Vault by Platform
| OS | Vault | Python library |
|---|---|---|
| Windows | Windows Credential Manager | `keyring` |
| macOS | Keychain | `keyring` |
| Linux | SecretService / libsecret | `keyring` |

### Slug Naming Convention
```
{PROJECT_PREFIX}_{SERVICE_NAME}_{KEY_TYPE}

Examples:
  ANTHROPIC_API_KEY                    ← global default
  PROJECT_ALPHA_ANTHROPIC_API_KEY      ← project-scoped
  PROJECT_ALPHA_BOX_CLIENT_ID          ← project-scoped service credential
```

Project-scoped slugs isolate billing, rate limits, and access per project.

### Reading a Secret at Runtime
```python
import keyring
value = keyring.get_password("layman-secrets", "PROJECT_ALPHA_ANTHROPIC_API_KEY")
```

### Setup Utility
The `layman_secrets_setup.py` script provides a novice-friendly guided flow:
- Accepts `SLUG=value` pairs via secure in-memory terminal paste
- Stores each to OS vault via `keyring`
- Outputs a clean `secrets.yaml` with slugs and metadata only
- Nothing written to disk except the final reference file

See [references/setup.md](references/setup.md) for installation and usage instructions.

### Expiry Awareness
Always check `expires` field in `secrets.yaml`. Warn the user if any credential
expires within 30 days. Never silently fail on expired credentials.

---

## Claude's Role in This System

Claude can act as:

| Role | Actions |
|---|---|
| **Observer** | Read status files, summarize health, surface blockers |
| **Commander** | Write directives to agent comms.yaml files |
| **Rollup Agent** | Synthesize all agent files into project rollup on request |
| **Setup Assistant** | Guide user through secrets setup, folder initialization, first run |
| **Auditor** | Run `git diff` to show what changed since last check |

Claude **never** writes secret values to any file, log, or response.
Claude **always** commits changes to git after writing directives or rollup files.

---

## Common Tasks

### Initialize a new project
```bash
mkdir -p command-hub/projects/{name}/rollup
mkdir -p command-hub/projects/{name}/agents/{node-id}
cd command-hub && git init  # if not already a git repo
```
Then create starter files using formats in [references/formats.md](references/formats.md).

### Check what changed since last run
```bash
git diff HEAD~1 HEAD -- projects/{name}/
```

### Trigger a manual rollup
Read all `agents/*/status.yaml` files, merge fields, write to `rollup/status.yaml` and `rollup/status.mermaid`, then commit.

### Reassign lead agent
Write to `rollup/comms.yaml`:
```yaml
task: reassign_lead
instructions: "Set lead_agent to node-02 effective immediately."
```

---

## Reference Files

- [references/formats.md](references/formats.md) — Full YAML and Mermaid format specs with examples
- [references/setup.md](references/setup.md) — Secrets setup utility installation and usage
