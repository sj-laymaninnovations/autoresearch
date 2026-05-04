# Layman Agent Platform — Format Specifications
## Full YAML and Mermaid examples for all six file types

---

## 1. status.mermaid — High Level Kanban

Mermaid graph format. Keep nodes brief — Mermaid's syntax enforces brevity by design.
One file per agent. Rollup version collapses to project-level milestones only.

### Agent-level example
```
graph LR
  A[Fetch Data] -->|done| B[Transform]
  B -->|in_progress| C[Load to Box]
  C -->|pending| D[Report]
  D -->|pending| E[Done]
```

### Project rollup example (milestone-level only)
```
graph LR
  M1[Data Pipeline] -->|done| M2[Transform Layer]
  M2 -->|in_progress| M3[Output Delivery]
  M3 -->|pending| M4[Sign-off]
```

### Status indicators
Use `|done|`, `|in_progress|`, `|pending|`, `|blocked|` as edge labels.

---

## 2. status.yaml — Details

Full project/task status report. One file per agent. Written after each significant step.

```yaml
# Agent status report
node: node-01
project: project-alpha
timestamp: 2026-04-29T09:15:00Z
status: in_progress          # done | in_progress | pending | blocked | stalled
lead_agent: node-01          # only present in rollup/status.yaml
last_completed: Transform dataset
current_task: Load to Box
next_task: Write final report
blockers: none               # describe blocker or "none"
notes: Processed 4,200 records. ETA 8 minutes.
last_updated: 2026-04-29T09:15:00Z
```

### Project rollup status.yaml (additional fields)
```yaml
# Project rollup — aggregate across all agents
node: rollup
project: project-alpha
timestamp: 2026-04-29T09:20:00Z
lead_agent: node-01
overall_status: in_progress
agents:
  node-01:
    status: in_progress
    current_task: Load to Box
    last_updated: 2026-04-29T09:15:00Z
    blockers: none
  node-02:
    status: done
    current_task: null
    last_updated: 2026-04-29T08:45:00Z
    blockers: none
tasks_summary:
  done: 4
  in_progress: 2
  pending: 3
  blocked: 0
notes: On track. Node-01 is the bottleneck — Box load in progress.
last_updated: 2026-04-29T09:20:00Z
```

---

## 3. refs.yaml — References / Supplemental

Supporting context, links, dependencies, external resources. One file per agent or shared at rollup.

```yaml
# References and supplemental context
node: node-01
project: project-alpha
last_updated: 2026-04-29T09:00:00Z

references:
  - slug: SOURCE_DATASET
    name: Q1 Sales Export
    path: /data/exports/q1-sales-2026.csv
    notes: Raw input for transform pipeline

  - slug: BOX_OUTPUT_FOLDER
    name: Project Alpha Box Output
    path: /Box/projects/project-alpha/outputs/
    notes: Target delivery folder

dependencies:
  - node: node-02
    on: Transform complete
    notes: node-01 load step requires node-02 transform output

external_links:
  - label: Project Brief
    url: https://box.com/...
  - label: API Docs
    url: https://docs.anthropic.com
```

---

## 4. secrets.yaml — Slug References Only

**NEVER include actual values.** Slugs only. Safe to commit to git.

```yaml
# secrets.yaml — slug references only
# Actual values stored in OS credential vault (Windows Credential Manager / macOS Keychain / Linux SecretService)
# Install: pip install keyring
# Read:    keyring.get_password("layman-secrets", "SLUG_NAME")

secrets:
  - slug: ANTHROPIC_API_KEY
    name: Anthropic API Key (default)
    description: General purpose inference key
    date: 2026-04-01
    last_updated: 2026-04-01
    expires: 2027-04-01

  - slug: PROJECT_ALPHA_ANTHROPIC_API_KEY
    name: Anthropic API Key — Project Alpha
    description: Isolated billing key for Project Alpha
    date: 2026-04-15
    last_updated: 2026-04-15
    expires: 2027-04-15

  - slug: BOX_CLIENT_ID
    name: Box OAuth Client ID
    description: Box API client ID for agent filesystem access
    date: 2026-03-01
    last_updated: 2026-03-01
    expires: null

  - slug: BOX_CLIENT_SECRET
    name: Box OAuth Client Secret
    description: Box API client secret
    date: 2026-03-01
    last_updated: 2026-03-01
    expires: null
```

### Metadata fields

| Field | Required | Purpose |
|---|---|---|
| `slug` | Yes | Unique reference — matches OS credential account name |
| `name` | Yes | Human-readable label |
| `description` | No | What this credential is used for |
| `date` | Yes | Date credential was created |
| `last_updated` | Yes | Date last rotated |
| `expires` | No | Expiry date — warn if within 30 days |

### Reading secrets in Python
```python
import keyring

SERVICE = "layman-secrets"

def get_secret(slug: str) -> str:
    value = keyring.get_password(SERVICE, slug)
    if value is None:
        raise ValueError(f"Secret not found in OS vault: {slug}")
    return value

# Usage
api_key = get_secret("PROJECT_ALPHA_ANTHROPIC_API_KEY")
```

---

## 5. comms.yaml — Command Bus

Async directives written by user, Claude, or lead agent. Agent polls this file every 15–20 seconds.

### Incoming directive (written by commander)
```yaml
# comms.yaml — directive for node-01
agent: node-01
project: project-alpha
name: Load Phase Directive
description: Initiate Box load sequence for transformed dataset
date: 2026-04-29T09:00:00Z
task: load_to_box
status: pending              # pending | acknowledged | complete | failed
priority: high               # low | medium | high
instructions: |
  Load output from /transforms/dataset-v3.csv to Box folder /project-alpha/outputs/.
  Write status.yaml update on completion.
  Set status to complete when done.
```

### After agent picks up directive
Agent updates `status` field in place:
```yaml
status: acknowledged         # agent has read and is executing
# ... later ...
status: complete             # agent finished
```

### Project-wide directive (written to rollup/comms.yaml)
```yaml
# Project-wide broadcast directive
agent: all
project: project-alpha
name: Pause for Rollup
description: All agents pause and write final status before rollup
date: 2026-04-29T10:00:00Z
task: write_final_status
status: pending
priority: high
instructions: |
  Complete current step. Write status.yaml and status.mermaid.
  Set status to pending and await next directive.
```

### Lead agent reassignment directive
```yaml
agent: rollup
project: project-alpha
name: Reassign Lead Agent
description: Transfer lead agent responsibility
date: 2026-04-29T10:05:00Z
task: reassign_lead
status: pending
instructions: |
  Set lead_agent field in rollup/status.yaml to node-02.
  node-01 is unresponsive. node-02 should begin rollup cadence immediately.
```

---

## Git Commit Convention

After any write operation, commit with a structured message:

```bash
# Directive written
git add . && git commit -m "directive: load_to_box → node-01"

# Status update
git add . && git commit -m "status: node-01 in_progress load_to_box"

# Rollup
git add . && git commit -m "rollup: project-alpha 2026-04-29T10:00:00Z"

# Secrets reference update
git add . && git commit -m "secrets: add PROJECT_ALPHA_OPENAI_KEY slug"
```

---

## Collision Handling

If two agents attempt simultaneous writes:
1. Append ISO timestamp suffix: `status-20260429T091500.yaml`
2. Lead agent detects multiple files on next rollup cycle
3. Lead merges — latest `last_updated` timestamp wins per field
4. Canonical file restored to `status.yaml`, timestamped copy archived or deleted
5. Git commit records the resolution

---

## Folder Initialization Checklist

When setting up a new project or agent folder, create these files in order:

1. `secrets.yaml` — slug references (run `layman_secrets_setup.py` first)
2. `status.yaml` — initial status with `status: pending`
3. `status.mermaid` — initial task graph
4. `refs.yaml` — known references and dependencies
5. `comms.yaml` — empty or first directive

Git init and first commit:
```bash
git init
git add .
git commit -m "init: project-alpha / node-01"
```
