---
name: layman-agent-platform
description: >
  Use this skill to set up, operate, inspect, or update a Layman Agent Awareness
  Platform: a git-backed shared-filesystem command hub for coordinating agents
  with status.yaml, status.mermaid, refs.yaml, comms.yaml, and secrets.yaml files.
  Trigger when the user mentions Layman platform, Layman Secrets, agent status,
  project rollup, command hub, command bus, node reporting, agent Kanban, comms
  directives, stalled agents, lead agents, shared filesystem coordination, OS
  vault secret slugs, or layman_secrets_setup.py.
---

# Layman Agent Platform

Use the Layman Agent Awareness Platform as a human-readable command bus over a shared filesystem, with git providing history and conflict handling. Keep secrets out of files: `secrets.yaml` contains only slugs and metadata; values live in the OS credential vault.

## First Moves

- To summarize project health, read `projects/{project}/rollup/status.yaml` and `status.mermaid`; then inspect `agents/*/status.yaml` only if more detail is needed.
- To send an instruction, write a directive to the target agent's `comms.yaml` or to `rollup/comms.yaml` for project-wide commands, then commit the change.
- To set up or audit secrets, read `references/layman-secrets.md`; use `scripts/layman_secrets_setup.py` when the user wants a guided setup utility.
- To initialize or validate file formats, read `references/formats.md` for schemas and examples.
- To explain installation/use of the setup utility, read `references/setup.md`.

## Command Hub Layout

```text
command-hub/
  projects/{project}/
    rollup/
      status.mermaid
      status.yaml
      comms.yaml
      refs.yaml
    agents/{node-id}/
      status.mermaid
      status.yaml
      comms.yaml
      refs.yaml
      secrets.yaml
  standalone-agents/{node-id}/
    status.mermaid
    status.yaml
    comms.yaml
    refs.yaml
    secrets.yaml
```

## Operating Rules

- Always check `schema_version` before parsing YAML. Current expected version is `1`.
- Never write secret values to git, markdown, YAML files, logs, terminal output, or responses. Use slug names only.
- Treat `comms.yaml` as active directives only: `pending` and `acknowledged`. Completed or failed directives older than 24 hours belong in `archive/comms-{YYYYMMDD}.yaml`.
- Determine stalls per agent: stalled when `now - last_updated > expected_cadence_s * 3`.
- When a rollup has `subscriptions`, read referenced rollups and include blockers/stalls that affect the subscribing project.
- After writing directives, rollups, status updates, refs, or secrets metadata, commit with a concise structured message.

## Status Workflow

1. Locate the rollup folder.
2. Read `status.yaml` for health, `lead_agent`, agent timestamps, blockers, and subscriptions.
3. Read `status.mermaid` for the visual flow.
4. Compare each agent's `last_updated` against its `expected_cadence_s`.
5. Summarize overall health, active work, blockers, stalled nodes, and any cross-project risks.

## Directive Workflow

1. Identify target: one node, all project nodes, rollup, or standalone agent.
2. Load `references/formats.md` if exact directive fields are needed.
3. Write a `schema_version: 1` directive with `agent`, `project`, `name`, `description`, `date`, `task`, `status`, `priority`, and `instructions`.
4. Use only secret slugs in instructions.
5. Commit the changed `comms.yaml`.

## Rollup Workflow

1. Read every `agents/*/status.yaml` and `status.mermaid`.
2. Merge statuses into `rollup/status.yaml`, preserving `schema_version: 1`.
3. Create a milestone-level `rollup/status.mermaid`; avoid copying every agent detail into the visual rollup.
4. If timestamp-suffixed collision files exist, merge latest `last_updated` values into canonical files and archive or remove resolved copies.
5. Commit with `rollup: {project} {timestamp}`.

## Secrets Workflow

- `secrets.yaml` is a reference manifest only. It is safe to commit because it contains no values.
- Slug convention: `{PROJECT_PREFIX}_{SERVICE_NAME}_{KEY_TYPE}`, for example `PROJECT_ALPHA_ANTHROPIC_API_KEY`.
- Python runtime lookup:

```python
import keyring
value = keyring.get_password("layman-secrets", "PROJECT_ALPHA_ANTHROPIC_API_KEY")
```

- Warn if any `expires` date is within 30 days.
- To create a manifest interactively, run:

```bash
python scripts/layman_secrets_setup.py
```

## Resources

- `references/formats.md`: full schema examples for `status.mermaid`, `status.yaml`, `refs.yaml`, `secrets.yaml`, and `comms.yaml`.
- `references/setup.md`: install and usage notes for the secrets setup utility.
- `references/layman-secrets.md`: conceptual one-page Layman Secrets standard.
- `scripts/layman_secrets_setup.py`: guided script that stores pasted `SLUG=value` pairs in the OS vault and writes a safe `secrets.yaml`.
