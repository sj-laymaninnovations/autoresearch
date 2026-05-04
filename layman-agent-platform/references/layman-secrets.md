# Layman Secrets
## OS-Native Credential Management Pattern
*v2 — Layman Innovations Open Standard*

> Store secrets where they belong: in your operating system's native vault — not in files, not in third-party services, not in git.

---

## The Problem

Open source and private projects routinely leak secrets by storing them in `.env` files, committing them to git, or relying on third-party secret managers that introduce new attack surfaces. Even experienced developers accidentally litter passphrases, API keys, and credentials into repos.

---

## The Pattern

### 1. `secrets.yaml` — Reference Manifest Only

Create a `secrets.yaml` in your repo listing **slug references only** — never actual values. Each entry includes metadata to support credential lifecycle management.

```yaml
# secrets.yaml
secrets:
  - slug: ANTHROPIC_API_KEY
    name: Anthropic API Key (default)
    description: General purpose inference key
    date: 2026-04-01
    last_updated: 2026-04-01
    expires: 2027-04-01

  - slug: PROJECT_ONE_ANTHROPIC_API_KEY
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
```

This file is **safe to commit**. It contains zero sensitive information.

---

### 2. OS Credential Store — Actual Values Live Here

Store actual secret values in your OS native vault using the slug as the account/credential name:

| OS | Native Vault |
|---|---|
| Windows | Windows Credential Manager |
| macOS | Keychain |
| Linux | `secret-tool` / `pass` |

Agents and scripts query the OS credential store at runtime using the slug name. The actual value is **never written to disk, never committed to git, never exposed in logs or reports.**

---

### 3. Project-Scoped Secrets

Prefix slugs with a project identifier to keep billing, rate limits, and access isolated:

```yaml
- slug: PROJECT_ALPHA_ANTHROPIC_API_KEY
- slug: PROJECT_BETA_ANTHROPIC_API_KEY
- slug: PROJECT_ALPHA_BOX_CLIENT_ID
```

Each project gets its own credentials. One project exhausting tokens or hitting rate limits does not impact others.

---

### 4. Secrets Metadata Fields

| Field | Purpose |
|---|---|
| `slug` | Unique reference name — matches OS credential account name |
| `name` | Human-readable label |
| `description` | Context for what this credential is used for |
| `date` | Date credential was created |
| `last_updated` | Date credential was last rotated |
| `expires` | Expiration date — drives rotation awareness |

The `expires` field forces credential rotation into the protocol. Agents or Claude can surface expiring credentials proactively.

---

## Benefits

- **Zero secrets in git** — repo can be fully public
- **Zero third-party risk** — no external secret manager to breach
- **No encryption overhead** — OS vault handles protection natively
- **Non-technical friendly** — just create an OS credential with the right slug name
- **Project isolation** — separate billing, rate limits, and access per project
- **Lifecycle awareness** — expiry metadata drives rotation discipline
- **De-risks open source** — contributors follow the same pattern, nobody commits values accidentally

---

## Agent Implementation

Agents follow a simple lookup pattern at startup:

1. Read `secrets.yaml` to get required slug names
2. Query OS credential store for each slug
3. Load values into runtime environment
4. **Never** write values to logs, markdown reports, Comms files, or git commits
5. Check `expires` field — surface warnings if credential expires within 30 days

---

## Adoption Path

1. Publish one-pager + agent implementation guide (this document)
2. Identify OSS projects with secrets-in-git risk
3. Submit PR with pattern applied — docs included
4. Let community decide — no sales pitch needed

---

## Launch

- Layman Innovations site — full writeup
- LinkedIn — professional reach
- X — visibility
- Reddit — community engagement
- Medium — broader tech audience

---

*A Layman Innovations open standard. Simple by design. Secure by default.*
