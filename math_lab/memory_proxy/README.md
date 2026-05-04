# memory_proxy — universal AI proxy + shared memory

Two surfaces, one canonical memory store:

| Surface | What it is | Who uses it |
|---|---|---|
| **HTTP proxy** at `http://127.0.0.1:4400` | OpenAI / Anthropic / Gemini compatible | Claude Code, Cursor, VSCode (Continue), Codex, AnythingLLM, anything that takes a base URL |
| **MCP server** over stdio | Model Context Protocol — exposes `memory_search`, `memory_add`, `memory_recent`, `memory_count` tools | Claude Desktop |

Both write to the **same `memories.jsonl` file** on a cloud-sync directory (auto-detected: OneDrive / Box / Google Drive / iCloud). Memory follows you across devices because the cloud sync ships the JSONL.

Every request and response is also logged to `events_YYYY-MM-DD.jsonl` for later analysis. Keeping that log lets us A/B compare memory backends (jsonl vs agentmemory vs quipu-kv) when those land.

## Quick start

```bash
# 1. Configure where the canonical memory dir lives. Pick a path that's
#    cloud-synced on every machine you work on — same path on all.
export MEMORY_PROXY_PATH="$HOME/Library/Mobile Documents/com~apple~CloudDocs/autoresearch"

# 2. (Optional) override backend. Default is "jsonl".
export MEMORY_PROXY_BACKEND=jsonl     # or: agentmemory | quipu_kv (when ready)

# 3. Start the HTTP proxy (4400 by default).
cd ~/Documents/autoresearch
python3 -m math_lab.memory_proxy.server

# 4. (Optional) for Claude Desktop — wire up the MCP server (see below).
```

## Provider routing

The proxy decides where to send a request based on the model id:

| client sends | provider | env var(s) |
|---|---|---|
| `abacus/<model>` (or any model with no prefix) | Abacus RouteLLM | `abacus_api_key` |
| `anthropic/<model>` | Anthropic native | `ANTHROPIC_API_KEY` |
| `openai/<model>` | OpenAI | `OPENAI_API_KEY` |
| `azure/<deployment>` | Azure OpenAI | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY` |
| `gemini/<model>` or `vertex/<model>` | Gemini / Vertex (OpenAI-compat endpoint) | `GEMINI_API_KEY` |

Keys can live in `math_lab/.env` (already used for `abacus_api_key`).

## Configuring each AI tool

### Claude Code (CLI)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4400
# Claude Code passes the model through; pick anthropic/<model> when chatting
```

### Cursor

Settings → AI → Custom OpenAI base URL → `http://127.0.0.1:4400/v1`. Set any non-empty API key.

### VSCode + Continue.dev

In `~/.continue/config.json`:
```json
{ "models": [
  { "title": "abacus-claude",
    "provider": "openai",
    "model": "abacus/claude-sonnet-4-7",
    "apiBase": "http://127.0.0.1:4400/v1",
    "apiKey": "any-non-empty-string" }
]}
```

### AnythingLLM

Settings → LLM Provider → "Generic OpenAI" → Base URL `http://127.0.0.1:4400/v1`, model `abacus/gpt-5` (or whichever).

### Codex

```bash
export OPENAI_API_BASE=http://127.0.0.1:4400/v1
```

### Claude Desktop (MCP)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (create if missing):
```json
{
  "mcpServers": {
    "autoresearch_memory": {
      "command": "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
      "args": ["-m", "math_lab.memory_proxy.mcp_server"],
      "cwd": "/Users/seanjosiah/Documents/autoresearch",
      "env": {
        "MEMORY_PROXY_PATH": "/Users/seanjosiah/Library/Mobile Documents/com~apple~CloudDocs/autoresearch",
        "MEMORY_PROXY_BACKEND": "jsonl"
      }
    }
  }
}
```
Quit + relaunch Claude Desktop. The four tools (`memory_search`, `memory_add`, `memory_recent`, `memory_count`) appear in the tool drawer.

## Memory backends

`MEMORY_PROXY_BACKEND` selects the active backend:

| value | what it does | status |
|---|---|---|
| `jsonl` (default) | append-only JSONL with naive Jaccard token search | ✅ ready |
| `agentmemory` | Chroma-backed semantic search via the `agentmemory` package | 🚧 skeleton — wraps JSONL canonical store, swap-in pending |
| `quipu_kv` | once quipu-kv is available | 🚧 skeleton |

Every backend wraps `JSONLMemory` as the canonical record. That means we never lose data when we swap — we re-index. The plan is to run the proxy with all three and capture event logs to compare retrieval quality, latency, and downstream completion quality.

## Files in the memory dir

```
<cloud-sync>/autoresearch_memory/
  memories.jsonl              ← canonical memory record (read+write by everything)
  events_2026-04-28.jsonl     ← daily structured event log
  events_2026-04-29.jsonl
  ...
```

## Logging schema (event log)

Every line is one JSON object. Common fields: `ts`, `kind`, `request_id`, `source`, `model`, `memory_backend`. Per-kind fields:

- `kind="request"`: `memory_hits` (list of memory ids injected), `payload.messages_preview`
- `kind="response"`: `duration_ms`, `input_tokens`, `output_tokens`, `memory_added`, `payload.completion_preview`
- `kind="memory_add"` / `kind="memory_search"`: backend-internal events
- `kind="mcp_tool"`: tool invocations from Claude Desktop
- `kind="error"`: exception details

This is the corpus we mine for the memory-backend comparison study.
