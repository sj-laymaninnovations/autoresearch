"""memory_proxy — universal AI proxy + shared memory layer.

Two surfaces, one memory store:

  1. LiteLLM proxy (OpenAI / Anthropic / Gemini compatible) — for tools that
     accept a base-URL override: Claude Code, Cursor, VSCode, Codex, AnythingLLM.
  2. MCP memory server — for tools that don't take a base URL but support
     MCP: Claude Desktop is the primary case.

Memory store sits on a cloud-sync path (iCloud / OneDrive / Box / Drive)
so it follows the user across devices. Backends are pluggable — current
default is `JSONLMemory` (append-only file). Future: `AgentMemory` (vector
store) and `QuipuKv`. The abstraction in `memory.py` lets us A/B compare.

All requests + responses are logged in JSONL alongside the memory file
for later analysis (memory_helpfulness, recall accuracy, latency, cost).
"""
__version__ = "0.1.0"
