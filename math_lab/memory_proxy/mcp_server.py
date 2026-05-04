"""
mcp_server.py — Model Context Protocol server exposing the SAME memory
store as the proxy. This is the path Claude Desktop (and any MCP-aware
client) uses to read & write memories.

Tools exposed:
  memory_search(query, k=5)           — return relevant memories
  memory_add(text, tags=[])           — persist a new memory
  memory_recent(n=20)                  — last N memories chronologically
  memory_count()                       — total in store

The store is the same JSONL file the HTTP proxy uses, so memories written
by Claude Code (via the proxy) are visible to Claude Desktop (via MCP),
and vice versa. Cloud-sync moves the JSONL across machines.

Configure Claude Desktop to load this MCP server by adding to
  ~/Library/Application Support/Claude/claude_desktop_config.json :

    {
      "mcpServers": {
        "autoresearch_memory": {
          "command": "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
          "args": ["-m", "math_lab.memory_proxy.mcp_server"],
          "cwd": "/Users/seanjosiah/Documents/autoresearch"
        }
      }
    }

Then restart Claude Desktop. The four tools become available in the chat.
"""
from __future__ import annotations

import asyncio, json, sys
from typing import Any

from .memory import Memory, get_backend
from .logger import log_event


# ---------------------------------------------------------------------------
# Minimal MCP server — JSON-RPC over stdio.
# We avoid the official `mcp` SDK to keep dependencies tight; the protocol
# we implement is a small subset sufficient for tool listing + invocation.
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"


def _ok(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


TOOLS = [
    {
        "name": "memory_search",
        "description":
            "Search the user's persistent cross-device memory for relevant "
            "context. Returns up to `k` past memories most related to `query`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k":     {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_add",
        "description":
            "Persist a memory the user wants remembered across devices and "
            "across AI tools (Claude Code / Claude Desktop / Cursor / etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memory_recent",
        "description": "Return the most recent N memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            },
        },
    },
    {
        "name": "memory_count",
        "description": "Return the total number of memories in the store.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle_initialize(req_id, params):
    return _ok(req_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "autoresearch_memory", "version": "0.1.0"},
    })


def handle_list_tools(req_id, params):
    return _ok(req_id, {"tools": TOOLS})


def handle_call_tool(req_id, params, backend):
    name = params.get("name")
    args = params.get("arguments", {}) or {}
    log_event("mcp_tool", source="claude_desktop",
               tool=name, args=args)

    if name == "memory_search":
        query = args.get("query", "")
        k = int(args.get("k", 5))
        # Cross-tool search by default — the whole point of universal memory.
        # Pass source="claude_desktop" only to scope to in-tool memories.
        only_self = bool(args.get("only_self", False))
        hits = backend.search(query, k=k,
                               source=("claude_desktop" if only_self else None))
        text = "\n".join(
            f"- [{h.created_at[:10]}, score {h.score:.2f}] {h.text}"
            for h in hits
        ) if hits else "(no matching memories)"
        return _ok(req_id, {"content": [{"type": "text", "text": text}]})

    if name == "memory_add":
        text = args.get("text", "").strip()
        tags = args.get("tags", []) or []
        if not text:
            return _err(req_id, -32602, "memory_add: 'text' is required and non-empty")
        mem = Memory.make(text=text, tags=tags,
                           source="claude_desktop", model="(unknown)")
        backend.add(mem)
        return _ok(req_id, {"content": [{"type": "text",
                                          "text": f"Saved memory id={mem.id}"}]})

    if name == "memory_recent":
        n = int(args.get("n", 20))
        items = list(backend.all())[-n:]
        items.reverse()
        text = "\n".join(
            f"- [{m.created_at[:10]}] {m.text}" for m in items
        ) if items else "(no memories yet)"
        return _ok(req_id, {"content": [{"type": "text", "text": text}]})

    if name == "memory_count":
        n = sum(1 for _ in backend.all())
        return _ok(req_id, {"content": [{"type": "text",
                                          "text": f"Total memories: {n}"}]})

    return _err(req_id, -32601, f"Unknown tool: {name!r}")


async def main():
    backend = get_backend()
    # MCP runs over stdio: read JSON-RPC line-delimited (well, technically
    # framed by Content-Length headers for the formal protocol, but most
    # clients also accept newline-delimited JSON for simple servers).
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    writer_transport, _ = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(writer_transport, _, None, loop)

    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.decode("utf-8").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params", {}) or {}

        if method == "initialize":
            resp = handle_initialize(req_id, params)
        elif method == "tools/list":
            resp = handle_list_tools(req_id, params)
        elif method == "tools/call":
            resp = handle_call_tool(req_id, params, backend)
        elif method == "notifications/initialized":
            continue  # nothing to acknowledge
        elif req_id is None:
            continue  # other notifications we don't handle
        else:
            resp = _err(req_id, -32601, f"Method not implemented: {method!r}")

        writer.write((json.dumps(resp) + "\n").encode("utf-8"))
        await writer.drain()


if __name__ == "__main__":
    asyncio.run(main())
