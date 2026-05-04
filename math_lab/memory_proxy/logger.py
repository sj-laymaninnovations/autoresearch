"""
logger.py — append-only structured JSONL event log.

Every request and response that flows through the proxy / MCP server gets
one line in `events.jsonl`. Co-located with `memories.jsonl` on the
cloud-sync path. Rotated daily.

Schema (one line per event):
{
  "ts":         ISO-8601 UTC,
  "kind":       "request" | "response" | "memory_add" | "memory_search" | "error",
  "source":     "claude_code" | "claude_desktop" | "cursor" | "vscode" | ...
  "model":      "claude-sonnet-4-7" | "gpt-5" | ...
  "request_id": correlates request and response for the same call,
  "duration_ms": int (response only),
  "input_tokens": int,
  "output_tokens": int,
  "memory_backend": "jsonl" | "agentmemory" | "quipu_kv",
  "memory_hits":  list of memory ids injected,
  "payload":    safe-truncated dict with messages, completion text, etc.
}

Designed to be the canonical thing we mine later when comparing
agentmemory vs quipu-kv.
"""
from __future__ import annotations

import datetime, json, os, threading, uuid
from pathlib import Path
from typing import Any, Optional

from .config import memory_root


_LOCK = threading.Lock()
_PATH_CACHE: dict[str, Path] = {}


def _today() -> str:
    return datetime.date.today().isoformat()


def event_log_path() -> Path:
    """Daily rotated path: events_YYYY-MM-DD.jsonl."""
    name = f"events_{_today()}.jsonl"
    if name not in _PATH_CACHE:
        p = memory_root() / name
        p.parent.mkdir(parents=True, exist_ok=True)
        _PATH_CACHE[name] = p
    return _PATH_CACHE[name]


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(kind: str, **fields: Any) -> None:
    """Append one line to today's log. Truncates string fields > 8 KB to keep
    the log tractable; full payloads are still recoverable via memory store."""
    rec = {
        "ts":   datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "kind": kind,
        **fields,
    }
    # Truncate huge strings but mark them as truncated
    def _truncate(v: Any, limit: int = 8192):
        if isinstance(v, str) and len(v) > limit:
            return v[:limit] + f"...<truncated {len(v) - limit} chars>"
        if isinstance(v, dict):
            return {k: _truncate(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_truncate(x) for x in v]
        return v
    rec = _truncate(rec)

    line = json.dumps(rec, default=str) + "\n"
    with _LOCK:
        with event_log_path().open("a", encoding="utf-8") as f:
            f.write(line)
