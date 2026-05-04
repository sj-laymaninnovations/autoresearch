"""
memory.py — abstract memory interface + concrete backends.

Two operations the proxy/MCP needs:
  search(query, k)  → list[Memory]    # retrieve relevant past memories
  add(text, tags)   → Memory          # persist a new memory

Backends:
  JSONLMemory      : append-only JSONL on disk, naive substring/keyword search.
                     Always works, fully cloud-sync friendly, low fidelity.
  AgentMemoryStore : (TODO) wraps `agentmemory` (chromadb-backed semantic search).
  QuipuKvStore     : (TODO) wraps quipu-kv when ready.

The active backend is chosen by env var MEMORY_PROXY_BACKEND
(default: "jsonl"). All backends share the same JSONL append log so the
record is portable — when we swap backends we don't lose history; we
re-index.
"""
from __future__ import annotations

import datetime, hashlib, json, os, re, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

from .config import memory_root


@dataclass
class Memory:
    id: str                    # stable hash, used for dedup across devices
    text: str                  # the memory content
    tags: list[str] = field(default_factory=list)
    source: str = ""           # which tool produced it (claude_code, claude_desktop, ...)
    model: str = ""            # which LLM was in the conversation
    created_at: str = ""       # ISO-8601
    score: float = 0.0         # populated only on retrieval

    @classmethod
    def make(cls, text: str, tags: list[str] | None = None,
             source: str = "", model: str = "") -> "Memory":
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return cls(
            id=h,
            text=text.strip(),
            tags=tags or [],
            source=source,
            model=model,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
        )


class MemoryBackend:
    """Abstract base. Subclasses must implement add/search."""

    name = "abstract"

    def add(self, mem: Memory) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 5,
               source: Optional[str] = None) -> list[Memory]:
        raise NotImplementedError

    def all(self) -> Iterable[Memory]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JSONL backend — append-only, cloud-sync friendly, primary canonical store
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class JSONLMemory(MemoryBackend):
    name = "jsonl"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (memory_root() / "memories.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        # Lightweight in-process cache. Reload on every search keeps multi-device
        # writes consistent at the cost of an O(N) scan per query — fine for
        # the foreseeable size (tens of thousands).
        self._cache: list[Memory] = []
        self._mtime: float = 0.0

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._mtime and self._cache:
            return
        out: list[Memory] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(Memory(**{k: v for k, v in obj.items()
                                          if k in Memory.__annotations__}))
                except Exception:
                    continue
        self._cache = out
        self._mtime = mtime

    def add(self, mem: Memory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(mem)) + "\n")
        self._mtime = 0.0  # invalidate cache

    def search(self, query: str, k: int = 5,
               source: Optional[str] = None) -> list[Memory]:
        self._refresh()
        if not self._cache:
            return []
        q_terms = set(t.lower() for t in _WORD_RE.findall(query))
        if not q_terms:
            return []
        scored: list[tuple[float, Memory]] = []
        for m in self._cache:
            if source and m.source and m.source != source:
                continue
            terms = set(t.lower() for t in _WORD_RE.findall(m.text))
            if not terms:
                continue
            inter = q_terms & terms
            if not inter:
                continue
            # Jaccard-ish, slightly tilted toward longer overlap
            score = len(inter) / (len(q_terms) + len(terms) - len(inter))
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, m in scored[:k]:
            m.score = score
            out.append(m)
        return out

    def all(self) -> Iterable[Memory]:
        self._refresh()
        return list(self._cache)


# ---------------------------------------------------------------------------
# agentmemory backend — TODO. Wraps `agentmemory` for vector search.
# ---------------------------------------------------------------------------

class AgentMemoryStore(MemoryBackend):
    """Vector-search backend via the `agentmemory` package.

    Skeleton for now — real implementation lands when we install agentmemory
    and design the swap. The constructor still appends to the same JSONL log
    via a co-instantiated JSONLMemory so we don't lose canonical record.
    """
    name = "agentmemory"

    def __init__(self, path: Optional[Path] = None):
        self.canonical = JSONLMemory(path)
        try:
            import agentmemory  # type: ignore
            self.am = agentmemory
        except ImportError:
            raise RuntimeError(
                "agentmemory not installed. `pip install agentmemory` first.")
        # TODO: configure persistent storage path inside memory_root()

    def add(self, mem: Memory) -> None:
        self.canonical.add(mem)
        # TODO: self.am.create_memory(...)

    def search(self, query: str, k: int = 5,
               source: Optional[str] = None) -> list[Memory]:
        # TODO: real semantic search via self.am.get_memories
        return self.canonical.search(query, k=k, source=source)

    def all(self):
        return self.canonical.all()


# ---------------------------------------------------------------------------
# Quipu-KV backend — TODO. Drops in once quipu-kv is ready for the
# comparison study against agentmemory.
# ---------------------------------------------------------------------------

class QuipuKvStore(MemoryBackend):
    name = "quipu_kv"

    def __init__(self, path: Optional[Path] = None):
        self.canonical = JSONLMemory(path)
        # TODO: instantiate quipu-kv client/store

    def add(self, mem: Memory) -> None:
        self.canonical.add(mem)
        # TODO: write to quipu-kv

    def search(self, query: str, k: int = 5,
               source: Optional[str] = None) -> list[Memory]:
        return self.canonical.search(query, k=k, source=source)

    def all(self):
        return self.canonical.all()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backend() -> MemoryBackend:
    name = os.environ.get("MEMORY_PROXY_BACKEND", "jsonl").lower()
    if name == "jsonl":
        return JSONLMemory()
    if name in ("agentmemory", "am"):
        return AgentMemoryStore()
    if name in ("quipu", "quipu_kv", "quipukv"):
        return QuipuKvStore()
    raise ValueError(f"Unknown MEMORY_PROXY_BACKEND={name!r}. "
                     f"Options: jsonl | agentmemory | quipu_kv")
