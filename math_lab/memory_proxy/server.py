"""
server.py — FastAPI proxy that fronts an LLM provider with shared memory.

OpenAI-compatible (/v1/chat/completions, /v1/models, /v1/messages for the
Anthropic shape used by Claude Code). Works with anything that accepts a
base-URL override:
  - Claude Code:   ANTHROPIC_BASE_URL=http://127.0.0.1:4400
  - Cursor:        Settings → "OpenAI Base URL" = http://127.0.0.1:4400/v1
  - Continue.dev:  apiBase = http://127.0.0.1:4400/v1
  - VSCode (Continue extension), Codex, AnythingLLM: same shape.

Backend providers (selected per request via the `model` field):
  - abacus/<model>      → Abacus RouteLLM (OpenAI shape)
  - anthropic/<model>   → Anthropic native API
  - openai/<model>      → OpenAI
  - azure/<model>       → Azure OpenAI
  - vertex/<model>      → Google Vertex AI
  - gemini/<model>      → Google AI Studio (Gemini)
  - <model>             → routed via Abacus RouteLLM by default

For each request we:
  1. Search memory with the user's last message → inject as a system note
  2. Forward the (possibly augmented) request to the backend
  3. On success, optionally extract & persist memory
  4. Log the round-trip to the event log

This file only implements the OpenAI-shape `/v1/chat/completions`
non-streaming path for v0. Streaming + Anthropic shape land in v0.1.
"""
from __future__ import annotations

import argparse, json, os, ssl, time, urllib.error, urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .config import describe_root, env_or_dotenv
from .logger import log_event, new_request_id
from .memory import Memory, get_backend


try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

ABACUS_BASE = "https://routellm.abacus.ai/v1"
ANTHROPIC_BASE = "https://api.anthropic.com"
OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
# Azure / Vertex are configured per-deployment (see env vars below).


def _openai_call(base: str, key: str, model: str,
                 messages: list[dict], extra: dict) -> dict:
    body = {"model": model, "messages": messages, **extra}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "memory-proxy/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def _anthropic_call(base: str, key: str, model: str,
                    messages: list[dict], extra: dict) -> dict:
    """Anthropic /v1/messages → translated to OpenAI-shape response."""
    # Anthropic separates `system` from `messages`
    sys_text = "\n\n".join(m["content"] for m in messages
                            if m.get("role") == "system" and isinstance(m.get("content"), str))
    user_assistant = [m for m in messages if m.get("role") in ("user", "assistant")]
    body: dict = {
        "model": model,
        "messages": user_assistant,
        "max_tokens": extra.get("max_tokens", 1024),
    }
    if sys_text:
        body["system"] = sys_text
    if "temperature" in extra:
        body["temperature"] = extra["temperature"]

    req = urllib.request.Request(
        f"{base}/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "memory-proxy/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
        ant = json.loads(r.read().decode("utf-8"))
    # Translate to OpenAI shape
    text = ""
    for block in ant.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    return {
        "id": ant.get("id"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ant.get("model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": ant.get("stop_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": ant.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": ant.get("usage", {}).get("output_tokens", 0),
            "total_tokens": (ant.get("usage", {}).get("input_tokens", 0)
                             + ant.get("usage", {}).get("output_tokens", 0)),
        },
    }


def route_to_backend(model_id: str, messages: list[dict], extra: dict) -> dict:
    """Pick provider from the model prefix and forward."""
    if "/" in model_id:
        prefix, model = model_id.split("/", 1)
    else:
        prefix, model = "abacus", model_id

    if prefix == "abacus":
        key = env_or_dotenv("abacus_api_key", "ABACUS_API_KEY")
        if not key:
            raise RuntimeError("abacus_api_key not configured")
        return _openai_call(ABACUS_BASE, key, model, messages, extra)

    if prefix == "openai":
        key = env_or_dotenv("OPENAI_API_KEY", "openai_api_key")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        return _openai_call(OPENAI_BASE, key, model, messages, extra)

    if prefix == "anthropic":
        key = env_or_dotenv("ANTHROPIC_API_KEY", "anthropic_api_key")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        return _anthropic_call(ANTHROPIC_BASE, key, model, messages, extra)

    if prefix == "azure":
        # Expect AZURE_OPENAI_ENDPOINT (e.g., https://my-azoai.openai.azure.com/openai/deployments/<dep>)
        # and AZURE_OPENAI_KEY. The model arg is the deployment name (already in URL).
        endpoint = env_or_dotenv("AZURE_OPENAI_ENDPOINT")
        key = env_or_dotenv("AZURE_OPENAI_KEY", "azure_openai_key")
        if not endpoint or not key:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY not configured")
        # Azure uses api-key header instead of Bearer
        body = {"messages": messages, **extra}
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/chat/completions?api-version=2024-08-01-preview",
            data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"api-key": key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
            return json.loads(r.read().decode("utf-8"))

    if prefix in ("gemini", "vertex"):
        key = env_or_dotenv("GEMINI_API_KEY", "gemini_api_key",
                            "VERTEX_API_KEY", "vertex_api_key")
        if not key:
            raise RuntimeError("GEMINI_API_KEY / VERTEX_API_KEY not configured")
        return _openai_call(GEMINI_BASE, key, model, messages, extra)

    raise RuntimeError(f"Unknown provider prefix: {prefix!r}")


# ---------------------------------------------------------------------------
# Memory injection / extraction
# ---------------------------------------------------------------------------

MEMORY_INJECT_TOP_K = int(os.environ.get("MEMORY_INJECT_K", "5"))


def inject_memory(messages: list[dict], source: str,
                  backend) -> tuple[list[dict], list[str]]:
    """Search backend for the user's last message, prepend a system note
    with the top-K hits. Returns (new_messages, hit_ids)."""
    last_user = next((m for m in reversed(messages)
                      if m.get("role") == "user"), None)
    if not last_user or not isinstance(last_user.get("content"), str):
        return messages, []
    hits = backend.search(last_user["content"], k=MEMORY_INJECT_TOP_K)
    if not hits:
        return messages, []
    # Format hits as a system-message preamble.
    bullet_lines = []
    for h in hits:
        ts = h.created_at[:10] if h.created_at else "—"
        tags = ",".join(h.tags) if h.tags else ""
        bullet_lines.append(f"- ({ts}{', '+tags if tags else ''}) {h.text}")
    note = ("Relevant context from prior conversations (auto-recalled):\n"
            + "\n".join(bullet_lines))
    new_msgs = [{"role": "system", "content": note}] + list(messages)
    return new_msgs, [h.id for h in hits]


def extract_memory(messages: list[dict], response: dict, source: str,
                   model: str, backend) -> list[str]:
    """Naive extraction: store the user's last message + the assistant's reply
    as one memory entry, tagged with the source and model. Future versions
    will use a small classifier or an LLM call to extract STRUCTURED facts."""
    last_user = next((m for m in reversed(messages)
                      if m.get("role") == "user"), None)
    if not last_user:
        return []
    user_text = last_user.get("content", "")
    if not isinstance(user_text, str):
        return []
    completion = ""
    try:
        completion = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return []
    if not user_text.strip() or not completion.strip():
        return []
    text = f"User asked: {user_text.strip()}\nAssistant said: {completion.strip()}"
    mem = Memory.make(text=text, tags=["conversation"],
                       source=source, model=model)
    backend.add(mem)
    return [mem.id]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def make_app() -> FastAPI:
    app = FastAPI(title="autoresearch memory_proxy",
                  version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    backend = get_backend()
    print(f"  {describe_root()}")
    print(f"  memory backend: {backend.name}")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "memory_backend": backend.name,
                "memory_root": str(__import__('math_lab.memory_proxy.config',
                                              fromlist=['memory_root']).memory_root())}

    @app.get("/v1/models")
    def list_models():
        # We don't enumerate provider-specific model lists here; clients
        # should know what model id to ask for. Surface a small curated set.
        return {"object": "list", "data": [
            {"id": "abacus/gpt-5", "object": "model"},
            {"id": "abacus/claude-sonnet-4-7", "object": "model"},
            {"id": "abacus/gemini-3.1-pro-preview", "object": "model"},
            {"id": "abacus/gpt-4o-mini", "object": "model"},
            {"id": "anthropic/claude-sonnet-4-7", "object": "model"},
            {"id": "openai/gpt-4o-mini", "object": "model"},
            {"id": "gemini/gemini-2.5-flash", "object": "model"},
        ]}

    @app.post("/v1/messages")
    async def anthropic_messages(req: Request):
        """Anthropic-shape endpoint — what Claude Code's SDK posts to.

        We do memory injection on the request, forward to Anthropic native
        API, and pass the response back unchanged so the SDK is happy.
        """
        rid = new_request_id()
        body = await req.json()
        source = req.headers.get("X-Tool", "claude_code")
        model = body.get("model", "claude-sonnet-4-7")
        ant_messages = body.get("messages", [])
        ant_system = body.get("system", "")

        # Convert to OpenAI-shape just for memory injection (so we reuse
        # the same `inject_memory` logic), then translate back.
        if isinstance(ant_system, str) and ant_system:
            oai_msgs = [{"role": "system", "content": ant_system}] + ant_messages
        elif isinstance(ant_system, list):
            sys_text = "\n\n".join(b.get("text", "") for b in ant_system
                                    if b.get("type") == "text")
            oai_msgs = ([{"role": "system", "content": sys_text}] if sys_text else []) + ant_messages
        else:
            oai_msgs = list(ant_messages)
        injected, mem_hits = inject_memory(oai_msgs, source, backend)

        log_event("request", request_id=rid, source=source, model=model,
                   memory_backend=backend.name, memory_hits=mem_hits,
                   payload={"messages_len": len(ant_messages),
                            "endpoint": "/v1/messages"})

        # Translate injected back to Anthropic shape: collapse all
        # `system`-role messages into a single `system` string for Anthropic.
        sys_chunks = [m["content"] for m in injected
                      if m.get("role") == "system" and isinstance(m.get("content"), str)]
        ant_msgs = [m for m in injected if m.get("role") in ("user", "assistant")]
        forward_body = dict(body)
        forward_body["messages"] = ant_msgs
        if sys_chunks:
            forward_body["system"] = "\n\n".join(sys_chunks)

        # Forward to Anthropic
        key = env_or_dotenv("ANTHROPIC_API_KEY", "anthropic_api_key")
        if not key:
            raise HTTPException(status_code=500,
                                 detail="ANTHROPIC_API_KEY not configured")
        ant_req = urllib.request.Request(
            f"{ANTHROPIC_BASE}/v1/messages",
            data=json.dumps(forward_body).encode("utf-8"), method="POST",
            headers={"x-api-key": key,
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json",
                     "User-Agent": "memory-proxy/0.1"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(ant_req, timeout=120, context=_SSL_CTX) as r:
                ant_resp_bytes = r.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:1000]
            log_event("error", request_id=rid, model=model, source=source,
                       http_status=e.code, body=err_body, endpoint="/v1/messages")
            raise HTTPException(status_code=e.code, detail=err_body)

        ant_resp = json.loads(ant_resp_bytes.decode("utf-8"))
        duration = int((time.time() - t0) * 1000)

        # Extract memory: pull assistant text from the Anthropic response
        completion = ""
        for block in ant_resp.get("content", []):
            if block.get("type") == "text":
                completion += block.get("text", "")
        oai_shape_resp = {"choices": [{"message": {"role": "assistant",
                                                     "content": completion}}]}
        new_mem_ids = extract_memory(oai_msgs, oai_shape_resp, source, model, backend)

        log_event("response", request_id=rid, source=source, model=model,
                   duration_ms=duration,
                   input_tokens=ant_resp.get("usage", {}).get("input_tokens"),
                   output_tokens=ant_resp.get("usage", {}).get("output_tokens"),
                   memory_backend=backend.name,
                   memory_added=new_mem_ids,
                   payload={"completion_preview": completion[:400]})

        return JSONResponse(ant_resp)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        rid = new_request_id()
        body = await req.json()
        source = req.headers.get("X-Tool", "unknown")
        model = body.get("model", "abacus/gpt-4o-mini")
        messages = body.get("messages", [])
        extra = {k: v for k, v in body.items()
                 if k not in ("model", "messages")}

        # Inject memory
        injected_msgs, mem_hits = inject_memory(messages, source, backend)

        log_event("request",
                   request_id=rid, source=source, model=model,
                   memory_backend=backend.name, memory_hits=mem_hits,
                   payload={"messages_len": len(messages),
                            "messages_preview": messages[-1] if messages else None})

        t0 = time.time()
        try:
            resp = route_to_backend(model, injected_msgs, extra)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:1000]
            log_event("error", request_id=rid, model=model, source=source,
                       http_status=e.code, body=err_body)
            raise HTTPException(status_code=e.code, detail=err_body)
        except Exception as e:
            log_event("error", request_id=rid, model=model, source=source,
                       error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        duration = int((time.time() - t0) * 1000)

        # Extract memory
        new_mem_ids = extract_memory(messages, resp, source, model, backend)

        usage = resp.get("usage", {}) or {}
        log_event("response",
                   request_id=rid, source=source, model=model,
                   duration_ms=duration,
                   input_tokens=usage.get("prompt_tokens"),
                   output_tokens=usage.get("completion_tokens"),
                   memory_backend=backend.name,
                   memory_added=new_mem_ids,
                   payload={"completion_preview":
                            (resp.get("choices") or [{}])[0]
                            .get("message", {}).get("content", "")[:400]})
        return JSONResponse(resp)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4400)
    args = ap.parse_args()
    app = make_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
