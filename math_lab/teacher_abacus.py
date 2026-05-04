"""
teacher_abacus.py — thin client around Abacus.AI's RouteLLM (OpenAI-compatible).

Loads the API key from `math_lab/.env` (env var name: `abacus_api_key`),
exposes a `Teacher` class with a synchronous `chat(messages, ...)` method.

Designed for batch synthetic-data generation: handles transient errors,
applies polite retry-with-backoff, never prints the key.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path
from typing import Optional

import ssl
import urllib.request
import urllib.error

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

DEFAULT_BASE = "https://routellm.abacus.ai/v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _load_env_file(path: Path) -> dict:
    """Minimal .env loader — handles `KEY=VALUE` lines, ignores blanks/comments."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        # Strip surrounding quotes if present
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def _resolve_key() -> str:
    """Load from env, then fall back to math_lab/.env. Never prints the value."""
    for var in ("abacus_api_key", "ABACUS_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    here = Path(__file__).parent
    for envp in (here / ".env", here.parent / ".env"):
        env = _load_env_file(envp)
        for var in ("abacus_api_key", "ABACUS_API_KEY"):
            if env.get(var):
                return env[var]
    raise RuntimeError(
        "abacus_api_key not found in environment or math_lab/.env. "
        "Add `abacus_api_key=...` to math_lab/.env."
    )


class Teacher:
    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE,
                 timeout: float = 60.0,
                 max_retries: int = 5):
        self.api_key = _resolve_key()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict],
             temperature: float = 0.7,
             max_tokens: int = 600) -> str:
        """Returns the assistant's content string. Raises after max_retries."""
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Some Cloudflare-fronted endpoints reject the default
                # `Python-urllib/X.Y` UA. Spoof a generic one.
                "User-Agent": "autoresearch-teacher/1.0",
            },
        )
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return content
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    backoff = min(60.0, 2 ** attempt)
                    time.sleep(backoff)
                    continue
                # 4xx other than 429 → fast fail with safe message
                raise RuntimeError(f"Teacher HTTP {e.code} {e.reason}") from e
            except urllib.error.URLError as e:
                last_err = e
                time.sleep(min(60.0, 2 ** attempt))
                continue
        raise RuntimeError(f"Teacher exhausted {self.max_retries} retries: {last_err}")


if __name__ == "__main__":
    # Smoke
    t = Teacher()
    out = t.chat([
        {"role": "user",
         "content": "Reply with exactly the word OK and nothing else."}
    ], temperature=0.0, max_tokens=10)
    print(f"Teacher OK: model={t.model!r}, response={out.strip()!r}")
