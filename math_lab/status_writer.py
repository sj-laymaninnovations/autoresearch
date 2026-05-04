"""
status_writer.py — Layman Agent Platform status.yaml writer for node-windows.

Writes a platform-compliant status.yaml to the command-hub and commits it so
the Quipu Trellis Dashboard (and any other observer) sees live experiment
progress from node-windows in real time.

Usage (from orchestrators):
    from math_lab.status_writer import write_agent_status, write_heartbeat

    # At experiment start:
    write_agent_status(current_task="finetune: add_2d seed=42 starting")

    # At epoch checkpoint (in-loop):
    write_agent_status(current_task=f"finetune: add_2d seed=42 ep={epoch}/10000 val_em={em:.1%}")

    # At experiment end:
    write_agent_status(
        status="in_progress",
        last_completed="add_2d seed=42 → GENERALIZED 100.0% EM",
        current_task="idle — awaiting next directive",
    )

Non-blocking: git failures are logged to stderr but never raise, so a broken
git state never crashes a training run.

Path resolution (in priority order):
  1. QUIPU_COMMAND_HUB environment variable
  2. ~/command-hub  (Layman Platform default, works on macbook + windows if Box Sync mounts there)
  3. ~/Documents/autoresearch/command-hub  (macbook local fallback)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_NODE_ID  = "node-windows"
_PROJECT  = "autoresearch"


def _resolve_command_hub() -> Path:
    """Resolve the command-hub root from env var or well-known defaults."""
    env = os.environ.get("QUIPU_COMMAND_HUB")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.home() / "command-hub",
        Path.home() / "Documents" / "autoresearch" / "command-hub",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return first candidate even if it doesn't exist yet — caller will fail
    # gracefully if the path is absent.
    return candidates[0]


def _agent_status_path(command_hub: Path) -> Path:
    return command_hub / "projects" / _PROJECT / "agents" / _NODE_ID / "status.yaml"


# ---------------------------------------------------------------------------
# YAML helpers (no external dependency — PyYAML may not be installed on all nodes)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml_simple(text: str) -> dict:
    """
    Minimal YAML loader for flat + simple-list status.yaml files.
    Preserves lines it does not understand so round-trip is lossless.
    """
    try:
        import yaml  # prefer PyYAML if available
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    out: dict = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped or ":" not in stripped or stripped.startswith(" ") or stripped.startswith("-"):
            continue
        k, _, v = stripped.partition(":")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v == "" or v.lower() == "null":
            out[k] = None
        elif v.lower() == "true":
            out[k] = True
        elif v.lower() == "false":
            out[k] = False
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _dump_yaml_simple(data: dict) -> str:
    """
    Minimal YAML dumper. Produces a clean status.yaml that the Layman Platform
    Rust types can deserialize. Falls back to PyYAML if available.
    """
    try:
        import yaml

        class _Dumper(yaml.Dumper):
            pass

        # Force block style strings (no | folded unless multiline)
        _Dumper.add_representer(
            str,
            lambda dumper, data: dumper.represent_scalar(
                "tag:yaml.org,2002:str", data,
                style="|" if "\n" in data else None,
            ),
        )
        return yaml.dump(data, Dumper=_Dumper, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
    except ImportError:
        pass

    lines = [f"# Agent status — {_NODE_ID}", f"schema_version: 1"]
    scalar_keys = [
        "schema_version", "node", "project", "timestamp", "status",
        "expected_cadence_s", "host", "last_completed", "current_task",
        "next_task", "blockers", "last_updated",
    ]
    block_keys = ["notes"]

    def _scalar(v) -> str:
        if v is None:
            return "null"
        s = str(v)
        if any(c in s for c in ":#{}[]|>&*!,'\"") or s.strip() != s:
            return f'"{s}"'
        return s

    # Write in canonical field order
    written = {"schema_version"}
    for key in scalar_keys:
        if key in data and key not in written:
            lines.append(f"{key}: {_scalar(data[key])}")
            written.add(key)

    # gpu / gpus
    for gk in ("gpu", "gpus"):
        if gk in data and gk not in written:
            v = data[gk]
            if isinstance(v, list):
                lines.append(f"{gk}:")
                for item in v:
                    if isinstance(item, dict):
                        first = True
                        for ik, iv in item.items():
                            prefix = "  - " if first else "    "
                            lines.append(f"{prefix}{ik}: {_scalar(iv)}")
                            first = False
                    else:
                        lines.append(f"  - {_scalar(item)}")
            else:
                lines.append(f"{gk}: {_scalar(v)}")
            written.add(gk)

    # running_services
    if "running_services" in data and "running_services" not in written:
        svc = data["running_services"]
        if isinstance(svc, list):
            lines.append("running_services:")
            for s in svc:
                lines.append(f"  - {_scalar(s)}")
        else:
            lines.append(f"running_services: {_scalar(svc)}")
        written.add("running_services")

    # notes (block scalar)
    for key in block_keys:
        if key in data and key not in written:
            v = data[key]
            if v and "\n" in str(v):
                lines.append(f"{key}: |")
                for l in str(v).splitlines():
                    lines.append(f"  {l}")
            elif v:
                lines.append(f"{key}: {_scalar(v)}")
            written.add(key)

    # Anything else
    for key, val in data.items():
        if key not in written:
            lines.append(f"{key}: {_scalar(val)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

def write_agent_status(
    current_task: Optional[str] = None,
    status: str = "in_progress",
    last_completed: Optional[str] = None,
    next_task: Optional[str] = None,
    blockers: str = "none",
    notes: Optional[str] = None,
    node_id: str = _NODE_ID,
    project: str = _PROJECT,
    commit: bool = True,
) -> bool:
    """
    Write a full status update to the agent's status.yaml and git-commit it.

    Returns True on success, False if the write or commit failed (non-fatal —
    the caller should continue normally either way).

    Fields not passed as arguments are preserved from the existing file.
    timestamp and last_updated are always set to now.
    """
    try:
        hub = _resolve_command_hub()
        path = _agent_status_path(hub)

        # Load existing data to preserve static fields (host, gpus, services…)
        existing: dict = {}
        if path.exists():
            try:
                existing = _load_yaml_simple(path.read_text(encoding="utf-8"))
            except Exception:
                pass

        now = _now_iso()

        # Build updated data — start from existing, overlay changed fields
        data: dict = dict(existing)
        data["schema_version"] = 1
        data["node"]    = node_id
        data["project"] = project
        data["timestamp"]    = now
        data["last_updated"] = now
        data["status"]  = status

        if current_task  is not None: data["current_task"]   = current_task
        if last_completed is not None: data["last_completed"] = last_completed
        if next_task      is not None: data["next_task"]      = next_task
        if blockers       is not None: data["blockers"]       = blockers
        if notes          is not None: data["notes"]          = notes

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump_yaml_simple(data), encoding="utf-8")

        if commit:
            _git_commit(hub, path,
                        f"status: {node_id} {status} {current_task or ''}".strip())

        return True

    except Exception as exc:
        print(f"[status_writer] WARNING: failed to write status: {exc}", file=sys.stderr)
        return False


def write_heartbeat(
    node_id: str = _NODE_ID,
    project: str = _PROJECT,
) -> bool:
    """
    Minimal heartbeat — updates only timestamp and last_updated.
    All other fields are preserved from the existing file.
    Used by the standalone heartbeat daemon.
    """
    try:
        hub  = _resolve_command_hub()
        path = _agent_status_path(hub)

        existing: dict = {}
        if path.exists():
            existing = _load_yaml_simple(path.read_text(encoding="utf-8"))

        now = _now_iso()
        existing["timestamp"]    = now
        existing["last_updated"] = now
        # Ensure mandatory fields present even on first write
        existing.setdefault("schema_version", 1)
        existing.setdefault("node",    node_id)
        existing.setdefault("project", project)
        existing.setdefault("status",  "in_progress")
        existing.setdefault("expected_cadence_s", 60)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump_yaml_simple(existing), encoding="utf-8")
        _git_commit(hub, path, f"heartbeat: {node_id} {now}")
        return True

    except Exception as exc:
        print(f"[status_writer] WARNING: heartbeat failed: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Git helper
# ---------------------------------------------------------------------------

def _git_commit(hub: Path, changed_file: Path, message: str) -> None:
    """
    Stage the changed file and commit it.
    Silently skips if git is not available, hub is not a repo, or nothing to commit.
    """
    try:
        rel = str(changed_file.relative_to(hub))
        subprocess.run(
            ["git", "add", rel],
            cwd=hub, check=True,
            capture_output=True, timeout=10,
        )
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=hub,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode not in (0, 1):  # 1 = nothing to commit
            print(f"[status_writer] git commit rc={result.returncode}: {result.stderr.strip()}",
                  file=sys.stderr)
    except FileNotFoundError:
        pass  # git not on PATH on this machine
    except Exception as exc:
        print(f"[status_writer] git commit failed: {exc}", file=sys.stderr)
