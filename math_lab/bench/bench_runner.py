"""
bench_runner.py — multi-platform tokens/s harness for math_lab/serve.py.

Hits each server's /v1/chat/completions in streaming mode, parses the SSE
usage payload that serve.py emits in its final frame, and writes one TSV
row per request. Prints a median(tok/s) rollup at the end.

Stdlib only (urllib, json, statistics, argparse).

Usage:
    python bench_runner.py \\
        --servers servers.json --prompts prompts.json \\
        --out runs/2026-05-09.tsv --repeats 5 --warmup 1
"""

import argparse, json, statistics, subprocess, sys, time
import urllib.request, urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TSV_COLS = [
    "timestamp", "git_commit", "platform", "host", "device", "gpu_name",
    "cpu_cores", "ram_total_gb", "param_count",
    "model_id", "prompt", "rep",
    "tokens", "elapsed_s", "tokens_per_sec", "note",
]


def git_commit_short() -> str:
    """Local repo HEAD short hash. Assumes server is running the same source."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def fetch_system(base: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(f"{base}/v1/system", timeout=timeout) as r:
        return json.loads(r.read())


def stream_one(base: str, model: str, prompt: str, max_tokens: int | None,
               timeout: int = 600) -> dict | None:
    """Send one streaming chat completion; return the final usage payload."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_usage = None
    buf = b""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for chunk in r:
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                if not frame.startswith(b"data:"):
                    continue
                payload = frame[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "usage" in obj:
                    last_usage = obj["usage"]
    return last_usage


def write_row(f, row: dict) -> None:
    cells = []
    for c in TSV_COLS:
        v = row.get(c, "")
        if isinstance(v, str):
            v = v.replace("\t", " ").replace("\n", " ")
        cells.append(str(v))
    f.write("\t".join(cells) + "\n")
    f.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True, type=Path,
                    help="JSON list of {name, base}")
    ap.add_argument("--prompts", required=True, type=Path,
                    help="JSON list of {model, prompt, max_tokens?}")
    ap.add_argument("--out", required=True, type=Path,
                    help="TSV output path (parent dir created)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="counted requests per (server, model)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded requests before counting (per model)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override per-prompt max_tokens for all rows")
    args = ap.parse_args()

    servers = json.loads(args.servers.read_text())
    prompts = json.loads(args.prompts.read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    commit = git_commit_short()
    medians: dict[tuple[str, str], list[float]] = defaultdict(list)

    with args.out.open("w") as f:
        f.write("\t".join(TSV_COLS) + "\n")

        for srv in servers:
            name, base = srv["name"], srv["base"].rstrip("/")
            try:
                sysinfo = fetch_system(base)
            except Exception as e:
                print(f"[{name}] /v1/system failed: {e}", file=sys.stderr)
                continue

            print(f"[{name}] host={sysinfo.get('hostname','?')} "
                  f"device={sysinfo.get('device','?')} "
                  f"gpu={sysinfo.get('gpu_name','-')} "
                  f"cores={sysinfo.get('cpu_cores','-')}",
                  file=sys.stderr, flush=True)

            meta = {
                "platform":     name,
                "host":         sysinfo.get("hostname", ""),
                "device":       sysinfo.get("device", ""),
                "gpu_name":     sysinfo.get("gpu_name", ""),
                "cpu_cores":    sysinfo.get("cpu_cores", ""),
                "ram_total_gb": sysinfo.get("ram_total_gb", ""),
                "param_count":  sysinfo.get("param_count", ""),
            }

            for p in prompts:
                model = p["model"]
                prompt = p["prompt"]
                mt = args.max_tokens if args.max_tokens is not None else p.get("max_tokens")

                for _ in range(args.warmup):
                    try:
                        stream_one(base, model, prompt, mt)
                    except Exception as e:
                        print(f"  warmup {model}: {e}", file=sys.stderr)

                for rep in range(args.repeats):
                    row = {**meta, "git_commit": commit,
                           "model_id": model, "prompt": prompt, "rep": rep}
                    try:
                        u = stream_one(base, model, prompt, mt)
                        if not u:
                            row["note"] = "no usage"
                        else:
                            row["tokens"]         = u.get("completion_tokens", "")
                            row["elapsed_s"]      = u.get("elapsed_s", "")
                            row["tokens_per_sec"] = u.get("tokens_per_sec", "")
                            row["note"]           = ""
                            tps = u.get("tokens_per_sec")
                            if isinstance(tps, (int, float)):
                                medians[(name, model)].append(float(tps))
                    except Exception as e:
                        row["note"] = f"error: {e}"
                    row["timestamp"] = datetime.now(timezone.utc).isoformat()
                    write_row(f, row)

                tps_list = medians.get((name, model), [])
                med_str = f"{statistics.median(tps_list):.1f}" if tps_list else "—"
                print(f"  {model:<24}  median tok/s = {med_str}",
                      file=sys.stderr, flush=True)

    print(file=sys.stderr)
    print("=== rollup (median tok/s) ===", file=sys.stderr)
    by_platform = sorted({k[0] for k in medians})
    by_model    = sorted({k[1] for k in medians})
    header = ["model"] + by_platform
    print("\t".join(header), file=sys.stderr)
    for m in by_model:
        cells = [m]
        for p in by_platform:
            vals = medians.get((p, m), [])
            cells.append(f"{statistics.median(vals):.1f}" if vals else "—")
        print("\t".join(cells), file=sys.stderr)

    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
