"""
serve.py — minimal OpenAI-compatible inference server for MathGPT skills.

Exposes /v1/chat/completions, /v1/completions, /v1/models on port 4321 by
default. Each "model" in the registry is a (skill_name, checkpoint_path)
pair pointing at an EM-peak checkpoint.

Behavior on a chat request: takes the *last* user message verbatim, runs it
through the canonical PROMPT_TEMPLATE as the math problem, returns the
model's generated CoT + answer. Multi-turn context is ignored — the model
is char-level and trained only on single-problem completions.

Usage:
    python math_lab/serve.py --host 0.0.0.0 --port 4321 --device cuda
"""
import argparse, json, sys, time, uuid, os, platform, multiprocessing
from pathlib import Path
from typing import Optional, List, Iterable

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    MathGPT, GPTConfig, PROMPT_TEMPLATE,
    char_encode, char_decode, EOS_ID, VOCAB_SIZE,
)
from eval_canonical import load_model, extract_answer


# ---------------------------------------------------------------------------
# Skill registry — EM-peak checkpoints. Edit paths to match deployment.
# Keys are the model id strings clients send in {"model": "..."}.
# ---------------------------------------------------------------------------
DEFAULT_SKILLS: dict[str, dict] = {

    # ── V1 — original 5M-step direct-answer models ───────────────────────────
    "arith-generalist-v1": {
        "ckpt": "results/checkpoints_v1/v1_arith_generalist_best.pt",
        "desc": "V1 Arithmetic generalist — add/sub/mul/div mixed, direct answer",
        "max_new_default": 80,
    },
    "curriculum-v1": {
        "ckpt": "results/checkpoints_v1/v1_curriculum_p3_best.pt",
        "desc": "V1 Curriculum phase-3 — progressive difficulty, direct answer",
        "max_new_default": 80,
    },
    "word-problem-v1": {
        "ckpt": "results/checkpoints_v1/v1_wordproblem_best.pt",
        "desc": "V1 Word problems — e.g. 'Sara has 3 apples and gets 4 more'",
        "max_new_default": 120,
    },

    # ── V2 — example-first CoT, answer-at-end ────────────────────────────────
    "add_1d-v2": {
        "ckpt": "results/checkpoints_v2/v2_add1d_best.pt",
        "desc": "V2 Add 1-digit — CoT e.g. '7 + 8' | 100% EM",
        "max_new_default": 180,
    },
    "add_2d-v2": {
        "ckpt": "results/checkpoints_v2/v2_add2d_best.pt",
        "desc": "V2 Add 2-digit — CoT e.g. '47 + 58' | 100% EM",
        "max_new_default": 200,
    },
    "mul_1d-v2": {
        "ckpt": "results/checkpoints_v2/v2_mul1d_best.pt",
        "desc": "V2 Multiply 1-digit — CoT e.g. '8 * 7' | 100% EM",
        "max_new_default": 180,
    },
    "mul_2d-v2": {
        "ckpt": "results/checkpoints_v2/v2_mul2d_best.pt",
        "desc": "V2 Multiply 2-digit — CoT e.g. '23 * 46' | 40% EM",
        "max_new_default": 250,
    },
    "div_1d-v2": {
        "ckpt": "results/checkpoints_v2/v2_div1d_best.pt",
        "desc": "V2 Divide 1-digit — CoT e.g. '56 / 7' | 60% EM",
        "max_new_default": 220,
    },
    "sub_2d_borrow-v2": {
        "ckpt": "results/checkpoints_v2/v2_sub2d_borrow_best.pt",
        "desc": "V2 Subtract 2-digit w/ borrow — CoT e.g. '82 - 47' | 100% EM",
        "max_new_default": 220,
    },
    "sub_2d_no_borrow-v2": {
        "ckpt": "results/checkpoints_v2/v2_sub2d_no_borrow_best.pt",
        "desc": "V2 Subtract 2-digit no borrow — CoT e.g. '87 - 34' | 100% EM",
        "max_new_default": 180,
    },
    "mod_1d-v2": {
        "ckpt": "results/checkpoints_v2/v2_mod1d_best.pt",
        "desc": "V2 Modulo 1-digit — CoT e.g. '17 % 5' | 20% EM",
        "max_new_default": 200,
    },

    # ── V3 — expanded datasets, higher epochs ────────────────────────────────
    "add_3d-v3": {
        "ckpt": "results/checkpoints_v3/v3_add3d_x2_best.pt",
        "desc": "V3 Add 3-digit — CoT e.g. '920 + 138' | 100% EM",
        "max_new_default": 220,
    },
    "div_1d_exp-v3": {
        "ckpt": "results/checkpoints_v3/v3_div1d_expanded_best.pt",
        "desc": "V3 Divide expanded — q up to 12, 8 variants | 60% EM",
        "max_new_default": 230,
    },
    "alg_1step-v3": {
        "ckpt": "results/checkpoints_v3/v3_alg1step_best.pt",
        "desc": "V3 Algebra 1-step — CoT e.g. '7x = 56' | 20% EM (use v4)",
        "max_new_default": 200,
    },
    "alg_2step-v3": {
        "ckpt": "results/checkpoints_v3/v3_alg2step_best.pt",
        "desc": "V3 Algebra 2-step — CoT e.g. '2x + 3 = 11' | 100% EM",
        "max_new_default": 220,
    },

    # ── V4 — curriculum-sequenced, wider coverage ─────────────────────────────
    "div_1d-v4": {
        "ckpt": "results/checkpoints_v4/v4_div1d_best.pt",
        "desc": "V4 Divide — curriculum-tiered, d≤12 q≤15 | 75% EM core / 33% q>9",
        "max_new_default": 250,
    },
    "alg_1step-v4": {
        "ckpt": "results/checkpoints_v4/v4_alg1step_best.pt",
        "desc": "V4 Algebra 1-step — ax=b, a≤15, neg x | 100% EM",
        "max_new_default": 220,
    },
    "alg_2step_hard-v4": {
        "ckpt": "results/checkpoints_v4/v4_alg2step_hard_best.pt",
        "desc": "V4 Algebra 2-step hard — neg solutions, both sides | 100% pos / 0% both-sides",
        "max_new_default": 250,
    },
    "alg_distribute-v4": {
        "ckpt": "results/checkpoints_v4/v4_alg_distribute_best.pt",
        "desc": "V4 Distribute/Combine — a(x+b)=c and ax+bx=c | expand 100% / combine 25%",
        "max_new_default": 250,
    },
}

LOADED: dict[str, MathGPT] = {}
DEVICE: str = "cuda"
USE_KV_CACHE: bool = True
COMPILE_MODE: str = "off"   # "off" | "reduce-overhead" | "default"

# Model architecture constants (all our models share the same config)
_ARCH = {"n_layer": 6, "n_head": 8, "n_embd": 256, "vocab_size": 96}

def _param_count(arch: dict = _ARCH) -> int:
    """Estimate parameter count from GPT config."""
    V, L, D = arch["vocab_size"], arch["n_layer"], arch["n_embd"]
    embed   = V * D                         # token embedding
    per_lyr = 4*D*D + D*D + 2*4*D*D        # QKV+proj + 2x FFN
    lm_head = V * D                         # output projection
    return embed + L * per_lyr + lm_head

def _flops_per_token(param_count: int, seq_len: int = 128) -> int:
    """Approx FLOPs for one forward-pass token (2×params rule of thumb)."""
    return 2 * param_count


def get_model(model_id: str, registry: dict) -> MathGPT:
    if model_id in LOADED:
        return LOADED[model_id]
    if model_id not in registry:
        raise HTTPException(status_code=404,
                            detail=f"Unknown model id: {model_id}")
    ckpt = Path(registry[model_id]["ckpt"])
    if not ckpt.exists():
        raise HTTPException(status_code=500,
                            detail=f"Checkpoint missing for {model_id}: {ckpt}")
    m, _ = load_model(ckpt, DEVICE)
    if COMPILE_MODE != "off":
        try:
            m = torch.compile(m, mode=COMPILE_MODE)
        except Exception as e:
            print(f"  warn: torch.compile({COMPILE_MODE}) failed for {model_id}: {e}")
    LOADED[model_id] = m
    return m


def stream_generate(model: MathGPT, prompt: str, max_new: int) -> Iterable[str]:
    """Yields one decoded character per emitted token (char-level tokenizer).

    Uncached path: re-runs the full forward over the trailing seq_len context
    every step. O(N²) total compute for an N-token completion. Kept as a
    correctness oracle for the KV-cached path below.
    """
    device = next(model.parameters()).device
    ids = torch.tensor(char_encode(prompt)[:-1], dtype=torch.long,
                       device=device).unsqueeze(0)
    seq_len = model.cfg.seq_len
    for _ in range(max_new):
        ids_cond = ids[:, -seq_len:]
        with torch.no_grad():
            logits = model(ids_cond)[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id == EOS_ID:
            break
        ch = char_decode([next_id])
        if ch:
            yield ch
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)


def stream_generate_kv(model: MathGPT, prompt: str, max_new: int) -> Iterable[str]:
    """KV-cached greedy decode: one prefill forward, then per-token decode.

    O(N) per step instead of O(N²) total — the dominant win on CUDA where
    per-kernel-launch overhead is the bottleneck for tiny models.
    """
    device = next(model.parameters()).device
    ids = torch.tensor(char_encode(prompt)[:-1], dtype=torch.long,
                       device=device).unsqueeze(0)

    # Prefill: one forward over the full prompt seeds the K/V cache.
    with torch.no_grad():
        logits, past_kvs = model(ids, use_cache=True)

    next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    if next_id == EOS_ID:
        return
    ch = char_decode([next_id])
    if ch:
        yield ch

    # Decode: one token at a time, reusing and growing the cache.
    seq_len = model.cfg.seq_len
    next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        # pos_emb is bounded by seq_len; stop before we'd index out of range.
        if past_kvs[0][0].shape[2] >= seq_len:
            break
        with torch.no_grad():
            logits, past_kvs = model(next_input, past_kvs=past_kvs, use_cache=True)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if next_id == EOS_ID:
            break
        ch = char_decode([next_id])
        if ch:
            yield ch
        next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# OpenAI request/response schemas (subset)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0   # ignored — we always greedy
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.0


def make_app(registry: dict) -> FastAPI:
    app = FastAPI(title="MathGPT OpenAI-compatible Server")

    # CORS — open since this is a LAN-only research server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "device": DEVICE,
                "loaded": list(LOADED.keys()),
                "registered": list(registry.keys())}

    # ── Static UI — serve chat.html at / and /chat ────────────────────────────
    _CHAT_HTML = Path(__file__).parent / "chat.html"

    @app.get("/", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    def serve_chat():
        if not _CHAT_HTML.exists():
            return HTMLResponse("chat.html not found next to serve.py", status_code=404)
        return HTMLResponse(_CHAT_HTML.read_text(encoding="utf-8"))

    _PARAM_COUNT = _param_count()
    _FLOPS_PER_TOKEN = _flops_per_token(_PARAM_COUNT)

    @app.get("/v1/models")
    def list_models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now,
                    "owned_by": "autoresearch",
                    "description": meta.get("desc", ""),
                    "param_count": _PARAM_COUNT,
                    "flops_per_token": _FLOPS_PER_TOKEN,
                    "arch": _ARCH,
                    "max_new_default": meta.get("max_new_default", 200),
                }
                for name, meta in registry.items()
            ],
        }

    @app.get("/v1/system")
    def system_info():
        """Returns server hardware and capacity estimates."""
        info: dict = {
            "hostname":     platform.node(),
            "platform":     platform.platform(),
            "python":       platform.python_version(),
            "cpu_cores":    multiprocessing.cpu_count(),
            "device":       DEVICE,
            "inference_path": "kv-cached" if USE_KV_CACHE else "uncached",
            "compile_mode": COMPILE_MODE,
            "loaded_models": list(LOADED.keys()),
            "registered_models": len(registry),
            "param_count":  _PARAM_COUNT,
            "flops_per_token": _FLOPS_PER_TOKEN,
        }
        # CPU GFLOPS estimate (very rough: cores × 8 GFLOPS per core @ ~2.5 GHz)
        info["cpu_gflops_est"] = round(info["cpu_cores"] * 8 * 2.5, 1)
        # Memory
        try:
            import psutil
            vm = psutil.virtual_memory()
            info["ram_total_gb"]  = round(vm.total / 1e9, 1)
            info["ram_avail_gb"]  = round(vm.available / 1e9, 1)
            info["ram_used_pct"]  = vm.percent
        except ImportError:
            pass
        # CUDA info
        if DEVICE == "cuda":
            try:
                import torch
                info["gpu_name"]      = torch.cuda.get_device_name(0)
                info["gpu_mem_total"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                info["gpu_mem_free"]  = round(torch.cuda.memory_reserved(0) / 1e9, 2)
                # Rough TFLOPS: RTX 3090 ≈ 35.6 TF fp32; 3080 ≈ 29.8 TF fp32
                gname = info["gpu_name"].lower()
                info["gpu_tflops_est"] = 35.6 if "3090" in gname else \
                                         29.8 if "3080" in gname else \
                                         10.0
            except Exception:
                pass
        return info

    FEEDBACK_FILE = Path("results/feedback.jsonl")
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    class FeedbackRequest(BaseModel):
        model_id:   str
        problem:    str
        response:   str
        thumb:      Optional[str]  = None   # "up" | "down" | None
        stars:      Optional[int]  = None   # 1-5
        comment:    Optional[str]  = None
        user:       Optional[str]  = "anon"
        session_id: Optional[str]  = None
        ts:         Optional[str]  = None

    @app.post("/v1/feedback")
    def post_feedback(fb: FeedbackRequest):
        record = {
            "ts":         fb.ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user":       fb.user or "anon",
            "model":      fb.model_id,
            "problem":    fb.problem,
            "response":   fb.response,
            "thumb":      fb.thumb,
            "stars":      fb.stars,
            "comment":    fb.comment,
            "session_id": fb.session_id,
        }
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        return {"ok": True, "saved": record["ts"]}

    @app.get("/v1/feedback")
    def get_feedback(limit: int = 200):
        if not FEEDBACK_FILE.exists():
            return {"data": []}
        lines = FEEDBACK_FILE.read_text().strip().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        return {"data": records[-limit:], "total": len(records)}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        # take the LAST user message — multi-turn ignored (model is single-problem)
        user_msg = next((m.content for m in reversed(req.messages)
                         if m.role == "user"), None)
        if not user_msg:
            raise HTTPException(400, "no user message found")

        model = get_model(req.model, registry)
        prompt_text = PROMPT_TEMPLATE.format(problem=user_msg.strip())
        max_new = req.max_tokens or registry[req.model].get("max_new_default", 200)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if req.stream:
            def sse():
                # initial role frame (OpenAI convention)
                yield ("data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant"},
                                 "finish_reason": None}],
                }) + "\n\n")
                # token deltas — track timing
                t0 = time.perf_counter()
                token_count = 0
                gen_fn = stream_generate_kv if USE_KV_CACHE else stream_generate
                for ch in gen_fn(model, prompt_text, max_new):
                    token_count += 1
                    yield ("data: " + json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0,
                                     "delta": {"content": ch},
                                     "finish_reason": None}],
                    }) + "\n\n")
                elapsed = time.perf_counter() - t0
                # final stop frame — include perf stats for client telemetry
                yield ("data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0,
                                 "delta": {},
                                 "finish_reason": "stop"}],
                    "usage": {
                        "completion_tokens": token_count,
                        "elapsed_s": round(elapsed, 3),
                        "tokens_per_sec": round(token_count / elapsed, 1) if elapsed > 0 else 0,
                        "mflops_total": round(token_count * _FLOPS_PER_TOKEN / 1e6, 1),
                        "flops_per_token": _FLOPS_PER_TOKEN,
                    },
                }) + "\n\n")
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        # non-streaming
        gen_fn = stream_generate_kv if USE_KV_CACHE else stream_generate
        out_chars = list(gen_fn(model, prompt_text, max_new))
        content = "".join(out_chars)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(content),
                "total_tokens": len(prompt_text) + len(content),
            },
        })

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        """Raw text-completion endpoint — sends `prompt` as the math problem."""
        model = get_model(req.model, registry)
        prompt_text = PROMPT_TEMPLATE.format(problem=req.prompt.strip())
        max_new = req.max_tokens or registry[req.model].get("max_new_default", 200)

        gen_fn = stream_generate_kv if USE_KV_CACHE else stream_generate
        out_chars = list(gen_fn(model, prompt_text, max_new))
        content = "".join(out_chars)
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": content,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(content),
                "total_tokens": len(prompt_text) + len(content),
            },
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=4321)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preload", action="store_true",
                    help="load all registered models at startup")
    ap.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=None,
                    help="use KV-cached decode path (default: on for MPS/CPU, off for CUDA — "
                         "CUDA is launch-bound, KV cache slightly hurts there)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="wrap models in torch.compile(mode='reduce-overhead') after load. "
                         "Default: on for CUDA (uses CUDA Graphs underneath), off otherwise. "
                         "First request to each model is slow (compilation).")
    args = ap.parse_args()

    global DEVICE, USE_KV_CACHE, COMPILE_MODE
    DEVICE = args.device
    # Platform-aware default: KV cache helps on compute-bound platforms (CPU,
    # MPS) but hurts on launch-bound CUDA. User can override either way.
    if args.kv_cache is None:
        USE_KV_CACHE = (DEVICE != "cuda")
    else:
        USE_KV_CACHE = args.kv_cache
    # torch.compile default: on for CUDA, off elsewhere (MPS support is
    # immature; CPU compile can help marginally but isn't the bottleneck).
    if args.compile is None:
        COMPILE_MODE = "reduce-overhead" if DEVICE == "cuda" else "off"
    else:
        COMPILE_MODE = "reduce-overhead" if args.compile else "off"
    print(f"  device={DEVICE}  kv_cache={USE_KV_CACHE}  compile={COMPILE_MODE}", flush=True)

    app = make_app(DEFAULT_SKILLS)

    if args.preload:
        for name in DEFAULT_SKILLS:
            try:
                get_model(name, DEFAULT_SKILLS)
                print(f"  preloaded: {name}")
            except Exception as e:
                print(f"  skip {name}: {e}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
