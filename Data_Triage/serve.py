"""
serve.py — OpenAI-compatible inference server for the EraGPT fleet.

Sibling of math_lab/serve.py. Difference: this serves the era-curriculum
PDP-11 SLM (cl100k_base BPE tokenizer + RoPE GPT or BitGPT) rather than
the math_lab MathGPT (96-char vocab).

Listens on port 4322 by default (math_lab serves :4321 so both can run
side-by-side). Loads per-checkpoint architecture from each run's
`run_config.json` sidecar; falls back to hardcoded Config A for the
BitNet variant where no sidecar exists.

Usage:
    python Data_Triage/serve.py --host 0.0.0.0 --port 4322 --device mps
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import tiktoken
import uvicorn

# training/ holds the model classes (GPT, BitGPT) and the ModelConfig dataclass.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))

from config import ModelConfig         # type: ignore
from train  import GPT                  # type: ignore
from bitnet import BitGPT               # type: ignore


# ---------------------------------------------------------------------------
# Fleet registry — each entry is one era checkpoint family. The serve loads
# the *latest* ckpt_*.pt from `ckpt_dir` on first request.
# ---------------------------------------------------------------------------

CKPT_DIR_BASE = ROOT / "checkpoints"

EraSpec = dict  # {ckpt_dir, is_bitnet, config_letter, desc, max_new_default,
                #  fallback_config?}

DEFAULT_ERA_MODELS: dict[str, EraSpec] = {
    "prehistoric-a-era": {
        "ckpt_dir":         "prehistoric_config_a",
        "is_bitnet":        False,
        "config_letter":    "A",
        "desc":             "Paleolithic gate (-30000 to -10000 BCE). Config A (~7M).",
        "max_new_default":  400,
    },
    "mixed-b-era": {
        "ckpt_dir":         "mixed_config_b",
        "is_bitnet":        False,
        "config_letter":    "B",
        "desc":             "Paleo + Ancient + Foundation mixed corpus. Config B (~25M).",
        "max_new_default":  400,
    },
    "egypt-a-era": {
        "ckpt_dir":         "test_egypt_a",
        "is_bitnet":        False,
        "config_letter":    "A",
        "desc":             "Ancient Egypt teacher-distilled. Config A (~7M, ckpt 9999).",
        "max_new_default":  400,
    },
    "bitnet-a-era": {
        "ckpt_dir":         "bitnet_a",
        "is_bitnet":        True,
        "config_letter":    "A",
        "desc":             "BitNet 1.58 (ternary weights) broad-pretrain on raw FineWeb-Edu.",
        "max_new_default":  400,
        # No run_config.json sidecar for the bitnet run — hardcode Config A.
        "fallback_config":  {
            "n_layer": 6, "n_head": 4, "d_model": 128, "d_ff": 512,
            "vocab_size": 100_277, "seq_len": 2048,
            "rope_theta": 500_000.0, "dropout": 0.0, "bias": False,
            # BitNet-specific (mtp_depth=1 → no drafter)
            "mtp_depth": 1,
        },
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

LOADED: dict[str, dict] = {}    # model_id -> {model, cfg, step, ckpt_name}
DEVICE: str             = "mps"
TOKENIZER: Optional[tiktoken.Encoding] = None
EOS_ID: int             = 100_257  # cl100k_base <|endoftext|>


def get_tokenizer() -> tiktoken.Encoding:
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return TOKENIZER


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_config(model_id: str, spec: EraSpec, ckpt_dir: Path) -> ModelConfig:
    """Build ModelConfig from run_config.json sidecar, or fallback."""
    rc = ckpt_dir / "run_config.json"
    if rc.exists():
        data = json.loads(rc.read_text())
        return ModelConfig(**data["model"])
    if "fallback_config" in spec:
        # BitNet-style ModelConfig may carry extra fields (e.g., mtp_depth).
        # ModelConfig dataclass doesn't know about mtp_depth; set as attribute.
        cfg_dict = dict(spec["fallback_config"])
        mtp_depth = cfg_dict.pop("mtp_depth", 1)
        cfg = ModelConfig(**cfg_dict)
        if spec["is_bitnet"]:
            setattr(cfg, "mtp_depth", mtp_depth)
        return cfg
    raise HTTPException(500,
        f"{model_id}: no run_config.json at {rc} and no fallback_config")


def _latest_ckpt(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise HTTPException(500, f"no ckpt_*.pt found in {ckpt_dir}")
    return ckpts[-1]


def get_model(model_id: str) -> dict:
    if model_id in LOADED:
        return LOADED[model_id]
    if model_id not in DEFAULT_ERA_MODELS:
        raise HTTPException(404, f"unknown model id: {model_id}")
    spec     = DEFAULT_ERA_MODELS[model_id]
    ckpt_dir = CKPT_DIR_BASE / spec["ckpt_dir"]
    if not ckpt_dir.exists():
        raise HTTPException(500, f"checkpoint dir missing: {ckpt_dir}")
    cfg      = _build_config(model_id, spec, ckpt_dir)
    ckpt     = _latest_ckpt(ckpt_dir)

    if spec["is_bitnet"]:
        model = BitGPT(cfg)
    else:
        model = GPT(cfg)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    sd = state.get("model", state)
    # Strip a potential `_orig_mod.` prefix that torch.compile leaves on keys.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    info = {
        "model":     model,
        "cfg":       cfg,
        "step":      state.get("step", 0),
        "ckpt_name": ckpt.name,
        "spec":      spec,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    LOADED[model_id] = info
    print(f"  loaded {model_id} <- {ckpt.name} step={info['step']:,} "
          f"params={info['param_count']/1e6:.2f}M")
    return info


# ---------------------------------------------------------------------------
# Streaming greedy/sampled generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_generate(model: torch.nn.Module, cfg: ModelConfig,
                    prompt_ids: List[int], max_new: int,
                    temperature: float = 0.0) -> Iterable[str]:
    """Yield decoded text fragments (1 BPE token's text per yield).

    Uses greedy when temperature == 0; otherwise samples with the implicit
    softmax. No KV cache — re-runs the full forward over the trailing
    seq_len each step. Good enough for the smoke/correctness pass; can be
    optimized later (KV cache won the inference benchmark on compute-bound
    paths but not on CUDA — same story applies here).
    """
    enc = get_tokenizer()
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        ids_cond = ids[:, -cfg.seq_len:]
        out = model(ids_cond)
        # GPT.forward returns (logits, _) when given y; logits alone otherwise.
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :]
        if temperature == 0.0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        if next_id == EOS_ID:
            break
        # Decode just this one token (tiktoken can decode partial tokens fine)
        try:
            text = enc.decode([next_id])
        except Exception:
            text = ""
        if text:
            yield text
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)


# ---------------------------------------------------------------------------
# OpenAI request/response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def make_app() -> FastAPI:
    app = FastAPI(title="EraGPT OpenAI-compatible Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"], expose_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "device": DEVICE,
                "loaded": list(LOADED.keys()),
                "registered": list(DEFAULT_ERA_MODELS.keys())}

    _CHAT_HTML = ROOT / "chat.html"

    @app.get("/", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    def serve_chat():
        if not _CHAT_HTML.exists():
            return HTMLResponse("chat.html not found next to serve.py", 404)
        return HTMLResponse(_CHAT_HTML.read_text(encoding="utf-8"))

    @app.get("/v1/models")
    def list_models():
        now = int(time.time())
        out = []
        for name, spec in DEFAULT_ERA_MODELS.items():
            # Param count is only known once loaded; fall back to a rough
            # estimate from the fallback_config or sidecar.
            param_count = LOADED.get(name, {}).get("param_count")
            if param_count is None:
                ckpt_dir = CKPT_DIR_BASE / spec["ckpt_dir"]
                try:
                    cfg = _build_config(name, spec, ckpt_dir)
                    V, L, D, F_ = cfg.vocab_size, cfg.n_layer, cfg.d_model, cfg.d_ff
                    param_count = V * D + L * (4*D*D + 2*D*F_) + V * D
                except Exception:
                    param_count = None
            out.append({
                "id":              name,
                "object":          "model",
                "created":         now,
                "owned_by":        "data_triage",
                "description":     spec["desc"],
                "config":          spec["config_letter"],
                "is_bitnet":       spec["is_bitnet"],
                "param_count":     param_count,
                "flops_per_token": 2 * param_count if param_count else None,
                "max_new_default": spec["max_new_default"],
                "ckpt_dir":        spec["ckpt_dir"],
            })
        return {"object": "list", "data": out}

    @app.get("/v1/system")
    def system_info():
        info: dict = {
            "hostname":          platform.node(),
            "platform":          platform.platform(),
            "python":            platform.python_version(),
            "cpu_cores":         multiprocessing.cpu_count(),
            "device":            DEVICE,
            "inference_path":    "uncached",
            "tokenizer":         "cl100k_base",
            "loaded_models":     list(LOADED.keys()),
            "registered_models": len(DEFAULT_ERA_MODELS),
        }
        info["cpu_gflops_est"] = round(info["cpu_cores"] * 8 * 2.5, 1)
        try:
            import psutil
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = round(vm.total / 1e9, 1)
            info["ram_avail_gb"] = round(vm.available / 1e9, 1)
            info["ram_used_pct"] = vm.percent
        except ImportError:
            pass
        if DEVICE == "cuda":
            try:
                info["gpu_name"]      = torch.cuda.get_device_name(0)
                info["gpu_mem_total"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                info["gpu_mem_free"]  = round(torch.cuda.memory_reserved(0) / 1e9, 2)
                gname = info["gpu_name"].lower()
                info["gpu_tflops_est"] = 35.6 if "3090" in gname else \
                                         29.8 if "3080" in gname else 10.0
            except Exception:
                pass
        return info

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        user_msg = next((m.content for m in reversed(req.messages)
                         if m.role == "user"), None)
        if not user_msg:
            raise HTTPException(400, "no user message found")

        sys_msg = next((m.content for m in req.messages
                        if m.role == "system"), None)
        prompt_text = (f"{sys_msg.strip()}\n\n{user_msg.strip()}"
                       if sys_msg else user_msg.strip())

        info     = get_model(req.model)
        spec     = info["spec"]
        model    = info["model"]
        cfg      = info["cfg"]
        max_new  = req.max_tokens or spec["max_new_default"]
        prompt_ids = get_tokenizer().encode_ordinary(prompt_text)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        flops_per_token = 2 * info["param_count"]

        if req.stream:
            def sse():
                yield ("data: " + json.dumps({
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": req.model,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant"},
                                 "finish_reason": None}],
                }) + "\n\n")
                t0 = time.perf_counter()
                token_count = 0
                for piece in stream_generate(
                        model, cfg, prompt_ids, max_new,
                        temperature=req.temperature or 0.0):
                    token_count += 1
                    yield ("data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": req.model,
                        "choices": [{"index": 0,
                                     "delta": {"content": piece},
                                     "finish_reason": None}],
                    }) + "\n\n")
                elapsed = time.perf_counter() - t0
                yield ("data: " + json.dumps({
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": req.model,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}],
                    "usage": {
                        "completion_tokens": token_count,
                        "elapsed_s":         round(elapsed, 3),
                        "tokens_per_sec":    round(token_count/elapsed, 1) if elapsed > 0 else 0,
                        "mflops_total":      round(token_count * flops_per_token / 1e6, 1),
                        "flops_per_token":   flops_per_token,
                    },
                }) + "\n\n")
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        out_pieces = list(stream_generate(
            model, cfg, prompt_ids, max_new,
            temperature=req.temperature or 0.0))
        content = "".join(out_pieces)
        return JSONResponse({
            "id": completion_id, "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens":     len(prompt_ids),
                "completion_tokens": len(out_pieces),
                "total_tokens":      len(prompt_ids) + len(out_pieces),
            },
        })

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        info  = get_model(req.model)
        model = info["model"]
        cfg   = info["cfg"]
        max_new = req.max_tokens or info["spec"]["max_new_default"]
        prompt_ids = get_tokenizer().encode_ordinary(req.prompt)
        content = "".join(stream_generate(
            model, cfg, prompt_ids, max_new,
            temperature=req.temperature or 0.0))
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens":     len(prompt_ids),
                "completion_tokens": len(content),
                "total_tokens":      len(prompt_ids) + len(content),
            },
        })

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",    default="0.0.0.0")
    ap.add_argument("--port",    type=int, default=4322)
    ap.add_argument("--device",  default="mps")
    ap.add_argument("--preload", action="store_true",
                    help="load all registered models at startup")
    args = ap.parse_args()

    global DEVICE
    DEVICE = args.device
    print(f"  EraGPT serve  device={DEVICE}  port={args.port}", flush=True)

    app = make_app()

    if args.preload:
        for name in DEFAULT_ERA_MODELS:
            try:
                get_model(name)
            except Exception as e:
                print(f"  skip {name}: {e}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
