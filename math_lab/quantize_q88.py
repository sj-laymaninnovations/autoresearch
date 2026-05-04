"""
quantize_q88.py — convert a safetensors model to Q8.8 fixed-point.

Format spec:
  Q8.8 = 16-bit signed integer, 8 integer bits + 8 fractional bits.
  Range: [-128, +127.99609375]
  Resolution: 1/256 ≈ 3.9e-3

Strategy: per-tensor symmetric quantization.
  scale_t = max(|W_t|) / 127.99609375     # so the max-magnitude weight maps
                                            # to the Q8.8 max value
  W_q[i] = round(W[i] / scale_t * 256)     # 256 = 2^8 (fractional shift)
  W_q[i] is stored as int16

At runtime:
  W_real[i] = W_q[i] / 256 * scale_t

This is symmetric per-tensor quant. The kernel can either dequantize on-the-fly
OR do integer math throughout (true PDP-11 vision). Both options are downstream.

This script writes:
  <out>/index.json       — list of tensor names + dtype + shape + scale
  <out>/q88_weights.bin  — concatenated int16 weights, sequentially in index
                           order, naturally aligned

Usage:
  python math_lab/quantize_q88.py \
      --model-dir ~/models/Qwen2.5-Math-1.5B-Instruct \
      --out       ~/models/Qwen2.5-Math-1.5B-Q88
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np


def quantize_tensor(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor Q8.8.

    Returns (int16_values, scale_float) so that:
      w ≈ int16_values / 256.0 * scale
    """
    if w.size == 0:
        return np.zeros(0, dtype=np.int16), 1.0
    abs_max = float(np.abs(w).max())
    if abs_max == 0.0:
        return np.zeros(w.shape, dtype=np.int16), 1.0
    # Q8.8 max = 127 + 255/256 ≈ 127.996
    Q88_MAX = 127.99609375
    # We pick scale so that max|w| maps exactly to Q88_MAX after the /256 shift.
    # i.e., max|w| / scale * 256 == Q88_MAX * 256 = 32767
    # => scale = max|w| / Q88_MAX
    scale = abs_max / Q88_MAX
    # Quantize: q = round(w / scale * 256)
    q = np.round(w.astype(np.float32) / scale * 256.0)
    # Clamp into int16 range just in case of rounding off-by-one
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q, scale


def measure_quant_error(orig: np.ndarray, q: np.ndarray, scale: float) -> dict:
    deq = q.astype(np.float32) / 256.0 * scale
    err = orig.astype(np.float32) - deq
    return {
        "max_abs_err":  float(np.abs(err).max()) if err.size else 0.0,
        "mean_abs_err": float(np.abs(err).mean()) if err.size else 0.0,
        "rms_err":      float(np.sqrt((err ** 2).mean())) if err.size else 0.0,
        "abs_max_orig": float(np.abs(orig).max()) if orig.size else 0.0,
    }


def quantize_model(model_dir: Path, out_dir: Path,
                    sample_metrics_every: int = 25):
    """Walk every .safetensors shard, quantize each tensor, write Q8.8."""
    from safetensors import safe_open
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise SystemExit(f"No .safetensors files in {model_dir}")

    index = {
        "source_dir":  str(model_dir),
        "format":      "Q8.8 symmetric per-tensor (int16 + float scale)",
        "fractional_bits": 8,
        "tensors":     [],
        "metrics":     [],  # sampled at sample_metrics_every
    }

    out_blob = out_dir / "q88_weights.bin"
    offset = 0
    n_total_params = 0
    n_tensors = 0
    with out_blob.open("wb") as fout:
        for shard in shard_files:
            with safe_open(str(shard), framework="np") as f:
                for name in f.keys():
                    w = f.get_tensor(name)  # numpy
                    # Quantize
                    q, scale = quantize_tensor(w)
                    # Write to blob
                    fout.write(q.tobytes())
                    nbytes = q.nbytes
                    record = {
                        "name":  name,
                        "shape": list(w.shape),
                        "dtype_orig": str(w.dtype),
                        "scale": scale,
                        "offset_bytes": offset,
                        "nbytes": nbytes,
                    }
                    index["tensors"].append(record)
                    if n_tensors % sample_metrics_every == 0:
                        m = measure_quant_error(w, q, scale)
                        m["name"] = name
                        index["metrics"].append(m)
                    offset += nbytes
                    n_total_params += w.size
                    n_tensors += 1

    index["total_params"]   = n_total_params
    index["total_bytes"]    = offset
    index["n_tensors"]      = n_tensors
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Copy the tokenizer & config for downstream use
    for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "special_tokens_map.json", "vocab.json", "merges.txt",
               "config.json", "generation_config.json"):
        src = model_dir / fn
        if src.exists():
            (out_dir / fn).write_bytes(src.read_bytes())

    print(f"  source dir   : {model_dir}")
    print(f"  shards seen  : {len(shard_files)}")
    print(f"  tensors quant: {n_tensors}")
    print(f"  total params : {n_total_params:,}")
    print(f"  output blob  : {out_blob}  ({offset / 1024 / 1024:.1f} MB)")
    print(f"  vs fp16 size : {n_total_params * 2 / 1024 / 1024:.1f} MB → "
          f"Q8.8 is {offset / (n_total_params * 2) * 100:.0f}% (same int16 width)")
    print(f"  index file   : {out_dir / 'index.json'}")
    print(f"  sampled error metrics ({len(index['metrics'])} tensors):")
    for m in index["metrics"][:5]:
        print(f"    {m['name'][:50]:50s}  abs_max_err={m['max_abs_err']:.4e}  "
              f"rms={m['rms_err']:.4e}  abs_max_orig={m['abs_max_orig']:.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--sample-metrics-every", type=int, default=25)
    args = ap.parse_args()
    quantize_model(args.model_dir.expanduser(), args.out.expanduser(),
                    sample_metrics_every=args.sample_metrics_every)


if __name__ == "__main__":
    main()
