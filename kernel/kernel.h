/**
 * kernel.h  —  C ABI for the ATTN11-x86 Integer Attention Kernel
 *
 * All functions use Win64 calling convention (also compatible with System V
 * via the platform shim in io_win64.asm / io_sysv.asm).
 *
 * Fixed-point format: Q8.16
 *   typedef int32_t q16_t;
 *   1.0 = 65536   (0x00010000)
 *  -1.0 = -65536  (0xFFFF0000)
 *   resolution ≈ 0.0000153   (1 / 65536)
 *   range: -32768.0 .. +32767.99998
 *
 * Quantized storage format (turboquant): Q8.8 stored as int16_t
 *   1.0 = 256   (0x0100)
 *   Used for weight matrices on disk / in KV cache entries.
 */

#ifndef ATTN11_KERNEL_H
#define ATTN11_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Fixed-point type ──────────────────────────────────────────────────── */
typedef int32_t  q16_t;    /* Q8.16 — the native compute format */
typedef int16_t  q8_t;     /* Q8.8  — quantized storage format  */

#define Q16_ONE   65536     /* 1.0 in Q8.16  */
#define Q16_MAX   2147483647
#define Q16_MIN   (-2147483648)
#define Q8_ONE    256       /* 1.0 in Q8.8   */
#define Q8_MAX    32767
#define Q8_MIN    (-32768)

/* ── fxmath: scalar fixed-point primitives ─────────────────────────────── */
q16_t fxmul  (q16_t a, q16_t b);
q16_t fxdiv  (q16_t a, q16_t b);
q16_t fxabs  (q16_t a);
q16_t fxclamp(q16_t val, q16_t lo, q16_t hi);
q16_t fxone  (void);

/* ── vecop: vector operations ──────────────────────────────────────────── */

/** Dot product: Σ x[i]*y[i] in Q8.16 (64-bit accumulator, no overflow) */
q16_t vdot  (const q16_t *x, const q16_t *y, int n);

/** z[i] = x[i] + y[i], saturated Q8.16 */
void  vadd  (const q16_t *x, const q16_t *y, q16_t *z, int n);

/** z[i] = x[i] - y[i], saturated Q8.16 */
void  vsub  (const q16_t *x, const q16_t *y, q16_t *z, int n);

/** y[i] = alpha * x[i] in Q8.16 */
void  vscl  (const q16_t *x, q16_t *y, int n, q16_t alpha);

/** Returns max element; writes its 0-based index to *idx_out */
q16_t vmax  (const q16_t *v, int n, int *idx_out);

/** dst[i] = src[i] */
void  vcpy  (const q16_t *src, q16_t *dst, int n);

/** v[i] = 0 */
void  vclr  (q16_t *v, int n);

/** dst[k] += (scalar * src[k]) >> 16  (scale-accumulate, Q8.16) */
void  vsadd (q16_t scalar, const q16_t *src, q16_t *dst, int n);

/* ── matop: matrix-vector operations (row-major, Q8.16) ────────────────── */

/** vout[i] = Σ_j mat[i][j]*vin[j]   (matrix × vector) */
void  mvmul (const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols);

/** vout[i] += Σ_j mat[i][j]*vin[j]  (matrix × vector, accumulate) */
void  mvadd (const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols);

/** vout[j] = Σ_i mat[i][j]*vin[i]   (matrix^T × vector) */
void  vtmul (const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols);

/** mat[i][j] += vx[i]*vy[j]          (outer product accumulate) */
void  outer (q16_t *mat, const q16_t *vx, const q16_t *vy, int rows, int cols);

/* ── actfn: activation functions ───────────────────────────────────────── */

/** v[i] = max(0, v[i])  in-place ReLU */
void  vrelu (q16_t *v, int n);

/** v[] = softmax(v[])  in-place, values in [0, Q16_ONE], sum ≈ Q16_ONE */
void  sftmx (q16_t *v, int n);

/* ── attn_kernel: self-attention forward pass ──────────────────────────── */

/**
 * Arguments for attn_forward().
 * Layout mirrors the PDP-11 ATTN parameter block from LAYER.MAC.
 */
typedef struct AttnArgs {
    q16_t   *xin;       /**< Input  [seq_len × d_model]             */
    q16_t   *wq;        /**< Query  weight [d_model × d_model]      */
    q16_t   *wk;        /**< Key    weight [d_model × d_model]      */
    q16_t   *wv;        /**< Value  weight [d_model × d_model]      */
    q16_t   *yout;      /**< Output [seq_len × d_model]             */
    q16_t   *work;      /**< Scratch ≥ (3*seq*dim + seq*seq) int32  */
    int32_t  seq_len;
    int32_t  d_model;
    int32_t  sqrt_shift; /**< Right-shift count ≈ log2(sqrt(d_model)) */
} AttnArgs;

/**
 * Self-attention forward pass (causal, single-head).
 * Steps: Q=X·Wq, K=X·Wk, V=X·Wv, S=Q·K^T>>shift, A=softmax(S), Y=V^T·A, Y+=X
 */
void attn_forward(AttnArgs *a);

/** Token + positional embedding lookup */
void embed(
    const int32_t *tokens,     /**< token IDs [seq_len]              */
    const q16_t   *token_emb,  /**< embedding table [vocab × d_model] */
    const q16_t   *pos_emb,    /**< position table  [seq_len × d_model] */
    q16_t         *xout,       /**< output [seq_len × d_model]        */
    int32_t        seq_len,
    int32_t        d_model
);

/** Output projection: logits[i] = Wout^T · Y[i] */
void proj(
    const q16_t *yin,      /**< input   [seq_len × d_model] */
    const q16_t *wout,     /**< weights [d_model × vocab]   */
    q16_t       *logits,   /**< output  [seq_len × vocab]   */
    int32_t      seq_len,
    int32_t      d_model,
    int32_t      vocab
);

/* ── qsparser: ASCII text tokenizer ───────────────────────────────────── */

/**
 * Tokenize a null-terminated ASCII string into integer token IDs.
 * Character-level baseline: token_id = ASCII byte value (0..127).
 * Non-ASCII bytes (≥128) are mapped to token ID 3 (<UNK>).
 *
 * Special token IDs:
 *   0 = <PAD>    1 = <BOS>    2 = <EOS>    3 = <UNK>
 *   4..127  = ASCII chars  (space=' '=32+4 offset? No—raw ASCII value)
 *
 * @param text       Null-terminated ASCII input
 * @param token_ids  Output array of int32_t token IDs
 * @param out_cap    Maximum number of tokens to write
 * @return           Number of tokens written (≤ out_cap)
 */
int qsparse(const char *text, int32_t *token_ids, int out_cap);

/**
 * Decode token IDs back to ASCII string.
 * @param token_ids Input token ID array
 * @param n         Number of tokens
 * @param buf       Output buffer (must be ≥ n+1 bytes)
 * @param buf_cap   Buffer capacity
 */
void qsdecode(const int32_t *token_ids, int n, char *buf, int buf_cap);

/* ── turboquant: integer weight quantization ───────────────────────────── */

/**
 * Quantize float32 array → int16_t Q8.8 (lossily).
 * scale: multiply float value by (256 * scale) before rounding.
 * If scale == 0.0f, auto-computes scale = 127 / max(|src|).
 * Returns the scale factor used (for dequantization).
 */
float tq_f32_to_q8(const float *src, q8_t *dst, int n, float scale);

/**
 * Dequantize int16_t Q8.8 → q16_t Q8.16 (for compute).
 * scale must match the value returned by tq_f32_to_q8().
 */
void  tq_q8_to_q16(const q8_t *src, q16_t *dst, int n, float scale);

/**
 * Quantize q16_t Q8.16 → q8_t Q8.8 (lossily).
 * Equivalent to halving the fractional bits (>> 8, then saturate to int16).
 */
void  tq_q16_to_q8(const q16_t *src, q8_t *dst, int n);

/**
 * Expand q8_t Q8.8 → q16_t Q8.16 (losslessly, just shift left 8).
 */
void  tq_q8_to_q16_exact(const q8_t *src, q16_t *dst, int n);

/* ── kvcache: key-value cache for autoregressive inference ─────────────── */

/** Opaque KV cache handle */
typedef struct KVCache KVCache;

/**
 * Allocate a KV cache.
 * Memory: n_layers × 2 × max_seq × d_kv × sizeof(q8_t) bytes.
 * (Keys and values stored in Q8.8 to halve memory; expanded on query.)
 */
KVCache *kvc_create(int n_layers, int max_seq, int d_kv);

/** Free a KV cache */
void kvc_free(KVCache *c);

/**
 * Insert K and V vectors for a single position.
 * @param layer  Layer index (0-based)
 * @param pos    Sequence position (0-based)
 * @param k      Key   vector [d_kv] in Q8.16
 * @param v      Value vector [d_kv] in Q8.16
 */
void kvc_insert(KVCache *c, int layer, int pos,
                const q16_t *k, const q16_t *v);

/**
 * Read out K and V matrices for all positions [0..max_pos-1].
 * Expands Q8.8 → Q8.16 on output.
 * @param k_out  [max_pos × d_kv] output for keys
 * @param v_out  [max_pos × d_kv] output for values
 */
void kvc_query(KVCache *c, int layer, int max_pos,
               q16_t *k_out, q16_t *v_out);

/** Reset position counter (reuse buffer without reallocating) */
void kvc_reset(KVCache *c);

/** Return current number of filled positions (all layers share one counter) */
int  kvc_pos(const KVCache *c);

/* ── io: console output helpers ────────────────────────────────────────── */
void putq16 (q16_t v);          /**< Print Q8.16 value as decimal (e.g. "1.234") */
void putvec16(const q16_t *v, int n); /**< Print vector: [1.000, -0.500, ...] */

#ifdef __cplusplus
}
#endif

#endif /* ATTN11_KERNEL_H */
