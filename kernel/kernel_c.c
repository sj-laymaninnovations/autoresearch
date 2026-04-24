/**
 * kernel_c.c  —  Pure C fallback implementation of kernel.h
 *
 * Targets platforms where the x86-64 NASM assembly cannot be compiled,
 * specifically macOS ARM64 (Apple Silicon / M-series).
 *
 * Build:
 *   clang -O2 -Wall -fPIC -dynamiclib -o kernel.dylib kernel_c.c -lm
 *
 * All fixed-point formats match the ASM kernel exactly:
 *   q16_t = int32_t, Q8.16:  1.0 = 65536  (Q16_ONE)
 *   q8_t  = int16_t, Q8.8:   1.0 = 256    (Q8_ONE)
 */

#include "kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── fxmath ─────────────────────────────────────────────────────────────── */

q16_t fxmul(q16_t a, q16_t b) {
    return (q16_t)(((int64_t)a * b) >> 16);
}

q16_t fxdiv(q16_t a, q16_t b) {
    if (b == 0) return a >= 0 ? Q16_MAX : Q16_MIN;
    return (q16_t)(((int64_t)a << 16) / b);
}

q16_t fxabs(q16_t a) {
    return a < 0 ? -a : a;
}

q16_t fxclamp(q16_t val, q16_t lo, q16_t hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

q16_t fxone(void) {
    return Q16_ONE;
}

/* ── vecop ──────────────────────────────────────────────────────────────── */

q16_t vdot(const q16_t *x, const q16_t *y, int n) {
    int64_t acc = 0;
    for (int i = 0; i < n; i++)
        acc += (int64_t)x[i] * y[i];
    return (q16_t)(acc >> 16);
}

void vadd(const q16_t *x, const q16_t *y, q16_t *z, int n) {
    for (int i = 0; i < n; i++) {
        int64_t s = (int64_t)x[i] + y[i];
        z[i] = (q16_t)(s > Q16_MAX ? Q16_MAX : s < Q16_MIN ? Q16_MIN : s);
    }
}

void vsub(const q16_t *x, const q16_t *y, q16_t *z, int n) {
    for (int i = 0; i < n; i++) {
        int64_t s = (int64_t)x[i] - y[i];
        z[i] = (q16_t)(s > Q16_MAX ? Q16_MAX : s < Q16_MIN ? Q16_MIN : s);
    }
}

void vscl(const q16_t *x, q16_t *y, int n, q16_t alpha) {
    for (int i = 0; i < n; i++)
        y[i] = (q16_t)(((int64_t)x[i] * alpha) >> 16);
}

q16_t vmax(const q16_t *v, int n, int *idx_out) {
    q16_t best = v[0];
    int   best_i = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > best) { best = v[i]; best_i = i; }
    }
    if (idx_out) *idx_out = best_i;
    return best;
}

void vcpy(const q16_t *src, q16_t *dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(q16_t));
}

void vclr(q16_t *v, int n) {
    memset(v, 0, (size_t)n * sizeof(q16_t));
}

void vsadd(q16_t scalar, const q16_t *src, q16_t *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += (q16_t)(((int64_t)scalar * src[i]) >> 16);
}

/* ── matop ──────────────────────────────────────────────────────────────── */

void mvmul(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        vout[i] = vdot(mat + i * cols, vin, cols);
}

void mvadd(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        vout[i] += vdot(mat + i * cols, vin, cols);
}

void vtmul(const q16_t *mat, const q16_t *vin, q16_t *vout, int rows, int cols) {
    vclr(vout, cols);
    for (int i = 0; i < rows; i++)
        vsadd(vin[i], mat + i * cols, vout, cols);
}

void outer(q16_t *mat, const q16_t *vx, const q16_t *vy, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i * cols + j] += (q16_t)(((int64_t)vx[i] * vy[j]) >> 16);
}

/* ── actfn ──────────────────────────────────────────────────────────────── */

void vrelu(q16_t *v, int n) {
    for (int i = 0; i < n; i++)
        if (v[i] < 0) v[i] = 0;
}

void sftmx(q16_t *v, int n) {
    /* Find max for numerical stability, then compute softmax via float exp */
    q16_t mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];

    double denom = 0.0;
    double fmx   = (double)mx / Q16_ONE;
    for (int i = 0; i < n; i++) {
        double e = exp((double)v[i] / Q16_ONE - fmx);
        denom += e;
        /* Store scaled value temporarily as Q16 (may overflow for large n, but
           typical attention sequence lengths are well within range) */
        v[i] = (q16_t)(int32_t)(e * Q16_ONE);
    }
    double inv = (denom > 1e-30) ? (1.0 / denom) : 0.0;
    for (int i = 0; i < n; i++)
        v[i] = (q16_t)(int32_t)((double)v[i] * inv);
}

/* ── attn_kernel ────────────────────────────────────────────────────────── */

void attn_forward(AttnArgs *a) {
    int T = a->seq_len, D = a->d_model, shift = a->sqrt_shift;

    /* work layout: Q[T×D], K[T×D], V[T×D], S[T×T] */
    q16_t *Q = a->work;
    q16_t *K = Q + T * D;
    q16_t *V = K + T * D;
    q16_t *S = V + T * D;

    for (int t = 0; t < T; t++) {
        mvmul(a->wq, a->xin + t * D, Q + t * D, D, D);
        mvmul(a->wk, a->xin + t * D, K + t * D, D, D);
        mvmul(a->wv, a->xin + t * D, V + t * D, D, D);
    }

    /* Scaled dot-product attention scores with causal mask */
    for (int i = 0; i < T; i++) {
        for (int j = 0; j <= i; j++) {
            int64_t acc = 0;
            for (int k = 0; k < D; k++)
                acc += (int64_t)Q[i*D+k] * K[j*D+k];
            S[i*T+j] = (q16_t)(acc >> (16 + shift));
        }
        for (int j = i + 1; j < T; j++)
            S[i*T+j] = Q16_MIN; /* causal mask: -inf for future positions */
    }

    for (int i = 0; i < T; i++)
        sftmx(S + i * T, T);

    /* Output = A · V, then add residual X */
    vclr(a->yout, T * D);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j <= i; j++)
            vsadd(S[i*T+j], V + j * D, a->yout + i * D, D);
    }
    for (int i = 0; i < T * D; i++) {
        int64_t s = (int64_t)a->yout[i] + a->xin[i];
        a->yout[i] = (q16_t)(s > Q16_MAX ? Q16_MAX : s < Q16_MIN ? Q16_MIN : s);
    }
}

void embed(const int32_t *tokens, const q16_t *token_emb, const q16_t *pos_emb,
           q16_t *xout, int32_t seq_len, int32_t d_model) {
    for (int t = 0; t < seq_len; t++) {
        const q16_t *te  = token_emb + (int)tokens[t] * d_model;
        const q16_t *pe  = pos_emb   + t * d_model;
        q16_t       *out = xout       + t * d_model;
        for (int d = 0; d < d_model; d++) {
            int64_t s = (int64_t)te[d] + pe[d];
            out[d] = (q16_t)(s > Q16_MAX ? Q16_MAX : s < Q16_MIN ? Q16_MIN : s);
        }
    }
}

void proj(const q16_t *yin, const q16_t *wout, q16_t *logits,
          int32_t seq_len, int32_t d_model, int32_t vocab) {
    for (int t = 0; t < seq_len; t++)
        mvmul(wout, yin + t * d_model, logits + t * vocab, vocab, d_model);
}

/* ── qsparser ────────────────────────────────────────────────────────────── */

int qsparse(const char *text, int32_t *token_ids, int out_cap) {
    int n = 0;
    if (n < out_cap) token_ids[n++] = 1; /* BOS */
    for (const char *p = text; *p && n < out_cap - 1; p++) {
        unsigned char c = (unsigned char)*p;
        token_ids[n++] = (c < 128) ? (int32_t)c : 3; /* UNK for non-ASCII */
    }
    if (n < out_cap) token_ids[n++] = 2; /* EOS */
    return n;
}

void qsdecode(const int32_t *token_ids, int n, char *buf, int buf_cap) {
    int out = 0;
    for (int i = 0; i < n && out < buf_cap - 1; i++) {
        int32_t id = token_ids[i];
        if (id == 2) break; /* EOS */
        if (id >= 4 && id < 128)
            buf[out++] = (char)id;
    }
    buf[out] = '\0';
}

/* ── turboquant ─────────────────────────────────────────────────────────── */

float tq_f32_to_q8(const float *src, q8_t *dst, int n, float scale) {
    if (scale == 0.0f) {
        float mx = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = src[i] < 0.0f ? -src[i] : src[i];
            if (a > mx) mx = a;
        }
        scale = (mx > 0.0f) ? 127.0f / mx : 1.0f;
    }
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)(src[i] * scale * Q8_ONE + 0.5f);
        if (v > Q8_MAX) v = Q8_MAX;
        if (v < Q8_MIN) v = Q8_MIN;
        dst[i] = (q8_t)v;
    }
    return scale;
}

void tq_q8_to_q16(const q8_t *src, q16_t *dst, int n, float scale) {
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    for (int i = 0; i < n; i++) {
        float f   = ((float)src[i] / Q8_ONE) * inv;
        int64_t v = (int64_t)(f * Q16_ONE);
        dst[i] = (q16_t)(v > Q16_MAX ? Q16_MAX : v < Q16_MIN ? Q16_MIN : v);
    }
}

void tq_q16_to_q8(const q16_t *src, q8_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        int32_t v = src[i] >> 8; /* Q8.16 → Q8.8: drop 8 fractional bits */
        dst[i] = (q8_t)(v > Q8_MAX ? Q8_MAX : v < Q8_MIN ? Q8_MIN : v);
    }
}

void tq_q8_to_q16_exact(const q8_t *src, q16_t *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = (q16_t)((int32_t)src[i] << 8); /* Q8.8 → Q8.16: shift left 8 */
}

/* ── kvcache ─────────────────────────────────────────────────────────────── */

struct KVCache {
    int   n_layers;
    int   max_seq;
    int   d_kv;
    int   pos;
    q8_t *data; /* [n_layers][2][max_seq][d_kv] in Q8.8 */
};

KVCache *kvc_create(int n_layers, int max_seq, int d_kv) {
    KVCache *c = (KVCache *)malloc(sizeof(KVCache));
    if (!c) return NULL;
    c->n_layers = n_layers;
    c->max_seq  = max_seq;
    c->d_kv     = d_kv;
    c->pos      = 0;
    size_t sz = (size_t)n_layers * 2 * max_seq * d_kv * sizeof(q8_t);
    c->data = (q8_t *)calloc(1, sz);
    if (!c->data) { free(c); return NULL; }
    return c;
}

void kvc_free(KVCache *c) {
    if (!c) return;
    free(c->data);
    free(c);
}

void kvc_insert(KVCache *c, int layer, int pos,
                const q16_t *k, const q16_t *v) {
    int D = c->d_kv, S = c->max_seq;
    q8_t *ks = c->data + ((size_t)(layer * 2 + 0) * S + pos) * D;
    q8_t *vs = c->data + ((size_t)(layer * 2 + 1) * S + pos) * D;
    tq_q16_to_q8(k, ks, D);
    tq_q16_to_q8(v, vs, D);
    if (pos + 1 > c->pos) c->pos = pos + 1;
}

void kvc_query(KVCache *c, int layer, int max_pos,
               q16_t *k_out, q16_t *v_out) {
    int D = c->d_kv, S = c->max_seq;
    for (int p = 0; p < max_pos; p++) {
        q8_t *ks = c->data + ((size_t)(layer * 2 + 0) * S + p) * D;
        q8_t *vs = c->data + ((size_t)(layer * 2 + 1) * S + p) * D;
        tq_q8_to_q16_exact(ks, k_out + p * D, D);
        tq_q8_to_q16_exact(vs, v_out + p * D, D);
    }
}

void kvc_reset(KVCache *c) { c->pos = 0; }

int kvc_pos(const KVCache *c) { return c->pos; }

/* ── io ──────────────────────────────────────────────────────────────────── */

void putq16(q16_t v) {
    int neg = (v < 0);
    if (neg) v = -v;
    int32_t whole = (int32_t)((uint32_t)v >> 16);
    int32_t frac  = (int32_t)((uint32_t)v & 0xFFFF);
    int32_t dec   = (int32_t)(((int64_t)frac * 10000) >> 16);
    if (neg) putchar('-');
    printf("%d.%04d", whole, dec);
}

void putvec16(const q16_t *v, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        putq16(v[i]);
        if (i < n - 1) printf(", ");
    }
    putchar(']');
}
