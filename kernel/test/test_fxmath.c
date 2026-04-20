/*
 * test_fxmath.c — Unit tests for fxmath.asm (Q8.16 fixed-point primitives)
 *
 * Build (Windows, MSVC):
 *   cl /nologo /O2 /I.. test_fxmath.c /link /LIBPATH:.. kernel.lib
 *
 * Build (Linux, gcc):
 *   gcc -O2 -I.. test_fxmath.c -L.. -lkernel -Wl,-rpath,.. -o test_fxmath
 *
 * Exit code: 0 = all pass, 1 = failure
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../kernel.h"

/* ── Test framework ────────────────────────────────────────────────────── */
static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) \
    do { \
        if (cond) { \
            printf("  PASS: " fmt "\n", ##__VA_ARGS__); \
            g_pass++; \
        } else { \
            printf("  FAIL: " fmt "\n", ##__VA_ARGS__); \
            g_fail++; \
        } \
    } while (0)

/* Convert float to Q8.16 */
static inline q16_t F(double v) {
    double r = v * 65536.0;
    if (r >  2147483647.0) return  2147483647;
    if (r < -2147483648.0) return -2147483648;
    return (q16_t)(long long)r;
}

/* Convert Q8.16 back to float */
static inline double Q(q16_t v) {
    return (double)v / 65536.0;
}

/* Tolerance check: |a - b| <= eps */
static int near(q16_t a, q16_t b, double eps_float) {
    double diff = fabs(Q(a) - Q(b));
    return diff <= eps_float;
}

/* ── Test cases ────────────────────────────────────────────────────────── */

void test_fxmul(void) {
    printf("\n[fxmul]\n");
    q16_t r;

    /* 1.0 * 1.0 = 1.0 */
    r = fxmul(F(1.0), F(1.0));
    CHECK(near(r, F(1.0), 0.00002), "1.0 * 1.0 = %.6f (expect 1.0)", Q(r));

    /* 2.0 * 3.0 = 6.0 */
    r = fxmul(F(2.0), F(3.0));
    CHECK(near(r, F(6.0), 0.001), "2.0 * 3.0 = %.6f (expect 6.0)", Q(r));

    /* 0.5 * 0.5 = 0.25 */
    r = fxmul(F(0.5), F(0.5));
    CHECK(near(r, F(0.25), 0.00002), "0.5 * 0.5 = %.6f (expect 0.25)", Q(r));

    /* -1.0 * 2.0 = -2.0 */
    r = fxmul(F(-1.0), F(2.0));
    CHECK(near(r, F(-2.0), 0.00002), "-1.0 * 2.0 = %.6f (expect -2.0)", Q(r));

    /* Saturation: 1000 * 1000 should clamp */
    r = fxmul(F(1000.0), F(1000.0));
    CHECK(r == Q16_MAX, "1000 * 1000 saturates to INT32_MAX (got %d)", r);

    /* 0.0 * anything = 0.0 */
    r = fxmul(F(0.0), F(123.456));
    CHECK(r == 0, "0.0 * 123.456 = 0 (got %d)", r);
}

void test_fxdiv(void) {
    printf("\n[fxdiv]\n");
    q16_t r;

    /* 1.0 / 1.0 = 1.0 */
    r = fxdiv(F(1.0), F(1.0));
    CHECK(near(r, F(1.0), 0.00002), "1.0 / 1.0 = %.6f (expect 1.0)", Q(r));

    /* 6.0 / 2.0 = 3.0 */
    r = fxdiv(F(6.0), F(2.0));
    CHECK(near(r, F(3.0), 0.001), "6.0 / 2.0 = %.6f (expect 3.0)", Q(r));

    /* 1.0 / 4.0 = 0.25 */
    r = fxdiv(F(1.0), F(4.0));
    CHECK(near(r, F(0.25), 0.00002), "1.0 / 4.0 = %.6f (expect 0.25)", Q(r));

    /* -4.0 / 2.0 = -2.0 */
    r = fxdiv(F(-4.0), F(2.0));
    CHECK(near(r, F(-2.0), 0.00002), "-4.0 / 2.0 = %.6f (expect -2.0)", Q(r));

    /* Division by zero: positive dividend → INT32_MAX */
    r = fxdiv(F(1.0), F(0.0));
    CHECK(r == Q16_MAX, "1.0 / 0.0 → INT32_MAX saturation (got %d)", r);

    /* Division by zero: negative dividend → INT32_MIN */
    r = fxdiv(F(-1.0), F(0.0));
    CHECK(r == Q16_MIN, "-1.0 / 0.0 → INT32_MIN saturation (got %d)", r);
}

void test_fxabs(void) {
    printf("\n[fxabs]\n");
    q16_t r;

    r = fxabs(F(3.14));
    CHECK(near(r, F(3.14), 0.0001), "fxabs(3.14) = %.6f", Q(r));

    r = fxabs(F(-3.14));
    CHECK(near(r, F(3.14), 0.0001), "fxabs(-3.14) = %.6f", Q(r));

    r = fxabs(F(0.0));
    CHECK(r == 0, "fxabs(0.0) = 0");

    r = fxabs(F(1.0));
    CHECK(near(r, F(1.0), 0.00002), "fxabs(1.0) = %.6f", Q(r));
}

void test_fxclamp(void) {
    printf("\n[fxclamp]\n");
    q16_t r;

    /* Value in range → unchanged */
    r = fxclamp(F(1.5), F(0.0), F(2.0));
    CHECK(near(r, F(1.5), 0.00002), "clamp(1.5, 0, 2) = %.6f", Q(r));

    /* Below lo → lo */
    r = fxclamp(F(-1.0), F(0.0), F(2.0));
    CHECK(near(r, F(0.0), 0.00002), "clamp(-1, 0, 2) = %.6f (expect 0)", Q(r));

    /* Above hi → hi */
    r = fxclamp(F(5.0), F(0.0), F(2.0));
    CHECK(near(r, F(2.0), 0.00002), "clamp(5, 0, 2) = %.6f (expect 2)", Q(r));
}

void test_fxmul_precision(void) {
    printf("\n[fxmul precision]\n");

    /* Test a range of values comparing to float reference */
    double errors[10];
    int i;
    double vals[] = {0.1, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.141, 7.99};
    int n = 10;
    double max_err = 0;

    for (i = 0; i < n; i++) {
        double a = vals[i];
        double b = vals[(i * 3) % n];
        double expected = a * b;
        q16_t result = fxmul(F(a), F(b));
        errors[i] = fabs(Q(result) - expected);
        if (errors[i] > max_err) max_err = errors[i];
    }

    CHECK(max_err < 0.001, "fxmul max error over 10 cases: %.8f (limit 0.001)", max_err);
}

int main(void) {
    printf("=== fxmath.asm unit tests (Q8.16 Fixed-Point) ===\n");
    printf("Q16_ONE = %d  (1.0)\n", Q16_ONE);
    printf("Q16_MAX = %d  (~%.1f)\n", Q16_MAX, Q(Q16_MAX));
    printf("Q16_MIN = %d  (~%.1f)\n", Q16_MIN, Q(Q16_MIN));

    test_fxmul();
    test_fxdiv();
    test_fxabs();
    test_fxclamp();
    test_fxmul_precision();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
