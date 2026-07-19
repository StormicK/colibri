/* AVX2 int8 FP4 kernels (native_quant_avx2.c) against the float reference
 * kernels and an fp64 exact oracle.
 *
 * - accuracy: both kernel families vs an exact double-precision dot of the
 *   dequantized weights with the RAW (unquantized) activations; the int8
 *   path must stay within its bound and not be materially worse than the
 *   E4M3-simulating float path.
 * - exactness: dual == 2x single and every batch item == single, bit for
 *   bit (they share one accumulation order by construction).
 * - dispatch: COLI_V4_AVX2=0 must reproduce the float kernels bit for bit.
 */
#include "../native_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int coli_fp4_matvec_float_ref(float *output, const ColiTensorView *weight,
                              const float *input);
int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a, const ColiTensorView *b,
                             const float *input);
int coli_fp4_dual_matvec_float_ref(float *output_a, float *output_b,
                                   const ColiTensorView *a,
                                   const ColiTensorView *b,
                                   const float *input);
int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch);
int coli_fp4_matmul_batch_float_v4(float *outputs,
                                   const ColiTensorView *weight,
                                   const float *inputs, int batch);
int coli_fp8_matvec_float_ref(float *output, const ColiTensorView *weight,
                              const float *input);
int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a, const ColiTensorView *b,
                             const float *input);
int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch);

#ifndef __AVX2__

int main(void) {
    printf("native quant AVX2 tests: skipped (built without AVX2)\n");
    return 0;
}

#else

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint32_t rng32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static float rng_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng32() >> 8) / 16777216.0f);
}

static void fill_fp4_tensor(ColiTensorView *view, uint8_t *data,
                            uint8_t *scales, int64_t rows, int64_t columns) {
    size_t data_bytes = (size_t)rows * (size_t)columns / 2;
    size_t scale_bytes = (size_t)rows * (size_t)columns / 32;
    for (size_t i = 0; i < data_bytes; i++) data[i] = (uint8_t)rng32();
    for (size_t i = 0; i < scale_bytes; i++)
        scales[i] = (uint8_t)(119 + rng32() % 15); /* 2^-8 .. 2^6 */
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = data;
    view->scales = scales;
    view->data_bytes = data_bytes;
    view->scale_bytes = scale_bytes;
    view->rows = rows;
    view->columns = columns;
    view->block_rows = 1;
    view->block_columns = 32;
}

static void fill_input(float *input, size_t columns) {
    for (size_t i = 0; i < columns; i++) {
        input[i] = rng_uniform(-2.0f, 2.0f);
        if (rng32() % 37 == 0) input[i] *= 8.0f; /* occasional outlier */
    }
}

/* Exact fp64 dot of dequantized weights with the raw activations. */
static void exact_matvec(double *output, const ColiTensorView *w,
                         const float *input) {
    const uint8_t *data = w->data, *scales = w->scales;
    size_t columns = (size_t)w->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    for (int64_t row = 0; row < w->rows; row++) {
        double sum = 0.0;
        for (size_t column = 0; column < columns; column++) {
            uint8_t byte = data[(size_t)row * packed_stride + column / 2];
            uint8_t code = column & 1 ? byte >> 4 : byte & 15;
            double scale = (double)coli_e8m0_decode(
                scales[(size_t)row * scale_stride + column / 32]);
            sum += (double)input[column] *
                   (double)coli_e2m1_decode(code) * scale;
        }
        output[row] = sum;
    }
}

static double rel_l2_error(const float *got, const double *truth,
                           int64_t rows) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        double diff = (double)got[i] - truth[i];
        num += diff * diff;
        den += truth[i] * truth[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static void set_avx2_dispatch(int enabled) {
#ifdef _WIN32
    _putenv(enabled ? "COLI_V4_AVX2=" : "COLI_V4_AVX2=0");
#else
    if (enabled)
        unsetenv("COLI_V4_AVX2");
    else
        setenv("COLI_V4_AVX2", "0", 1);
#endif
}

static int check_shape(int64_t rows, int64_t columns) {
    size_t r = (size_t)rows, c = (size_t)columns;
    uint8_t *data_a = malloc(r * c / 2), *scales_a = malloc(r * c / 32);
    uint8_t *data_b = malloc(r * c / 2), *scales_b = malloc(r * c / 32);
    float *input = malloc(c * sizeof(float));
    float *out_avx2 = malloc(r * sizeof(float));
    float *out_float = malloc(r * sizeof(float));
    float *dual_a = malloc(r * sizeof(float));
    float *dual_b = malloc(r * sizeof(float));
    double *truth = malloc(r * sizeof(double));
    ColiTensorView a, b;
    int failed = 1;
    if (!data_a || !scales_a || !data_b || !scales_b || !input ||
        !out_avx2 || !out_float || !dual_a || !dual_b || !truth)
        goto done;
    fill_fp4_tensor(&a, data_a, scales_a, rows, columns);
    fill_fp4_tensor(&b, data_b, scales_b, rows, columns);
    fill_input(input, c);

    /* accuracy vs the exact oracle */
    if (coli_fp4_matvec_ref(out_avx2, &a, input) ||
        coli_fp4_matvec_float_ref(out_float, &a, input)) {
        fprintf(stderr, "matvec failed (%lld x %lld)\n",
                (long long)rows, (long long)columns);
        goto done;
    }
    exact_matvec(truth, &a, input);
    double err_avx2 = rel_l2_error(out_avx2, truth, rows);
    double err_float = rel_l2_error(out_float, truth, rows);
    if (err_avx2 > 0.05 || err_avx2 > err_float * 1.5 + 1e-4) {
        fprintf(stderr,
                "accuracy: avx2 %.5f vs float %.5f (%lld x %lld)\n",
                err_avx2, err_float, (long long)rows, (long long)columns);
        goto done;
    }

    /* dual == two singles, bit for bit */
    if (coli_fp4_dual_matvec_ref(dual_a, dual_b, &a, &b, input) ||
        memcmp(dual_a, out_avx2, r * sizeof(float))) {
        fprintf(stderr, "dual != single (a)\n");
        goto done;
    }
    if (coli_fp4_matvec_ref(out_float, &b, input) ||
        memcmp(dual_b, out_float, r * sizeof(float))) {
        fprintf(stderr, "dual != single (b)\n");
        goto done;
    }

    failed = 0;
done:
    free(truth); free(dual_b); free(dual_a); free(out_float);
    free(out_avx2); free(input); free(data_b); free(scales_b);
    free(data_a); free(scales_a);
    return failed;
}

static int check_batch(int64_t rows, int64_t columns, int batch) {
    size_t r = (size_t)rows, c = (size_t)columns;
    uint8_t *data = malloc(r * c / 2), *scales = malloc(r * c / 32);
    float *inputs = malloc((size_t)batch * c * sizeof(float));
    float *outputs = malloc((size_t)batch * r * sizeof(float));
    float *single = malloc(r * sizeof(float));
    ColiTensorView w;
    int failed = 1;
    if (!data || !scales || !inputs || !outputs || !single) goto done;
    fill_fp4_tensor(&w, data, scales, rows, columns);
    for (int item = 0; item < batch; item++)
        fill_input(inputs + (size_t)item * c, c);
    if (coli_fp4_matmul_batch_ref(outputs, &w, inputs, batch)) {
        fprintf(stderr, "batch matmul failed (batch=%d)\n", batch);
        goto done;
    }
    for (int item = 0; item < batch; item++) {
        if (coli_fp4_matvec_ref(single, &w, inputs + (size_t)item * c) ||
            memcmp(outputs + (size_t)item * r, single, r * sizeof(float))) {
            fprintf(stderr, "batch item %d != single (batch=%d)\n",
                    item, batch);
            goto done;
        }
    }
    failed = 0;
done:
    free(single); free(outputs); free(inputs); free(scales); free(data);
    return failed;
}

static void fill_fp8_tensor(ColiTensorView *view, uint8_t *data,
                            uint8_t *scales, int64_t rows, int64_t columns,
                            int denormals_only) {
    size_t data_bytes = (size_t)rows * (size_t)columns;
    size_t scale_bytes = (size_t)((rows + 127) / 128) *
                         ((size_t)columns / 128);
    for (size_t i = 0; i < data_bytes; i++) {
        uint8_t byte = (uint8_t)rng32();
        if (denormals_only) byte &= 0x87;         /* e=0: sign + mantissa */
        while ((byte & 0x7f) == 0x7f) byte = (uint8_t)rng32(); /* no NaN */
        data[i] = byte;
    }
    for (size_t i = 0; i < scale_bytes; i++)
        scales[i] = (uint8_t)(119 + rng32() % 15);
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = data;
    view->scales = scales;
    view->data_bytes = data_bytes;
    view->scale_bytes = scale_bytes;
    view->rows = rows;
    view->columns = columns;
    view->block_rows = 128;
    view->block_columns = 128;
}

/* fp64 dot of the dequantized FP8 weights with an already-QDQ'd activation:
 * both kernel families use this exact math, so they must land within float
 * summation-order distance of it. */
static void exact_fp8_matvec(double *output, const ColiTensorView *w,
                             const float *activation) {
    const uint8_t *data = w->data, *scales = w->scales;
    size_t columns = (size_t)w->columns, scale_columns = columns / 128;
    for (int64_t row = 0; row < w->rows; row++) {
        double sum = 0.0;
        size_t scale_base = ((size_t)row / 128) * scale_columns;
        for (size_t column = 0; column < columns; column++)
            sum += (double)activation[column] *
                   (double)coli_e4m3fn_decode(data[(size_t)row * columns +
                                                   column]) *
                   (double)coli_e8m0_decode(scales[scale_base + column / 128]);
        output[row] = sum;
    }
}

static int check_fp8(int64_t rows, int64_t columns, int denormals_only) {
    size_t r = (size_t)rows, c = (size_t)columns;
    size_t scale_bytes = (size_t)((rows + 127) / 128) * (c / 128);
    uint8_t *data_a = malloc(r * c), *scales_a = malloc(scale_bytes);
    uint8_t *data_b = malloc(r * c), *scales_b = malloc(scale_bytes);
    float *input = malloc(c * sizeof(float));
    float *qdq = malloc(c * sizeof(float));
    uint8_t *qdq_scales = malloc(c / 128);
    float *out_avx2 = malloc(r * sizeof(float));
    float *out_float = malloc(r * sizeof(float));
    float *dual_a = malloc(r * sizeof(float));
    float *dual_b = malloc(r * sizeof(float));
    double *truth = malloc(r * sizeof(double));
    ColiTensorView a, b;
    int failed = 1;
    if (!data_a || !scales_a || !data_b || !scales_b || !input || !qdq ||
        !qdq_scales || !out_avx2 || !out_float || !dual_a || !dual_b ||
        !truth)
        goto done;
    fill_fp8_tensor(&a, data_a, scales_a, rows, columns, denormals_only);
    fill_fp8_tensor(&b, data_b, scales_b, rows, columns, denormals_only);
    fill_input(input, c);

    if (coli_fp8_matvec_ref(out_avx2, &a, input) ||
        coli_fp8_matvec_float_ref(out_float, &a, input) ||
        coli_fp8_activation_qdq_ref(qdq, qdq_scales, input, c, 128)) {
        fprintf(stderr, "fp8 matvec failed (%lld x %lld)\n",
                (long long)rows, (long long)columns);
        goto done;
    }
    exact_fp8_matvec(truth, &a, qdq);
    double err_avx2 = rel_l2_error(out_avx2, truth, rows);
    double err_float = rel_l2_error(out_float, truth, rows);
    if (err_avx2 > 1e-4 || err_float > 1e-4) {
        fprintf(stderr, "fp8 accuracy: avx2 %.3g float %.3g (%lld x %lld%s)\n",
                err_avx2, err_float, (long long)rows, (long long)columns,
                denormals_only ? ", denormals" : "");
        goto done;
    }

    if (coli_fp8_dual_matvec_ref(dual_a, dual_b, &a, &b, input) ||
        memcmp(dual_a, out_avx2, r * sizeof(float))) {
        fprintf(stderr, "fp8 dual != single (a)\n");
        goto done;
    }
    if (coli_fp8_matvec_ref(out_float, &b, input) ||
        memcmp(dual_b, out_float, r * sizeof(float))) {
        fprintf(stderr, "fp8 dual != single (b)\n");
        goto done;
    }

    failed = 0;
done:
    free(truth); free(dual_b); free(dual_a); free(out_float);
    free(out_avx2); free(qdq_scales); free(qdq); free(input);
    free(data_b); free(scales_b); free(data_a); free(scales_a);
    return failed;
}

static int check_fp8_batch(int64_t rows, int64_t columns, int batch) {
    size_t r = (size_t)rows, c = (size_t)columns;
    size_t scale_bytes = (size_t)((rows + 127) / 128) * (c / 128);
    uint8_t *data = malloc(r * c), *scales = malloc(scale_bytes);
    float *inputs = malloc((size_t)batch * c * sizeof(float));
    float *outputs = malloc((size_t)batch * r * sizeof(float));
    float *single = malloc(r * sizeof(float));
    ColiTensorView w;
    int failed = 1;
    if (!data || !scales || !inputs || !outputs || !single) goto done;
    fill_fp8_tensor(&w, data, scales, rows, columns, 0);
    for (int item = 0; item < batch; item++)
        fill_input(inputs + (size_t)item * c, c);
    if (coli_fp8_matmul_batch_ref(outputs, &w, inputs, batch)) {
        fprintf(stderr, "fp8 batch matmul failed (batch=%d)\n", batch);
        goto done;
    }
    for (int item = 0; item < batch; item++) {
        if (coli_fp8_matvec_ref(single, &w, inputs + (size_t)item * c) ||
            memcmp(outputs + (size_t)item * r, single, r * sizeof(float))) {
            fprintf(stderr, "fp8 batch item %d != single (batch=%d)\n",
                    item, batch);
            goto done;
        }
    }
    failed = 0;
done:
    free(single); free(outputs); free(inputs); free(scales); free(data);
    return failed;
}

static int check_dispatch_off(void) {
    enum { ROWS = 24, COLUMNS = 256, BATCH = 3 };
    uint8_t data[ROWS * COLUMNS / 2], scales[ROWS * COLUMNS / 32];
    float inputs[BATCH * COLUMNS];
    /* layout: [single row block | BATCH batch row blocks] */
    float got[(BATCH + 1) * ROWS], want[(BATCH + 1) * ROWS];
    float got_a[ROWS], got_b[ROWS], want_a[ROWS], want_b[ROWS];
    ColiTensorView w;
    fill_fp4_tensor(&w, data, scales, ROWS, COLUMNS);
    for (int item = 0; item < BATCH; item++)
        fill_input(inputs + (size_t)item * COLUMNS, COLUMNS);
    set_avx2_dispatch(0);
    int rc = coli_fp4_matvec_ref(got, &w, inputs) ||
             coli_fp4_dual_matvec_ref(got_a, got_b, &w, &w, inputs) ||
             coli_fp4_matmul_batch_ref(got + ROWS, &w, inputs, BATCH);
    set_avx2_dispatch(1);
    if (rc ||
        coli_fp4_matvec_float_ref(want, &w, inputs) ||
        coli_fp4_dual_matvec_float_ref(want_a, want_b, &w, &w, inputs) ||
        coli_fp4_matmul_batch_float_v4(want + ROWS, &w, inputs, BATCH))
        return 1;
    if (memcmp(got, want, sizeof(float) * (BATCH + 1) * ROWS) ||
        memcmp(got_a, want_a, sizeof(float) * ROWS) ||
        memcmp(got_b, want_b, sizeof(float) * ROWS)) {
        fprintf(stderr, "COLI_V4_AVX2=0 did not reproduce the float path\n");
        return 1;
    }
    return 0;
}

int main(void) {
    static const int64_t shapes[][2] = {
        {1, 128}, {3, 256}, {16, 7168}, {127, 1024},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        if (check_shape(shapes[i][0], shapes[i][1])) return 1;

    /* zero input: every kernel must produce exact zeros */
    {
        enum { ROWS = 8, COLUMNS = 128 };
        uint8_t data[ROWS * COLUMNS / 2], scales[ROWS * COLUMNS / 32];
        float input[COLUMNS] = {0}, out[ROWS];
        ColiTensorView w;
        fill_fp4_tensor(&w, data, scales, ROWS, COLUMNS);
        if (coli_fp4_matvec_ref(out, &w, input)) return 1;
        for (int i = 0; i < ROWS; i++)
            if (out[i] != 0.0f) return 1;
    }

    if (check_batch(64, 512, 1)) return 1;
    if (check_batch(64, 512, 5)) return 1;
    if (check_batch(33, 1024, 33)) return 1;
    if (check_batch(16, 7168, 24)) return 1;

    /* FP8: random tensors, a >1-row-block shape, and a denormal-only one */
    if (check_fp8(64, 512, 0)) return 1;
    if (check_fp8(300, 4096, 0)) return 1;
    if (check_fp8(64, 256, 1)) return 1;
    if (check_fp8_batch(64, 512, 1)) return 1;
    if (check_fp8_batch(300, 1024, 24)) return 1;

    if (check_dispatch_off()) return 1;

    printf("native quant AVX2 tests: ok\n");
    return 0;
}

#endif /* __AVX2__ */
