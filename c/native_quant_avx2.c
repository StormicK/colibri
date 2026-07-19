/* AVX2 integer-dot kernels for the native FP4 expert tensors.
 *
 * The FP4 expert format (32-column blocks of E2M1 nibbles with one UE8M0
 * scale per block) is the MXFP4 layout, and doubled E2M1 values are small
 * integers: 2*{0,.5,1,1.5,2,3,4,6} = {0,1,2,3,4,6,8,12}. That makes the
 * fastest AVX2 recipe an int8 dot product (the same maddubs idiom as the
 * GLM engine's quantized kernels and ggml's MXFP4 path):
 *
 *   nibbles --pshufb LUT--> signed int8 weights (exact, value*2)
 *   activations --absmax per 32-block--> int8 + one f32 scale per block
 *   maddubs/madd            -> exact int32 dot per block
 *   int32 * (wscale*ascale/2) accumulated in f32 across blocks
 *
 * Numerics note: the reference path simulates E4M3 activation quantization
 * (3 mantissa bits, blocks of 128) and accumulates f32 per element. This
 * path uses int8 activations (blocks of 32) and exact integer accumulation
 * inside each block, which measures tighter than the E4M3 round-trip - but
 * it rounds differently, so greedy token streams may fork (same
 * kernel-family sensitivity as GLM #100). COLI_V4_AVX2=0 restores the
 * float kernels at runtime.
 *
 * The float implementations stay compiled in as validation partners and as
 * the fallback for !AVX2 builds, disabled dispatch, and unusual shapes.
 */
#include "native_quant_parallel.c"
#include "native_quant_dual.c"

/* Float batch kernels (native_quant_batch_avx512.c). */
int coli_fp4_matmul_batch_float_v4(float *outputs, const ColiTensorView *weight,
                                   const float *inputs, int batch);
int coli_fp8_matmul_batch_float_v6(float *outputs, const ColiTensorView *weight,
                                   const float *inputs, int batch);

static int avx2_dispatch_enabled(void) {
    const char *env = getenv("COLI_V4_AVX2");
    return !(env && env[0] == '0' && !env[1]);
}

static int fp4_view_standard(const ColiTensorView *w) {
    return w && w->format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           w->scale_format == COLI_SCALE_UE8M0 && w->data && w->scales &&
           w->rows >= 1 && w->columns >= 1 && !(w->columns % 128) &&
           w->block_rows == 1 && w->block_columns == 32 &&
           w->data_bytes == (size_t)w->rows * (size_t)w->columns / 2 &&
           w->scale_bytes == (size_t)w->rows * (size_t)w->columns / 32;
}

static int fp8_view_standard(const ColiTensorView *w) {
    return w && w->format == COLI_TENSOR_FP8_E4M3_BLOCK &&
           w->scale_format == COLI_SCALE_UE8M0 && w->data && w->scales &&
           w->rows >= 1 && w->columns >= 1 && !(w->columns % 128) &&
           w->block_rows == 128 && w->block_columns == 128 &&
           w->data_bytes == (size_t)w->rows * (size_t)w->columns &&
           w->scale_bytes == (size_t)((w->rows + 127) / 128) *
                             ((size_t)w->columns / 128);
}

#ifdef __AVX2__

#include <immintrin.h>

/* Idempotent fill of the UE8M0 decode table; a benign race writes the same
 * deterministic values. */
static const float *e8m0_table(void) {
    static float table[256];
    static volatile int ready;
    if (!ready) {
        for (int i = 0; i < 256; i++)
            table[i] = coli_e8m0_decode((uint8_t)i);
        ready = 1;
    }
    return table;
}

/* Signed doubled-E2M1 values, repeated across both 128-bit lanes. */
static inline __m256i fp4_value_lut(void) {
    return _mm256_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
}

/* 16 packed bytes -> 32 signed int8 weights (value*2) in column order:
 * even columns sit in the low nibble, odd columns in the high nibble. */
static inline __m256i fp4_weights32(const uint8_t *packed, __m256i lut) {
    __m128i raw = _mm_loadu_si128((const __m128i *)packed);
    __m128i mask = _mm_set1_epi8(15);
    __m128i lo = _mm_and_si128(raw, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    __m256i codes = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                     _mm_unpacklo_epi8(lo, hi));
    return _mm256_shuffle_epi8(lut, codes);
}

/* Exact int32 dot of 32 weight/activation int8 pairs (8 partial lanes).
 * maddubs needs one unsigned operand: |w| <= 12 is the unsigned side and
 * the weight sign moves onto the activation. Pair sums stay < 2^12. */
static inline __m256i fp4_dot32(__m256i w, __m256i aq) {
    __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w),
                                       _mm256_sign_epi8(aq, w));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

/* Symmetric absmax int8 activation quantization, one f32 scale per
 * 32-column block (matching the weight block granularity). */
static void quantize_activation_i8(int8_t *quantized, float *scales,
                                   const float *input, size_t columns) {
    for (size_t base = 0; base < columns; base += 32) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; i++) {
            float magnitude = fabsf(input[base + i]);
            if (magnitude > amax) amax = magnitude;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inverse = 1.0f / scale;
        scales[base / 32] = scale;
        for (size_t i = 0; i < 32; i++) {
            long rounded = lrintf(input[base + i] * inverse);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            quantized[base + i] = (int8_t)rounded;
        }
    }
}

/* One row x quantized activation. The 0.5f undoes the doubled LUT. */
static inline float fp4_row_dot(const uint8_t *row_data,
                                const uint8_t *row_scales,
                                const int8_t *aq, const float *ascales,
                                const float *e8, size_t columns,
                                __m256i lut) {
    __m256 acc = _mm256_setzero_ps();
    for (size_t base = 0; base < columns; base += 32) {
        __m256i w = fp4_weights32(row_data + base / 2, lut);
        __m256i aqv = _mm256_loadu_si256((const __m256i *)(aq + base));
        float s = e8[row_scales[base / 32]] * ascales[base / 32] * 0.5f;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(fp4_dot32(w, aqv)),
                              _mm256_set1_ps(s), acc);
    }
    return hsum256(acc);
}

int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !input || !fp4_view_standard(weight) ||
        !avx2_dispatch_enabled())
        return coli_fp4_matvec_float_ref(output, weight, input);
    size_t columns = (size_t)weight->columns;
    int8_t *aq = malloc(columns);
    float *ascales = malloc(columns / 32 * sizeof(*ascales));
    if (!aq || !ascales) {
        free(ascales); free(aq);
        return coli_fp4_matvec_float_ref(output, weight, input);
    }
    quantize_activation_i8(aq, ascales, input, columns);
    const float *e8 = e8m0_table();
    const __m256i lut = fp4_value_lut();
    const uint8_t *data = weight->data, *scales = weight->scales;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++)
        output[row] = fp4_row_dot(data + (size_t)row * packed_stride,
                                  scales + (size_t)row * scale_stride,
                                  aq, ascales, e8, columns, lut);
    free(ascales); free(aq);
    return 0;
}

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b,
                             const float *input) {
    if (!output_a || !output_b || !input ||
        !fp4_view_standard(a) || !fp4_view_standard(b) ||
        a->rows != b->rows || a->columns != b->columns ||
        !avx2_dispatch_enabled())
        return coli_fp4_dual_matvec_float_ref(output_a, output_b, a, b,
                                              input);
    size_t columns = (size_t)a->columns;
    int8_t *aq = malloc(columns);
    float *ascales = malloc(columns / 32 * sizeof(*ascales));
    if (!aq || !ascales) {
        free(ascales); free(aq);
        return coli_fp4_dual_matvec_float_ref(output_a, output_b, a, b,
                                              input);
    }
    quantize_activation_i8(aq, ascales, input, columns);
    const float *e8 = e8m0_table();
    const __m256i lut = fp4_value_lut();
    const uint8_t *data_a = a->data, *scales_a = a->scales;
    const uint8_t *data_b = b->data, *scales_b = b->scales;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < a->rows; row++) {
        size_t row_data = (size_t)row * packed_stride;
        size_t row_scale = (size_t)row * scale_stride;
        output_a[row] = fp4_row_dot(data_a + row_data, scales_a + row_scale,
                                    aq, ascales, e8, columns, lut);
        output_b[row] = fp4_row_dot(data_b + row_data, scales_b + row_scale,
                                    aq, ascales, e8, columns, lut);
    }
    free(ascales); free(aq);
    return 0;
}

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    enum { BATCH_MAX = 64 };
    if (!outputs || !inputs || batch < 1 || batch > BATCH_MAX ||
        !fp4_view_standard(weight) || !avx2_dispatch_enabled())
        return coli_fp4_matmul_batch_float_v4(outputs, weight, inputs,
                                              batch);
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    int8_t *aq = malloc((size_t)batch * columns);
    float *ascales = malloc((size_t)batch * (columns / 32) *
                            sizeof(*ascales));
    if (!aq || !ascales) {
        free(ascales); free(aq);
        return coli_fp4_matmul_batch_float_v4(outputs, weight, inputs,
                                              batch);
    }
    for (int item = 0; item < batch; item++)
        quantize_activation_i8(aq + (size_t)item * columns,
                               ascales + (size_t)item * (columns / 32),
                               inputs + (size_t)item * columns, columns);
    const float *e8 = e8m0_table();
    const __m256i lut = fp4_value_lut();
    const uint8_t *data = weight->data, *scales = weight->scales;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    /* Weights unpack once per block and stay in registers across the batch;
     * per-item accumulation order matches fp4_row_dot exactly, so each
     * item's column equals the single-matvec result bit for bit. */
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++) {
        const uint8_t *row_data = data + (size_t)row * packed_stride;
        const uint8_t *row_scales = scales + (size_t)row * scale_stride;
        __m256 acc[BATCH_MAX];
        for (int item = 0; item < batch; item++)
            acc[item] = _mm256_setzero_ps();
        for (size_t base = 0; base < columns; base += 32) {
            __m256i w = fp4_weights32(row_data + base / 2, lut);
            float wscale = e8[row_scales[base / 32]] * 0.5f;
            for (int item = 0; item < batch; item++) {
                const int8_t *item_aq = aq + (size_t)item * columns + base;
                __m256i aqv = _mm256_loadu_si256((const __m256i *)item_aq);
                float s = wscale *
                          ascales[(size_t)item * scale_stride + base / 32];
                acc[item] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(fp4_dot32(w, aqv)),
                    _mm256_set1_ps(s), acc[item]);
            }
        }
        for (int item = 0; item < batch; item++)
            outputs[(size_t)item * rows + (size_t)row] = hsum256(acc[item]);
    }
    free(ascales); free(aq);
    return 0;
}

/* ---- FP8 (E4M3, 128x128 UE8M0 block scales): the attention projections.
 * Same float math as the reference - E4M3-simulated activations, f32
 * accumulation - but the weight dequant runs 8 lanes wide: fp32 bits are
 * built directly from the E4M3 fields (bias rebase 7 -> 127), with a
 * branchless blend for e=0 denormals (value = m * 2^-9). The two NaN
 * encodings (0x7f/0xff) decode to finite 480 instead of NaN; NaN weights
 * would already poison the reference path, so no valid checkpoint hits
 * this. Accumulation is per 128-column block, then one FMA by the block
 * scale - the same product grouping as the reference, so results differ
 * only by float summation order. */
static inline __m256 fp8_values8(const uint8_t *weights) {
    __m256i v = _mm256_cvtepu8_epi32(
        _mm_loadl_epi64((const __m128i *)weights));
    __m256i mag = _mm256_and_si256(v, _mm256_set1_epi32(0x7f));
    __m256i sign = _mm256_slli_epi32(
        _mm256_and_si256(v, _mm256_set1_epi32(0x80)), 24);
    __m256 normal = _mm256_castsi256_ps(_mm256_add_epi32(
        _mm256_slli_epi32(mag, 20), _mm256_set1_epi32(120 << 23)));
    __m256 denormal = _mm256_mul_ps(_mm256_cvtepi32_ps(mag),
                                    _mm256_set1_ps(1.0f / 512.0f));
    __m256 magnitude = _mm256_blendv_ps(normal, denormal,
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(
            _mm256_srli_epi32(mag, 3), _mm256_setzero_si256())));
    return _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_castps_si256(magnitude), sign));
}

static inline float fp8_row_dot(const uint8_t *row_data,
                                const uint8_t *scales, size_t scale_base,
                                const float *activation, const float *e8,
                                size_t columns) {
    __m256 acc = _mm256_setzero_ps();
    for (size_t base = 0; base < columns; base += 128) {
        __m256 block = _mm256_setzero_ps();
        for (size_t i = 0; i < 128; i += 8)
            block = _mm256_fmadd_ps(fp8_values8(row_data + base + i),
                                    _mm256_loadu_ps(activation + base + i),
                                    block);
        acc = _mm256_fmadd_ps(block,
                              _mm256_set1_ps(e8[scales[scale_base + base / 128]]),
                              acc);
    }
    return hsum256(acc);
}

/* Shared activation prep: the reference's E4M3 QDQ in blocks of 128. */
static float *fp8_prepare_activation(const float *input, size_t columns,
                                     int items) {
    float *activation = malloc((size_t)items * columns * sizeof(*activation));
    uint8_t *scratch = malloc(columns / 128);
    if (!activation || !scratch) {
        free(scratch); free(activation);
        return NULL;
    }
    for (int item = 0; item < items; item++)
        if (coli_fp8_activation_qdq_ref(activation + (size_t)item * columns,
                                        scratch, input + (size_t)item * columns,
                                        columns, 128)) {
            free(scratch); free(activation);
            return NULL;
        }
    free(scratch);
    return activation;
}

int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !input || !fp8_view_standard(weight) ||
        !avx2_dispatch_enabled())
        return coli_fp8_matvec_float_ref(output, weight, input);
    size_t columns = (size_t)weight->columns;
    size_t scale_columns = columns / 128;
    float *activation = fp8_prepare_activation(input, columns, 1);
    if (!activation)
        return coli_fp8_matvec_float_ref(output, weight, input);
    const float *e8 = e8m0_table();
    const uint8_t *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++)
        output[row] = fp8_row_dot(data + (size_t)row * columns, scales,
                                  ((size_t)row / 128) * scale_columns,
                                  activation, e8, columns);
    free(activation);
    return 0;
}

int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b, const float *input) {
    if (!output_a || !output_b || !input ||
        !fp8_view_standard(a) || !fp8_view_standard(b) ||
        a->rows != b->rows || a->columns != b->columns ||
        !avx2_dispatch_enabled())
        return coli_fp8_dual_matvec_float_ref(output_a, output_b, a, b,
                                              input);
    size_t columns = (size_t)a->columns;
    size_t scale_columns = columns / 128;
    float *activation = fp8_prepare_activation(input, columns, 1);
    if (!activation)
        return coli_fp8_dual_matvec_float_ref(output_a, output_b, a, b,
                                              input);
    const float *e8 = e8m0_table();
    const uint8_t *data_a = a->data, *scales_a = a->scales;
    const uint8_t *data_b = b->data, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < a->rows; row++) {
        size_t row_data = (size_t)row * columns;
        size_t scale_base = ((size_t)row / 128) * scale_columns;
        output_a[row] = fp8_row_dot(data_a + row_data, scales_a, scale_base,
                                    activation, e8, columns);
        output_b[row] = fp8_row_dot(data_b + row_data, scales_b, scale_base,
                                    activation, e8, columns);
    }
    free(activation);
    return 0;
}

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    enum { BATCH_MAX = 64 };
    if (!outputs || !inputs || batch < 1 || batch > BATCH_MAX ||
        !fp8_view_standard(weight) || !avx2_dispatch_enabled())
        return coli_fp8_matmul_batch_float_v6(outputs, weight, inputs,
                                              batch);
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t scale_columns = columns / 128;
    float *activation = fp8_prepare_activation(inputs, columns, batch);
    if (!activation)
        return coli_fp8_matmul_batch_float_v6(outputs, weight, inputs,
                                              batch);
    const float *e8 = e8m0_table();
    const uint8_t *data = weight->data, *scales = weight->scales;
    /* Weights convert once per 8 columns and stay in registers across the
     * batch; per-item block grouping matches fp8_row_dot exactly, so each
     * item's column equals the single-matvec result bit for bit. */
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++) {
        const uint8_t *row_data = data + (size_t)row * columns;
        size_t scale_base = ((size_t)row / 128) * scale_columns;
        __m256 acc[BATCH_MAX], block[BATCH_MAX];
        for (int item = 0; item < batch; item++)
            acc[item] = _mm256_setzero_ps();
        for (size_t base = 0; base < columns; base += 128) {
            for (int item = 0; item < batch; item++)
                block[item] = _mm256_setzero_ps();
            for (size_t i = 0; i < 128; i += 8) {
                __m256 values = fp8_values8(row_data + base + i);
                for (int item = 0; item < batch; item++)
                    block[item] = _mm256_fmadd_ps(values,
                        _mm256_loadu_ps(activation + (size_t)item * columns +
                                        base + i),
                        block[item]);
            }
            __m256 scale = _mm256_set1_ps(
                e8[scales[scale_base + base / 128]]);
            for (int item = 0; item < batch; item++)
                acc[item] = _mm256_fmadd_ps(block[item], scale, acc[item]);
        }
        for (int item = 0; item < batch; item++)
            outputs[(size_t)item * rows + (size_t)row] = hsum256(acc[item]);
    }
    free(activation);
    return 0;
}

#else /* !__AVX2__: keep the exported names, forward to the float kernels. */

int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    (void)avx2_dispatch_enabled; (void)fp4_view_standard;
    return coli_fp4_matvec_float_ref(output, weight, input);
}

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b, const float *input) {
    return coli_fp4_dual_matvec_float_ref(output_a, output_b, a, b, input);
}

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    return coli_fp4_matmul_batch_float_v4(outputs, weight, inputs, batch);
}

int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    (void)fp8_view_standard;
    return coli_fp8_matvec_float_ref(output, weight, input);
}

int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b, const float *input) {
    return coli_fp8_dual_matvec_float_ref(output_a, output_b, a, b, input);
}

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    return coli_fp8_matmul_batch_float_v6(outputs, weight, inputs, batch);
}

#endif /* __AVX2__ */
