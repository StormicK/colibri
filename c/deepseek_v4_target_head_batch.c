#include "deepseek_v4_target_head_batch.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "deepseek_v4_head_cache.h"
#include "deepseek_v4_math.h"
#include "native_quant.h"
#include "tensor_io.h"

int coli_v4_target_load_embeddings(float *states_hc,
                                    const ColiSafetensorsIndex *index,
                                    const ColiDeepSeekV4Config *config,
                                    const int *tokens, int batch) {
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    if (!states_hc || !index || !config || !tokens || batch < 1 || batch > 64 ||
        !embed || embed->dtype != COLI_ST_BF16) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    uint16_t *raw = malloc((size_t)d * sizeof(*raw));
    if (!raw) return -1;
    for (int item = 0; item < batch; item++) {
        if (tokens[item] < 0 || tokens[item] >= config->vocab_size ||
            coli_st_read_at(index, embed->shard,
                            embed->offset + (uint64_t)tokens[item] * d * 2,
                            (size_t)d * sizeof(*raw), raw)) {
            free(raw); return -1;
        }
        float *state = states_hc + (size_t)item * hc * d;
        for (int copy = 0; copy < hc; copy++)
            for (int i = 0; i < d; i++)
                state[(size_t)copy * d + i] = coli_bf16_decode(raw[i]);
    }
    free(raw); return 0;
}

int coli_v4_target_head_argmax_batch(
    const float *states_hc, const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, int batch,
    int *tokens, float *logits, char *error, size_t error_size) {
    if (!states_hc || !index || !config || batch < 1 || batch > 64 || !tokens)
        return -1;
    ColiFloatTensor function = {0}, base = {0}, scale = {0}, norm = {0};
    if (coli_tensor_load_f32(&function, index, "hc_head_fn", error, error_size) ||
        coli_tensor_load_f32(&base, index, "hc_head_base", error, error_size) ||
        coli_tensor_load_f32(&scale, index, "hc_head_scale", error, error_size) ||
        coli_tensor_load_f32(&norm, index, "norm.weight", error, error_size))
        return -1;
    int d = config->hidden_size, hc = config->hc_mult, flat = hc * d;
    float *hidden = malloc((size_t)batch * d * sizeof(*hidden));
    if (!hidden || hc > 16) { free(hidden); return -1; }
    for (int item = 0; item < batch; item++) {
        const float *state = states_hc + (size_t)item * flat;
        float *out = hidden + (size_t)item * d;
        float square = 0.0f, pre[16];
        for (int i = 0; i < flat; i++) square += state[i] * state[i];
        float inverse = 1.0f / sqrtf(square / flat + config->rms_norm_eps);
        for (int copy = 0; copy < hc; copy++) {
            float mix = 0.0f;
            for (int i = 0; i < flat; i++)
                mix += function.data[(size_t)copy * flat + i] * state[i];
            float z = mix * inverse * scale.data[0] + base.data[copy];
            float sigmoid = z >= 0.0f ? 1.0f / (1.0f + expf(-z))
                                      : expf(z) / (1.0f + expf(z));
            pre[copy] = sigmoid + config->hc_eps;
        }
        for (int i = 0; i < d; i++) {
            float value = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                value += pre[copy] * state[(size_t)copy * d + i];
            out[i] = coli_bf16_round(value);
        }
        if (coli_v4_rmsnorm(out, out, norm.data, d, config->rms_norm_eps))
            return -1;
        coli_bf16_round_array(out, (size_t)d);
        tokens[item] = -1; if (logits) logits[item] = -FLT_MAX;
    }
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
#ifndef COLI_V4_DISABLE_ZERO_COPY_HEAD
    const uint16_t *resident = head ? coli_v4_head_cache_data(
        head->shard, head->offset, (size_t)head->nbytes) : NULL;
    if (resident) {
        float *resident_scores = malloc(
            (size_t)batch * config->vocab_size * sizeof(*resident_scores));
        if (!resident_scores) return -1;
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < config->vocab_size; row++) {
            float sums[64] = {0};
            const uint16_t *weight = resident + (size_t)row * d;
            for (int i = 0; i < d; i++) {
                float decoded = coli_bf16_decode(weight[i]);
                for (int item = 0; item < batch; item++)
                    sums[item] += decoded * hidden[(size_t)item * d + i];
            }
            for (int item = 0; item < batch; item++)
                resident_scores[(size_t)item * config->vocab_size + row] =
                    sums[item];
        }
        for (int item = 0; item < batch; item++)
            for (int row = 0; row < config->vocab_size; row++) {
                float score = resident_scores[
                    (size_t)item * config->vocab_size + row];
                float current = logits ? logits[item] :
                    (tokens[item] < 0 ? -FLT_MAX : 0.0f);
                if (tokens[item] < 0 || score > current) {
                    tokens[item] = row;
                    if (logits) logits[item] = score;
                }
            }
        free(resident_scores);
        goto head_complete;
    }
#endif
    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * d * sizeof(*raw));
    float *scores = malloc((size_t)batch * ROWS * sizeof(*scores));
    if (!head || head->dtype != COLI_ST_BF16 || !raw || !scores) return -1;
    for (int start = 0; start < config->vocab_size; start += ROWS) {
        int rows = config->vocab_size - start < ROWS
            ? config->vocab_size - start : ROWS;
        if (coli_st_read_at(index, head->shard,
                            head->offset + (uint64_t)start * d * 2,
                            (size_t)rows * d * sizeof(*raw), raw)) return -1;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int item = 0; item < batch; item++) for (int row = 0; row < rows; row++) {
            float sum = 0.0f; const uint16_t *weight = raw + (size_t)row * d;
            const float *input = hidden + (size_t)item * d;
            for (int i = 0; i < d; i++) sum += coli_bf16_decode(weight[i]) * input[i];
            scores[(size_t)item * ROWS + row] = sum;
        }
        for (int item = 0; item < batch; item++) for (int row = 0; row < rows; row++) {
            float score = scores[(size_t)item * ROWS + row];
            float current = logits ? logits[item] :
                (tokens[item] < 0 ? -FLT_MAX : 0.0f);
            if (tokens[item] < 0 || score > current) {
                tokens[item] = start + row;
                if (logits) logits[item] = score;
            }
        }
    }
    free(scores); free(raw);
#ifndef COLI_V4_DISABLE_ZERO_COPY_HEAD
head_complete:
#endif
    free(hidden);
    coli_float_tensor_free(&norm); coli_float_tensor_free(&scale);
    coli_float_tensor_free(&base); coli_float_tensor_free(&function);
    return 0;
}
