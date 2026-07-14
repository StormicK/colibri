#include "deepseek_v4_compressor.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_math.h"
#include "native_quant.h"

struct ColiDeepSeekV4CompressorState {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    int ratio;
    int layer;
    int hidden;
    int head_dim;
    int projection_dim;
    int state_rows;
    int rope_dim;
    int rotate_fp4;
    char prefix[96];
    float *kv_state;
    float *score_state;
};

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_value(const ColiDeepSeekV4LayerWeights *weights,
                               const char *suffix) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, NULL);
}

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size) {
    ColiDeepSeekV4CompressorOptions options = {
        "attn.compressor", config ? config->head_dim : 0, 0
    };
    return coli_v4_compressor_create_with_options(
        output, weights, config, &options, error, error_size);
}

int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **output,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size) {
    if (!output || !weights || !config || !options || !options->prefix ||
        !options->prefix[0] || options->head_dimension <= 0 ||
        (weights->plan.compression_ratio != 128 &&
         weights->plan.compression_ratio != 4))
        return set_error(error, error_size, "unsupported compressor ratio");
    if (strlen(options->prefix) >= sizeof(((ColiDeepSeekV4CompressorState *)0)->prefix))
        return set_error(error, error_size, "compressor prefix is too long");
    *output = NULL;
    ColiDeepSeekV4CompressorState *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating compressor");
    state->weights = weights;
    state->config = config;
    state->ratio = weights->plan.compression_ratio;
    state->layer = weights->plan.layer;
    state->hidden = config->hidden_size;
    state->head_dim = options->head_dimension;
    state->rotate_fp4 = options->rotate_fp4 != 0;
    memcpy(state->prefix, options->prefix, strlen(options->prefix) + 1);
    int overlap = state->ratio == 4;
    state->projection_dim = (1 + overlap) * state->head_dim;
    state->state_rows = (1 + overlap) * state->ratio;
    state->rope_dim = config->qk_rope_head_dim;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    state->kv_state = calloc(count, sizeof(*state->kv_state));
    state->score_state = malloc(count * sizeof(*state->score_state));
    if (!state->kv_state || !state->score_state) {
        coli_v4_compressor_destroy(state);
        return set_error(error, error_size, "out of memory allocating compressor state");
    }
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
    *output = state;
    return 0;
}

int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size) {
    if (!state || !weights ||
        weights->plan.layer != state->layer ||
        weights->plan.compression_ratio != state->ratio)
        return set_error(error, error_size, "incompatible compressor weights");
    state->weights = weights;
    return 0;
}

void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    memset(state->kv_state, 0, count * sizeof(*state->kv_state));
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
}

void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    free(state->score_state);
    free(state->kv_state);
    free(state);
}

int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size) {
    if (!state || !produced || !input || position < 0)
        return set_error(error, error_size, "invalid compressor step arguments");
    *produced = 0;
    int slot = position % state->ratio;
    int hidden = state->hidden, dimension = state->head_dim;
    int projection = state->projection_dim;
    int state_row = state->ratio == 4 ? state->ratio + slot : slot;
    char suffix[128];
    snprintf(suffix, sizeof(suffix), "%s.wkv.weight", state->prefix);
    const uint16_t *wkv = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.wgate.weight", state->prefix);
    const uint16_t *wgate = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.ape", state->prefix);
    const float *ape = layer_value(state->weights, suffix);
    if (!wkv || !wgate || !ape)
        return set_error(error, error_size, "missing compressor tensor for %s", state->prefix);
    float *kv_row = state->kv_state + (size_t)state_row * projection;
    float *score_row = state->score_state + (size_t)state_row * projection;
    #pragma omp parallel for
    for (int row = 0; row < projection; row++) {
        float kv_sum = 0.0f, gate_sum = 0.0f;
        const uint16_t *kv_weight = wkv + (size_t)row * hidden;
        const uint16_t *gate_weight = wgate + (size_t)row * hidden;
        for (int column = 0; column < hidden; column++) {
            float value = input[column];
            kv_sum += coli_bf16_decode(kv_weight[column]) * value;
            gate_sum += coli_bf16_decode(gate_weight[column]) * value;
        }
        kv_row[row] = kv_sum;
        score_row[row] = gate_sum + ape[(size_t)slot * projection + row];
    }
    if ((position + 1) % state->ratio != 0) return 0;
    if (!output) return set_error(error, error_size, "compressor output is required");

    #pragma omp parallel for
    for (int column = 0; column < dimension; column++) {
        float maximum = -INFINITY;
        int pool_rows = state->ratio == 4 ? 2 * state->ratio : state->ratio;
        for (int row = 0; row < pool_rows; row++) {
            int source_row = row;
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float score = state->score_state[
                (size_t)source_row * projection + source_column];
            if (score > maximum) maximum = score;
        }
        float total = 0.0f, weighted = 0.0f;
        for (int row = 0; row < pool_rows; row++) {
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float weight = expf(state->score_state[
                (size_t)row * projection + source_column] - maximum);
            total += weight;
            weighted += state->kv_state[
                (size_t)row * projection + source_column] * weight;
        }
        output[column] = weighted / total;
    }
    if (state->ratio == 4) {
        memcpy(state->kv_state,
               state->kv_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->kv_state));
        memcpy(state->score_state,
               state->score_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->score_state));
    }
    coli_bf16_round_array(output, (size_t)dimension);
    snprintf(suffix, sizeof(suffix), "%s.norm.weight", state->prefix);
    const uint16_t *raw_norm = layer_value(state->weights, suffix);
    float *norm = malloc((size_t)dimension * sizeof(*norm));
    if (!raw_norm || !norm) {
        free(norm);
        return set_error(error, error_size, "missing compressor norm");
    }
    for (int i = 0; i < dimension; i++) norm[i] = coli_bf16_decode(raw_norm[i]);
    coli_v4_rmsnorm(output, output, norm, dimension, state->config->rms_norm_eps);
    coli_bf16_round_array(output, (size_t)dimension);
    free(norm);

    int rope_position = position + 1 - state->ratio;
    int pairs = state->rope_dim / 2;
    size_t table_count = (size_t)(rope_position + 1) * pairs;
    float *cosines = malloc(table_count * sizeof(*cosines));
    float *sines = malloc(table_count * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_precompute(
            cosines, sines, state->rope_dim, rope_position + 1,
            state->config->original_max_position_embeddings,
            state->config->compress_rope_theta, state->config->rope_factor,
            state->config->rope_beta_fast, state->config->rope_beta_slow)) {
        free(sines); free(cosines);
        return set_error(error, error_size, "cannot create compressor RoPE table");
    }
    float *rope = output + dimension - state->rope_dim;
    coli_v4_rope_apply(rope, 1, state->rope_dim,
                       cosines + (size_t)rope_position * pairs,
                       sines + (size_t)rope_position * pairs, 0);
    coli_bf16_round_array(rope, (size_t)state->rope_dim);
    free(sines); free(cosines);

    size_t quantized = state->rotate_fp4
        ? (size_t)dimension : (size_t)(dimension - state->rope_dim);
    size_t block = state->rotate_fp4 ? 32u : 64u;
    float *qdq = malloc(quantized * sizeof(*qdq));
    uint8_t *scales = malloc((quantized + block - 1) / block);
    if (!qdq || !scales) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    int quant_error = 0;
    if (state->rotate_fp4)
        quant_error = coli_hadamard_bf16_ref(output, (size_t)dimension) ||
                      coli_fp4_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    else
        quant_error = coli_fp8_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    if (quant_error) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    memcpy(output, qdq, quantized * sizeof(*output));
    coli_bf16_round_array(output, quantized);
    free(scales); free(qdq);
    *produced = 1;
    return 0;
}
