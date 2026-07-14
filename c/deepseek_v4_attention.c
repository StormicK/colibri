#include "deepseek_v4_attention.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_math.h"
#include "deepseek_v4_compressor.h"
#include "deepseek_v4_indexer.h"
#include "deepseek_v4_sparse_attention.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...);

struct ColiDeepSeekV4WindowAttentionState {
    int window_size;
    int head_dim;
    int layer;
    int ratio;
    float *kv;
    ColiDeepSeekV4CompressorState *compressor;
    ColiDeepSeekV4Indexer *indexer;
    float *compressed;
    int compressed_count;
    int compressed_capacity;
};

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = config->sliding_window;
    (*output)->head_dim = config->head_dim;
    (*output)->layer = -1;
    (*output)->kv = calloc((size_t)config->sliding_window * config->head_dim,
                           sizeof(*(*output)->kv));
    if (!(*output)->kv) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    memset(state->kv, 0,
           (size_t)state->window_size * state->head_dim * sizeof(*state->kv));
    state->compressed_count = 0;
    if (state->compressor) coli_v4_compressor_reset(state->compressor);
    if (state->indexer) coli_v4_indexer_reset(state->indexer);
}

void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    coli_v4_indexer_destroy(state->indexer);
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state->kv);
    free(state);
}

static int prepare_compressed_state(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, char *error, size_t error_size) {
    int ratio = weights->plan.compression_ratio;
    if (!ratio) return 0;
    if (state->layer < 0) {
        state->layer = weights->plan.layer;
        state->ratio = ratio;
        state->compressed_capacity = 16;
        state->compressed = calloc((size_t)state->compressed_capacity * state->head_dim,
                                   sizeof(*state->compressed));
        if (!state->compressed || coli_v4_compressor_create(
                &state->compressor, weights, config, error, error_size)) return -1;
        if (ratio == 4 && coli_v4_indexer_create(
                &state->indexer, weights, config, config->max_position_embeddings,
                error, error_size)) return -1;
    } else if (state->layer != weights->plan.layer || state->ratio != ratio) {
        return set_error(error, error_size, "attention state belongs to another layer");
    }
    if (coli_v4_compressor_bind_weights(state->compressor, weights,
                                        error, error_size)) return -1;
    if (state->indexer && coli_v4_indexer_bind_weights(
            state->indexer, weights, error, error_size)) return -1;
    return 0;
}

static int grow_compressed_state(ColiDeepSeekV4WindowAttentionState *state,
                                 char *error, size_t error_size) {
    if (state->compressed_count < state->compressed_capacity) return 0;
    int capacity = state->compressed_capacity * 2;
    float *grown = realloc(state->compressed,
        (size_t)capacity * state->head_dim * sizeof(*grown));
    if (!grown) return set_error(error, error_size, "cannot grow compressed KV cache");
    state->compressed = grown;
    state->compressed_capacity = capacity;
    return 0;
}

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_data(const ColiDeepSeekV4LayerWeights *weights,
                              const char *suffix,
                              const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *weight_spec = NULL, *scale_spec = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = layer_data(weights, suffix, &weight_spec);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = layer_data(weights, suffix, &scale_spec);
    if (!data || !scales || !weight_spec || !scale_spec ||
        weight_spec->dtype != COLI_ST_F8_E4M3 ||
        scale_spec->dtype != COLI_ST_F8_E8M0 || weight_spec->rank != 2)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0, data, scales,
        (size_t)(weight_spec->shape[0] * weight_spec->shape[1]),
        (size_t)(scale_spec->shape[0] * scale_spec->shape[1]),
        weight_spec->shape[0], weight_spec->shape[1], 128, 128
    };
    return 0;
}

static int decode_bf16(float *output, const void *data, size_t count) {
    if (!output || !data) return -1;
    const uint16_t *values = data;
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(values[i]);
    return 0;
}

static int attention_token_impl(float *output,
                                ColiDeepSeekV4WindowAttentionState *state,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    if (!output || !weights || !config || !input || position < 0 ||
        (!state && weights->plan.compression_ratio != 0 && position != 0))
        return set_error(error, error_size, "invalid uncompressed attention arguments");
    int hidden = config->hidden_size;
    int heads = config->num_attention_heads;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank;
    int groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    if (hidden < 1 || heads < 1 || head_dim < 1 || rope_dim < 2 ||
        rope_dim > head_dim || q_rank < 1 || groups < 1 || heads % groups)
        return set_error(error, error_size, "unsupported attention dimensions");

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing native FP8 attention tensor");

    float *qa = calloc((size_t)q_rank, sizeof(*qa));
    float *q = calloc((size_t)heads * head_dim, sizeof(*q));
    float *kv = calloc((size_t)head_dim, sizeof(*kv));
    float *attended = calloc((size_t)heads * head_dim, sizeof(*attended));
    float *oa = calloc((size_t)groups * o_rank, sizeof(*oa));
    float *norm_weight = calloc((size_t)(q_rank > head_dim ? q_rank : head_dim),
                                sizeof(*norm_weight));
    float *cosines = calloc((size_t)rope_dim / 2, sizeof(*cosines));
    float *sines = calloc((size_t)rope_dim / 2, sizeof(*sines));
    int *compressed_indices = NULL;
    int compressed_selected = 0;
    if (!qa || !q || !kv || !attended || !oa || !norm_weight || !cosines || !sines) {
        free(sines); free(cosines); free(norm_weight); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in attention");
    }

    int result = coli_fp8_matvec_ref(qa, &wq_a, input);
    coli_bf16_round_array(qa, (size_t)q_rank);
    const void *q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!q_norm || decode_bf16(norm_weight, q_norm, (size_t)q_rank) ||
                    coli_v4_rmsnorm(qa, qa, norm_weight, q_rank,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(qa, (size_t)q_rank);
    if (!result && state && weights->plan.compression_ratio) {
        result = prepare_compressed_state(state, weights, config,
                                          error, error_size);
        if (!result && (position + 1) % state->ratio == 0)
            result = grow_compressed_state(state, error, error_size);
        int produced = 0;
        if (!result) result = coli_v4_compressor_step(
            state->compressor,
            state->compressed + (size_t)state->compressed_count * head_dim,
            &produced, input, position, error, error_size);
        if (!result && produced) state->compressed_count++;
        if (!result && state->indexer) {
            compressed_indices = malloc((size_t)config->index_topk *
                                        sizeof(*compressed_indices));
            if (!compressed_indices) result = -1;
            else compressed_selected = coli_v4_indexer_step(
                state->indexer, compressed_indices, config->index_topk,
                qa, input, position, error, error_size);
            if (compressed_selected < 0) result = -1;
        }
    }
    if (!result) result = coli_fp8_matvec_ref(q, &wq_b, qa);
    if (!result) coli_bf16_round_array(q, (size_t)heads * head_dim);
    for (int head = 0; !result && head < heads; head++) {
        float *values = q + (size_t)head * head_dim;
        float mean_square = 0.0f;
        for (int i = 0; i < head_dim; i++) mean_square += values[i] * values[i];
        float scale = 1.0f / sqrtf(mean_square / head_dim + config->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) values[i] = coli_bf16_round(values[i] * scale);
    }

    if (!result) result = coli_fp8_matvec_ref(kv, &wkv, input);
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);
    const void *kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!kv_norm || decode_bf16(norm_weight, kv_norm, (size_t)head_dim) ||
                    coli_v4_rmsnorm(kv, kv, norm_weight, head_dim,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);

    if (!result) {
        float *all_cos = calloc((size_t)(position + 1) * rope_dim / 2, sizeof(*all_cos));
        float *all_sin = calloc((size_t)(position + 1) * rope_dim / 2, sizeof(*all_sin));
        int compressed = weights->plan.compression_ratio != 0;
        if (!all_cos || !all_sin || coli_v4_rope_precompute(
                all_cos, all_sin, rope_dim, position + 1,
                compressed ? config->original_max_position_embeddings : 0,
                compressed ? config->compress_rope_theta : config->rope_theta,
                config->rope_factor,
                config->rope_beta_fast, config->rope_beta_slow)) result = -1;
        if (!result) {
            memcpy(cosines, all_cos + (size_t)position * rope_dim / 2,
                   (size_t)rope_dim / 2 * sizeof(*cosines));
            memcpy(sines, all_sin + (size_t)position * rope_dim / 2,
                   (size_t)rope_dim / 2 * sizeof(*sines));
        }
        free(all_sin); free(all_cos);
    }
    if (!result) {
        for (int head = 0; head < heads; head++) {
            float *rope = q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, cosines, sines, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(qdq, scales, kv, nope, 64))
            result = -1;
        if (!result) {
            memcpy(kv, qdq, nope * sizeof(*kv));
            coli_bf16_round_array(kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    if (!result && state) {
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, kv,
               (size_t)head_dim * sizeof(*kv));
        if (!state->indexer) compressed_selected = state->compressed_count;
        int topk = state->window_size + compressed_selected;
        int kv_count = state->window_size + state->compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = state->compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (state->compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1) {
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            } else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *kv_values = state->kv;
            if (state->compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)state->compressed_count * head_dim * sizeof(*all_kv));
                kv_values = all_kv;
            }
            for (int i = 0; i < compressed_selected; i++) {
                int ordinal = state->indexer ? compressed_indices[i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                attended, q, kv_values, sinks, indices, heads, head_dim,
                kv_count, topk,
                1.0f / sqrtf((float)head_dim));
        }
        free(all_kv);
        free(indices);
    } else for (int head = 0; !result && head < heads; head++) {
        float *query = q + (size_t)head * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) score += query[i] * kv[i];
        score *= 1.0f / sqrtf((float)head_dim);
        float attention_weight = 1.0f / (1.0f + expf(sinks[head] - score));
        float *head_output = attended + (size_t)head * head_dim;
        for (int i = 0; i < head_dim; i++)
            head_output[i] = coli_bf16_round(kv[i] * attention_weight);
    }
    for (int head = 0; !result && head < heads; head++) {
        float *head_output = attended + (size_t)head * head_dim;
        float *rope = head_output + head_dim - rope_dim;
        coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 1);
        coli_bf16_round_array(rope, (size_t)rope_dim);
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (hidden + 127) / 128;
    int scale_rows_per_group = (o_rank + 127) / 128;
    for (int group = 0; !result && group < groups; group++) {
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
            (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
            (size_t)group * scale_rows_per_group * scale_columns;
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes = (size_t)scale_rows_per_group * scale_columns;
        result = coli_fp8_matvec_ref(oa + (size_t)group * o_rank, &group_view,
                                     attended + (size_t)group * group_width);
    }
    if (!result) coli_bf16_round_array(oa, (size_t)groups * o_rank);
    if (!result) result = coli_fp8_matvec_ref(output, &wo_b, oa);
    if (!result) coli_bf16_round_array(output, (size_t)hidden);

    free(compressed_indices);
    free(sines); free(cosines); free(norm_weight); free(oa);
    free(attended); free(kv); free(q); free(qa);
    if (result) return set_error(error, error_size, "attention computation failed");
    return 0;
}

int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    return attention_token_impl(output, NULL, weights, config, input, position,
                                error, error_size);
}

int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    return attention_token_impl(output, state, weights, config, input, position,
                                error, error_size);
}
