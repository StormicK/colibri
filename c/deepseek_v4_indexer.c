#include "deepseek_v4_indexer.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_compressor.h"
#include "deepseek_v4_math.h"
#include "native_quant.h"

struct ColiDeepSeekV4Indexer {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    ColiDeepSeekV4CompressorState *compressor;
    int layer;
    int capacity;
    int count;
    float *compressed;
};

typedef struct { float score; int index; } IndexScore;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
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
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = value(weights, suffix, &ws);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = value(weights, suffix, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2 ||
        ws->dtype != COLI_ST_F8_E4M3 || ss->dtype != COLI_ST_F8_E8M0)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]),
        ws->shape[0], ws->shape[1], 128, 128
    };
    return 0;
}

static int descending_score(const void *left, const void *right) {
    const IndexScore *a = left, *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **output,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size) {
    if (!output || !weights || !config || !weights->plan.has_indexer ||
        max_context < 4 || config->index_head_dim < 1 ||
        config->index_n_heads < 1)
        return set_error(error, error_size, "invalid indexer options");
    *output = NULL;
    ColiDeepSeekV4Indexer *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating indexer");
    state->weights = weights;
    state->config = config;
    state->layer = weights->plan.layer;
    state->capacity = (max_context + 3) / 4;
    if (state->capacity > 128) state->capacity = 128;
    state->compressed = calloc((size_t)state->capacity * config->index_head_dim,
                               sizeof(*state->compressed));
    ColiDeepSeekV4CompressorOptions options = {
        "attn.indexer.compressor", config->index_head_dim, 1
    };
    if (!state->compressed || coli_v4_compressor_create_with_options(
            &state->compressor, weights, config, &options, error, error_size)) {
        coli_v4_indexer_destroy(state);
        return set_error(error, error_size, "cannot create indexer compressor");
    }
    *output = state;
    return 0;
}

int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size) {
    if (!state || !weights || weights->plan.layer != state->layer ||
        !weights->plan.has_indexer)
        return set_error(error, error_size, "incompatible indexer weights");
    state->weights = weights;
    return coli_v4_compressor_bind_weights(state->compressor, weights,
                                            error, error_size);
}

void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    state->count = 0;
    memset(state->compressed, 0,
           (size_t)state->capacity * state->config->index_head_dim * sizeof(float));
    coli_v4_compressor_reset(state->compressor);
}

void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state);
}

static int apply_position_rope(float *queries,
                               const ColiDeepSeekV4Config *config,
                               int position) {
    int heads = config->index_n_heads, dimension = config->index_head_dim;
    int rope_dim = config->qk_rope_head_dim, pairs = rope_dim / 2;
    size_t count = (size_t)(position + 1) * pairs;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_precompute(
            cosines, sines, rope_dim, position + 1,
            config->original_max_position_embeddings,
            config->compress_rope_theta, config->rope_factor,
            config->rope_beta_fast, config->rope_beta_slow)) {
        free(sines); free(cosines); return -1;
    }
    for (int head = 0; head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        coli_v4_rope_apply(query + dimension - rope_dim, 1, rope_dim,
                           cosines + (size_t)position * pairs,
                           sines + (size_t)position * pairs, 0);
        coli_bf16_round_array(query + dimension - rope_dim, (size_t)rope_dim);
    }
    free(sines); free(cosines);
    return 0;
}

int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size) {
    if (!state || !indices || index_capacity < 1 || !query_rank || !input ||
        position < 0)
        return set_error(error, error_size, "invalid indexer step arguments");
    int dimension = state->config->index_head_dim;
    int heads = state->config->index_n_heads;
    int produced = 0;
    if ((position + 1) % 4 == 0 && state->count >= state->capacity) {
        int next_capacity = state->capacity * 2;
        float *grown = realloc(state->compressed,
            (size_t)next_capacity * dimension * sizeof(*grown));
        if (!grown) return set_error(error, error_size, "cannot grow indexer cache");
        memset(grown + (size_t)state->capacity * dimension, 0,
               (size_t)(next_capacity - state->capacity) * dimension * sizeof(*grown));
        state->compressed = grown;
        state->capacity = next_capacity;
    }
    float *next = state->count < state->capacity
        ? state->compressed + (size_t)state->count * dimension : NULL;
    if (coli_v4_compressor_step(state->compressor, next, &produced, input,
                                position, error, error_size)) return -1;
    if (produced) {
        if (state->count >= state->capacity)
            return set_error(error, error_size, "indexer cache capacity exceeded");
        state->count++;
    }
    if (!state->count) return 0;

    ColiTensorView wq;
    if (fp8_view(&wq, state->weights, "attn.indexer.wq_b"))
        return set_error(error, error_size, "missing indexer query weight");
    float *queries = malloc((size_t)heads * dimension * sizeof(*queries));
    float *head_weights = malloc((size_t)heads * sizeof(*head_weights));
    IndexScore *scores = malloc((size_t)state->count * sizeof(*scores));
    uint8_t *scales = malloc((size_t)dimension / 32);
    float *qdq = malloc((size_t)dimension * sizeof(*qdq));
    const uint16_t *raw_weights = value(
        state->weights, "attn.indexer.weights_proj.weight", NULL);
    if (!queries || !head_weights || !scores || !scales || !qdq || !raw_weights) {
        free(qdq); free(scales); free(scores); free(head_weights); free(queries);
        return set_error(error, error_size, "out of memory scoring indexer");
    }
    int result = coli_fp8_matvec_ref(queries, &wq, query_rank);
    if (!result) coli_bf16_round_array(queries, (size_t)heads * dimension);
    if (!result) result = apply_position_rope(queries, state->config, position);
    for (int head = 0; !result && head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        result = coli_hadamard_bf16_ref(query, (size_t)dimension);
        if (!result) result = coli_fp4_activation_qdq_ref(
            qdq, scales, query, (size_t)dimension, 32);
        if (!result) {
            memcpy(query, qdq, (size_t)dimension * sizeof(*query));
            coli_bf16_round_array(query, (size_t)dimension);
        }
    }
    float weight_scale = 1.0f / sqrtf((float)(dimension * heads));
    for (int head = 0; !result && head < heads; head++) {
        float sum = 0.0f;
        const uint16_t *row = raw_weights + (size_t)head * state->config->hidden_size;
        for (int column = 0; column < state->config->hidden_size; column++)
            sum += coli_bf16_decode(row[column]) * input[column];
        head_weights[head] = sum * weight_scale;
    }
    for (int candidate = 0; !result && candidate < state->count; candidate++) {
        const float *key = state->compressed + (size_t)candidate * dimension;
        float score = 0.0f;
        for (int head = 0; head < heads; head++) {
            const float *query = queries + (size_t)head * dimension;
            float dot = 0.0f;
            for (int i = 0; i < dimension; i++) dot += query[i] * key[i];
            score += fmaxf(dot, 0.0f) * head_weights[head];
        }
        scores[candidate] = (IndexScore){score, candidate};
    }
    if (!result) qsort(scores, (size_t)state->count, sizeof(*scores), descending_score);
    int selected = state->count;
    if (selected > state->config->index_topk) selected = state->config->index_topk;
    if (selected > index_capacity) selected = index_capacity;
    for (int i = 0; !result && i < selected; i++) indices[i] = scores[i].index;
    free(qdq); free(scales); free(scores); free(head_weights); free(queries);
    return result ? set_error(error, error_size, "indexer scoring failed") : selected;
}

const float *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state) {
    return state ? state->compressed : NULL;
}

int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state) {
    return state ? state->count : 0;
}
