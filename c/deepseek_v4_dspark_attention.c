/* ---- begin inlined deepseek_v4_dspark_attention.c ---- */
#include "deepseek_v4_dspark_attention.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_math.h"
#include "native_quant.h"
#include "native_quant_batch.h"

struct ColiV4DSparkAttentionState {
    int window_size;
    int head_dim;
    int valid;
    float *kv;
    int *positions;
};

static int ds_attn_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static const void *ds_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                                 const char *suffix,
                                 const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int ds_fp8_view(ColiTensorView *view,
                       const ColiDeepSeekV4LayerWeights *weights,
                       const char *prefix) {
    char name[128]; const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = ds_layer_data(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = ds_layer_data(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2 || ss->rank != 2)
        return -1;
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = data; view->scales = scales;
    view->rows = ws->shape[0]; view->columns = ws->shape[1];
    view->data_bytes = (size_t)view->rows * view->columns;
    view->scale_bytes = (size_t)ss->shape[0] * ss->shape[1];
    view->block_rows = 128; view->block_columns = 128;
    return 0;
}

int coli_v4_dspark_attention_create(ColiV4DSparkAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = NULL;
    ColiV4DSparkAttentionState *state = calloc(1, sizeof(*state));
    if (!state) return -1;
    state->window_size = config->sliding_window;
    state->head_dim = config->head_dim;
    state->kv = calloc((size_t)state->window_size * state->head_dim,
                       sizeof(*state->kv));
    state->positions = malloc((size_t)state->window_size *
                              sizeof(*state->positions));
    if (!state->kv || !state->positions) {
        coli_v4_dspark_attention_destroy(state); return -1;
    }
    coli_v4_dspark_attention_reset(state); *output = state; return 0;
}

void coli_v4_dspark_attention_reset(ColiV4DSparkAttentionState *state) {
    if (!state) return;
    state->valid = 0;
    memset(state->kv, 0, (size_t)state->window_size * state->head_dim *
                         sizeof(*state->kv));
    for (int i = 0; i < state->window_size; i++) state->positions[i] = -1;
}

void coli_v4_dspark_attention_destroy(ColiV4DSparkAttentionState *state) {
    if (!state) return;
    free(state->positions); free(state->kv); free(state);
}

int coli_v4_dspark_attention_context_count(
    const ColiV4DSparkAttentionState *state) { return state ? state->valid : 0; }

int coli_v4_dspark_attention_precompute_context(
    ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *main_x, int start_position, int batch,
    char *error, size_t error_size) {
    if (!state || !weights || !config || !main_x || start_position < 0 ||
        batch < 1 || batch > 64 || state->head_dim != config->head_dim)
        return ds_attn_error(error, error_size, "invalid DSpark context KV input");
    ColiTensorView wkv;
    if (ds_fp8_view(&wkv, weights, "attn.wkv"))
        return ds_attn_error(error, error_size, "missing DSpark wkv");
    int d = config->head_dim, rope = config->qk_rope_head_dim;
    float *kv = malloc((size_t)batch * d * sizeof(*kv));
    float *norm = malloc((size_t)d * sizeof(*norm));
    int end = start_position + batch;
    size_t pairs = (size_t)rope / 2;
    float *cosines = malloc((size_t)end * pairs * sizeof(*cosines));
    float *sines = malloc((size_t)end * pairs * sizeof(*sines));
    const uint16_t *raw_norm = ds_layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!kv || !norm || !cosines || !sines || !raw_norm) {
        free(sines); free(cosines); free(norm); free(kv);
        return ds_attn_error(error, error_size, "out of memory in DSpark context KV");
    }
    for (int i = 0; i < d; i++) norm[i] = coli_bf16_decode(raw_norm[i]);
    int result = coli_fp8_matmul_batch_ref(kv, &wkv, main_x, batch);
    if (!result) coli_bf16_round_array(kv, (size_t)batch * d);
    if (!result) result = coli_v4_rope_precompute(
        cosines, sines, rope, end, 0, config->rope_theta,
        config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
    for (int item = 0; !result && item < batch; item++) {
        float *item_kv = kv + (size_t)item * d;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, d, config->rms_norm_eps);
        if (result) break;
        coli_bf16_round_array(item_kv, (size_t)d);
        int position = start_position + item;
        coli_v4_rope_apply(item_kv + d - rope, 1, rope,
                           cosines + (size_t)position * pairs,
                           sines + (size_t)position * pairs, 0);
        coli_bf16_round_array(item_kv + d - rope, (size_t)rope);
        size_t nope = (size_t)(d - rope);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64)) result = -1;
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(*item_kv));
            coli_bf16_round_array(item_kv, nope);
            int slot = position % state->window_size;
            memcpy(state->kv + (size_t)slot * d, item_kv,
                   (size_t)d * sizeof(*item_kv));
            state->positions[slot] = position;
            if (state->valid < state->window_size) state->valid++;
        }
        free(scales); free(qdq);
    }
    free(sines); free(cosines); free(norm); free(kv);
    return result ? ds_attn_error(error, error_size,
                                  "DSpark context KV precompute failed") : 0;
}
/* ---- end inlined deepseek_v4_dspark_attention.c ---- */


#include "deepseek_v4_dspark_attention_block.h"
#include "deepseek_v4_sparse_attention.h"

int coli_v4_dspark_attention_block(
    float *outputs, ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int query_start_position, int batch, char *error, size_t error_size) {
    if (!outputs || !state || !weights || !config || !inputs ||
        query_start_position < 0 || batch < 1 || batch > 64)
        return ds_attn_error(error, error_size, "invalid DSpark query block");
    int hidden = config->hidden_size, heads = config->num_attention_heads;
    int d = config->head_dim, rope = config->qk_rope_head_dim;
    int qr = config->q_lora_rank, groups = config->o_groups;
    int orank = config->o_lora_rank;
    size_t qwidth = (size_t)heads * d;
    size_t oawidth = (size_t)groups * orank;
    ColiTensorView wqa, wqb, wkv, woa, wob;
    if (ds_fp8_view(&wqa, weights, "attn.wq_a") ||
        ds_fp8_view(&wqb, weights, "attn.wq_b") ||
        ds_fp8_view(&wkv, weights, "attn.wkv") ||
        ds_fp8_view(&woa, weights, "attn.wo_a") ||
        ds_fp8_view(&wob, weights, "attn.wo_b"))
        return ds_attn_error(error, error_size, "missing DSpark attention matrix");
    float *qa = calloc((size_t)batch * qr, sizeof(float));
    float *q = calloc((size_t)batch * qwidth, sizeof(float));
    float *kv = calloc((size_t)batch * d, sizeof(float));
    float *attended = calloc((size_t)batch * qwidth, sizeof(float));
    float *oa = calloc((size_t)batch * oawidth, sizeof(float));
    float *norm = malloc((size_t)(qr > d ? qr : d) * sizeof(float));
    int end = query_start_position + batch; size_t pairs = (size_t)rope / 2;
    float *cosines = malloc((size_t)end * pairs * sizeof(float));
    float *sines = malloc((size_t)end * pairs * sizeof(float));
    int context = state->valid, kv_count = context + batch;
    float *all_kv = malloc((size_t)kv_count * d * sizeof(float));
    int *indices = malloc((size_t)kv_count * sizeof(int));
    if (!qa || !q || !kv || !attended || !oa || !norm || !cosines ||
        !sines || !all_kv || !indices) {
        free(indices); free(all_kv); free(sines); free(cosines); free(norm);
        free(oa); free(attended); free(kv); free(q); free(qa);
        return ds_attn_error(error, error_size, "out of memory in DSpark block");
    }
    int result = coli_fp8_matmul_batch_ref(qa, &wqa, inputs, batch);
    if (!result) coli_bf16_round_array(qa, (size_t)batch * qr);
    const uint16_t *qnorm = ds_layer_data(weights, "attn.q_norm.weight", NULL);
    if (!qnorm) result = -1;
    for (int i = 0; !result && i < qr; i++) norm[i] = coli_bf16_decode(qnorm[i]);
    for (int item = 0; !result && item < batch; item++) {
        float *row = qa + (size_t)item * qr;
        result = coli_v4_rmsnorm(row, row, norm, qr, config->rms_norm_eps);
        if (!result) coli_bf16_round_array(row, (size_t)qr);
    }
    if (!result) result = coli_fp8_matmul_batch_ref(q, &wqb, qa, batch);
    if (!result) coli_bf16_round_array(q, (size_t)batch * qwidth);
    for (int item = 0; !result && item < batch; item++)
        for (int head = 0; head < heads; head++) {
            float *row = q + (size_t)item * qwidth + (size_t)head * d;
            float squares = 0.0f;
            for (int i = 0; i < d; i++) squares += row[i] * row[i];
            float scale = 1.0f / sqrtf(squares / d + config->rms_norm_eps);
            for (int i = 0; i < d; i++) row[i] = coli_bf16_round(row[i] * scale);
        }
    if (!result) result = coli_fp8_matmul_batch_ref(kv, &wkv, inputs, batch);
    if (!result) coli_bf16_round_array(kv, (size_t)batch * d);
    const uint16_t *knorm = ds_layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!knorm) result = -1;
    for (int i = 0; !result && i < d; i++) norm[i] = coli_bf16_decode(knorm[i]);
    if (!result) result = coli_v4_rope_precompute(
        cosines, sines, rope, end, 0, config->rope_theta,
        config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
    for (int item = 0; !result && item < batch; item++) {
        int position = query_start_position + item;
        float *item_kv = kv + (size_t)item * d;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, d, config->rms_norm_eps);
        if (result) break;
        coli_bf16_round_array(item_kv, (size_t)d);
        for (int head = 0; head < heads; head++) {
            float *r = q + (size_t)item * qwidth + (size_t)head * d + d - rope;
            coli_v4_rope_apply(r, 1, rope,
                               cosines + (size_t)position * pairs,
                               sines + (size_t)position * pairs, 0);
            coli_bf16_round_array(r, (size_t)rope);
        }
        coli_v4_rope_apply(item_kv + d - rope, 1, rope,
                           cosines + (size_t)position * pairs,
                           sines + (size_t)position * pairs, 0);
        coli_bf16_round_array(item_kv + d - rope, (size_t)rope);
        size_t nope = (size_t)(d - rope);
        float *qdq = malloc(nope * sizeof(float));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64)) result = -1;
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(float));
            coli_bf16_round_array(item_kv, nope);
        }
        free(scales); free(qdq);
    }
    int copied = 0;
    for (int slot = 0; !result && slot < state->window_size; slot++)
        if (state->positions[slot] >= 0) {
            memcpy(all_kv + (size_t)copied * d,
                   state->kv + (size_t)slot * d, (size_t)d * sizeof(float));
            copied++;
        }
    if (copied != context) result = -1;
    if (!result) memcpy(all_kv + (size_t)context * d, kv,
                        (size_t)batch * d * sizeof(float));
    for (int i = 0; i < kv_count; i++) indices[i] = i;
    const float *sinks = ds_layer_data(weights, "attn.attn_sink", NULL);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_sparse_attention_ref(
            attended + (size_t)item * qwidth,
            q + (size_t)item * qwidth, all_kv, sinks, indices,
            heads, d, kv_count, kv_count, 1.0f / sqrtf((float)d));
        int position = query_start_position + item;
        for (int head = 0; !result && head < heads; head++) {
            float *r = attended + (size_t)item * qwidth +
                       (size_t)head * d + d - rope;
            coli_v4_rope_apply(r, 1, rope,
                               cosines + (size_t)position * pairs,
                               sines + (size_t)position * pairs, 1);
            coli_bf16_round_array(r, (size_t)rope);
        }
    }
    int heads_per_group = heads / groups;
    int group_width = heads_per_group * d;
    int scale_columns = (hidden + 127) / 128;
    int scale_rows = (orank + 127) / 128;
    float *group_inputs = malloc((size_t)batch * group_width * sizeof(float));
    float *group_outputs = malloc((size_t)batch * orank * sizeof(float));
    if (!group_inputs || !group_outputs) result = -1;
    for (int group = 0; !result && group < groups; group++) {
        for (int item = 0; item < batch; item++)
            memcpy(group_inputs + (size_t)item * group_width,
                   attended + (size_t)item * qwidth +
                   (size_t)group * group_width,
                   (size_t)group_width * sizeof(float));
        ColiTensorView view = woa;
        view.rows = orank; view.columns = group_width;
        view.data = (const uint8_t *)woa.data +
                    (size_t)group * orank * group_width;
        view.scales = (const uint8_t *)woa.scales +
                      (size_t)group * scale_rows * scale_columns;
        view.data_bytes = (size_t)orank * group_width;
        view.scale_bytes = (size_t)scale_rows * scale_columns;
        result = coli_fp8_matmul_batch_ref(group_outputs, &view,
                                            group_inputs, batch);
        for (int item = 0; !result && item < batch; item++)
            memcpy(oa + (size_t)item * oawidth + (size_t)group * orank,
                   group_outputs + (size_t)item * orank,
                   (size_t)orank * sizeof(float));
    }
    if (!result) coli_bf16_round_array(oa, (size_t)batch * oawidth);
    if (!result) result = coli_fp8_matmul_batch_ref(outputs, &wob, oa, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * hidden);
    free(group_outputs); free(group_inputs); free(indices); free(all_kv);
    free(sines); free(cosines); free(norm); free(oa); free(attended);
    free(kv); free(q); free(qa);
    return result ? ds_attn_error(error, error_size,
                                  "DSpark non-causal attention failed") : 0;
}
