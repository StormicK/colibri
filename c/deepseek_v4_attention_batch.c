#define coli_v4_window_attention_create coli_v4_window_attention_batch_create_copy
#define coli_v4_window_attention_reset coli_v4_window_attention_batch_reset_copy
#define coli_v4_window_attention_destroy coli_v4_window_attention_batch_destroy_copy
#define coli_v4_attention_token_ref coli_v4_attention_token_batch_serial_copy
#define coli_v4_attention_window_token_ref coli_v4_attention_window_token_batch_serial_copy
#include "deepseek_v4_attention.c"
#undef coli_v4_window_attention_create
#undef coli_v4_window_attention_reset
#undef coli_v4_window_attention_destroy
#undef coli_v4_attention_token_ref
#undef coli_v4_attention_window_token_ref

#include "deepseek_v4_attention_batch.h"
#include "native_quant_batch.h"

int coli_v4_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size) {
    if (!outputs || !state || !weights || !config || !inputs ||
        start_position < 0 || batch < 1 || batch > 64)
        return set_error(error, error_size, "invalid batched attention arguments");
    int hidden = config->hidden_size, heads = config->num_attention_heads;
    int head_dim = config->head_dim, rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank, groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    size_t q_width = (size_t)heads * head_dim;
    size_t oa_width = (size_t)groups * o_rank;

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing batched attention tensor");

    float *qa = calloc((size_t)batch * q_rank, sizeof(*qa));
    float *q = calloc((size_t)batch * q_width, sizeof(*q));
    float *kv = calloc((size_t)batch * head_dim, sizeof(*kv));
    float *attended = calloc((size_t)batch * q_width, sizeof(*attended));
    float *oa = calloc((size_t)batch * oa_width, sizeof(*oa));
    float *norm = malloc((size_t)(q_rank > head_dim ? q_rank : head_dim) *
                         sizeof(*norm));
    int *selected_counts = calloc((size_t)batch, sizeof(*selected_counts));
    int *compressed_counts = calloc((size_t)batch, sizeof(*compressed_counts));
    int *compressed_indices = malloc((size_t)batch * config->index_topk *
                                     sizeof(*compressed_indices));
    int end_position = start_position + batch;
    size_t rope_pairs = (size_t)rope_dim / 2;
    float *cosines = malloc((size_t)end_position * rope_pairs * sizeof(*cosines));
    float *sines = malloc((size_t)end_position * rope_pairs * sizeof(*sines));
    if (!qa || !q || !kv || !attended || !oa || !norm || !selected_counts ||
        !compressed_counts || !compressed_indices || !cosines || !sines) {
        free(sines); free(cosines); free(compressed_indices);
        free(compressed_counts); free(selected_counts); free(norm); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in batched attention");
    }

    int result = coli_fp8_matmul_batch_ref(qa, &wq_a, inputs, batch);
    if (!result) coli_bf16_round_array(qa, (size_t)batch * q_rank);
    const void *raw_q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!raw_q_norm || decode_bf16(norm, raw_q_norm, q_rank))) result = -1;
    for (int item = 0; !result && item < batch; item++) {
        float *item_qa = qa + (size_t)item * q_rank;
        result = coli_v4_rmsnorm(item_qa, item_qa, norm, q_rank,
                                 config->rms_norm_eps);
        if (!result) coli_bf16_round_array(item_qa, (size_t)q_rank);
    }

    for (int item = 0; !result && item < batch; item++) {
        int position = start_position + item;
        if (weights->plan.compression_ratio) {
            result = prepare_compressed_state(state, weights, config,
                                              error, error_size);
            if (!result && (position + 1) % state->ratio == 0)
                result = grow_compressed_state(state, error, error_size);
            int produced = 0;
            if (!result) result = coli_v4_compressor_step(
                state->compressor,
                state->compressed + (size_t)state->compressed_count * head_dim,
                &produced, inputs + (size_t)item * hidden, position,
                error, error_size);
            if (!result && produced) state->compressed_count++;
            compressed_counts[item] = state->compressed_count;
            if (!result && state->indexer) {
                selected_counts[item] = coli_v4_indexer_step(
                    state->indexer,
                    compressed_indices + (size_t)item * config->index_topk,
                    config->index_topk, qa + (size_t)item * q_rank,
                    inputs + (size_t)item * hidden, position,
                    error, error_size);
                if (selected_counts[item] < 0) result = -1;
            } else if (!result) {
                selected_counts[item] = state->compressed_count;
            }
        }
    }

    if (!result) result = coli_fp8_matmul_batch_ref(q, &wq_b, qa, batch);
    if (!result) coli_bf16_round_array(q, (size_t)batch * q_width);
    for (int item = 0; !result && item < batch; item++)
        for (int head = 0; head < heads; head++) {
            float *values = q + (size_t)item * q_width + (size_t)head * head_dim;
            float square = 0.0f;
            for (int i = 0; i < head_dim; i++) square += values[i] * values[i];
            float scale = 1.0f / sqrtf(square / head_dim + config->rms_norm_eps);
            for (int i = 0; i < head_dim; i++)
                values[i] = coli_bf16_round(values[i] * scale);
        }

    if (!result) result = coli_fp8_matmul_batch_ref(kv, &wkv, inputs, batch);
    if (!result) coli_bf16_round_array(kv, (size_t)batch * head_dim);
    const void *raw_kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!raw_kv_norm || decode_bf16(norm, raw_kv_norm, head_dim))) result = -1;
    for (int item = 0; !result && item < batch; item++) {
        float *item_kv = kv + (size_t)item * head_dim;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, head_dim,
                                 config->rms_norm_eps);
        if (!result) coli_bf16_round_array(item_kv, (size_t)head_dim);
    }

    int compressed = weights->plan.compression_ratio != 0;
    if (!result) result = coli_v4_rope_precompute(
        cosines, sines, rope_dim, end_position,
        compressed ? config->original_max_position_embeddings : 0,
        compressed ? config->compress_rope_theta : config->rope_theta,
        config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
    for (int item = 0; !result && item < batch; item++) {
        int position = start_position + item;
        const float *item_cos = cosines + (size_t)position * rope_pairs;
        const float *item_sin = sines + (size_t)position * rope_pairs;
        float *item_q = q + (size_t)item * q_width;
        float *item_kv = kv + (size_t)item * head_dim;
        for (int head = 0; head < heads; head++) {
            float *rope = item_q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, item_cos, item_sin, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = item_kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, item_cos, item_sin, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64)) result = -1;
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(*item_kv));
            coli_bf16_round_array(item_kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    for (int item = 0; !result && item < batch; item++) {
        int position = start_position + item;
        float *item_kv = kv + (size_t)item * head_dim;
        float *item_q = q + (size_t)item * q_width;
        float *item_attended = attended + (size_t)item * q_width;
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, item_kv,
               (size_t)head_dim * sizeof(*item_kv));
        int selected = selected_counts[item];
        int compressed_count = compressed_counts[item];
        int topk = state->window_size + selected;
        int kv_count = state->window_size + compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1)
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *values = state->kv;
            if (compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)compressed_count * head_dim * sizeof(*all_kv));
                values = all_kv;
            }
            for (int i = 0; i < selected; i++) {
                int ordinal = state->indexer
                    ? compressed_indices[(size_t)item * config->index_topk + i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                item_attended, item_q, values, sinks, indices, heads, head_dim,
                kv_count, topk, 1.0f / sqrtf((float)head_dim));
        }
        free(all_kv); free(indices);
        const float *item_cos = cosines + (size_t)position * rope_pairs;
        const float *item_sin = sines + (size_t)position * rope_pairs;
        for (int head = 0; !result && head < heads; head++) {
            float *rope = item_attended + (size_t)head * head_dim +
                          head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, item_cos, item_sin, 1);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (hidden + 127) / 128;
    int scale_rows = (o_rank + 127) / 128;
    float *group_inputs = malloc((size_t)batch * group_width * sizeof(*group_inputs));
    float *group_outputs = malloc((size_t)batch * o_rank * sizeof(*group_outputs));
    if (!group_inputs || !group_outputs) result = -1;
    for (int group = 0; !result && group < groups; group++) {
        for (int item = 0; item < batch; item++)
            memcpy(group_inputs + (size_t)item * group_width,
                   attended + (size_t)item * q_width + (size_t)group * group_width,
                   (size_t)group_width * sizeof(*group_inputs));
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
                          (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
                            (size_t)group * scale_rows * scale_columns;
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes = (size_t)scale_rows * scale_columns;
        result = coli_fp8_matmul_batch_ref(
            group_outputs, &group_view, group_inputs, batch);
        for (int item = 0; !result && item < batch; item++)
            memcpy(oa + (size_t)item * oa_width + (size_t)group * o_rank,
                   group_outputs + (size_t)item * o_rank,
                   (size_t)o_rank * sizeof(*oa));
    }
    if (!result) coli_bf16_round_array(oa, (size_t)batch * oa_width);
    if (!result) result = coli_fp8_matmul_batch_ref(outputs, &wo_b, oa, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * hidden);

    free(group_outputs); free(group_inputs); free(sines); free(cosines);
    free(compressed_indices); free(compressed_counts); free(selected_counts);
    free(norm); free(oa); free(attended); free(kv); free(q); free(qa);
    return result ? set_error(error, error_size, "batched attention failed") : 0;
}
