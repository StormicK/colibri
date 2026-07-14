#define coli_v4_block_token_ref coli_v4_block_token_batch_serial_ref
#define coli_v4_block_window_token_ref coli_v4_block_window_token_batch_serial_ref
#include "deepseek_v4_block.c"
#undef coli_v4_block_token_ref
#undef coli_v4_block_window_token_ref

#include "deepseek_v4_block_batch.h"
#include "native_quant_batch.h"

static int shared_expert_batch(float *outputs, const ColiTensorView *w1,
                               const ColiTensorView *w2,
                               const ColiTensorView *w3,
                               const float *inputs, int batch,
                               float swiglu_limit) {
    int intermediate = (int)w1->rows, d = (int)w2->rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp8_matmul_batch_ref(gate, w1, inputs, batch) ||
                 coli_fp8_matmul_batch_ref(up, w3, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        if (!result) coli_bf16_round_array(item_activated,
                                           (size_t)intermediate);
    }
    if (!result) result = coli_fp8_matmul_batch_ref(
        outputs, w2, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int routed_expert_batch(float *outputs, const ColiExpertView *expert,
                               const float *inputs, const float *route_weights,
                               int batch, float swiglu_limit) {
    int intermediate = (int)expert->gate.rows;
    int d = (int)expert->down.rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_matmul_batch_ref(gate, &expert->gate, inputs, batch) ||
                 coli_fp4_matmul_batch_ref(up, &expert->up, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        for (int i = 0; !result && i < intermediate; i++)
            item_activated[i] = coli_bf16_round(
                item_activated[i] * route_weights[item]);
    }
    if (!result) result = coli_fp4_matmul_batch_ref(
        outputs, &expert->down, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int moe_batch(float *outputs,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *inputs,
                     const int *tokens, int batch) {
    int d = config->hidden_size, n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    float *gate = malloc((size_t)n * d * sizeof(*gate));
    float *route_weights = malloc((size_t)batch * topk * sizeof(*route_weights));
    int *indices = malloc((size_t)batch * topk * sizeof(*indices));
    float *shared = malloc((size_t)batch * d * sizeof(*shared));
    float *compact_inputs = malloc((size_t)batch * d * sizeof(*compact_inputs));
    float *compact_outputs = malloc((size_t)batch * d * sizeof(*compact_outputs));
    float *compact_weights = malloc((size_t)batch * sizeof(*compact_weights));
    int *compact_items = malloc((size_t)batch * sizeof(*compact_items));
    if (!gate || !route_weights || !indices || !shared || !compact_inputs ||
        !compact_outputs || !compact_weights || !compact_items) {
        free(compact_items); free(compact_weights); free(compact_outputs);
        free(compact_inputs); free(shared); free(indices); free(route_weights);
        free(gate); return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), (size_t)n * d);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        int *item_indices = indices + (size_t)item * topk;
        float *item_weights = route_weights + (size_t)item * topk;
        if (tokens[item] < 0 || tokens[item] >= config->vocab_size) result = -1;
        if (!result && weights->plan.uses_hash_router) {
            if (!table) result = -1;
            else for (int i = 0; i < topk; i++)
                item_indices[i] = (int)table[(size_t)tokens[item] * topk + i];
        }
        if (!result) result = coli_v4_route(
            item_weights, item_indices, inputs + (size_t)item * d,
            gate, bias, weights->plan.uses_hash_router ? item_indices : NULL,
            n, d, topk, config->routed_scaling_factor);
    }
    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = shared_expert_batch(
        shared, &w1, &w2, &w3, inputs, batch, config->swiglu_limit);
    if (!result) memset(outputs, 0, (size_t)batch * d * sizeof(*outputs));

    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int count = 0;
        for (int item = 0; item < batch; item++) {
            for (int rank = 0; rank < topk; rank++) {
                if (indices[(size_t)item * topk + rank] == expert_id) {
                    compact_items[count] = item;
                    compact_weights[count] =
                        route_weights[(size_t)item * topk + rank];
                    memcpy(compact_inputs + (size_t)count * d,
                           inputs + (size_t)item * d, (size_t)d * sizeof(float));
                    count++;
                }
            }
        }
        if (!count) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) { result = -1; break; }
        result = routed_expert_batch(compact_outputs, &expert, compact_inputs,
                                     compact_weights, count,
                                     config->swiglu_limit);
        coli_expert_release(store, &expert);
        for (int compact = 0; !result && compact < count; compact++) {
            float *destination = outputs + (size_t)compact_items[compact] * d;
            const float *source = compact_outputs + (size_t)compact * d;
            for (int i = 0; i < d; i++) destination[i] += source[i];
        }
    }
    for (int item = 0; !result && item < batch; item++)
        for (int i = 0; i < d; i++)
            outputs[(size_t)item * d + i] = coli_bf16_round(
                outputs[(size_t)item * d + i] + shared[(size_t)item * d + i]);
    free(compact_items); free(compact_weights); free(compact_outputs);
    free(compact_inputs); free(shared); free(indices); free(route_weights);
    free(gate); return result;
}

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size) {
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(*states));
    float *normalized_ffn = malloc((size_t)batch * d * sizeof(*normalized_ffn));
    float *branches = malloc((size_t)batch * d * sizeof(*branches));
    float *posts = malloc((size_t)batch * hc * sizeof(*posts));
    float *combs = malloc((size_t)batch * hc * hc * sizeof(*combs));
    float *residual = malloc(hd * sizeof(*residual));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!states || !normalized_ffn || !branches || !posts || !combs ||
        !residual || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized); free(reduced);
        free(residual); free(combs); free(posts); free(branches);
        free(normalized_ffn); free(states); return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        const float *input = inputs_hc + (size_t)item * hd;
        float *state = states + (size_t)item * hd;
        memcpy(residual, input, hd * sizeof(float));
        result = normalized_hc_pre(reduced, post, comb, normalized, input,
                                   weights, config, "attn", "attn_norm.weight");
        if (!result) result = coli_v4_attention_window_token_ref(
            branch, attention, weights, config, normalized,
            start_position + item, error, error_size);
        if (!result) result = coli_v4_hc_post(state, branch, residual,
                                              post, comb, hc, d);
        if (!result) coli_bf16_round_array(state, hd);
        if (!result) result = normalized_hc_pre(
            reduced, posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc,
            normalized_ffn + (size_t)item * d, state,
            weights, config, "ffn", "ffn_norm.weight");
    }
    if (!result) result = moe_batch(branches, weights, config, experts,
                                    normalized_ffn, tokens, batch);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd,
            branches + (size_t)item * d,
            states + (size_t)item * hd,
            posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(
            outputs_hc + (size_t)item * hd, hd);
    }
    free(comb); free(post); free(branch); free(normalized); free(reduced);
    free(residual); free(combs); free(posts); free(branches);
    free(normalized_ffn); free(states);
    return result ? set_error(error, error_size, "batched block failed") : 0;
}
