#include "deepseek_v4_block.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_attention.h"
#include "deepseek_v4_expert.h"
#include "deepseek_v4_math.h"
#include "native_quant.h"

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
    char name[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = value(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = value(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2) return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]),
        ws->shape[0], ws->shape[1], 128, 128
    };
    return 0;
}

static void decode_bf16(float *output, const uint16_t *input, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(input[i]);
}

static int normalized_hc_pre(float *reduced, float *post, float *comb,
                             float *normalized, const float *input_hc,
                             const ColiDeepSeekV4LayerWeights *weights,
                             const ColiDeepSeekV4Config *config,
                             const char *branch, const char *norm_name) {
    char name[64];
    snprintf(name, sizeof(name), "hc_%s_fn", branch);
    const float *function = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_scale", branch);
    const float *scale = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_base", branch);
    const float *base = value(weights, name, NULL);
    const uint16_t *raw_norm = value(weights, norm_name, NULL);
    int d = config->hidden_size;
    float *norm = malloc((size_t)d * sizeof(*norm));
    if (!function || !scale || !base || !raw_norm || !norm) {
        free(norm);
        return -1;
    }
    decode_bf16(norm, raw_norm, (size_t)d);
    int result = coli_v4_hc_pre(reduced, post, comb, input_hc, function,
                                scale, base, config->hc_mult, d,
                                config->hc_sinkhorn_iters,
                                config->rms_norm_eps, config->hc_eps);
    if (!result) {
        coli_bf16_round_array(reduced, (size_t)d);
        result = coli_v4_rmsnorm(normalized, reduced, norm, d,
                                 config->rms_norm_eps);
        coli_bf16_round_array(normalized, (size_t)d);
    }
    free(norm);
    return result;
}

static int moe_token(float *output,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *input, int token) {
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (!gate || !route_weights || !indices || !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(indices);
        free(route_weights); free(gate);
        return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router) {
        if (!table) result = -1;
    }
    if (!result && weights->plan.uses_hash_router) {
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
    }
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int rank = -1;
        for (int candidate = 0; candidate < topk; candidate++)
            if (indices[candidate] == expert_id) rank = candidate;
        if (rank < 0) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) {
            result = -1;
            break;
        }
        result = coli_v4_expert_forward_ref(expert_output, &expert, input,
                                             route_weights[rank],
                                             config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);
    free(shared_output); free(expert_output); free(indices);
    free(route_weights); free(gate);
    return result;
}

static int block_token_impl(float *output_hc,
                            ColiDeepSeekV4WindowAttentionState *attention,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual);
        return set_error(error, error_size, "out of memory in block");
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);

    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn", "ffn_norm.weight");
    if (!result) result = moe_token(branch, weights, config, experts, normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);

    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_impl(output_hc, NULL, weights, config, experts, input_hc,
                            token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_impl(output_hc, attention, weights, config, experts,
                            input_hc, token, position, error, error_size);
}
