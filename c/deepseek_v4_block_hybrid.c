/* Accepted decode pipeline plus batched causal attention for prompt prefill. */
#include "deepseek_v4_block_pipeline.c"

#include "deepseek_v4_attention_batch.h"
#include "deepseek_v4_block_batch.h"

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
    float *normalized = malloc((size_t)batch * d * sizeof(*normalized));
    float *branches = malloc((size_t)batch * d * sizeof(*branches));
    float *posts = malloc((size_t)batch * hc * sizeof(*posts));
    float *combs = malloc((size_t)batch * hc * hc * sizeof(*combs));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *ffn_normalized = malloc((size_t)d * sizeof(*ffn_normalized));
    float *ffn_branch = malloc((size_t)d * sizeof(*ffn_branch));
    float *ffn_post = malloc((size_t)hc * sizeof(*ffn_post));
    float *ffn_comb = malloc((size_t)hc * hc * sizeof(*ffn_comb));
    if (!states || !normalized || !branches || !posts || !combs || !reduced ||
        !ffn_normalized || !ffn_branch || !ffn_post || !ffn_comb) {
        free(ffn_comb); free(ffn_post); free(ffn_branch); free(ffn_normalized);
        free(reduced); free(combs); free(posts); free(branches);
        free(normalized); free(states); return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc,
            normalized + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            weights, config, "attn", "attn_norm.weight");
    if (!result) result = coli_v4_attention_window_batch_ref(
        branches, attention, weights, config, normalized,
        start_position, batch, error, error_size);
    for (int item = 0; !result && item < batch; item++) {
        float *state = states + (size_t)item * hd;
        result = coli_v4_hc_post(
            state, branches + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(state, hd);
        if (!result) result = normalized_hc_pre(
            reduced, ffn_post, ffn_comb, ffn_normalized, state,
            weights, config, "ffn", "ffn_norm.weight");
        if (!result) result = moe_token_pipeline(
            ffn_branch, weights, config, experts,
            ffn_normalized, tokens[item]);
        if (!result) result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd, ffn_branch, state,
            ffn_post, ffn_comb, hc, d);
        if (!result) coli_bf16_round_array(
            outputs_hc + (size_t)item * hd, hd);
    }
    free(ffn_comb); free(ffn_post); free(ffn_branch); free(ffn_normalized);
    free(reduced); free(combs); free(posts); free(branches);
    free(normalized); free(states);
    return result ? set_error(error, error_size, "hybrid batched block failed") : 0;
}
