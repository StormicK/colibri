#define coli_v4_block_window_batch_ref coli_v4_dspark_causal_block_unused
#include "deepseek_v4_block_batch.c"
#undef coli_v4_block_window_batch_ref

#include "deepseek_v4_dspark_attention_block.h"
#include "deepseek_v4_dspark_block.h"

int coli_v4_dspark_block(
    float *outputs_hc, ColiV4DSparkAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens,
    int query_start_position, int batch,
    char *error, size_t error_size) {
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(float));
    float *attn_norm = malloc((size_t)batch * d * sizeof(float));
    float *attn_branch = malloc((size_t)batch * d * sizeof(float));
    float *attn_post = malloc((size_t)batch * hc * sizeof(float));
    float *attn_comb = malloc((size_t)batch * hc * hc * sizeof(float));
    float *ffn_norm = malloc((size_t)batch * d * sizeof(float));
    float *ffn_branch = malloc((size_t)batch * d * sizeof(float));
    float *ffn_post = malloc((size_t)batch * hc * sizeof(float));
    float *ffn_comb = malloc((size_t)batch * hc * hc * sizeof(float));
    float *reduced = malloc((size_t)d * sizeof(float));
    if (!states || !attn_norm || !attn_branch || !attn_post || !attn_comb ||
        !ffn_norm || !ffn_branch || !ffn_post || !ffn_comb || !reduced) {
        free(reduced); free(ffn_comb); free(ffn_post); free(ffn_branch);
        free(ffn_norm); free(attn_comb); free(attn_post); free(attn_branch);
        free(attn_norm); free(states); return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, attn_post + (size_t)item * hc,
            attn_comb + (size_t)item * hc * hc,
            attn_norm + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            weights, config, "attn", "attn_norm.weight");
    if (!result) result = coli_v4_dspark_attention_block(
        attn_branch, attention, weights, config, attn_norm,
        query_start_position, batch, error, error_size);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            states + (size_t)item * hd,
            attn_branch + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            attn_post + (size_t)item * hc,
            attn_comb + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(states + (size_t)item * hd, hd);
    }
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, ffn_post + (size_t)item * hc,
            ffn_comb + (size_t)item * hc * hc,
            ffn_norm + (size_t)item * d,
            states + (size_t)item * hd,
            weights, config, "ffn", "ffn_norm.weight");
    if (!result) result = moe_batch(
        ffn_branch, weights, config, experts, ffn_norm, tokens, batch);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd,
            ffn_branch + (size_t)item * d,
            states + (size_t)item * hd,
            ffn_post + (size_t)item * hc,
            ffn_comb + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(outputs_hc + (size_t)item * hd, hd);
    }
    free(reduced); free(ffn_comb); free(ffn_post); free(ffn_branch);
    free(ffn_norm); free(attn_comb); free(attn_post); free(attn_branch);
    free(attn_norm); free(states);
    return result ? set_error(error, error_size, "DSpark block failed") : 0;
}
