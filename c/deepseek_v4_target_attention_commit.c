#include "deepseek_v4_attention_batch.h"
#define coli_v4_block_token_batch_serial_ref \
    coli_v4_commit_block_token_serial_ref
#define coli_v4_block_window_token_batch_serial_ref \
    coli_v4_commit_block_window_token_serial_ref
/* ---- begin inlined deepseek_v4_target_attention_commit.c ---- */
#define coli_v4_block_window_batch_ref coli_v4_commit_unused_full_block
#include "deepseek_v4_block_batch.c"
#undef coli_v4_block_window_batch_ref

#include "deepseek_v4_target_attention_commit.h"

int coli_v4_target_attention_commit_batch(
    ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *inputs_hc, int start_position, int batch,
    char *error, size_t error_size) {
    if (!attention || !weights || !config || !inputs_hc ||
        start_position < 0 || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *normalized = malloc((size_t)batch * d * sizeof(*normalized));
    float *discarded = malloc((size_t)batch * d * sizeof(*discarded));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!normalized || !discarded || !reduced || !post || !comb) {
        free(comb); free(post); free(reduced); free(discarded); free(normalized);
        return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, post, comb, normalized + (size_t)item * d,
            inputs_hc + (size_t)item * hd, weights, config,
            "attn", "attn_norm.weight");
    if (!result) result = coli_v4_attention_window_batch_ref(
        discarded, attention, weights, config, normalized,
        start_position, batch, error, error_size);
    free(comb); free(post); free(reduced); free(discarded); free(normalized);
    return result;
}
/* ---- end inlined deepseek_v4_target_attention_commit.c ---- */

#undef coli_v4_block_window_token_batch_serial_ref
#undef coli_v4_block_token_batch_serial_ref
