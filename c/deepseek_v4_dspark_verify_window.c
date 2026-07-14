#include <stdlib.h>

#include "deepseek_v4_dspark_heads.h"
#include "deepseek_v4_dspark_runner.h"
#include "deepseek_v4_runtime.h"

static int verify_available;
static int verify_limit;
static int head_slot;
static int cached_tokens[64];
static float cached_logits[64];

static int configured_limit(int available) {
    int requested = coli_v4_runtime_options()->verify_drafts;
    if (requested < 1) return available;
    return requested < available ? requested : available;
}

int __real_coli_v4_dspark_runner_block_size(
    const ColiV4DSparkRunner *runner);
int __wrap_coli_v4_dspark_runner_block_size(
    const ColiV4DSparkRunner *runner) {
    verify_available = __real_coli_v4_dspark_runner_block_size(runner);
    verify_limit = configured_limit(verify_available);
    head_slot = 0;
    return verify_limit;
}

int __wrap_coli_v4_dspark_biased_argmax(
    ColiV4DSparkHeads *heads, const ColiSafetensorsIndex *target_index,
    const float *hidden, int previous_token,
    int *best_token, float *best_logit) {
    int slot = head_slot++;
    if (verify_available > 0) head_slot %= verify_available;
    if (slot == 0 && verify_limit > 0 &&
        coli_v4_dspark_biased_argmax_batch(
            heads, target_index, hidden, previous_token,
            cached_tokens, cached_logits, verify_limit)) return -1;
    if (slot >= 0 && slot < verify_limit) {
        if (best_token) *best_token = cached_tokens[slot];
        if (best_logit) *best_logit = cached_logits[slot];
        return 0;
    }
    if (best_token) *best_token = previous_token;
    if (best_logit) *best_logit = 0.0f;
    return 0;
}
