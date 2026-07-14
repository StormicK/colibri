#ifndef COLIBRI_DEEPSEEK_V4_TARGET_VERIFY_H
#define COLIBRI_DEEPSEEK_V4_TARGET_VERIFY_H

#include "deepseek_v4_attention.h"
#include "deepseek_v4_config.h"
#include "deepseek_v4_expert_store.h"
#include "deepseek_v4_speculative.h"

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size);

#endif
