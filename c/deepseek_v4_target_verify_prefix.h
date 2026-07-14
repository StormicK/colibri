#ifndef COLIBRI_DEEPSEEK_V4_TARGET_VERIFY_PREFIX_H
#define COLIBRI_DEEPSEEK_V4_TARGET_VERIFY_PREFIX_H

#include "deepseek_v4_target_verify.h"

/* The target has already processed the anchor and draft_tokens[0] matched its
 * prediction. Verify drafts 1..3 with the same 2+2 schedule as the prior
 * verifier, while preserving the prefix main_x row for the next DSpark round. */
int coli_v4_target_verify_after_prefix_v69(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *draft_tokens, int draft_count, int prefix_position,
    const float *prefix_main_x, char *error, size_t error_size);

#endif
