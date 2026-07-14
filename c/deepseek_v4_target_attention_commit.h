#ifndef COLIBRI_DEEPSEEK_V4_TARGET_ATTENTION_COMMIT_H
#define COLIBRI_DEEPSEEK_V4_TARGET_ATTENTION_COMMIT_H

#include "deepseek_v4_attention.h"
#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

/* Advance only a target layer's causal attention/cache for already-computed
 * layer inputs. Used after speculative rollback to avoid replaying HC post and
 * the complete MoE branch. */
int coli_v4_target_attention_commit_batch(
    ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *inputs_hc, int start_position, int batch,
    char *error, size_t error_size);

#endif
