#ifndef COLIBRI_DEEPSEEK_V4_BLOCK_BATCH_H
#define COLIBRI_DEEPSEEK_V4_BLOCK_BATCH_H

#include "deepseek_v4_attention.h"
#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"
#include "expert_store.h"

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size);

#endif
