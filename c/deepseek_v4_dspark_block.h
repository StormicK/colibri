#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_BLOCK_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_BLOCK_H

#include "deepseek_v4_dspark_attention.h"
#include "deepseek_v4_expert_store.h"

int coli_v4_dspark_block(
    float *outputs_hc, ColiV4DSparkAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens,
    int query_start_position, int batch,
    char *error, size_t error_size);

#endif
