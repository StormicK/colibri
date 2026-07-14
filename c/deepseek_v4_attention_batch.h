#ifndef COLIBRI_DEEPSEEK_V4_ATTENTION_BATCH_H
#define COLIBRI_DEEPSEEK_V4_ATTENTION_BATCH_H

#include "deepseek_v4_attention.h"

int coli_v4_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size);

#endif
