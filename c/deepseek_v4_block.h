#ifndef COLIBRI_DEEPSEEK_V4_BLOCK_H
#define COLIBRI_DEEPSEEK_V4_BLOCK_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_attention.h"
#include "deepseek_v4_layer.h"
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size);
int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
