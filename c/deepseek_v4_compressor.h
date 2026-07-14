#ifndef COLIBRI_DEEPSEEK_V4_COMPRESSOR_H
#define COLIBRI_DEEPSEEK_V4_COMPRESSOR_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4CompressorState ColiDeepSeekV4CompressorState;

typedef struct {
    const char *prefix;
    int head_dimension;
    int rotate_fp4;
} ColiDeepSeekV4CompressorOptions;

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **state,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size);
int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);
void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state);
int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size);
void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state);

/* Processes one decode token. produced is set to one only when a complete
 * compression window emits a KV vector. output may be NULL on other steps. */
int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
