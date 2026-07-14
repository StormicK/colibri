#ifndef COLIBRI_DEEPSEEK_V4_LAYER_H
#define COLIBRI_DEEPSEEK_V4_LAYER_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_config.h"
#include "safetensors_index.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_V4_MAX_LAYER_TENSORS 48
#define COLI_V4_MAX_TENSOR_NAME 160

typedef struct {
    char name[COLI_V4_MAX_TENSOR_NAME];
    ColiSafetensorsDType dtype;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiDeepSeekV4TensorSpec;

typedef struct {
    int layer;
    int compression_ratio;
    int uses_hash_router;
    int has_compressor;
    int has_indexer;
    size_t tensor_count;
    ColiDeepSeekV4TensorSpec tensors[COLI_V4_MAX_LAYER_TENSORS];
} ColiDeepSeekV4LayerPlan;

typedef struct {
    size_t tensor_count;
    uint64_t total_bytes;
    uint64_t bf16_bytes;
    uint64_t f32_bytes;
    uint64_t fp8_weight_bytes;
    uint64_t fp8_scale_bytes;
    uint64_t i64_bytes;
} ColiDeepSeekV4LayerStats;

typedef struct {
    ColiDeepSeekV4LayerPlan plan;
    ColiDeepSeekV4LayerStats stats;
    void *data[COLI_V4_MAX_LAYER_TENSORS];
} ColiDeepSeekV4LayerWeights;

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size);
int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size);
int coli_v4_layer_load(ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size);
void coli_v4_layer_free(ColiDeepSeekV4LayerWeights *weights);
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec);

#ifdef __cplusplus
}
#endif

#endif
