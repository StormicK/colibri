#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

#define COLI_V4_DSPARK_MAX_STAGES 8
#define COLI_V4_DSPARK_MAX_TARGETS 8

typedef struct {
    int stage_count;
    int block_size;
    int noise_token_id;
    int markov_rank;
    int target_count;
    int target_layer_ids[COLI_V4_DSPARK_MAX_TARGETS];
    uint64_t common_stage_bytes[COLI_V4_DSPARK_MAX_STAGES];
    uint64_t special_bytes;
} ColiDeepSeekV4DSparkManifest;

int coli_v4_dspark_inspect(const char *model_dir,
                           const ColiDeepSeekV4Config *config,
                           ColiDeepSeekV4DSparkManifest *manifest,
                           char *error, size_t error_size);
int coli_v4_dspark_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                              const ColiDeepSeekV4Config *config, int stage,
                              char *error, size_t error_size);

#endif
