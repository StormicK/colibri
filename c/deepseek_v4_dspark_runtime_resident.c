#define coli_v4_dspark_layer_load coli_v4_dspark_layer_reference_load
#include "deepseek_v4_dspark_runtime.c"
#undef coli_v4_dspark_layer_load

#include "deepseek_v4_dspark_runtime_resident.h"
#include "deepseek_v4_runtime.h"

static ColiDeepSeekV4LayerWeights resident_dspark_layers[COLI_V4_DSPARK_MAX_STAGES];
static unsigned char resident_dspark_ready[COLI_V4_DSPARK_MAX_STAGES];
static const ColiSafetensorsIndex *resident_dspark_index;
static uint64_t resident_dspark_bytes;

static int dspark_stages_resident(void) {
    return coli_v4_runtime_options()->dspark_resident;
}

int coli_v4_dspark_layer_load(ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size) {
    if (!dspark_stages_resident())
        return coli_v4_dspark_layer_reference_load(
            weights, config, index, stage, error, error_size);
    if (!weights || !config || !index || stage < 0 ||
        stage >= COLI_V4_DSPARK_MAX_STAGES) return -1;
    if (resident_dspark_index && resident_dspark_index != index) {
        if (error && error_size)
            snprintf(error, error_size,
                     "resident DSpark stages cannot switch model instances");
        return -1;
    }
    resident_dspark_index = index;
    if (!resident_dspark_ready[stage]) {
        if (coli_v4_dspark_layer_reference_load(
                &resident_dspark_layers[stage], config, index, stage,
                error, error_size)) return -1;
        resident_dspark_ready[stage] = 1;
        resident_dspark_bytes += resident_dspark_layers[stage].stats.total_bytes;
        fprintf(stderr, "dspark_stage_resident stage=%d total=%.3fGiB\n",
                stage, resident_dspark_bytes / 1073741824.0);
    }
    *weights = resident_dspark_layers[stage]; return 0;
}

void coli_v4_dspark_layer_release(ColiDeepSeekV4LayerWeights *weights) {
    if (!weights) return;
    int stage = weights->plan.layer;
    if (dspark_stages_resident() && stage >= 0 &&
        stage < COLI_V4_DSPARK_MAX_STAGES && resident_dspark_ready[stage] &&
        weights->data[0] == resident_dspark_layers[stage].data[0]) {
        memset(weights, 0, sizeof(*weights)); return;
    }
    coli_v4_layer_free(weights);
}
