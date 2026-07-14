#define coli_v4_layer_load coli_v4_layer_resident_reference_load
#define coli_v4_layer_free coli_v4_layer_resident_reference_free
#include "deepseek_v4_layer.c"
#undef coli_v4_layer_free
#undef coli_v4_layer_load

#include "deepseek_v4_runtime.h"

enum { COLI_V4_RESIDENT_MAX_LAYERS_V2 = 128 };
static ColiDeepSeekV4LayerWeights resident_layers_v2[COLI_V4_RESIDENT_MAX_LAYERS_V2];
static unsigned char resident_ready_v2[COLI_V4_RESIDENT_MAX_LAYERS_V2];
static const ColiDeepSeekV4Config *resident_config_v2;
static const ColiSafetensorsIndex *resident_index_v2;
static uint64_t resident_total_bytes_v2;

static int resident_enabled_v2(void) {
    return coli_v4_runtime_options()->dense_resident;
}

int coli_v4_layer_load(ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size) {
    if (!resident_enabled_v2())
        return coli_v4_layer_resident_reference_load(
            weights, config, index, layer, error, error_size);
    if (!weights || !config || !index || layer < 0 ||
        layer >= config->num_hidden_layers ||
        layer >= COLI_V4_RESIDENT_MAX_LAYERS_V2) return -1;
    if ((resident_config_v2 && resident_config_v2 != config) ||
        (resident_index_v2 && resident_index_v2 != index)) {
        if (error && error_size)
            snprintf(error, error_size,
                     "resident V4 dense cache cannot switch model instances");
        return -1;
    }
    resident_config_v2 = config; resident_index_v2 = index;
    if (!resident_ready_v2[layer]) {
        if (coli_v4_layer_resident_reference_load(
                &resident_layers_v2[layer], config, index, layer,
                error, error_size)) return -1;
        resident_ready_v2[layer] = 1;
        resident_total_bytes_v2 += resident_layers_v2[layer].stats.total_bytes;
        if (layer == config->num_hidden_layers - 1)
            fprintf(stderr, "v4_dense_resident layers=%d bytes=%.3fGiB\n",
                    config->num_hidden_layers,
                    resident_total_bytes_v2 / 1073741824.0);
    }
    *weights = resident_layers_v2[layer]; return 0;
}

void coli_v4_layer_free(ColiDeepSeekV4LayerWeights *weights) {
    if (!weights) return;
    int layer = weights->plan.layer;
    if (layer >= 0 && layer < COLI_V4_RESIDENT_MAX_LAYERS_V2 &&
        resident_ready_v2[layer] &&
        weights->plan.tensor_count == resident_layers_v2[layer].plan.tensor_count &&
        weights->data[0] == resident_layers_v2[layer].data[0]) {
        memset(weights, 0, sizeof(*weights)); return;
    }
    coli_v4_layer_resident_reference_free(weights);
}
