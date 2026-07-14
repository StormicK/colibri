#include "deepseek_v4_dspark_runtime.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int runtime_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static void add_stats(ColiDeepSeekV4LayerStats *stats,
                      const ColiSafetensorsTensor *tensor) {
    stats->tensor_count++; stats->total_bytes += tensor->nbytes;
    switch (tensor->dtype) {
        case COLI_ST_BF16: stats->bf16_bytes += tensor->nbytes; break;
        case COLI_ST_F32: stats->f32_bytes += tensor->nbytes; break;
        case COLI_ST_F8_E4M3: stats->fp8_weight_bytes += tensor->nbytes; break;
        case COLI_ST_F8_E8M0: stats->fp8_scale_bytes += tensor->nbytes; break;
        case COLI_ST_I64: stats->i64_bytes += tensor->nbytes; break;
        default: break;
    }
}

int coli_v4_dspark_layer_load(ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size) {
    if (!weights || !config || !index)
        return runtime_error(error, error_size, "invalid DSpark layer load");
    memset(weights, 0, sizeof(*weights));
    if (coli_v4_dspark_layer_plan(&weights->plan, config, stage,
                                  error, error_size)) return -1;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        ColiDeepSeekV4TensorSpec *spec = &weights->plan.tensors[i];
        char actual[COLI_V4_MAX_TENSOR_NAME];
        strcpy(actual, spec->name);
        const ColiSafetensorsTensor *tensor = coli_st_find(index, actual);
        if (!tensor || tensor->dtype != spec->dtype ||
            tensor->rank != spec->rank) {
            coli_v4_layer_free(weights);
            return runtime_error(error, error_size,
                                 "invalid DSpark tensor: %s", actual);
        }
        for (int dimension = 0; dimension < spec->rank; dimension++)
            if (tensor->shape[dimension] != spec->shape[dimension]) {
                coli_v4_layer_free(weights);
                return runtime_error(error, error_size,
                                     "DSpark shape mismatch: %s", actual);
            }
        weights->data[i] = malloc((size_t)tensor->nbytes);
        if (!weights->data[i] ||
            coli_st_read_tensor(index, tensor, weights->data[i])) {
            coli_v4_layer_free(weights);
            return runtime_error(error, error_size,
                                 "cannot load DSpark tensor: %s", actual);
        }
        add_stats(&weights->stats, tensor);
        const char *suffix = actual + strlen("mtp.0");
        int prefix = snprintf(spec->name, sizeof(spec->name),
                              "layers.%d", stage);
        size_t suffix_length = strlen(suffix);
        if (prefix < 0 || (size_t)prefix + suffix_length >= sizeof(spec->name)) {
            coli_v4_layer_free(weights);
            return runtime_error(error, error_size,
                                 "DSpark runtime tensor name is too long");
        }
        memcpy(spec->name + prefix, suffix, suffix_length + 1);
    }
    return 0;
}
