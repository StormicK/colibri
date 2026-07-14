#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "../deepseek_v4_config.h"
#include "../deepseek_v4_layer.h"
#include "../safetensors_index.h"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL_DIR [LOAD_LAYER]\n", argv[0]);
        return 2;
    }
    char error[512];
    ColiDeepSeekV4Config config;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) != 0) {
        fprintf(stderr, "config: %s\n", error);
        return 1;
    }
    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, argv[1], error, sizeof(error)) != 0) {
        fprintf(stderr, "index: %s\n", error);
        return 1;
    }
    uint64_t total = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        ColiDeepSeekV4LayerPlan plan;
        ColiDeepSeekV4LayerStats stats;
        if (coli_v4_layer_plan(&plan, &config, layer, error, sizeof(error)) != 0 ||
            coli_v4_layer_validate(&plan, index, &stats, error, sizeof(error)) != 0) {
            fprintf(stderr, "layer %d: %s\n", layer, error);
            coli_st_index_close(index);
            return 1;
        }
        total += stats.total_bytes;
        printf("layer %2d ratio=%3d indexer=%d tensors=%2zu resident=%9.2f MiB\n",
               layer, plan.compression_ratio, plan.has_indexer, stats.tensor_count,
               stats.total_bytes / 1048576.0);
    }
    printf("validated %d layers, planned resident tensors %.2f GiB (%" PRIu64 " bytes)\n",
           config.num_hidden_layers, total / 1073741824.0, total);
    if (argc == 3) {
        int layer = atoi(argv[2]);
        ColiDeepSeekV4LayerWeights weights;
        if (coli_v4_layer_load(&weights, &config, index, layer,
                               error, sizeof(error)) != 0) {
            fprintf(stderr, "load layer %d: %s\n", layer, error);
            coli_st_index_close(index);
            return 1;
        }
        printf("loaded layer %d in native checkpoint formats: %.2f MiB\n",
               layer, weights.stats.total_bytes / 1048576.0);
        coli_v4_layer_free(&weights);
    }
    coli_st_index_close(index);
    return 0;
}
