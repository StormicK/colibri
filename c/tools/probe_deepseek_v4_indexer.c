#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../deepseek_v4_config.h"
#include "../deepseek_v4_indexer.h"
#include "../native_quant.h"
#include "../safetensors_index.h"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) return 2;
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiDeepSeekV4LayerWeights layer;
    ColiDeepSeekV4Indexer *state = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_v4_layer_load(&layer, &config, index, 2, error, sizeof(error)) ||
        coli_v4_indexer_create(&state, &layer, &config, 32,
                               error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    float *input = malloc((size_t)config.hidden_size * sizeof(*input));
    float *query_rank = malloc((size_t)config.q_lora_rank * sizeof(*query_rank));
    int indices[16], selected = 0;
    for (int position = 0; position < 8; position++) {
        for (int i = 0; i < config.hidden_size; i++) {
            int integer = (position * 131 + i * 17) % 257 - 128;
            input[i] = coli_bf16_round(integer / 128.0f);
        }
        for (int i = 0; i < config.q_lora_rank; i++) {
            int integer = (position * 73 + i * 29) % 193 - 96;
            query_rank[i] = coli_bf16_round(integer / 96.0f);
        }
        selected = coli_v4_indexer_step(state, indices, 16, query_rank,
                                        input, position, error, sizeof(error));
        if (selected < 0) { fprintf(stderr, "%s\n", error); return 1; }
    }
    const float *compressed = coli_v4_indexer_compressed_values(state);
    int count = coli_v4_indexer_compressed_count(state);
    double sum = 0.0, square = 0.0;
    for (int i = 0; i < count * config.index_head_dim; i++) {
        sum += compressed[i]; square += (double)compressed[i] * compressed[i];
    }
    printf("layer=2 count=%d selected=%d indices=", count, selected);
    for (int i = 0; i < selected; i++) printf("%s%d", i ? "," : "", indices[i]);
    printf(" sum=%.9g l2=%.9g\n", sum, sqrt(square));
    if (argc == 3) {
        FILE *stream = fopen(argv[2], "wb");
        if (!stream || fwrite(compressed, sizeof(*compressed),
                              (size_t)count * config.index_head_dim, stream) !=
                       (size_t)count * config.index_head_dim) return 1;
        fclose(stream);
    }
    free(query_rank); free(input); coli_v4_indexer_destroy(state);
    coli_v4_layer_free(&layer); coli_st_index_close(index);
    return 0;
}
