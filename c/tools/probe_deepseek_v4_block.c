#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_block.h"
#include "../deepseek_v4_config.h"
#include "../deepseek_v4_expert_store.h"
#include "../native_quant.h"
#include "../safetensors_index.h"

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL_DIR TOKEN_ID [OUTPUT_F32]\n", argv[0]);
        return 2;
    }
    int token = atoi(argv[2]);
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiDeepSeekV4LayerWeights layer;
    ColiExpertStore *store = NULL;
    memset(&layer, 0, sizeof(layer));
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_v4_layer_load(&layer, &config, index, 0, error, sizeof(error)) ||
        coli_deepseek_v4_expert_store_open(
            &(ColiDeepSeekV4ExpertStoreOptions){
                argv[1], config.num_hidden_layers, config.n_routed_experts,
                UINT64_C(4) * 1024 * 1024 * 1024,
            }, &store, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    int d = config.hidden_size, hc = config.hc_mult;
    uint16_t *row = malloc((size_t)d * sizeof(*row));
    float *input = malloc((size_t)hc * d * sizeof(*input));
    float *output = malloc((size_t)hc * d * sizeof(*output));
    if (!embed || !row || !input || !output || token < 0 || token >= config.vocab_size ||
        coli_st_read_at(index, embed->shard,
                        embed->offset + (uint64_t)token * d * sizeof(*row),
                        (size_t)d * sizeof(*row), row)) {
        fprintf(stderr, "cannot load embedding\n");
        return 1;
    }
    for (int copy = 0; copy < hc; copy++)
        for (int i = 0; i < d; i++) input[(size_t)copy * d + i] = coli_bf16_decode(row[i]);
    if (coli_v4_block_token_ref(output, &layer, &config, store, input,
                                token, 0, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    double sum = 0.0, square = 0.0;
    for (int i = 0; i < hc * d; i++) {
        sum += output[i];
        square += (double)output[i] * output[i];
    }
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    printf("token=%d layer=0 block sum=%.9g l2=%.9g expert_reads=%llu bytes=%llu\n",
           token, sum, sqrt(square), (unsigned long long)stats.misses,
           (unsigned long long)stats.bytes_read);
    if (argc == 4) {
        FILE *stream = fopen(argv[3], "wb");
        if (!stream || fwrite(output, sizeof(*output), (size_t)hc * d, stream) != (size_t)hc * d)
            return 1;
        fclose(stream);
    }
    free(output); free(input); free(row);
    store->ops->destroy(store);
    coli_v4_layer_free(&layer);
    coli_st_index_close(index);
    return 0;
}
