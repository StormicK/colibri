#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../deepseek_v4_attention.h"
#include "../deepseek_v4_config.h"
#include "../native_quant.h"
#include "../safetensors_index.h"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiDeepSeekV4LayerWeights layer;
    ColiDeepSeekV4WindowAttentionState *state = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_v4_layer_load(&layer, &config, index, 2, error, sizeof(error)) ||
        coli_v4_window_attention_create(&state, &config)) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    float *input = malloc((size_t)config.hidden_size * sizeof(*input));
    float *output = malloc((size_t)config.hidden_size * sizeof(*output));
    for (int position = 0; position < 4; position++) {
        for (int i = 0; i < config.hidden_size; i++) {
            int integer = (position * 131 + i * 17) % 257 - 128;
            input[i] = coli_bf16_round(integer / 128.0f);
        }
        if (coli_v4_attention_window_token_ref(output, state, &layer, &config,
                                               input, position,
                                               error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
    }
    double sum = 0.0, square = 0.0;
    for (int i = 0; i < config.hidden_size; i++) {
        sum += output[i]; square += (double)output[i] * output[i];
    }
    printf("layer=2 positions=4 sum=%.9g l2=%.9g\n", sum, sqrt(square));
    FILE *stream = fopen(argv[2], "wb");
    if (!stream || fwrite(output, sizeof(*output),
                          (size_t)config.hidden_size, stream) !=
                   (size_t)config.hidden_size) return 1;
    fclose(stream);
    free(output); free(input); coli_v4_window_attention_destroy(state);
    coli_v4_layer_free(&layer); coli_st_index_close(index);
    return 0;
}
