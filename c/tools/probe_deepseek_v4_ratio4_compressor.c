#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../deepseek_v4_compressor.h"
#include "../deepseek_v4_config.h"
#include "../native_quant.h"
#include "../safetensors_index.h"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) return 2;
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiDeepSeekV4LayerWeights layer;
    ColiDeepSeekV4CompressorState *state = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_v4_layer_load(&layer, &config, index, 2, error, sizeof(error)) ||
        coli_v4_compressor_create(&state, &layer, &config, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    float *input = malloc((size_t)config.hidden_size * sizeof(*input));
    float *output = malloc((size_t)config.head_dim * sizeof(*output));
    int produced = 0, emissions = 0;
    for (int position = 0; position < 8; position++) {
        for (int i = 0; i < config.hidden_size; i++) {
            int integer = (position * 131 + i * 17) % 257 - 128;
            input[i] = coli_bf16_round(integer / 128.0f);
        }
        if (coli_v4_compressor_step(state, output, &produced, input, position,
                                    error, sizeof(error))) return 1;
        emissions += produced;
    }
    if (emissions != 2) return 1;
    double sum = 0.0, square = 0.0;
    for (int i = 0; i < config.head_dim; i++) {
        sum += output[i]; square += (double)output[i] * output[i];
    }
    printf("layer=2 ratio=4 emissions=2 sum=%.9g l2=%.9g first=%g,%g,%g,%g\n",
           sum, sqrt(square), output[0], output[1], output[2], output[3]);
    if (argc == 3) {
        FILE *stream = fopen(argv[2], "wb");
        if (!stream || fwrite(output, sizeof(*output),
                              (size_t)config.head_dim, stream) !=
                       (size_t)config.head_dim) return 1;
        fclose(stream);
    }
    free(output); free(input); coli_v4_compressor_destroy(state);
    coli_v4_layer_free(&layer); coli_st_index_close(index);
    return 0;
}
