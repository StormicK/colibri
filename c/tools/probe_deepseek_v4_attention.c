#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_attention.h"
#include "../deepseek_v4_config.h"
#include "../deepseek_v4_math.h"
#include "../native_quant.h"
#include "../safetensors_index.h"

static const void *layer_value(const ColiDeepSeekV4LayerWeights *weights,
                               const char *suffix) {
    char name[160];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, NULL);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL_DIR TOKEN_ID [OUTPUT_F32]\n", argv[0]);
        return 2;
    }
    int token = atoi(argv[2]);
    char error[512];
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiDeepSeekV4LayerWeights layer;
    memset(&layer, 0, sizeof(layer));
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        token < 0 || token >= config.vocab_size ||
        coli_v4_layer_load(&layer, &config, index, 0, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        coli_st_index_close(index);
        return 1;
    }
    int d = config.hidden_size, hc = config.hc_mult;
    float *expanded = malloc((size_t)hc * d * sizeof(*expanded));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *output = malloc((size_t)d * sizeof(*output));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    float *norm_weight = malloc((size_t)d * sizeof(*norm_weight));
    uint16_t *embedding = malloc((size_t)d * sizeof(*embedding));
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    if (!expanded || !reduced || !normalized || !output || !post || !comb ||
        !norm_weight || !embedding || !embed || embed->dtype != COLI_ST_BF16 ||
        coli_st_read_at(index, embed->shard,
                        embed->offset + (uint64_t)token * d * sizeof(*embedding),
                        (size_t)d * sizeof(*embedding), embedding)) {
        fprintf(stderr, "cannot load embedding row\n");
        return 1;
    }
    const uint16_t *norm = layer_value(&layer, "attn_norm.weight");
    for (int i = 0; i < d; i++) {
        float value = coli_bf16_decode(embedding[i]);
        norm_weight[i] = coli_bf16_decode(norm[i]);
        for (int copy = 0; copy < hc; copy++) expanded[copy * d + i] = value;
    }
    if (coli_v4_hc_pre(reduced, post, comb, expanded,
                       layer_value(&layer, "hc_attn_fn"),
                       layer_value(&layer, "hc_attn_scale"),
                       layer_value(&layer, "hc_attn_base"), hc, d,
                       config.hc_sinkhorn_iters, config.rms_norm_eps,
                       config.hc_eps) ||
        (coli_bf16_round_array(reduced, (size_t)d),
         coli_v4_rmsnorm(normalized, reduced, norm_weight, d,
                         config.rms_norm_eps)) ||
        (coli_bf16_round_array(normalized, (size_t)d),
         coli_v4_attention_token_ref(output, &layer, &config, normalized, 0,
                                     error, sizeof(error)))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    double sum = 0.0, square = 0.0;
    for (int i = 0; i < d; i++) {
        sum += output[i];
        square += (double)output[i] * output[i];
    }
    printf("token=%d layer=0 attention sum=%.9g l2=%.9g first=%g,%g,%g,%g\n",
           token, sum, sqrt(square), output[0], output[1], output[2], output[3]);
    if (argc == 4) {
        FILE *stream = fopen(argv[3], "wb");
        if (!stream || fwrite(output, sizeof(*output), (size_t)d, stream) != (size_t)d) {
            fprintf(stderr, "cannot write output\n");
            return 1;
        }
        fclose(stream);
    }
    free(embedding); free(norm_weight); free(comb); free(post); free(output);
    free(normalized); free(reduced); free(expanded);
    coli_v4_layer_free(&layer);
    coli_st_index_close(index);
    return 0;
}
