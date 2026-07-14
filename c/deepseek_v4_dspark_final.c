#include "native_quant.h"
/* ---- begin inlined deepseek_v4_dspark_final.c ---- */
#include "deepseek_v4_dspark_final.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "deepseek_v4_math.h"
#include "safetensors_index.h"
#include "tensor_io.h"

struct ColiV4DSparkFinal {
    int hidden, hc;
    float eps, hc_eps;
    ColiFloatTensor function, base, scale, norm;
};

void coli_v4_dspark_final_close(ColiV4DSparkFinal *final) {
    if (!final) return;
    coli_float_tensor_free(&final->norm);
    coli_float_tensor_free(&final->scale);
    coli_float_tensor_free(&final->base);
    coli_float_tensor_free(&final->function);
    free(final);
}

int coli_v4_dspark_final_open(ColiV4DSparkFinal **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size) {
    if (!output || !model_dir || !config || !manifest ||
        manifest->stage_count < 1) return -1;
    *output = NULL; ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    ColiV4DSparkFinal *final = calloc(1, sizeof(*final));
    if (!final) { coli_st_index_close(index); return -1; }
    final->hidden = config->hidden_size; final->hc = config->hc_mult;
    final->eps = config->rms_norm_eps; final->hc_eps = config->hc_eps;
    char name[160]; int last = manifest->stage_count - 1, failed = 0;
    snprintf(name, sizeof(name), "mtp.%d.hc_head_fn", last);
    failed |= coli_tensor_load_f32(&final->function, index, name,
                                   error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.hc_head_base", last);
    failed |= coli_tensor_load_f32(&final->base, index, name, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.hc_head_scale", last);
    failed |= coli_tensor_load_f32(&final->scale, index, name, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.norm.weight", last);
    failed |= coli_tensor_load_f32(&final->norm, index, name, error, error_size);
    coli_st_index_close(index);
    if (failed) { coli_v4_dspark_final_close(final); return -1; }
    *output = final; return 0;
}

int coli_v4_dspark_final_hidden(ColiV4DSparkFinal *final,
                                float *outputs, const float *states_hc,
                                int batch) {
    if (!final || !outputs || !states_hc || batch < 1 || batch > 64) return -1;
    int d = final->hidden, hc = final->hc, flattened = d * hc;
    if (hc > 16) return -1;
    for (int item = 0; item < batch; item++) {
        const float *state = states_hc + (size_t)item * flattened;
        float *output = outputs + (size_t)item * d;
        float square = 0.0f, pre[16];
        for (int i = 0; i < flattened; i++) square += state[i] * state[i];
        float inverse = 1.0f / sqrtf(square / flattened + final->eps);
        for (int copy = 0; copy < hc; copy++) {
            float mix = 0.0f;
            for (int i = 0; i < flattened; i++)
                mix += final->function.data[(size_t)copy * flattened + i] *
                       state[i];
            float z = mix * inverse * final->scale.data[0] +
                      final->base.data[copy];
            float sigmoid = z >= 0.0f ? 1.0f / (1.0f + expf(-z))
                                      : expf(z) / (1.0f + expf(z));
            pre[copy] = sigmoid + final->hc_eps;
        }
        for (int i = 0; i < d; i++) {
            float value = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                value += pre[copy] * state[(size_t)copy * d + i];
            output[i] = coli_bf16_round(value);
        }
        if (coli_v4_rmsnorm(output, output, final->norm.data, d, final->eps))
            return -1;
        coli_bf16_round_array(output, (size_t)d);
    }
    return 0;
}
/* ---- end inlined deepseek_v4_dspark_final.c ---- */

