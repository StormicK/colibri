#define coli_v4_dspark_capture_main_x \
    coli_v4_dspark_capture_main_x_underlying_v4
/* ---- begin inlined deepseek_v4_dspark_capture_v3.c ---- */
/* ---- begin inlined deepseek_v4_dspark_capture_v2.c ---- */
/* ---- begin inlined deepseek_v4_dspark_capture.c ---- */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_block.h"
#include "deepseek_v4_block_batch.h"
#include "deepseek_v4_dspark_heads.h"
#include "deepseek_v4_runtime.h"

static ColiV4DSparkHeads *capture_heads;
static ColiDeepSeekV4DSparkManifest capture_manifest;
static float *capture_states[COLI_V4_DSPARK_MAX_TARGETS];
static size_t capture_capacity;

static int capture_init(const ColiDeepSeekV4Config *config) {
    if (capture_heads) return 0;
    const char *model = coli_v4_runtime_options()->dspark_model_dir;
    char error[512];
    if (!model || coli_v4_dspark_inspect(model, config, &capture_manifest,
                                         error, sizeof(error)) ||
        coli_v4_dspark_heads_open(&capture_heads, model, config,
                                  &capture_manifest, error, sizeof(error))) {
        fprintf(stderr, "dspark capture disabled: %s\n", model ? error :
                "DSpark model path is unset");
        return -1;
    }
    return 0;
}

static int target_ordinal(int layer) {
    for (int i = 0; i < capture_manifest.target_count; i++)
        if (capture_manifest.target_layer_ids[i] == layer) return i;
    return -1;
}

static void capture_publish(const ColiDeepSeekV4Config *config, int batch) {
    size_t token_size = (size_t)config->hc_mult * config->hidden_size;
    float *joined = malloc((size_t)capture_manifest.target_count *
                           token_size * sizeof(float));
    float *main_x = malloc((size_t)config->hidden_size * sizeof(float));
    if (!joined || !main_x) { free(main_x); free(joined); return; }
    for (int item = 0; item < batch; item++) {
        for (int target = 0; target < capture_manifest.target_count; target++)
            memcpy(joined + (size_t)target * token_size,
                   capture_states[target] + (size_t)item * token_size,
                   token_size * sizeof(float));
        if (coli_v4_dspark_combine_hidden(capture_heads, main_x, joined)) break;
        double checksum = 0.0;
        for (int i = 0; i < config->hidden_size; i++)
            checksum += fabs((double)main_x[i]);
        if (item == batch - 1)
            fprintf(stderr, "dspark_main_x batch=%d first=%.9g l1=%.9g\n",
                    batch, main_x[0], checksum);
    }
    free(main_x); free(joined);
}

static void capture_layer(const ColiDeepSeekV4LayerWeights *weights,
                          const ColiDeepSeekV4Config *config,
                          const float *outputs, int batch) {
    if (capture_init(config)) return;
    int ordinal = target_ordinal(weights->plan.layer);
    if (ordinal < 0) return;
    size_t count = (size_t)batch * config->hc_mult * config->hidden_size;
    if (count > capture_capacity) {
        for (int i = 0; i < capture_manifest.target_count; i++) {
            float *allocation = realloc(capture_states[i], count * sizeof(float));
            if (!allocation) return;
            capture_states[i] = allocation;
        }
        capture_capacity = count;
    }
    memcpy(capture_states[ordinal], outputs, count * sizeof(float));
    if (ordinal == capture_manifest.target_count - 1)
        capture_publish(config, batch);
}

int __real_coli_v4_block_window_batch_ref(
    float *, ColiDeepSeekV4WindowAttentionState *,
    const ColiDeepSeekV4LayerWeights *, const ColiDeepSeekV4Config *,
    ColiExpertStore *, const float *, const int *, int, int, char *, size_t);

int __wrap_coli_v4_block_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs, const int *tokens, int start, int batch,
    char *error, size_t error_size) {
    int result = __real_coli_v4_block_window_batch_ref(
        outputs, attention, weights, config, experts, inputs, tokens,
        start, batch, error, error_size);
    if (!result) capture_layer(weights, config, outputs, batch);
    return result;
}

int __real_coli_v4_block_window_token_ref(
    float *, ColiDeepSeekV4WindowAttentionState *,
    const ColiDeepSeekV4LayerWeights *, const ColiDeepSeekV4Config *,
    ColiExpertStore *, const float *, int, int, char *, size_t);

int __wrap_coli_v4_block_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input, int token, int position,
    char *error, size_t error_size) {
    int result = __real_coli_v4_block_window_token_ref(
        output, attention, weights, config, experts, input, token, position,
        error, error_size);
    if (!result) capture_layer(weights, config, output, 1);
    return result;
}
/* ---- end inlined deepseek_v4_dspark_capture.c ---- */


/* ---- begin inlined deepseek_v4_dspark_capture_v2.h ---- */
#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V2_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V2_H

#include "deepseek_v4_config.h"

int coli_v4_dspark_capture_main_x(float *outputs, int batch,
                                  const ColiDeepSeekV4Config *config);

#endif
/* ---- end inlined deepseek_v4_dspark_capture_v2.h ---- */


int coli_v4_dspark_capture_main_x(float *outputs, int batch,
                                  const ColiDeepSeekV4Config *config) {
    if (!outputs || !config || batch < 1 ||
        capture_init(config) ||
        capture_capacity < (size_t)batch * config->hc_mult * config->hidden_size)
        return -1;
    size_t token_size = (size_t)config->hc_mult * config->hidden_size;
    float *joined = malloc((size_t)capture_manifest.target_count *
                           token_size * sizeof(*joined));
    if (!joined) return -1;
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        for (int target = 0; target < capture_manifest.target_count; target++)
            memcpy(joined + (size_t)target * token_size,
                   capture_states[target] + (size_t)item * token_size,
                   token_size * sizeof(*joined));
        result = coli_v4_dspark_combine_hidden(
            capture_heads, outputs + (size_t)item * config->hidden_size, joined);
    }
    free(joined); return result;
}
/* ---- end inlined deepseek_v4_dspark_capture_v2.c ---- */

/* ---- begin inlined deepseek_v4_dspark_capture_v3.h ---- */
#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V3_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V3_H

/* ---- begin inlined deepseek_v4_dspark_capture_v2.h ---- */
#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V2_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V2_H

#include "deepseek_v4_config.h"

int coli_v4_dspark_capture_main_x(float *outputs, int batch,
                                  const ColiDeepSeekV4Config *config);

#endif
/* ---- end inlined deepseek_v4_dspark_capture_v2.h ---- */

#include "deepseek_v4_dspark_heads.h"

ColiV4DSparkHeads *coli_v4_dspark_capture_heads(void);

#endif
/* ---- end inlined deepseek_v4_dspark_capture_v3.h ---- */


ColiV4DSparkHeads *coli_v4_dspark_capture_heads(void) { return capture_heads; }
/* ---- end inlined deepseek_v4_dspark_capture_v3.c ---- */

#undef coli_v4_dspark_capture_main_x

/* ---- begin inlined deepseek_v4_dspark_capture_v4.h ---- */
#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V4_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_V4_H

int coli_v4_dspark_capture_stage_main_x(const float *values, int batch,
                                        int hidden_size);

#endif
/* ---- end inlined deepseek_v4_dspark_capture_v4.h ---- */


static float *staged_main_x;
static int staged_batch;
static int staged_hidden;

int coli_v4_dspark_capture_stage_main_x(const float *values, int batch,
                                        int hidden_size) {
    if (!values || batch < 1 || batch > 64 || hidden_size < 1) return -1;
    size_t count = (size_t)batch * hidden_size;
    float *copy = realloc(staged_main_x, count * sizeof(*copy));
    if (!copy) return -1;
    staged_main_x = copy;
    memcpy(staged_main_x, values, count * sizeof(*copy));
    staged_batch = batch; staged_hidden = hidden_size;
    return 0;
}

int coli_v4_dspark_capture_main_x(float *outputs, int batch,
                                  const ColiDeepSeekV4Config *config) {
    if (staged_batch) {
        if (!outputs || !config || batch < 1 || batch > staged_batch ||
            config->hidden_size != staged_hidden) return -1;
        memcpy(outputs, staged_main_x,
               (size_t)batch * staged_hidden * sizeof(*outputs));
        staged_batch = 0;
        return 0;
    }
    return coli_v4_dspark_capture_main_x_underlying_v4(
        outputs, batch, config);
}
