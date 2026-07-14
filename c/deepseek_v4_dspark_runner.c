#include "native_quant.h"
#include "deepseek_v4_runtime.h"
#include "deepseek_v4_dspark_runtime_resident.h"

#define coli_v4_layer_free coli_v4_dspark_layer_release
#define coli_v4_dspark_runner_open coli_v4_dspark_runner_owned_open
#define coli_v4_dspark_runner_close coli_v4_dspark_runner_owned_close
/* ---- begin inlined deepseek_v4_dspark_runner.c ---- */
#include "deepseek_v4_dspark_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark_attention.h"
#include "deepseek_v4_dspark_block.h"
#include "deepseek_v4_dspark_final.h"
#include "deepseek_v4_dspark_heads.h"
#include "deepseek_v4_dspark_runtime.h"
#include "deepseek_v4_expert_store.h"
#include "safetensors_index.h"

struct ColiV4DSparkRunner {
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4DSparkManifest manifest;
    char *model_dir;
    ColiSafetensorsIndex *dspark_index;
    ColiSafetensorsIndex *target_index;
    ColiV4DSparkHeads *heads;
    ColiV4DSparkFinal *final;
    ColiExpertStore *experts;
    ColiV4DSparkAttentionState *attention[COLI_V4_DSPARK_MAX_STAGES];
    uint64_t loaded_stage_peak;
};

static int runner_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

static int runner_embedding(ColiV4DSparkRunner *runner, float *state,
                            int token) {
    const ColiSafetensorsTensor *embed =
        coli_st_find(runner->dspark_index, "embed.weight");
    int d = runner->config.hidden_size, hc = runner->config.hc_mult;
    uint16_t *raw = malloc((size_t)d * sizeof(*raw));
    if (!embed || embed->dtype != COLI_ST_BF16 || !raw || token < 0 ||
        token >= runner->config.vocab_size ||
        coli_st_read_at(runner->dspark_index, embed->shard,
                        embed->offset + (uint64_t)token * d * sizeof(*raw),
                        (size_t)d * sizeof(*raw), raw)) {
        free(raw); return -1;
    }
    for (int copy = 0; copy < hc; copy++)
        for (int i = 0; i < d; i++)
            state[(size_t)copy * d + i] = coli_bf16_decode(raw[i]);
    free(raw); return 0;
}

void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner) {
    if (!runner) return;
    for (int stage = 0; stage < runner->manifest.stage_count; stage++)
        coli_v4_dspark_attention_destroy(runner->attention[stage]);
    if (runner->experts) runner->experts->ops->destroy(runner->experts);
    coli_v4_dspark_final_close(runner->final);
    coli_v4_dspark_heads_close(runner->heads);
    coli_st_index_close(runner->target_index);
    coli_st_index_close(runner->dspark_index);
    free(runner->model_dir); free(runner);
}

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size) {
    if (!output || !dspark_model_dir || !target_model_dir || !config) return -1;
    *output = NULL; ColiV4DSparkRunner *runner = calloc(1, sizeof(*runner));
    if (!runner) return -1;
    runner->config = *config;
    runner->model_dir = malloc(strlen(dspark_model_dir) + 1);
    if (!runner->model_dir) { coli_v4_dspark_runner_close(runner); return -1; }
    strcpy(runner->model_dir, dspark_model_dir);
    if (coli_v4_dspark_inspect(dspark_model_dir, config, &runner->manifest,
                               error, error_size) ||
        coli_st_index_open(&runner->dspark_index, dspark_model_dir,
                           error, error_size) ||
        coli_st_index_open(&runner->target_index, target_model_dir,
                           error, error_size) ||
        coli_v4_dspark_heads_open(&runner->heads, dspark_model_dir, config,
                                  &runner->manifest, error, error_size) ||
        coli_v4_dspark_final_open(&runner->final, dspark_model_dir, config,
                                  &runner->manifest, error, error_size)) {
        coli_v4_dspark_runner_close(runner); return -1;
    }
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        if (runner->manifest.common_stage_bytes[stage] > runner->loaded_stage_peak)
            runner->loaded_stage_peak = runner->manifest.common_stage_bytes[stage];
        if (coli_v4_dspark_attention_create(&runner->attention[stage], config)) {
            coli_v4_dspark_runner_close(runner); return -1;
        }
    }
    ColiDeepSeekV4ExpertStoreOptions options = {
        dspark_model_dir, runner->manifest.stage_count,
        config->n_routed_experts, expert_cache_bytes};
    if (coli_deepseek_v4_dspark_expert_store_open(
            &options, &runner->experts, error, error_size)) {
        coli_v4_dspark_runner_close(runner); return -1;
    }
    *output = runner; return 0;
}

int coli_v4_dspark_runner_prefill(ColiV4DSparkRunner *runner,
                                  const float *main_x,
                                  int start_position, int batch,
                                  char *error, size_t error_size) {
    if (!runner || !main_x || start_position < 0 || batch < 1 || batch > 64)
        return runner_error(error, error_size, "invalid DSpark prefill");
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        ColiDeepSeekV4LayerWeights weights;
        if (coli_v4_dspark_layer_load(&weights, &runner->config,
                                      runner->dspark_index, stage,
                                      error, error_size)) return -1;
        int result = coli_v4_dspark_attention_precompute_context(
            runner->attention[stage], &weights, &runner->config,
            main_x, start_position, batch, error, error_size);
        coli_v4_layer_free(&weights);
        if (result) return -1;
    }
    return 0;
}

int coli_v4_dspark_runner_draft(ColiV4DSparkRunner *runner,
                                const float *main_x, int anchor_token,
                                int position, int *draft_tokens,
                                float *draft_logits,
                                char *error, size_t error_size) {
    if (!runner || !main_x || !draft_tokens || position < 0 ||
        anchor_token < 0 || anchor_token >= runner->config.vocab_size)
        return runner_error(error, error_size, "invalid DSpark draft");
    int batch = runner->manifest.block_size;
    int d = runner->config.hidden_size, hc = runner->config.hc_mult;
    size_t hd = (size_t)hc * d;
    float *state = calloc((size_t)batch * hd, sizeof(*state));
    float *next = calloc((size_t)batch * hd, sizeof(*next));
    float *hidden = malloc((size_t)batch * d * sizeof(*hidden));
    int *input_tokens = malloc((size_t)batch * sizeof(*input_tokens));
    if (!state || !next || !hidden || !input_tokens) {
        free(input_tokens); free(hidden); free(next); free(state);
        return runner_error(error, error_size, "out of memory in DSpark draft");
    }
    for (int item = 0; item < batch; item++) {
        input_tokens[item] = item ? runner->manifest.noise_token_id : anchor_token;
        if (runner_embedding(runner, state + (size_t)item * hd,
                             input_tokens[item])) {
            free(input_tokens); free(hidden); free(next); free(state);
            return runner_error(error, error_size, "DSpark embedding read failed");
        }
    }
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        ColiDeepSeekV4LayerWeights weights;
        if (coli_v4_dspark_layer_load(&weights, &runner->config,
                                      runner->dspark_index, stage,
                                      error, error_size)) {
            free(input_tokens); free(hidden); free(next); free(state); return -1;
        }
        int result = coli_v4_dspark_attention_precompute_context(
            runner->attention[stage], &weights, &runner->config,
            main_x, position, 1, error, error_size);
        if (!result) result = coli_v4_dspark_block(
            next, runner->attention[stage], &weights, &runner->config,
            runner->experts, state, input_tokens, position + 1, batch,
            error, error_size);
        coli_v4_layer_free(&weights);
        if (result) {
            free(input_tokens); free(hidden); free(next); free(state); return -1;
        }
        float *swap = state; state = next; next = swap;
    }
    int result = coli_v4_dspark_final_hidden(runner->final, hidden, state, batch);
    int previous = anchor_token;
    for (int item = 0; !result && item < batch; item++) {
        float logit = 0.0f; int token = -1;
        result = coli_v4_dspark_biased_argmax(
            runner->heads, runner->target_index, hidden + (size_t)item * d,
            previous, &token, &logit);
        draft_tokens[item] = token;
        if (draft_logits) draft_logits[item] = logit;
        previous = token;
    }
    free(input_tokens); free(hidden); free(next); free(state);
    return result ? runner_error(error, error_size, "DSpark draft head failed") : 0;
}

int coli_v4_dspark_runner_block_size(const ColiV4DSparkRunner *runner) {
    return runner ? runner->manifest.block_size : 0;
}

uint64_t coli_v4_dspark_runner_loaded_stage_peak(
    const ColiV4DSparkRunner *runner) {
    return runner ? runner->loaded_stage_peak : 0;
}
/* ---- end inlined deepseek_v4_dspark_runner.c ---- */

#undef coli_v4_dspark_runner_close
#undef coli_v4_dspark_runner_open
#undef coli_v4_layer_free

#include "deepseek_v4_dspark_runner_shared.h"

static ColiV4DSparkHeads *runner_v5_shared_heads;

static uint64_t runner_v5_cache_bytes(uint64_t fallback) {
    uint64_t configured =
        coli_v4_runtime_options()->dspark_expert_cache_bytes;
    return configured ? configured : fallback;
}

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size) {
    return coli_v4_dspark_runner_owned_open(
        output, dspark_model_dir, target_model_dir, config,
        runner_v5_cache_bytes(expert_cache_bytes), error, error_size);
}

int coli_v4_dspark_runner_use_shared_heads(ColiV4DSparkRunner *runner,
                                           ColiV4DSparkHeads *heads) {
    if (!runner || !heads) return -1;
    coli_v4_dspark_heads_close(runner->heads);
    runner->heads = heads; runner_v5_shared_heads = heads; return 0;
}

void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner) {
    if (runner && runner->heads == runner_v5_shared_heads) runner->heads = NULL;
    coli_v4_dspark_runner_owned_close(runner);
}
