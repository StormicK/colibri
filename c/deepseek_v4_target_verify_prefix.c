#include "deepseek_v4_target_verify_prefix.h"

#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_attention_transaction.h"
#include "deepseek_v4_block_batch.h"
#include "deepseek_v4_dspark_capture.h"
#include "deepseek_v4_layer.h"
#include "deepseek_v4_target_attention_commit.h"
#include "deepseek_v4_target_head_batch.h"

static void prefix_destroy_snapshots(ColiV4AttentionSnapshot **snapshots,
                                     int layers) {
    if (!snapshots) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots);
}

static int prefix_create_snapshots(
    ColiV4AttentionSnapshot ***output,
    ColiDeepSeekV4WindowAttentionState **attention, int layers) {
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)layers, sizeof(*snapshots));
    if (!snapshots) return -1;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) {
            prefix_destroy_snapshots(snapshots, layers); return -1;
        }
    *output = snapshots;
    return 0;
}

static int prefix_restore_snapshots(
    ColiDeepSeekV4WindowAttentionState **attention,
    ColiV4AttentionSnapshot **snapshots, int layers) {
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_restore(attention[layer],
                                               snapshots[layer])) return -1;
    return 0;
}

static int prefix_run_target_batch(
    float **state_ptr, float **next_ptr,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *tokens, int start, int batch, float *layer_inputs,
    char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        memcpy(layer_inputs + (size_t)layer_id * batch * hd, state,
               (size_t)batch * hd * sizeof(*state));
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(&layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_batch_ref(
            next, attention[layer_id], &layer, config, experts,
            state, tokens, start, batch, error, error_size);
        coli_v4_layer_free(&layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
    }
    *state_ptr = state;
    *next_ptr = next;
    return 0;
}

static int prefix_commit_attention(
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config,
    const float *layer_inputs, int recorded_batch,
    int start, int commit_batch, char *error, size_t error_size) {
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(&layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_target_attention_commit_batch(
            attention[layer_id], &layer, config,
            layer_inputs + (size_t)layer_id * recorded_batch * hd,
            start, commit_batch, error, error_size);
        coli_v4_layer_free(&layer);
        if (result) return -1;
    }
    return 0;
}

int coli_v4_target_verify_after_prefix_v69(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *draft_tokens, int draft_count, int prefix_position,
    const float *prefix_main_x, char *error, size_t error_size) {
    enum { FIRST_BATCH = 2, TAIL_BATCH = 2, TOTAL_ROWS = 5 };
    if (!verification || !output_tokens || output_capacity < TOTAL_ROWS ||
        !attention || !index || !config || !experts || !draft_tokens ||
        draft_count != 4 || prefix_position < 0 || !prefix_main_x) return -1;

    int result = 0;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)d * hc;
    float *combined_main_x = malloc((size_t)TOTAL_ROWS * d * sizeof(float));
    float *state = malloc((size_t)FIRST_BATCH * hd * sizeof(float));
    float *next = malloc((size_t)FIRST_BATCH * hd * sizeof(float));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 FIRST_BATCH * hd * sizeof(float));
    ColiV4AttentionSnapshot **snapshots = NULL;
    if (!combined_main_x || !state || !next || !layer_inputs ||
        prefix_create_snapshots(&snapshots, attention,
                                config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    memcpy(combined_main_x, prefix_main_x, (size_t)d * sizeof(float));

    int first_inputs[FIRST_BATCH] = {draft_tokens[0], draft_tokens[1]};
    int first_targets[FIRST_BATCH];
    float first_logits[FIRST_BATCH];
    result = coli_v4_target_load_embeddings(
                 state, index, config, first_inputs, FIRST_BATCH) ||
             prefix_run_target_batch(
                 &state, &next, attention, index, config, experts,
                 first_inputs, prefix_position + 1, FIRST_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, FIRST_BATCH,
                 first_targets, first_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(
                 combined_main_x + d, FIRST_BATCH, config);
    if (result) goto cleanup;

    output_tokens[0] = draft_tokens[0];
    int accepted = 0;
    while (accepted < FIRST_BATCH &&
           draft_tokens[accepted + 1] == first_targets[accepted]) {
        output_tokens[accepted + 1] = draft_tokens[accepted + 1];
        accepted++;
    }
    if (accepted < FIRST_BATCH) {
        int commit = accepted + 1;
        output_tokens[accepted + 1] = first_targets[accepted];
        verification->accepted_draft_tokens = accepted + 1;
        verification->output_count = accepted + 2;
        verification->mismatch_index = accepted + 1;
        if (prefix_restore_snapshots(attention, snapshots,
                                     config->num_hidden_layers) ||
            prefix_commit_attention(
                attention, index, config, layer_inputs, FIRST_BATCH,
                prefix_position + 1, commit, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(
                combined_main_x, commit + 1, d)) result = -1;
        goto cleanup;
    }

    prefix_destroy_snapshots(snapshots, config->num_hidden_layers);
    snapshots = NULL;
    free(layer_inputs); free(next); free(state);
    layer_inputs = NULL; next = NULL; state = NULL;

    state = malloc((size_t)TAIL_BATCH * hd * sizeof(float));
    next = malloc((size_t)TAIL_BATCH * hd * sizeof(float));
    layer_inputs = malloc((size_t)config->num_hidden_layers *
                          TAIL_BATCH * hd * sizeof(float));
    if (!state || !next || !layer_inputs ||
        prefix_create_snapshots(&snapshots, attention,
                                config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    int tail_inputs[TAIL_BATCH] = {draft_tokens[2], draft_tokens[3]};
    int tail_targets[TAIL_BATCH];
    float tail_logits[TAIL_BATCH];
    result = coli_v4_target_load_embeddings(
                 state, index, config, tail_inputs, TAIL_BATCH) ||
             prefix_run_target_batch(
                 &state, &next, attention, index, config, experts,
                 tail_inputs, prefix_position + 3, TAIL_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, TAIL_BATCH,
                 tail_targets, tail_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(
                 combined_main_x + (size_t)3 * d, TAIL_BATCH, config);
    if (result) goto cleanup;

    output_tokens[1] = draft_tokens[1];
    output_tokens[2] = draft_tokens[2];
    if (draft_tokens[3] == tail_targets[0]) {
        output_tokens[3] = draft_tokens[3];
        output_tokens[4] = tail_targets[1];
        verification->accepted_draft_tokens = 4;
        verification->output_count = 5;
        verification->mismatch_index = -1;
        result = coli_v4_dspark_capture_stage_main_x(
            combined_main_x, TOTAL_ROWS, d);
    } else {
        output_tokens[3] = tail_targets[0];
        verification->accepted_draft_tokens = 3;
        verification->output_count = 4;
        verification->mismatch_index = 3;
        if (prefix_restore_snapshots(attention, snapshots,
                                     config->num_hidden_layers) ||
            prefix_commit_attention(
                attention, index, config, layer_inputs, TAIL_BATCH,
                prefix_position + 3, 1, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(
                combined_main_x, 4, d)) result = -1;
    }

cleanup:
    prefix_destroy_snapshots(snapshots, config->num_hidden_layers);
    free(layer_inputs); free(next); free(state); free(combined_main_x);
    return result ? -1 : 0;
}
