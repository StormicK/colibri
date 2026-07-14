#define coli_v4_target_verify_greedy_batch \
    coli_v4_target_verify_greedy_batch_v2_fallback
/* ---- begin inlined deepseek_v4_target_verify_v2.c ---- */
#include "deepseek_v4_target_verify.h"

#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_attention_transaction.h"
#include "deepseek_v4_block_batch.h"
#include "deepseek_v4_layer.h"
#include "deepseek_v4_target_attention_commit.h"
#include "deepseek_v4_target_head_batch.h"

static int run_target_block_record(
    float **state_ptr, float **next_ptr,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *tokens, int start, int batch, float *layer_inputs,
    char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        if (layer_inputs)
            memcpy(layer_inputs + (size_t)layer_id * batch * hd, state,
                   (size_t)batch * hd * sizeof(float));
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
    *state_ptr = state; *next_ptr = next; return 0;
}

static int commit_target_attention(
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

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size) {
    if (!verification || !output_tokens || !attention || !index || !config ||
        !experts || !draft_tokens || draft_count < 1 || draft_count > 63 ||
        output_capacity < draft_count + 1) return -1;
    int batch = draft_count + 1;
    int *inputs = malloc((size_t)batch * sizeof(*inputs));
    int *targets = malloc((size_t)batch * sizeof(*targets));
    float *logits = malloc((size_t)batch * sizeof(*logits));
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    float *state = malloc((size_t)batch * hd * sizeof(*state));
    float *next = malloc((size_t)batch * hd * sizeof(*next));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 batch * hd * sizeof(*layer_inputs));
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)config->num_hidden_layers, sizeof(*snapshots));
    if (!inputs || !targets || !logits || !state || !next || !layer_inputs ||
        !snapshots) return -1;
    inputs[0] = anchor_token;
    for (int i = 0; i < draft_count; i++) inputs[i + 1] = draft_tokens[i];
    for (int layer = 0; layer < config->num_hidden_layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) return -1;
    if (coli_v4_target_load_embeddings(state, index, config, inputs, batch) ||
        run_target_block_record(&state, &next, attention, index, config,
                                experts, inputs, start_position, batch,
                                layer_inputs, error, error_size) ||
        coli_v4_target_head_argmax_batch(state, index, config, batch,
                                         targets, logits, error, error_size) ||
        coli_v4_verify_greedy(verification, output_tokens, output_capacity,
                              draft_tokens, draft_count, targets, batch))
        return -1;
    if (verification->accepted_draft_tokens < draft_count) {
        for (int layer = 0; layer < config->num_hidden_layers; layer++)
            if (coli_v4_attention_snapshot_restore(attention[layer],
                                                   snapshots[layer])) return -1;
        int commit = verification->accepted_draft_tokens + 1;
        if (commit_target_attention(attention, index, config, layer_inputs,
                                    batch, start_position, commit,
                                    error, error_size)) return -1;
    }
    for (int layer = 0; layer < config->num_hidden_layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots); free(layer_inputs); free(next); free(state);
    free(logits); free(targets); free(inputs); return 0;
}
/* ---- end inlined deepseek_v4_target_verify_v2.c ---- */

#undef coli_v4_target_verify_greedy_batch

#include "deepseek_v4_dspark_capture.h"


static void destroy_snapshots(ColiV4AttentionSnapshot **snapshots, int layers) {
    if (!snapshots) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots);
}

static int create_snapshots(
    ColiV4AttentionSnapshot ***output,
    ColiDeepSeekV4WindowAttentionState **attention, int layers) {
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)layers, sizeof(*snapshots));
    if (!snapshots) return -1;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) {
            destroy_snapshots(snapshots, layers); return -1;
        }
    *output = snapshots; return 0;
}

static int restore_snapshots(
    ColiDeepSeekV4WindowAttentionState **attention,
    ColiV4AttentionSnapshot **snapshots, int layers) {
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_restore(attention[layer],
                                               snapshots[layer])) return -1;
    return 0;
}

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size) {
    if (draft_count != 4)
        return coli_v4_target_verify_greedy_batch_v2_fallback(
            verification, output_tokens, output_capacity, attention, index,
            config, experts, anchor_token, draft_tokens, draft_count,
            start_position, error, error_size);
    if (!verification || !output_tokens || output_capacity < 5 || !attention ||
        !index || !config || !experts || !draft_tokens) return -1;

    enum { FIRST_BATCH = 3, TAIL_BATCH = 2 };
    int first_inputs[FIRST_BATCH] = {
        anchor_token, draft_tokens[0], draft_tokens[1]
    };
    int first_targets[FIRST_BATCH]; float first_logits[FIRST_BATCH];
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)d * hc;
    size_t layer_stride_first = (size_t)FIRST_BATCH * hd;
    float *state = malloc((size_t)FIRST_BATCH * hd * sizeof(*state));
    float *next = malloc((size_t)FIRST_BATCH * hd * sizeof(*next));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 layer_stride_first * sizeof(*layer_inputs));
    float *combined_main_x = malloc((FIRST_BATCH + TAIL_BATCH) *
                                    (size_t)d * sizeof(*combined_main_x));
    ColiV4AttentionSnapshot **snapshots = NULL;
    if (!state || !next || !layer_inputs || !combined_main_x ||
        create_snapshots(&snapshots, attention, config->num_hidden_layers))
        return -1;
    int result = coli_v4_target_load_embeddings(
                     state, index, config, first_inputs, FIRST_BATCH) ||
                 run_target_block_record(
                     &state, &next, attention, index, config, experts,
                     first_inputs, start_position, FIRST_BATCH, layer_inputs,
                     error, error_size) ||
                 coli_v4_target_head_argmax_batch(
                     state, index, config, FIRST_BATCH,
                     first_targets, first_logits, error, error_size) ||
                 coli_v4_dspark_capture_main_x(
                     combined_main_x, FIRST_BATCH, config);
    if (result) goto cleanup;
    int accepted = 0;
    while (accepted < FIRST_BATCH &&
           draft_tokens[accepted] == first_targets[accepted]) accepted++;
    if (accepted < FIRST_BATCH) {
        for (int i = 0; i < accepted; i++) output_tokens[i] = draft_tokens[i];
        output_tokens[accepted] = first_targets[accepted];
        verification->accepted_draft_tokens = accepted;
        verification->output_count = accepted + 1;
        verification->mismatch_index = accepted;
        int commit = accepted + 1;
        if (commit < FIRST_BATCH &&
            (restore_snapshots(attention, snapshots,
                               config->num_hidden_layers) ||
             commit_target_attention(
                 attention, index, config, layer_inputs, FIRST_BATCH,
                 start_position, commit, error, error_size))) result = -1;
        goto cleanup;
    }
    destroy_snapshots(snapshots, config->num_hidden_layers); snapshots = NULL;
    free(layer_inputs); layer_inputs = NULL;
    free(next); free(state); next = NULL; state = NULL;

    int tail_inputs[TAIL_BATCH] = {draft_tokens[2], draft_tokens[3]};
    int tail_targets[TAIL_BATCH]; float tail_logits[TAIL_BATCH];
    state = malloc((size_t)TAIL_BATCH * hd * sizeof(*state));
    next = malloc((size_t)TAIL_BATCH * hd * sizeof(*next));
    layer_inputs = malloc((size_t)config->num_hidden_layers * TAIL_BATCH *
                          hd * sizeof(*layer_inputs));
    if (!state || !next || !layer_inputs ||
        create_snapshots(&snapshots, attention, config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    result = coli_v4_target_load_embeddings(
                 state, index, config, tail_inputs, TAIL_BATCH) ||
             run_target_block_record(
                 &state, &next, attention, index, config, experts,
                 tail_inputs, start_position + FIRST_BATCH, TAIL_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, TAIL_BATCH,
                 tail_targets, tail_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(
                 combined_main_x + (size_t)FIRST_BATCH * d,
                 TAIL_BATCH, config);
    if (result) goto cleanup;
    for (int i = 0; i < FIRST_BATCH; i++) output_tokens[i] = draft_tokens[i];
    if (draft_tokens[3] == tail_targets[0]) {
        output_tokens[3] = draft_tokens[3];
        output_tokens[4] = tail_targets[1];
        verification->accepted_draft_tokens = 4;
        verification->output_count = 5;
        verification->mismatch_index = -1;
        result = coli_v4_dspark_capture_stage_main_x(
            combined_main_x, 5, d);
    } else {
        output_tokens[3] = tail_targets[0];
        verification->accepted_draft_tokens = 3;
        verification->output_count = 4;
        verification->mismatch_index = 3;
        if (restore_snapshots(attention, snapshots,
                              config->num_hidden_layers) ||
            commit_target_attention(
                attention, index, config, layer_inputs, TAIL_BATCH,
                start_position + FIRST_BATCH, 1, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(combined_main_x, 4, d))
            result = -1;
    }

cleanup:
    destroy_snapshots(snapshots, config->num_hidden_layers);
    free(combined_main_x); free(layer_inputs); free(next); free(state);
    return result;
}
