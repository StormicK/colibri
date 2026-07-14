#define coli_v4_compressor_create snapshot_copy_compressor_create
#define coli_v4_compressor_create_with_options snapshot_copy_compressor_create_with_options
#define coli_v4_compressor_reset snapshot_copy_compressor_reset
#define coli_v4_compressor_bind_weights snapshot_copy_compressor_bind_weights
#define coli_v4_compressor_destroy snapshot_copy_compressor_destroy
#define coli_v4_compressor_step snapshot_copy_compressor_step
#include "deepseek_v4_compressor.c"
#undef coli_v4_compressor_step
#undef coli_v4_compressor_destroy
#undef coli_v4_compressor_bind_weights
#undef coli_v4_compressor_reset
#undef coli_v4_compressor_create_with_options
#undef coli_v4_compressor_create

#include "deepseek_v4_compressor_snapshot.h"

struct ColiV4CompressorSnapshot {
    size_t count;
    float *kv_state;
    float *score_state;
};

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output) {
    if (!state || !output) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->count = (size_t)state->state_rows * state->projection_dim;
    (*output)->kv_state = malloc((*output)->count * sizeof(float));
    (*output)->score_state = malloc((*output)->count * sizeof(float));
    if (!(*output)->kv_state || !(*output)->score_state) {
        coli_v4_compressor_snapshot_destroy(*output); *output = NULL; return -1;
    }
    memcpy((*output)->kv_state, state->kv_state,
           (*output)->count * sizeof(float));
    memcpy((*output)->score_state, state->score_state,
           (*output)->count * sizeof(float));
    return 0;
}

int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot) {
    if (!state || !snapshot || snapshot->count !=
        (size_t)state->state_rows * state->projection_dim) return -1;
    memcpy(state->kv_state, snapshot->kv_state, snapshot->count * sizeof(float));
    memcpy(state->score_state, snapshot->score_state,
           snapshot->count * sizeof(float));
    return 0;
}

void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->score_state); free(snapshot->kv_state); free(snapshot);
}
