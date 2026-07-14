#define coli_v4_indexer_create snapshot_copy_indexer_create
#define coli_v4_indexer_bind_weights snapshot_copy_indexer_bind_weights
#define coli_v4_indexer_reset snapshot_copy_indexer_reset
#define coli_v4_indexer_destroy snapshot_copy_indexer_destroy
#define coli_v4_indexer_step snapshot_copy_indexer_step
#define coli_v4_indexer_compressed_values snapshot_copy_indexer_values
#define coli_v4_indexer_compressed_count snapshot_copy_indexer_count
#include "deepseek_v4_indexer.c"
#undef coli_v4_indexer_compressed_count
#undef coli_v4_indexer_compressed_values
#undef coli_v4_indexer_step
#undef coli_v4_indexer_destroy
#undef coli_v4_indexer_reset
#undef coli_v4_indexer_bind_weights
#undef coli_v4_indexer_create

#include "deepseek_v4_compressor_snapshot.h"
#include "deepseek_v4_indexer_snapshot.h"

struct ColiV4IndexerSnapshot {
    int count;
    int head_dim;
    float *compressed;
    ColiV4CompressorSnapshot *compressor;
};

int coli_v4_indexer_snapshot_create(const ColiDeepSeekV4Indexer *state,
                                    ColiV4IndexerSnapshot **output) {
    if (!state || !output || !state->config) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->count = state->count;
    (*output)->head_dim = state->config->index_head_dim;
    if (state->count) {
        (*output)->compressed = malloc((size_t)state->count *
                                       (*output)->head_dim * sizeof(float));
        if (!(*output)->compressed) {
            coli_v4_indexer_snapshot_destroy(*output); *output = NULL; return -1;
        }
        memcpy((*output)->compressed, state->compressed,
               (size_t)state->count * (*output)->head_dim * sizeof(float));
    }
    if (coli_v4_compressor_snapshot_create(state->compressor,
                                            &(*output)->compressor)) {
        coli_v4_indexer_snapshot_destroy(*output); *output = NULL; return -1;
    }
    return 0;
}

int coli_v4_indexer_snapshot_restore(ColiDeepSeekV4Indexer *state,
                                     const ColiV4IndexerSnapshot *snapshot) {
    if (!state || !snapshot || !state->config ||
        state->config->index_head_dim != snapshot->head_dim ||
        snapshot->count > state->capacity) return -1;
    state->count = snapshot->count;
    if (snapshot->count)
        memcpy(state->compressed, snapshot->compressed,
               (size_t)snapshot->count * snapshot->head_dim * sizeof(float));
    return coli_v4_compressor_snapshot_restore(state->compressor,
                                                snapshot->compressor);
}

void coli_v4_indexer_snapshot_destroy(ColiV4IndexerSnapshot *snapshot) {
    if (!snapshot) return;
    coli_v4_compressor_snapshot_destroy(snapshot->compressor);
    free(snapshot->compressed); free(snapshot);
}
