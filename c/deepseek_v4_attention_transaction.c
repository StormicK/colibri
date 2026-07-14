#define coli_v4_window_attention_create transaction_copy_attention_create
#define coli_v4_window_attention_reset transaction_copy_attention_reset
#define coli_v4_window_attention_destroy transaction_copy_attention_destroy
#define coli_v4_attention_token_ref transaction_copy_attention_token
#define coli_v4_attention_window_token_ref transaction_copy_attention_window_token
#include "deepseek_v4_attention.c"
#undef coli_v4_attention_window_token_ref
#undef coli_v4_attention_token_ref
#undef coli_v4_window_attention_destroy
#undef coli_v4_window_attention_reset
#undef coli_v4_window_attention_create

#include "deepseek_v4_attention_transaction.h"
#include "deepseek_v4_compressor_snapshot.h"
#include "deepseek_v4_indexer_snapshot.h"

struct ColiV4AttentionSnapshot {
    int window_size, head_dim, compressed_count;
    float *kv;
    float *compressed;
    ColiV4CompressorSnapshot *compressor;
    ColiV4IndexerSnapshot *indexer;
};

int coli_v4_attention_snapshot_create(
    const ColiDeepSeekV4WindowAttentionState *state,
    ColiV4AttentionSnapshot **output) {
    if (!state || !output) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = state->window_size;
    (*output)->head_dim = state->head_dim;
    (*output)->compressed_count = state->compressed_count;
    size_t kv_count = (size_t)state->window_size * state->head_dim;
    (*output)->kv = malloc(kv_count * sizeof(float));
    if (!(*output)->kv) goto failed;
    memcpy((*output)->kv, state->kv, kv_count * sizeof(float));
    if (state->compressed_count) {
        size_t count = (size_t)state->compressed_count * state->head_dim;
        (*output)->compressed = malloc(count * sizeof(float));
        if (!(*output)->compressed) goto failed;
        memcpy((*output)->compressed, state->compressed, count * sizeof(float));
    }
    if (state->compressor && coli_v4_compressor_snapshot_create(
            state->compressor, &(*output)->compressor)) goto failed;
    if (state->indexer && coli_v4_indexer_snapshot_create(
            state->indexer, &(*output)->indexer)) goto failed;
    return 0;
failed:
    coli_v4_attention_snapshot_destroy(*output); *output = NULL; return -1;
}

int coli_v4_attention_snapshot_restore(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot) {
    if (!state || !snapshot || state->window_size != snapshot->window_size ||
        state->head_dim != snapshot->head_dim ||
        snapshot->compressed_count > state->compressed_capacity) return -1;
    memcpy(state->kv, snapshot->kv,
           (size_t)state->window_size * state->head_dim * sizeof(float));
    state->compressed_count = snapshot->compressed_count;
    if (snapshot->compressed_count)
        memcpy(state->compressed, snapshot->compressed,
               (size_t)snapshot->compressed_count * state->head_dim * sizeof(float));
    if ((state->compressor != NULL) != (snapshot->compressor != NULL) ||
        (state->indexer != NULL) != (snapshot->indexer != NULL)) return -1;
    if (state->compressor && coli_v4_compressor_snapshot_restore(
            state->compressor, snapshot->compressor)) return -1;
    if (state->indexer && coli_v4_indexer_snapshot_restore(
            state->indexer, snapshot->indexer)) return -1;
    return 0;
}

void coli_v4_attention_snapshot_destroy(ColiV4AttentionSnapshot *snapshot) {
    if (!snapshot) return;
    coli_v4_indexer_snapshot_destroy(snapshot->indexer);
    coli_v4_compressor_snapshot_destroy(snapshot->compressor);
    free(snapshot->compressed); free(snapshot->kv); free(snapshot);
}
