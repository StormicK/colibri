#ifndef COLIBRI_DEEPSEEK_V4_INDEXER_H
#define COLIBRI_DEEPSEEK_V4_INDEXER_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

typedef struct ColiDeepSeekV4Indexer ColiDeepSeekV4Indexer;

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **state,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size);
int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size);
void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state);
void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state);

/* Updates the overlap compressor, then returns compressed-cache ordinals in
 * descending index score order. query_rank is the normalized q_lora vector. */
int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size);
const float *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state);
int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state);

#endif
