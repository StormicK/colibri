#ifndef COLIBRI_DEEPSEEK_V4_INDEXER_SNAPSHOT_H
#define COLIBRI_DEEPSEEK_V4_INDEXER_SNAPSHOT_H

#include "deepseek_v4_indexer.h"

typedef struct ColiV4IndexerSnapshot ColiV4IndexerSnapshot;

int coli_v4_indexer_snapshot_create(const ColiDeepSeekV4Indexer *state,
                                    ColiV4IndexerSnapshot **output);
int coli_v4_indexer_snapshot_restore(ColiDeepSeekV4Indexer *state,
                                     const ColiV4IndexerSnapshot *snapshot);
void coli_v4_indexer_snapshot_destroy(ColiV4IndexerSnapshot *snapshot);

#endif
