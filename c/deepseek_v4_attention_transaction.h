#ifndef COLIBRI_DEEPSEEK_V4_ATTENTION_TRANSACTION_H
#define COLIBRI_DEEPSEEK_V4_ATTENTION_TRANSACTION_H

#include "deepseek_v4_attention.h"

typedef struct ColiV4AttentionSnapshot ColiV4AttentionSnapshot;

int coli_v4_attention_snapshot_create(
    const ColiDeepSeekV4WindowAttentionState *state,
    ColiV4AttentionSnapshot **output);
int coli_v4_attention_snapshot_restore(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot);
void coli_v4_attention_snapshot_destroy(ColiV4AttentionSnapshot *snapshot);

#endif
