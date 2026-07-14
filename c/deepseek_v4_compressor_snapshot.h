#ifndef COLIBRI_DEEPSEEK_V4_COMPRESSOR_SNAPSHOT_H
#define COLIBRI_DEEPSEEK_V4_COMPRESSOR_SNAPSHOT_H

#include "deepseek_v4_compressor.h"

typedef struct ColiV4CompressorSnapshot ColiV4CompressorSnapshot;

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output);
int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot);
void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot);

#endif
