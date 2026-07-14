#ifndef COLIBRI_MODEL_H
#define COLIBRI_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_MODEL_ABI_VERSION 1u

typedef struct ColiModel ColiModel;
typedef struct ColiModelContext ColiModelContext;

typedef struct {
    const char *model_dir;
    ColiExpertStore *experts;
    uint64_t ram_budget_bytes;
    int max_context;
} ColiModelLoadOptions;

typedef struct {
    uint32_t abi_version;
    const char *architecture;
    int (*probe)(const char *model_dir);
    int (*load)(const ColiModelLoadOptions *options, ColiModel **model);
    void (*destroy)(ColiModel *model);
    int (*vocab_size)(const ColiModel *model);
    int (*max_context)(const ColiModel *model);
    int (*context_create)(ColiModel *model, int max_context,
                          ColiModelContext **context);
    void (*context_reset)(ColiModel *model, ColiModelContext *context);
    void (*context_destroy)(ColiModel *model, ColiModelContext *context);
    /* Forward writes vocab_size() floats to logits and returns zero on success. */
    int (*forward)(ColiModel *model, ColiModelContext *context,
                   const int *tokens, int token_count, int position,
                   float *logits, size_t logits_count);
} ColiModelOps;

#ifdef __cplusplus
}
#endif

#endif
