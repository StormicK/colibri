#ifndef COLIBRI_EXPERT_STORE_H
#define COLIBRI_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiExpertStore ColiExpertStore;

typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

typedef struct {
    ColiExpertKey key;
    ColiTensorView gate;
    ColiTensorView down;
    ColiTensorView up;
    void *lease;
} ColiExpertView;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t bytes_read;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;
} ColiExpertStoreStats;

typedef struct {
    /* Returns zero on success. The view remains valid until release(). */
    int (*lookup)(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view);
    void (*release)(ColiExpertStore *store, ColiExpertView *view);
    /* Prefetch is advisory. Unsupported or rejected requests return zero. */
    int (*prefetch)(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count);
    void (*stats)(const ColiExpertStore *store, ColiExpertStoreStats *stats);
    void (*destroy)(ColiExpertStore *store);
} ColiExpertStoreOps;

struct ColiExpertStore {
    const ColiExpertStoreOps *ops;
    void *state;
};

static inline int coli_expert_lookup(ColiExpertStore *store,
                                     ColiExpertKey key,
                                     ColiExpertView *view) {
    return store && store->ops && store->ops->lookup
        ? store->ops->lookup(store, key, view) : -1;
}

static inline void coli_expert_release(ColiExpertStore *store,
                                       ColiExpertView *view) {
    if (store && store->ops && store->ops->release)
        store->ops->release(store, view);
}

#ifdef __cplusplus
}
#endif

#endif
