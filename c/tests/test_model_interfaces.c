#include "../model.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    int held;
    ColiExpertStoreStats stats;
} MockState;

static int mock_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    static const unsigned char weights[4] = {1, 2, 3, 4};
    MockState *state = (MockState *)store->state;
    if (state->held || !view) return -1;
    memset(view, 0, sizeof(*view));
    view->key = key;
    view->gate.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->gate.scale_format = COLI_SCALE_UE8M0;
    view->gate.data = weights;
    view->gate.data_bytes = sizeof(weights);
    view->lease = state;
    state->held = 1;
    state->stats.requests++;
    state->stats.misses++;
    state->stats.bytes_read += sizeof(weights);
    return 0;
}

static void mock_release(ColiExpertStore *store, ColiExpertView *view) {
    MockState *state = (MockState *)store->state;
    if (view && view->lease == state) {
        state->held = 0;
        view->lease = NULL;
    }
}

static int mock_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                         size_t count) {
    MockState *state = (MockState *)store->state;
    (void)keys;
    state->stats.prefetched += count;
    return (int)count;
}

static void mock_stats(const ColiExpertStore *store,
                       ColiExpertStoreStats *stats) {
    *stats = ((const MockState *)store->state)->stats;
}

static void mock_destroy(ColiExpertStore *store) { (void)store; }

int main(void) {
    static const ColiExpertStoreOps ops = {
        mock_lookup, mock_release, mock_prefetch, mock_stats, mock_destroy
    };
    MockState state = {0};
    ColiExpertStore store = {&ops, &state};
    ColiExpertView view;
    ColiExpertKey key = {7, 19};
    ColiExpertStoreStats stats;

    if (coli_expert_lookup(&store, key, &view) != 0) return 1;
    if (view.key.layer != 7 || view.key.expert != 19) return 1;
    if (view.gate.format != COLI_TENSOR_FP4_NATIVE_BLOCK) return 1;
    if (view.gate.scale_format != COLI_SCALE_UE8M0) return 1;
    if (coli_expert_lookup(&store, key, &view) == 0) return 1;
    coli_expert_release(&store, &view);
    if (state.held) return 1;
    if (store.ops->prefetch(&store, &key, 1) != 1) return 1;
    store.ops->stats(&store, &stats);
    if (stats.requests != 1 || stats.misses != 1 ||
        stats.prefetched != 1 || stats.bytes_read != 4) return 1;
    if (COLI_MODEL_ABI_VERSION != 1u) return 1;
    puts("model interface tests: ok");
    return 0;
}
