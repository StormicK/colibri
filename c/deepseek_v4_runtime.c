#include "deepseek_v4_runtime.h"

#include <string.h>

static ColiDeepSeekV4RuntimeOptions runtime_options;

void coli_v4_runtime_reset(void) {
    memset(&runtime_options, 0, sizeof(runtime_options));
    runtime_options.context_tokens = 4096;
    runtime_options.pin_slots_per_layer = -1;
}

ColiDeepSeekV4RuntimeOptions *coli_v4_runtime_options(void) {
    if (!runtime_options.context_tokens) coli_v4_runtime_reset();
    return &runtime_options;
}
