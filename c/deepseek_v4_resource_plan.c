#include "deepseek_v4_resource_plan.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MIB UINT64_C(1048576)

static int plan_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *output) {
    if (UINT64_MAX - a < b) return -1;
    *output = a + b;
    return 0;
}

static int multiply_u64(uint64_t a, uint64_t b, uint64_t *output) {
    if (a && b > UINT64_MAX / a) return -1;
    *output = a * b;
    return 0;
}

uint64_t coli_v4_os_available_memory(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t)status.ullAvailPhys : 0;
#else
    FILE *stream = fopen("/proc/meminfo", "r");
    if (stream) {
        char line[256];
        unsigned long long kib = 0;
        while (fgets(line, sizeof(line), stream))
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
        fclose(stream);
        if (kib) return (uint64_t)kib * 1024;
    }
    long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

int coli_v4_resource_plan_compute(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size) {
    if (!plan || !inputs || !inputs->available_bytes ||
        !inputs->maximum_layer_bytes || !inputs->expert_record_bytes ||
        inputs->sparse_layers < 1 || inputs->routed_topk < 1 ||
        inputs->experts_per_layer < inputs->routed_topk)
        return plan_error(error, error_size, "invalid V4 resource-plan inputs");
    memset(plan, 0, sizeof(*plan));
    plan->os_available_bytes = inputs->available_bytes;
    uint64_t available = inputs->available_bytes;
    int explicit_process_limit = inputs->user_limit_bytes &&
        inputs->user_limit_bytes < available;
    if (explicit_process_limit)
        available = inputs->user_limit_bytes;
    plan->planner_available_bytes = available;

    /* A process limit leaves all RAM outside the limit to the OS. Automatic
     * mode starts from MemAvailable and therefore reserves that share here. */
    uint64_t system = explicit_process_limit ? 0 : available / 8;
    if (!explicit_process_limit && system < 512 * MIB) system = 512 * MIB;
    if (system > 4096 * MIB) system = 4096 * MIB;
    plan->system_reserve_bytes = system;

    uint64_t layers_twice;
    if (multiply_u64(inputs->maximum_layer_bytes, 2, &layers_twice) ||
        add_u64(layers_twice, inputs->runtime_other_bytes,
                &plan->runtime_reserve_bytes))
        return plan_error(error, error_size, "V4 runtime reserve overflow");

    uint64_t per_slot;
    if (multiply_u64((uint64_t)inputs->sparse_layers,
                     inputs->expert_record_bytes, &per_slot) ||
        multiply_u64(per_slot, (uint64_t)inputs->routed_topk,
                     &plan->minimum_expert_bytes))
        return plan_error(error, error_size, "V4 expert-cache size overflow");

    uint64_t fixed;
    if (add_u64(system, plan->runtime_reserve_bytes, &fixed) || fixed >= available)
        return plan_error(error, error_size,
                          "available RAM cannot hold V4 runtime reserves");
    uint64_t usable = available - fixed;
    if (usable < plan->minimum_expert_bytes)
        return plan_error(error, error_size,
            "V4 minimum expert cache needs %.2f GiB but only %.2f GiB remains",
            plan->minimum_expert_bytes / 1073741824.0,
            usable / 1073741824.0);

    uint64_t slots = usable / per_slot;
    if (slots > (uint64_t)inputs->experts_per_layer)
        slots = (uint64_t)inputs->experts_per_layer;
    if (slots < (uint64_t)inputs->routed_topk)
        slots = (uint64_t)inputs->routed_topk;
    plan->slots_per_layer = (int)slots;
    if (multiply_u64(per_slot, slots, &plan->expert_cache_bytes) ||
        add_u64(fixed, plan->expert_cache_bytes, &plan->projected_bytes))
        return plan_error(error, error_size, "V4 projected memory overflow");
    if (plan->projected_bytes > available)
        return plan_error(error, error_size, "V4 plan exceeds available RAM");
    return 0;
}
