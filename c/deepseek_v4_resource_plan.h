#ifndef COLIBRI_DEEPSEEK_V4_RESOURCE_PLAN_H
#define COLIBRI_DEEPSEEK_V4_RESOURCE_PLAN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t available_bytes;
    uint64_t user_limit_bytes;
    uint64_t maximum_layer_bytes;
    uint64_t runtime_other_bytes;
    uint64_t expert_record_bytes;
    int sparse_layers;
    int routed_topk;
    int experts_per_layer;
} ColiDeepSeekV4ResourceInputs;

typedef struct {
    uint64_t os_available_bytes;
    uint64_t planner_available_bytes;
    uint64_t system_reserve_bytes;
    uint64_t runtime_reserve_bytes;
    uint64_t minimum_expert_bytes;
    uint64_t expert_cache_bytes;
    uint64_t projected_bytes;
    int slots_per_layer;
} ColiDeepSeekV4ResourcePlan;

uint64_t coli_v4_os_available_memory(void);
int coli_v4_resource_plan_compute(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size);

#endif
