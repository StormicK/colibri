#include "../deepseek_v4_resource_plan.h"

#include <stdint.h>
#include <stdio.h>

#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

static ColiDeepSeekV4ResourceInputs fixture(uint64_t available) {
    ColiDeepSeekV4ResourceInputs input = {
        available, 0, 160 * MIB, 600 * MIB, 13369344,
        43, 6, 256,
    };
    return input;
}

int main(void) {
    char error[256];
    ColiDeepSeekV4ResourcePlan low, high, capped, auto_24, capped_24;
    ColiDeepSeekV4ResourceInputs input = fixture(8 * GIB);
    if (coli_v4_resource_plan_compute(&low, &input, error, sizeof(error)) ||
        low.slots_per_layer < 6 || low.projected_bytes > 8 * GIB)
        return 1;
    input = fixture(64 * GIB);
    if (coli_v4_resource_plan_compute(&high, &input, error, sizeof(error)) ||
        high.slots_per_layer <= low.slots_per_layer ||
        high.projected_bytes > 64 * GIB)
        return 1;
    input = fixture(64 * GIB);
    input.user_limit_bytes = 8 * GIB;
    if (coli_v4_resource_plan_compute(&capped, &input, error, sizeof(error)) ||
        capped.planner_available_bytes != 8 * GIB ||
        capped.system_reserve_bytes != 0 ||
        capped.slots_per_layer < low.slots_per_layer ||
        capped.projected_bytes > 8 * GIB)
        return 1;
    input = fixture(24 * GIB);
    if (coli_v4_resource_plan_compute(&auto_24, &input, error, sizeof(error)))
        return 1;
    input = fixture(64 * GIB);
    input.user_limit_bytes = 24 * GIB;
    if (coli_v4_resource_plan_compute(
            &capped_24, &input, error, sizeof(error)) ||
        capped_24.system_reserve_bytes != 0 ||
        capped_24.slots_per_layer <= auto_24.slots_per_layer ||
        capped_24.projected_bytes > 24 * GIB)
        return 1;
    input = fixture(4 * GIB);
    if (coli_v4_resource_plan_compute(&capped, &input, error, sizeof(error)) == 0)
        return 1;
    puts("DeepSeek V4 resource plan tests: ok");
    return 0;
}
