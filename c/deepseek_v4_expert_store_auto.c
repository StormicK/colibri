#include "deepseek_v4_dspark.h"
/* ---- begin inlined deepseek_v4_expert_store_auto_v5.c ---- */
#define __wrap_coli_deepseek_v4_expert_store_open \
    coli_v4_expert_store_auto_v1_unused
/* ---- begin inlined deepseek_v4_expert_store_auto.c ---- */
#include "deepseek_v4_resource_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_expert_store.h"
#include "deepseek_v4_layer.h"
#include "deepseek_v4_runtime.h"
#include "safetensors_index.h"

#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

int __real_coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *, ColiExpertStore **,
    char *, size_t);


static uint64_t expert_record_bytes(const ColiSafetensorsIndex *index) {
    static const char *parts[] = {
        "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
        "layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w2.scale",
        "layers.0.ffn.experts.0.w3.weight", "layers.0.ffn.experts.0.w3.scale",
    };
    uint64_t total = 0;
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        const ColiSafetensorsTensor *tensor = coli_st_find(index, parts[i]);
        if (!tensor || UINT64_MAX - total < tensor->nbytes) return 0;
        total += tensor->nbytes;
    }
    return total;
}

static uint64_t context_bytes(const ColiDeepSeekV4Config *config, int context) {
    uint64_t total = (uint64_t)config->num_hidden_layers *
        config->sliding_window * config->head_dim * sizeof(float);
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        int ratio = config->compress_ratios[layer];
        if (!ratio) continue;
        uint64_t compressed = ((uint64_t)context + (uint64_t)ratio - 1) /
                              (uint64_t)ratio;
        total += compressed * config->head_dim * sizeof(float);
        if (ratio == 4)
            total += compressed * config->index_head_dim * sizeof(float);
    }
    return total;
}

static int build_runtime_plan(const ColiDeepSeekV4ExpertStoreOptions *options,
                              ColiDeepSeekV4ResourcePlan *plan,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    if (coli_v4_config_load(&config, options->model_dir, error, error_size) ||
        coli_st_index_open(&index, options->model_dir, error, error_size))
        return -1;
    uint64_t maximum_layer = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        ColiDeepSeekV4LayerPlan layer_plan;
        ColiDeepSeekV4LayerStats stats;
        if (coli_v4_layer_plan(&layer_plan, &config, layer,
                               error, error_size) ||
            coli_v4_layer_validate(&layer_plan, index, &stats,
                                   error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        if (stats.total_bytes > maximum_layer) maximum_layer = stats.total_bytes;
    }
    uint64_t record = expert_record_bytes(index);
    coli_st_index_close(index);
    if (!record) {
        snprintf(error, error_size, "cannot determine V4 expert record size");
        return -1;
    }
    ColiDeepSeekV4RuntimeOptions *runtime = coli_v4_runtime_options();
    int context = runtime->context_tokens;
    if (context > config.max_position_embeddings)
        context = config.max_position_embeddings;
    uint64_t hidden = (uint64_t)64 * config.hc_mult * config.hidden_size *
                      sizeof(float) * 2;
    uint64_t scratch = 512 * MIB;
    uint64_t runtime_other = context_bytes(&config, context) + hidden + scratch;
    uint64_t available = coli_v4_os_available_memory();
    if (!available) {
        snprintf(error, error_size, "cannot determine OS available memory");
        return -1;
    }
    ColiDeepSeekV4ResourceInputs inputs = {
        available, runtime->memory_limit_bytes, maximum_layer,
        runtime_other, record, config.num_hidden_layers,
        config.num_experts_per_tok, config.n_routed_experts,
    };
    return coli_v4_resource_plan_compute(plan, &inputs, error, error_size);
}

int __wrap_coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!options) return -1;
    ColiDeepSeekV4ResourcePlan plan;
    if (build_runtime_plan(options, &plan, error, error_size)) return -1;

    ColiDeepSeekV4ExpertStoreOptions automatic = *options;
    automatic.cache_bytes = plan.expert_cache_bytes;
    fprintf(stderr,
        "ram_plan available=%.2fGiB reserve=%.2fGiB runtime=%.2fGiB "
        "expert_min=%.2fGiB expert_cache=%.2fGiB slots_per_layer=%d "
        "head=streamed-bf16 projected=%.2fGiB\n",
        plan.os_available_bytes / (double)GIB,
        plan.system_reserve_bytes / (double)GIB,
        plan.runtime_reserve_bytes / (double)GIB,
        plan.minimum_expert_bytes / (double)GIB,
        automatic.cache_bytes / (double)GIB,
        (int)(automatic.cache_bytes /
              (plan.minimum_expert_bytes / plan.slots_per_layer)),
        (plan.system_reserve_bytes + plan.runtime_reserve_bytes +
         automatic.cache_bytes) / (double)GIB);
    return __real_coli_deepseek_v4_expert_store_open(
        &automatic, output, error, error_size);
}
/* ---- end inlined deepseek_v4_expert_store_auto.c ---- */

#undef __wrap_coli_deepseek_v4_expert_store_open

#include <time.h>

#include "deepseek_v4_dspark_memory.h"
#include "deepseek_v4_head_cache.h"


static int v5_dense_inventory(const char *model_dir,
                              uint64_t *bytes,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config; ColiSafetensorsIndex *index = NULL;
    if (coli_v4_config_load(&config, model_dir, error, error_size) ||
        coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    uint64_t total = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        ColiDeepSeekV4LayerPlan layer_plan; ColiDeepSeekV4LayerStats stats;
        if (coli_v4_layer_plan(&layer_plan, &config, layer,
                               error, error_size) ||
            coli_v4_layer_validate(&layer_plan, index, &stats,
                                   error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        total += stats.total_bytes;
    }
    coli_st_index_close(index); *bytes = total; return 0;
}

int __wrap_coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!options) return -1;
    ColiDeepSeekV4ResourcePlan plan;
    ColiDeepSeekV4RuntimeOptions *runtime = coli_v4_runtime_options();
    if (build_runtime_plan(options, &plan, error, error_size)) return -1;
    uint64_t per_slot = plan.expert_cache_bytes /
                        (uint64_t)plan.slots_per_layer;
    uint64_t head_bytes = 0, dense_bytes = 0;
    if (coli_v4_head_cache_probe(options->model_dir, &head_bytes,
                                 error, error_size) ||
        v5_dense_inventory(options->model_dir, &dense_bytes,
                           error, error_size)) return -1;

    uint64_t dspark_bytes = 0, dspark_full_bytes = 0;
    uint64_t dspark_full_experts = 0;
    const char *dspark_model = runtime->dspark_model_dir;
    ColiV4DSparkMemoryPlan dspark = {0};
    if (dspark_model && *dspark_model) {
        ColiDeepSeekV4Config ds_config;
        ColiDeepSeekV4DSparkManifest manifest;
        if (coli_v4_config_load(&ds_config, dspark_model, error, error_size) ||
            coli_v4_dspark_memory_plan(dspark_model, &ds_config, &dspark,
                                       error, error_size) ||
            coli_v4_dspark_inspect(dspark_model, &ds_config, &manifest,
                                   error, error_size)) return -1;
        uint64_t all_stages = 0;
        for (int stage = 0; stage < manifest.stage_count; stage++)
            all_stages += manifest.common_stage_bytes[stage];
        dspark_full_experts = dspark.expert_record_bytes *
            (uint64_t)manifest.stage_count * ds_config.n_routed_experts;
        dspark_full_bytes = dspark.resident_heads_bytes + all_stages +
            dspark_full_experts + dspark.working_bytes;
        dspark_bytes = dspark.incremental_reserve_bytes;
    }

    uint64_t fixed = plan.system_reserve_bytes + plan.runtime_reserve_bytes;
    uint64_t full_required = fixed + head_bytes + dense_bytes +
        dspark_full_bytes + plan.minimum_expert_bytes;
    int full_resident = dspark_full_bytes &&
        full_required <= plan.planner_available_bytes;
    if (full_resident) {
        dspark_bytes = dspark_full_bytes;
        runtime->dense_resident = 1;
        runtime->dspark_resident = 1;
        runtime->dspark_expert_cache_bytes = dspark_full_experts;
    } else {
        dense_bytes = 0;
        runtime->dense_resident = 0;
        runtime->dspark_resident = 0;
        runtime->dspark_expert_cache_bytes = 0;
    }
    if (dspark_bytes + dense_bytes > plan.planner_available_bytes - fixed) {
        snprintf(error, error_size, "resident V4 tiers exceed available RAM");
        return -1;
    }
    uint64_t safe_payload = plan.planner_available_bytes - fixed -
                            dspark_bytes - dense_bytes;
    int requested_head = -1;
    int resident_head = safe_payload >= plan.minimum_expert_bytes +
                                      head_bytes + 256 * MIB;
    if (requested_head == 0) resident_head = 0;
    if (requested_head == 1 && !resident_head) {
        snprintf(error, error_size, "resident BF16 head does not fit RAM plan");
        return -1;
    }
    uint64_t cache_limit = safe_payload - (resident_head ? head_bytes : 0);

    if (cache_limit < plan.minimum_expert_bytes) {
        snprintf(error, error_size, "resident tiers leave too little target cache");
        return -1;
    }
    int slots = (int)(cache_limit / per_slot);
    if (slots > plan.slots_per_layer) slots = plan.slots_per_layer;
    if (slots < options->experts_per_layer && slots < 6) slots = 6;
    plan.expert_cache_bytes = (uint64_t)slots * per_slot;
    plan.projected_bytes = fixed + dense_bytes + dspark_bytes +
        plan.expert_cache_bytes + (resident_head ? head_bytes : 0);
    if (resident_head && coli_v4_head_cache_load(
            options->model_dir, error, error_size)) return -1;
    fprintf(stderr,
        "ram_tiers available=%.2fGiB dense=%s(%.2fGiB) "
        "dspark=%s(%.2fGiB) dspark_experts=%.2fGiB "
        "target_slots=%d target_cache=%.2fGiB head=%s projected=%.2fGiB\n",
        plan.planner_available_bytes / (double)GIB,
        full_resident ? "resident" : "streamed", dense_bytes / (double)GIB,
        full_resident ? "resident" : "streamed", dspark_bytes / (double)GIB,
        dspark_full_experts / (double)GIB, slots,
        plan.expert_cache_bytes / (double)GIB,
        resident_head ? "resident-bf16" : "streamed-bf16",
        plan.projected_bytes / (double)GIB);
    ColiDeepSeekV4ExpertStoreOptions automatic = *options;
    automatic.cache_bytes = plan.expert_cache_bytes;
    return __real_coli_deepseek_v4_expert_store_open(
        &automatic, output, error, error_size);
}
/* ---- end inlined deepseek_v4_expert_store_auto_v5.c ---- */

