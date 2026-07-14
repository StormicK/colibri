#include <string.h>

static char *dspark_second_dot(const char *text, int character) {
    char *first = strchr(text, character);
    return first ? strchr(first + 1, character) : NULL;
}

#define strchr dspark_second_dot
/* ---- begin inlined deepseek_v4_dspark.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "safetensors_index.h"

static int dspark_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static char *read_config(const char *model_dir) {
    size_t length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(length);
    if (!path) return NULL;
    snprintf(path, length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb"); free(path);
    if (!stream) return NULL;
    fseek(stream, 0, SEEK_END); long bytes = ftell(stream); rewind(stream);
    char *text = bytes > 0 ? malloc((size_t)bytes + 1) : NULL;
    if (!text || fread(text, 1, (size_t)bytes, stream) != (size_t)bytes) {
        free(text); fclose(stream); return NULL;
    }
    text[bytes] = 0; fclose(stream); return text;
}

static int json_integer(jval *root, const char *name, int *output) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM) return -1;
    *output = (int)value->num; return 0;
}

int coli_v4_dspark_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                              const ColiDeepSeekV4Config *config, int stage,
                              char *error, size_t error_size) {
    if (!plan || !config || stage < 0 || stage >= COLI_V4_DSPARK_MAX_STAGES)
        return dspark_error(error, error_size, "invalid DSpark layer plan");
    ColiDeepSeekV4Config common = *config;
    common.num_hash_layers = 0;
    common.compress_ratios[0] = 0;
    if (coli_v4_layer_plan(plan, &common, 0, error, error_size)) return -1;
    plan->layer = stage;
    plan->compression_ratio = 0;
    plan->uses_hash_router = 0;
    plan->has_compressor = 0;
    plan->has_indexer = 0;
    for (size_t i = 0; i < plan->tensor_count; i++) {
        const char *suffix = strchr(plan->tensors[i].name, '.');
        if (!suffix) return dspark_error(error, error_size,
                                         "invalid common layer tensor name");
        char rewritten[COLI_V4_MAX_TENSOR_NAME];
        int written = snprintf(rewritten, sizeof(rewritten), "mtp.%d%s",
                               stage, suffix);
        if (written < 0 || (size_t)written >= sizeof(rewritten))
            return dspark_error(error, error_size, "DSpark tensor name too long");
        strcpy(plan->tensors[i].name, rewritten);
    }
    return 0;
}

static int require_tensor(const ColiSafetensorsIndex *index, const char *name,
                          ColiSafetensorsDType dtype, int rank,
                          const int64_t *shape, uint64_t *bytes,
                          char *error, size_t error_size) {
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor) return dspark_error(error, error_size,
                                     "missing DSpark tensor: %s", name);
    if (tensor->dtype != dtype || tensor->rank != rank)
        return dspark_error(error, error_size,
                            "DSpark dtype/rank mismatch: %s", name);
    for (int i = 0; i < rank; i++)
        if (tensor->shape[i] != shape[i])
            return dspark_error(error, error_size,
                                "DSpark shape mismatch: %s", name);
    *bytes += tensor->nbytes; return 0;
}

int coli_v4_dspark_inspect(const char *model_dir,
                           const ColiDeepSeekV4Config *config,
                           ColiDeepSeekV4DSparkManifest *manifest,
                           char *error, size_t error_size) {
    if (!model_dir || !config || !manifest)
        return dspark_error(error, error_size, "invalid DSpark inspect arguments");
    memset(manifest, 0, sizeof(*manifest));
    char *text = read_config(model_dir), *arena = NULL;
    jval *root = text ? json_parse(text, &arena) : NULL;
    if (!root || root->t != J_OBJ ||
        json_integer(root, "dspark_block_size", &manifest->block_size) ||
        json_integer(root, "dspark_noise_token_id", &manifest->noise_token_id) ||
        json_integer(root, "dspark_markov_rank", &manifest->markov_rank)) {
        free(arena); free(text);
        return dspark_error(error, error_size, "missing DSpark config fields");
    }
    jval *targets = json_get(root, "dspark_target_layer_ids");
    if (!targets || targets->t != J_ARR || targets->len < 1 ||
        targets->len > COLI_V4_DSPARK_MAX_TARGETS) {
        free(arena); free(text);
        return dspark_error(error, error_size, "invalid DSpark target layers");
    }
    manifest->target_count = targets->len;
    for (int i = 0; i < targets->len; i++) {
        if (targets->kids[i]->t != J_NUM) {
            free(arena); free(text);
            return dspark_error(error, error_size, "invalid DSpark target layer");
        }
        manifest->target_layer_ids[i] = (int)targets->kids[i]->num;
    }
    free(arena); free(text);

    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    char name[160];
    for (int stage = 0; stage < COLI_V4_DSPARK_MAX_STAGES; stage++) {
        snprintf(name, sizeof(name), "mtp.%d.attn_norm.weight", stage);
        if (!coli_st_find(index, name)) break;
        manifest->stage_count++;
        ColiDeepSeekV4LayerPlan plan;
        ColiDeepSeekV4LayerStats stats;
        if (coli_v4_dspark_layer_plan(&plan, config, stage, error, error_size) ||
            coli_v4_layer_validate(&plan, index, &stats, error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        manifest->common_stage_bytes[stage] = stats.total_bytes;
    }
    if (manifest->stage_count < 1) {
        coli_st_index_close(index);
        return dspark_error(error, error_size, "checkpoint has no DSpark stages");
    }
    int64_t main_weight[] = {config->hidden_size,
        (int64_t)config->hidden_size * manifest->target_count};
    int64_t main_scale[] = {(config->hidden_size + 127) / 128,
        (main_weight[1] + 127) / 128};
    int64_t hidden[] = {config->hidden_size};
    int last = manifest->stage_count - 1;
    int64_t markov[] = {config->vocab_size, manifest->markov_rank};
    int64_t confidence[] = {1, config->hidden_size + manifest->markov_rank};
    snprintf(name, sizeof(name), "mtp.0.main_proj.weight");
    int failed = require_tensor(index, name, COLI_ST_F8_E4M3, 2, main_weight,
                                &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.0.main_proj.scale");
    failed |= require_tensor(index, name, COLI_ST_F8_E8M0, 2, main_scale,
                             &manifest->special_bytes, error, error_size);
    failed |= require_tensor(index, "mtp.0.main_norm.weight", COLI_ST_BF16, 1,
                             hidden, &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.norm.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 1, hidden,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w1.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, markov,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w2.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, markov,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.confidence_head.proj.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, confidence,
                             &manifest->special_bytes, error, error_size);
    coli_st_index_close(index);
    return failed ? -1 : 0;
}
/* ---- end inlined deepseek_v4_dspark.c ---- */

#undef strchr
