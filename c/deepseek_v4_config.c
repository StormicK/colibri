#include "deepseek_v4_config.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int required_int(jval *root, const char *name, int *output,
                        char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM)
        return set_error(error, error_size, "missing integer config field: %s", name);
    *output = (int)value->num;
    return 0;
}

static int required_float(jval *root, const char *name, float *output,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM)
        return set_error(error, error_size, "missing numeric config field: %s", name);
    *output = (float)value->num;
    return 0;
}

static int require_string(jval *root, const char *name, const char *expected,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_STR || strcmp(value->str, expected))
        return set_error(error, error_size, "unsupported %s (expected %s)",
                         name, expected);
    return 0;
}

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size) {
    if (!config || !json)
        return set_error(error, error_size, "invalid DeepSeek-V4 config arguments");
    memset(config, 0, sizeof(*config));
    char *arena = NULL;
    jval *root = json_parse(json, &arena);
    if (!root || root->t != J_OBJ) {
        free(arena);
        return set_error(error, error_size, "DeepSeek-V4 config is not an object");
    }
    int failed =
        require_string(root, "model_type", "deepseek_v4", error, error_size) ||
        require_string(root, "expert_dtype", "fp4", error, error_size) ||
        require_string(root, "scoring_func", "sqrtsoftplus", error, error_size) ||
        require_string(root, "topk_method", "noaux_tc", error, error_size) ||
        required_int(root, "hidden_size", &config->hidden_size, error, error_size) ||
        required_int(root, "num_hidden_layers", &config->num_hidden_layers, error, error_size) ||
        required_int(root, "num_attention_heads", &config->num_attention_heads, error, error_size) ||
        required_int(root, "head_dim", &config->head_dim, error, error_size) ||
        required_int(root, "q_lora_rank", &config->q_lora_rank, error, error_size) ||
        required_int(root, "qk_rope_head_dim", &config->qk_rope_head_dim, error, error_size) ||
        required_int(root, "o_groups", &config->o_groups, error, error_size) ||
        required_int(root, "o_lora_rank", &config->o_lora_rank, error, error_size) ||
        required_int(root, "sliding_window", &config->sliding_window, error, error_size) ||
        required_int(root, "index_n_heads", &config->index_n_heads, error, error_size) ||
        required_int(root, "index_head_dim", &config->index_head_dim, error, error_size) ||
        required_int(root, "index_topk", &config->index_topk, error, error_size) ||
        required_int(root, "n_routed_experts", &config->n_routed_experts, error, error_size) ||
        required_int(root, "num_experts_per_tok", &config->num_experts_per_tok, error, error_size) ||
        required_int(root, "n_shared_experts", &config->n_shared_experts, error, error_size) ||
        required_int(root, "moe_intermediate_size", &config->moe_intermediate_size, error, error_size) ||
        required_int(root, "num_hash_layers", &config->num_hash_layers, error, error_size) ||
        required_int(root, "num_nextn_predict_layers", &config->num_nextn_predict_layers, error, error_size) ||
        required_int(root, "hc_mult", &config->hc_mult, error, error_size) ||
        required_int(root, "hc_sinkhorn_iters", &config->hc_sinkhorn_iters, error, error_size) ||
        required_int(root, "vocab_size", &config->vocab_size, error, error_size) ||
        required_int(root, "max_position_embeddings", &config->max_position_embeddings, error, error_size) ||
        required_float(root, "rms_norm_eps", &config->rms_norm_eps, error, error_size) ||
        required_float(root, "hc_eps", &config->hc_eps, error, error_size) ||
        required_float(root, "routed_scaling_factor", &config->routed_scaling_factor, error, error_size) ||
        required_float(root, "swiglu_limit", &config->swiglu_limit, error, error_size) ||
        required_float(root, "rope_theta", &config->rope_theta, error, error_size) ||
        required_float(root, "compress_rope_theta", &config->compress_rope_theta, error, error_size);
    if (failed) {
        free(arena);
        return -1;
    }
    jval *rope = json_get(root, "rope_scaling");
    if (!rope || rope->t != J_OBJ ||
        required_int(rope, "original_max_position_embeddings",
                     &config->original_max_position_embeddings, error, error_size) ||
        required_int(rope, "beta_fast", &config->rope_beta_fast, error, error_size) ||
        required_int(rope, "beta_slow", &config->rope_beta_slow, error, error_size) ||
        required_float(rope, "factor", &config->rope_factor, error, error_size)) {
        free(arena);
        return -1;
    }
    jval *ratios = json_get(root, "compress_ratios");
    if (!ratios || ratios->t != J_ARR || ratios->len < 1 ||
        ratios->len > COLI_V4_MAX_LAYERS) {
        free(arena);
        return set_error(error, error_size, "invalid compress_ratios");
    }
    config->compress_ratio_count = ratios->len;
    for (int index = 0; index < ratios->len; index++) {
        if (ratios->kids[index]->t != J_NUM) {
            free(arena);
            return set_error(error, error_size, "non-numeric compress ratio");
        }
        config->compress_ratios[index] = (int)ratios->kids[index]->num;
    }
    jval *quantization = json_get(root, "quantization_config");
    if (!quantization || quantization->t != J_OBJ ||
        require_string(quantization, "fmt", "e4m3", error, error_size) ||
        require_string(quantization, "scale_fmt", "ue8m0", error, error_size)) {
        free(arena);
        return -1;
    }
    if (config->hidden_size < 1 || config->num_hidden_layers < 1 ||
        config->num_attention_heads < 1 || config->n_routed_experts < 1 ||
        config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > config->n_routed_experts ||
        config->n_shared_experts != 1 || config->hc_mult < 1 ||
        config->compress_ratio_count != config->num_hidden_layers +
                                        config->num_nextn_predict_layers) {
        free(arena);
        return set_error(error, error_size, "inconsistent DeepSeek-V4 config dimensions");
    }
    free(arena);
    return 0;
}

int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size) {
    if (!config || !model_dir)
        return set_error(error, error_size, "invalid DeepSeek-V4 config path");
    size_t path_length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(path_length);
    if (!path) return set_error(error, error_size, "out of memory building config path");
    snprintf(path, path_length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        int result = set_error(error, error_size, "cannot open %s", path);
        free(path);
        return result;
    }
    fseek(stream, 0, SEEK_END);
    long length = ftell(stream);
    rewind(stream);
    if (length < 1) {
        fclose(stream);
        free(path);
        return set_error(error, error_size, "empty config: %s", model_dir);
    }
    char *text = malloc((size_t)length + 1);
    if (!text || fread(text, 1, (size_t)length, stream) != (size_t)length) {
        free(text);
        fclose(stream);
        free(path);
        return set_error(error, error_size, "cannot read config: %s", model_dir);
    }
    text[length] = 0;
    fclose(stream);
    int result = coli_v4_config_parse(config, text, error, error_size);
    free(text);
    free(path);
    return result;
}
