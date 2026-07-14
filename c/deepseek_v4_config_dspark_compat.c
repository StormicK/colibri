#include "deepseek_v4_dspark.h"
/* ---- begin inlined deepseek_v4_config_dspark_compat.c ---- */
#define coli_v4_config_parse coli_v4_config_parse_strict
#define coli_v4_config_load coli_v4_config_load_strict
#include "deepseek_v4_config.c"
#undef coli_v4_config_load
#undef coli_v4_config_parse

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size) {
    int result = coli_v4_config_parse_strict(config, json, error, error_size);
    if (!result) return 0;
    if (!config || !json || !strstr(json, "\"dspark_block_size\"") ||
        config->num_hidden_layers < 1 ||
        config->compress_ratio_count <= config->num_hidden_layers)
        return result;
    int inferred = config->compress_ratio_count - config->num_hidden_layers;
    if (inferred < 1 || inferred > COLI_V4_DSPARK_MAX_STAGES)
        return result;
    config->num_nextn_predict_layers = inferred;
    if (config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > config->n_routed_experts ||
        config->n_shared_experts != 1 || config->hc_mult < 1)
        return result;
    if (error && error_size) error[0] = 0;
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
    if (!stream) { free(path); return set_error(error, error_size,
                                                "cannot open config"); }
    fseek(stream, 0, SEEK_END); long length = ftell(stream); rewind(stream);
    char *text = length > 0 ? malloc((size_t)length + 1) : NULL;
    if (!text || fread(text, 1, (size_t)length, stream) != (size_t)length) {
        free(text); fclose(stream); free(path);
        return set_error(error, error_size, "cannot read config");
    }
    text[length] = 0; fclose(stream);
    int result = coli_v4_config_parse(config, text, error, error_size);
    free(text); free(path); return result;
}
/* ---- end inlined deepseek_v4_config_dspark_compat.c ---- */

