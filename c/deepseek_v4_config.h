#ifndef COLIBRI_DEEPSEEK_V4_CONFIG_H
#define COLIBRI_DEEPSEEK_V4_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_V4_MAX_LAYERS 128

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int head_dim;
    int q_lora_rank;
    int qk_rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int sliding_window;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;
    int moe_intermediate_size;
    int num_hash_layers;
    int num_nextn_predict_layers;
    int hc_mult;
    int hc_sinkhorn_iters;
    int vocab_size;
    int max_position_embeddings;
    int original_max_position_embeddings;
    int compress_ratio_count;
    int compress_ratios[COLI_V4_MAX_LAYERS];
    float rms_norm_eps;
    float hc_eps;
    float routed_scaling_factor;
    float swiglu_limit;
    float rope_theta;
    float rope_factor;
    float compress_rope_theta;
    int rope_beta_fast;
    int rope_beta_slow;
} ColiDeepSeekV4Config;

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size);
int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
