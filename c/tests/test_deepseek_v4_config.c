#include "../deepseek_v4_config.h"

#include <stdio.h>

int main(int argc, char **argv) {
    static const char config_json[] =
        "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
        "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
        "\"hidden_size\":128,\"num_hidden_layers\":3,"
        "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
        "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
        "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
        "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
        "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
        "\"num_hash_layers\":1,\"num_nextn_predict_layers\":1,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
        "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
        "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
        "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
        "\"compress_ratios\":[0,4,128,0],"
        "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
        "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
        "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";
    ColiDeepSeekV4Config config;
    char error[256];
    if (coli_v4_config_parse(&config, config_json, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (config.hidden_size != 128 || config.num_hidden_layers != 3 ||
        config.n_routed_experts != 8 || config.num_experts_per_tok != 2 ||
        config.compress_ratio_count != 4 || config.compress_ratios[1] != 4 ||
        config.compress_ratios[2] != 128 || config.rope_factor != 4.0f)
        return 1;
    if (argc > 1) {
        if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) != 0) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
        if (config.hidden_size != 4096 || config.num_hidden_layers != 43 ||
            config.n_routed_experts != 256 || config.num_experts_per_tok != 6 ||
            config.compress_ratio_count != 44 || config.compress_ratios[2] != 4 ||
            config.compress_ratios[3] != 128 || config.hc_mult != 4)
            return 1;
    }
    puts("DeepSeek-V4 config tests: ok");
    return 0;
}
