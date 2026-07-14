#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../deepseek_v4_layer.h"

static const ColiDeepSeekV4TensorSpec *find_spec(
        const ColiDeepSeekV4LayerPlan *plan, const char *suffix) {
    for (size_t i = 0; i < plan->tensor_count; i++)
        if (strstr(plan->tensors[i].name, suffix)) return &plan->tensors[i];
    return NULL;
}

int main(void) {
    ColiDeepSeekV4Config config = {0};
    config.hidden_size = 4096;
    config.num_hidden_layers = 43;
    config.num_attention_heads = 64;
    config.head_dim = 512;
    config.q_lora_rank = 1024;
    config.o_groups = 8;
    config.o_lora_rank = 1024;
    config.index_n_heads = 64;
    config.index_head_dim = 128;
    config.n_routed_experts = 256;
    config.num_experts_per_tok = 6;
    config.n_shared_experts = 1;
    config.moe_intermediate_size = 2048;
    config.num_hash_layers = 3;
    config.hc_mult = 4;
    config.vocab_size = 129280;
    config.compress_ratio_count = 43;
    config.compress_ratios[2] = 4;
    config.compress_ratios[3] = 128;

    char error[256];
    ColiDeepSeekV4LayerPlan plan;
    assert(coli_v4_layer_plan(&plan, &config, 0, error, sizeof(error)) == 0);
    assert(plan.uses_hash_router && !plan.has_compressor && !plan.has_indexer);
    assert(plan.tensor_count == 29);
    const ColiDeepSeekV4TensorSpec *hash = find_spec(&plan, "tid2eid");
    assert(hash && hash->dtype == COLI_ST_I64);
    assert(hash->shape[0] == 129280 && hash->shape[1] == 6);

    assert(coli_v4_layer_plan(&plan, &config, 2, error, sizeof(error)) == 0);
    assert(plan.uses_hash_router && plan.has_compressor && plan.has_indexer);
    assert(plan.tensor_count == 40);
    const ColiDeepSeekV4TensorSpec *index_q = find_spec(&plan, "indexer.wq_b.weight");
    assert(index_q && index_q->shape[0] == 8192 && index_q->shape[1] == 1024);
    const ColiDeepSeekV4TensorSpec *ape = find_spec(&plan, "attn.compressor.ape");
    assert(ape && ape->shape[0] == 4 && ape->shape[1] == 1024);

    assert(coli_v4_layer_plan(&plan, &config, 3, error, sizeof(error)) == 0);
    assert(!plan.uses_hash_router && plan.has_compressor && !plan.has_indexer);
    assert(plan.tensor_count == 33);
    ape = find_spec(&plan, "attn.compressor.ape");
    assert(ape && ape->shape[0] == 128 && ape->shape[1] == 512);
    assert(find_spec(&plan, "ffn.gate.bias") != NULL);

    puts("deepseek_v4_layer tests passed");
    return 0;
}
