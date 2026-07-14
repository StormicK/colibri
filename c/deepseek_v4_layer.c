#include "deepseek_v4_layer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int add_spec(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                    int rank, const int64_t *shape, const char *suffix,
                    char *error, size_t error_size) {
    if (plan->tensor_count >= COLI_V4_MAX_LAYER_TENSORS)
        return set_error(error, error_size, "too many tensors in layer %d", plan->layer);
    ColiDeepSeekV4TensorSpec *spec = &plan->tensors[plan->tensor_count++];
    int written = snprintf(spec->name, sizeof(spec->name), "layers.%d.%s",
                           plan->layer, suffix);
    if (written < 0 || (size_t)written >= sizeof(spec->name))
        return set_error(error, error_size, "tensor name is too long: %s", suffix);
    spec->dtype = dtype;
    spec->rank = rank;
    memcpy(spec->shape, shape, (size_t)rank * sizeof(*shape));
    return 0;
}

static int add_1d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, const char *name, char *error, size_t size) {
    int64_t shape[] = {d0};
    return add_spec(plan, dtype, 1, shape, name, error, size);
}

static int add_2d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, int64_t d1, const char *name,
                  char *error, size_t size) {
    int64_t shape[] = {d0, d1};
    return add_spec(plan, dtype, 2, shape, name, error, size);
}

static int add_fp8(ColiDeepSeekV4LayerPlan *plan, int64_t rows, int64_t columns,
                   const char *prefix, char *error, size_t size) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (add_2d(plan, COLI_ST_F8_E4M3, rows, columns, name, error, size) != 0) return -1;
    snprintf(name, sizeof(name), "%s.scale", prefix);
    return add_2d(plan, COLI_ST_F8_E8M0, (rows + 127) / 128,
                  (columns + 127) / 128, name, error, size);
}

#define ADD(call) do { if ((call) != 0) return -1; } while (0)

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size) {
    if (!plan || !config || layer < 0 || layer >= config->num_hidden_layers ||
        layer >= config->compress_ratio_count)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer plan arguments");
    memset(plan, 0, sizeof(*plan));
    plan->layer = layer;
    plan->compression_ratio = config->compress_ratios[layer];
    plan->uses_hash_router = layer < config->num_hash_layers;
    plan->has_compressor = plan->compression_ratio != 0;
    plan->has_indexer = plan->compression_ratio == 4;

    const int64_t hidden = config->hidden_size;
    const int64_t heads = config->num_attention_heads;
    const int64_t head_dim = config->head_dim;
    const int64_t q_rank = config->q_lora_rank;
    const int64_t o_width = (int64_t)config->o_groups * config->o_lora_rank;
    const int64_t experts = config->n_routed_experts;
    const int64_t moe = config->moe_intermediate_size;
    const int64_t hc = config->hc_mult;
    const int64_t hc_params = 2 * hc * (hc - 1);

    ADD(add_1d(plan, COLI_ST_F32, heads, "attn.attn_sink", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, head_dim, "attn.kv_norm.weight", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, q_rank, "attn.q_norm.weight", error, error_size));
    ADD(add_fp8(plan, head_dim, hidden, "attn.wkv", error, error_size));
    ADD(add_fp8(plan, o_width, hidden, "attn.wo_a", error, error_size));
    ADD(add_fp8(plan, hidden, o_width, "attn.wo_b", error, error_size));
    ADD(add_fp8(plan, q_rank, hidden, "attn.wq_a", error, error_size));
    ADD(add_fp8(plan, heads * head_dim, q_rank, "attn.wq_b", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "attn_norm.weight", error, error_size));

    if (plan->has_compressor) {
        int64_t ratio = plan->compression_ratio;
        int64_t coff = ratio == 4 ? 2 : 1;
        ADD(add_2d(plan, COLI_ST_F32, ratio, coff * head_dim,
                   "attn.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, head_dim,
                   "attn.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wkv.weight", error, error_size));
    }
    if (plan->has_indexer) {
        int64_t ih = config->index_head_dim;
        int64_t in = config->index_n_heads;
        ADD(add_2d(plan, COLI_ST_F32, 4, 2 * ih,
                   "attn.indexer.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, ih,
                   "attn.indexer.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wkv.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, in, hidden,
                   "attn.indexer.weights_proj.weight", error, error_size));
        ADD(add_fp8(plan, in * ih, q_rank, "attn.indexer.wq_b", error, error_size));
    }

    ADD(add_2d(plan, COLI_ST_BF16, experts, hidden,
               "ffn.gate.weight", error, error_size));
    if (plan->uses_hash_router)
        ADD(add_2d(plan, COLI_ST_I64, config->vocab_size,
                   config->num_experts_per_tok, "ffn.gate.tid2eid", error, error_size));
    else
        ADD(add_1d(plan, COLI_ST_F32, experts, "ffn.gate.bias", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w1", error, error_size));
    ADD(add_fp8(plan, hidden, moe, "ffn.shared_experts.w2", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w3", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "ffn_norm.weight", error, error_size));

    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_attn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_attn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, hc - 1, "hc_attn_scale", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_ffn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_ffn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, hc - 1, "hc_ffn_scale", error, error_size));
    return 0;
}

int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size) {
    if (!plan || !index)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer validation arguments");
    ColiDeepSeekV4LayerStats local = {0};
    for (size_t i = 0; i < plan->tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &plan->tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        if (!tensor)
            return set_error(error, error_size, "missing tensor: %s", spec->name);
        if (tensor->dtype != spec->dtype || tensor->rank != spec->rank)
            return set_error(error, error_size, "dtype/rank mismatch: %s", spec->name);
        for (int dimension = 0; dimension < spec->rank; dimension++)
            if (tensor->shape[dimension] != spec->shape[dimension])
                return set_error(error, error_size, "shape mismatch: %s", spec->name);
        local.tensor_count++;
        local.total_bytes += tensor->nbytes;
        switch (tensor->dtype) {
            case COLI_ST_BF16: local.bf16_bytes += tensor->nbytes; break;
            case COLI_ST_F32: local.f32_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E4M3: local.fp8_weight_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E8M0: local.fp8_scale_bytes += tensor->nbytes; break;
            case COLI_ST_I64: local.i64_bytes += tensor->nbytes; break;
            default: break;
        }
    }
    if (stats) *stats = local;
    return 0;
}

void coli_v4_layer_free(ColiDeepSeekV4LayerWeights *weights) {
    if (!weights) return;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) free(weights->data[i]);
    memset(weights, 0, sizeof(*weights));
}

int coli_v4_layer_load(ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size) {
    if (!weights) return set_error(error, error_size, "missing layer weights output");
    memset(weights, 0, sizeof(*weights));
    if (coli_v4_layer_plan(&weights->plan, config, layer, error, error_size) != 0 ||
        coli_v4_layer_validate(&weights->plan, index, &weights->stats,
                               error, error_size) != 0)
        return -1;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &weights->plan.tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        weights->data[i] = malloc((size_t)tensor->nbytes);
        if (!weights->data[i]) {
            coli_v4_layer_free(weights);
            return set_error(error, error_size, "out of memory loading: %s", spec->name);
        }
        if (coli_st_read_tensor(index, tensor, weights->data[i]) != 0) {
            coli_v4_layer_free(weights);
            return set_error(error, error_size, "cannot read tensor: %s", spec->name);
        }
    }
    return 0;
}

const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec) {
    if (spec) *spec = NULL;
    if (!weights || !name) return NULL;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        if (strcmp(weights->plan.tensors[i].name, name) == 0) {
            if (spec) *spec = &weights->plan.tensors[i];
            return weights->data[i];
        }
    }
    return NULL;
}
