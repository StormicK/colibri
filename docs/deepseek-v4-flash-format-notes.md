# DeepSeek-V4-Flash checkpoint format notes

These notes record facts measured from the official checkpoint at
`/root/DeepSeek-V4-Flash`. They refine the assumptions in the development plan.

## Inventory

| Item | Measured value |
|---|---:|
| Shards | 46 |
| Tensors | 69,187 |
| Total tensor bytes | 159,609,485,896 |
| Resident/non-routed tensor bytes | 9,017,195,080 |
| Routed expert bytes | 147,169,738,752 |
| MTP routed expert bytes | 3,422,552,064 |
| Transformer layers | 43 |
| Routed experts per layer | 256 |
| Expert records | 11,008 |
| Bytes per expert record | 13,369,344 |

All 11,008 expert records are complete and contained within a single shard. Each
record has six tensors: `w1/w2/w3` weight and scale.

## Native FP4 expert layout

The official inference code identifies the expert format as
`float4_e2m1fn_x2`. Two FP4 values are packed into one byte along the K
dimension. The low nibble is the first value and the high nibble is the second.

The nibble lookup table is:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6,
0, -0.5, -1, -1.5, -2, -3, -4, -6
```

Every 32 logical FP4 values along K share one `float8_e8m0fnu`/UE8M0 scale.
The checkpoint exposes packed FP4 payloads as safetensors `I8`; the semantic
format comes from `expert_dtype=fp4` and must not be interpreted as signed INT8.

| Part | Stored dtype/shape | Logical matrix | Bytes |
|---|---|---|---:|
| `w1.weight` | I8 `[2048, 2048]` | `[2048, 4096]` FP4 | 4,194,304 |
| `w1.scale` | F8_E8M0 `[2048, 128]` | 1 scale / 32 K | 262,144 |
| `w2.weight` | I8 `[4096, 1024]` | `[4096, 2048]` FP4 | 4,194,304 |
| `w2.scale` | F8_E8M0 `[4096, 64]` | 1 scale / 32 K | 262,144 |
| `w3.weight` | I8 `[2048, 2048]` | `[2048, 4096]` FP4 | 4,194,304 |
| `w3.scale` | F8_E8M0 `[2048, 128]` | 1 scale / 32 K | 262,144 |

Native expert execution is `FP8 activation × FP4 weight`, with FP32
accumulation in the official reference kernel.

## Dense FP8 layout

Dense quantized tensors use E4M3 weights with UE8M0 scales and a `128 × 128`
weight block. Activations are dynamically quantized to FP8 in blocks of 128.
The complete checkpoint dtype byte totals are:

| Safetensors dtype | Bytes |
|---|---:|
| BF16 | 2,830,518,528 |
| F32 | 144,672,072 |
| F8_E4M3 | 6,023,020,544 |
| F8_E8M0 | 8,858,737,664 |
| I64 | 18,616,320 |
| I8 | 141,733,920,768 |

The large `I8` total is primarily packed native FP4 expert data, not INT8
weights. The large E8M0 total includes the fine-grained FP4 expert scales.

## Direct SSD streaming decision

Within each expert record, the three scale tensors are adjacent and the three
weight tensors are adjacent. The scale group and weight group are separated by
other records, so reading the raw min-to-max span would cause a median 138.7x
read amplification. Merging adjacent ranges produces exactly two reads:

```text
scale read    786,432 bytes
weight read   12,582,912 bytes
total         13,369,344 bytes
```

Therefore the first streaming implementation should read the official shards
directly using two coalesced `pread` operations per cache miss. A second 160 GB
reordered container is not required for correctness and is now an optional
optimization only.

The runtime should build a small manifest from safetensors headers containing:

```text
(layer, expert)
  shard path
  scale offset + length
  weight offset + length
  w1/w2/w3 sub-offsets and shapes
  semantic format = native FP4 E2M1
  scale format = UE8M0
```

## Architecture facts affecting the CPU reference

- `hidden_size = 4096`
- `num_hidden_layers = 43`
- every layer has 256 routed experts; top-k is 6
- `moe_intermediate_size = 2048`
- `hc_mult = 4`, Sinkhorn iterations = 20
- compression ratios alternate between 4 and 128 after the first two layers,
  with an uncompressed final entry used by the next-token layer
- ratio-4 attention has an indexer; `index_topk = 512`
- sliding window = 128
- one next-token-prediction layer is configured
- router scoring is `sqrtsoftplus` with `noaux_tc`
- SwiGLU input is limited by `swiglu_limit = 10.0`

These facts are inputs to the tiny reference fixtures and must be covered by
more than one fixture: uncompressed, ratio-4/indexed, and ratio-128 layers take
different forward paths.

## Validated resident layer plans

`deepseek_v4_layer.c` now derives an exact tensor plan from `config.json` and
checks every planned tensor's name, dtype, rank, and shape against the real
safetensors index. All 43 official layers pass this validation.

| Layer kind | Planned tensors | Native resident bytes per layer |
|---|---:|---:|
| Uncompressed hash-router (layers 0-1) | 29 | 136.94 MiB |
| Ratio-4 + indexer, hash-router (layer 2) | 40 | 165.47 MiB |
| Ratio-4 + indexer, learned router | 40 | 159.55 MiB |
| Ratio-128 compressor, learned router | 33 | 139.28 MiB |

The planned transformer-layer resident tensors total 6,727,565,512 bytes
(6.27 GiB). The larger 9,017,195,080-byte resident inventory additionally
contains global tensors such as embeddings, output heads, and non-expert MTP
weights. The MTP layer's 3,422,552,064 bytes of routed experts are classified
separately and must use streaming when MTP is enabled.

The layer loader reads one layer at a time in checkpoint-native formats. It
does not expand BF16 or FP8 payloads to FP32, so its temporary resident working
set is bounded to roughly 137-165 MiB plus compute buffers. Routed experts are
not included in this loader and remain under `StreamingExpertStore` control.

## Reproduction

Run the repository inventory tool without loading tensor payloads:

```bash
cd c
python3 tools/inspect_deepseek_v4.py ~/DeepSeek-V4-Flash
python3 tools/inspect_deepseek_v4.py ~/DeepSeek-V4-Flash --json
python3 tools/inspect_deepseek_v4.py ~/DeepSeek-V4-Flash --layer 2
make probe-v4-layers MODEL=~/DeepSeek-V4-Flash LAYER=2
```

## Real layer-0 CPU reference milestone

The complete uncompressed layer-0 path now runs from a real BF16 embedding row
through attention and MoE to the four-copy mHC output. It includes:

- native FP8 Q/KV/O projections with BF16 boundaries;
- head RMS normalization, RoPE/de-rotation, block-64 KV QDQ, and attention sink;
- grouped `wo_a` execution;
- hash routing from `tid2eid` and score-derived top-6 weights;
- one native-FP8 shared expert;
- six native-FP4 routed experts loaded through the SSD streaming store;
- both mHC pre/post residual branches.

For real token ID 1000, layer 0 reads six expert records totaling 80,216,064
bytes. Against the independent PyTorch CPU oracle, the 16,384-element output
has maximum absolute error 0.0078125, mean absolute error 0.00022785515,
84.497070% bit-exact elements, and cosine similarity 0.9999879003. The remaining
1-2 BF16 ULP differences are expected from scalar versus tiled FP32 reduction
order; attention alone is bit-exact across all 4,096 output values.

Reproduce the checks with:

```bash
/root/.venvs/colibri-v4/bin/python \
  tools/verify_deepseek_v4_attention_torch.py ~/DeepSeek-V4-Flash --token 1000
/root/.venvs/colibri-v4/bin/python \
  tools/verify_deepseek_v4_block_torch.py ~/DeepSeek-V4-Flash --token 1000
```

## Full-checkpoint first-token milestone

The position-zero CPU path now executes all 43 transformer layers with the
complete checkpoint. Position zero has no compressed KV candidates, so all
compression ratios share the single-window-KV attention path; routed experts
still use their native FP4 representation and the SSD streaming store.

Reproducible results on the development machine:

```text
input_token=1000 output_token=201 logit=20.6316776
input_token=0    output_token=5   logit=16.7998447
expert cache misses per run: 258
SSD expert bytes per run:    3,449,290,752
```

Token 0 is the checkpoint's configured BOS token. The BOS run therefore proves
an actual one-token prompt forward through embedding, 43 layers, mHC head,
final RMSNorm, streamed BF16 lm_head, and greedy argmax. Runtime was roughly
75-92 seconds on the reference CPU implementation.

Reproduce with:

```bash
cd c
make deepseek-v4-first-token MODEL=~/DeepSeek-V4-Flash TOKEN=0
```

## Persistent decode and compressed-attention milestone

The CPU reference now keeps per-layer decode state across tokens. Each of the
43 layers owns a 128-entry window KV ring. Compressed layers additionally own a
dynamically growing compressed-KV cache, so cache memory follows the actual
sequence length instead of reserving the checkpoint's maximum context.
Layer weights are still loaded one layer at a time and released immediately;
the persistent compressor/indexer state rebinds to the current layer load.

Both checkpoint compression modes are active in the full block executor:

- ratio 128 appends every completed compressed KV block to sparse attention;
- ratio 4 uses the overlapping main compressor plus the separate 64-head,
  128-dimensional FP4/Hadamard indexer and selects up to 512 compressed blocks;
- window and compressed indices share one sparse-attention call, matching the
  official cache offset convention.

Real-checkpoint oracle coverage:

```text
layer 2 ratio-4 indexer compressor: 256/256 FP32 values bit exact
layer 2 position-3 integrated attention: max abs error 0.03125 (one BF16 ULP)
layer 3 position-128 integrated attention: max abs error 0.0390625,
                                           cosine 0.9999710321
```

The full 43-layer BOS greedy decode crosses the first ratio-4 compression
boundary and emits:

```text
position 0: 0   -> 5    logit 16.7998447
position 1: 5   -> 223  logit 17.8514805
position 2: 223 -> 939  logit 16.1286011
position 3: 939 -> 22   logit 21.0966263
expert reads: 922
expert bytes: 12,326,535,168
```

WSL and native Windows 11 MinGW-w64/UCRT64 produce the exact same tokens,
logits, expert-read count, and byte count when both builds use
`-O3 -march=native` (`make ARCH=native`). The distributable Windows default,
`ARCH=x86-64-v3`, emits the same four token IDs but not bit-identical logits;
this is an ISA/code-generation difference, not a checkpoint or cache-format
difference.

Reproduce the principal checks with:

```bash
cd c/tools
/root/.venvs/colibri-v4/bin/python3 \
  verify_deepseek_v4_indexer_torch.py ~/DeepSeek-V4-Flash
/root/.venvs/colibri-v4/bin/python3 \
  verify_deepseek_v4_attention_sequence_torch.py ~/DeepSeek-V4-Flash
/root/.venvs/colibri-v4/bin/python3 \
  verify_deepseek_v4_ratio128_attention_torch.py \
  ~/DeepSeek-V4-Flash --integrated
```
