# DeepSeek V4 native FP4 rows16

Date: 2026-07-13  
Platform: Windows 11 native MinGW-w64, CPU only

## Promoted build

These kernels are part of the stable `make deepseek-v4` binary. See
`deepseek-v4-cli.md` for build and run instructions.

## Design

The model remains in its native FP4 plus UE8M0 format. No FP4-to-INT4
requantization is performed.

The SSD representation remains official row-major FP4. When the existing hot
policy pins an expert in resident RAM, its three matrices are rearranged in
place into sixteen-row tiles. Each AVX-512 lane owns one output row and visits
columns in the original order. This removes horizontal reductions and retains
the exact scalar accumulation order for every output.

Cold experts retain the official layout and use the established v4 kernel.
Non-AVX-512 builds never rearrange a slot and use the same fallback.

The relevant implementation files are:

- `native_quant_fp4_rows16.c`: rows16 packing and single/dual kernels;
- `deepseek_v4_expert_store_hot_rows16.c`: conversion integrated into the
  existing pin/repin policy;
- `deepseek_v4_expert_rows16.c`: one `block_rows` dispatch check followed
  by either rows16 or the original expert forward.

## Why ExpertStore integration matters

The production-size microbenchmark showed the kernel was useful:

| Matrix | Existing v4 | rows16 | Speedup |
| --- | ---: | ---: | ---: |
| gate/up, 2048x4096 dual | 3.044 ms | 2.314 ms | 1.316x |
| down, 4096x2048 | 1.482 ms | 1.163 ms | 1.274x |

Earlier eager conversion of every cache miss regressed short runs. Delayed
conversion with a second lease/usage manager fixed conversion volume but still
cost about 0.8-1.0 seconds per short run.

The promoted path does not duplicate hotness tracking. The ExpertStore already
knows which four experts per layer are pinned, so it converts those slots once
and marks their `TensorView.block_rows` as 16. The forward path adds only a
field check.

## Correctness

The rows16 microbenchmark and expert tests are bit-exact. End-to-end 10, 20,
and 40-token runs preserve:

- the established target token sequence and generated text;
- expert reads and bytes;
- progressive verification acceptance prefixes;
- target-authoritative sampling.

The 40-token result is:

```text
expert_reads=5882
dspark_rounds=9
proposed=36
accepted=31
acceptance=0.861
```

## Paired Windows-native performance

Historical paired measurements under a slower thermal/system state:

| Tokens | prior total | rows16 total | prior after first | rows16 after first | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 54.307 s | 53.697 s | 35.368 s | 34.957 s | 1.12% |
| 20 | 108.065 s | 107.297 s | 89.174 s | 89.041 s | 0.71% |
| 40 | 189.918 s | 185.038 s | 171.082 s | 166.427 s | 2.57% |

The 40-token after-first improvement is 2.72%. The increasing long-session
benefit is consistent with paying the rearrangement cost once and reusing
pinned experts afterward.

## Memory impact

Resident expert capacity does not increase because weights and scales are
permuted in place. Conversion temporarily allocates scratch space for one
matrix while holding the slot. The hot policy still pins at most four experts
per layer and reserves the six working slots required by routed top-k.

The 32 GiB plan remains unchanged:

```text
dense=6.27 GiB
dspark=10.45 GiB
target expert cache=9.10 GiB / 17 slots
head=resident BF16
projected=31.70 GiB
```
