# DeepSeek V4 Flash target-batch optimization

Date: 2026-07-13  
Platform: Windows 11 native MinGW-w64, CPU only

## Promoted build

The AVX-512 target-batch kernel is part of the stable `make deepseek-v4`
binary. See `deepseek-v4-cli.md` for build and run instructions.

The batch path lives in `native_quant_batch_avx512.c`. On AVX-512 systems it
decodes sixteen FP8 weights at once and reuses them across the target
verification batch. Products are accumulated in the original scalar lane
order, so output bits do not change. Builds without AVX-512 use the existing
portable batch kernel.

The memory planner and resident tensors are unchanged. The 32 GiB plan remains:

```text
dense=6.27 GiB
dspark=10.45 GiB
target expert cache=9.10 GiB / 17 slots
head=resident BF16
projected=31.70 GiB
```

## Correctness gates

Both the native `-march=native` test and the portable `-march=x86-64` test
reported:

```text
native quant batch tests: bit-exact
```

The end-to-end 10- and 20-token runs retained the established token sequence,
expert-read counts, and acceptance prefixes. The 20-token acceptance sequence
remains `2,4,1,4,4`:

```text
10 tokens: proposed=8  accepted=6  rate=0.750  expert_reads=1886
20 tokens: proposed=20 accepted=15 rate=0.750  expert_reads=3338
```

## Windows-native performance

Prompt: `The capital of France is`; RAM tier: 32 GiB.

| Build | Tokens | TTFT | After first | Total |
| --- | ---: | ---: | ---: | ---: |
| v35 | 10 | 16.906 s | 32.313 s | 49.219 s |
| v38 | 10 | 16.639 s | 30.668 s | 47.307 s |
| v35 | 20 | 16.876 s | 84.286 s | 101.161 s |
| v38 | 20 | 16.660 s | 79.188 s | 95.848 s |

v38 improves total time by 3.9% for 10 tokens and 5.25% for 20 tokens. Relative
to the first aligned v29 20-token result of 166.699 seconds, the cumulative
reduction is 42.5%.

Substage profiling attributes approximately 10.7 seconds of the v38 10-token
run to target batch attention, down from about 12.2 seconds in v35. The target
head is about 1.31 seconds. The remaining target block work is split across HC,
shared expert, and routed experts rather than concentrated in a few layers.

A full-run quantized-kernel profile measured these inclusive totals:

| Kernel | Calls | Time |
| --- | ---: | ---: |
| FP8 batch | 2163 | 17.147 s |
| FP4 dual gate/up | 3612 | 11.000 s |
| FP4 down | 3612 | 5.121 s |
| FP8 matvec | 1380 | 2.603 s |
| FP8 dual | 602 | 1.877 s |

These numbers cover prompt prefill, target verification, and DSpark; they are
for hotspot ordering, not mutually exclusive target-verifier phase totals.

## Rejected experiments

- v37 padded FP8 batches to eight lanes. It stayed bit-exact but measured
  49.472 seconds for 10 tokens because target batches are normally two or
  three inputs.
- v39 batches only the shared expert and keeps routed experts sequential. It
  retained exact tokens and reads but measured 48.915 seconds; small-batch
  setup cost exceeded shared-weight reuse.
- v42 applies AVX-512 across columns of single-token FP4 expert matvecs. It
  passed native and portable bit-exact tests but regressed the end-to-end run
  to 52.444 seconds. The extra nibble expansion and AVX-512 frequency cost make
  this unsuitable for the single-token routed path.

## Next optimization boundary

The next target should not be another global batch-union variant. The previous
union implementation and the shared-only batch both regressed. A production
attempt should instead add a persistent routed-expert worker pool and a cheap
per-layer overlap cost model:

1. compute the expert intersection/union for the two- or three-input target
   verification chunk;
2. use existing sequential pipelining when overlap is low;
3. reserve and load only overlapping experts through persistent workers when
   predicted reuse exceeds dispatch cost;
4. retain v38 tokens, `2,4,1,4,4` acceptance, reads, portable fallback, and the
   forced accepted-prefix-3 regression as promotion gates.

This direction attacks the remaining 16.12 seconds of full-run FP4 work without
changing FP4 to INT4, increasing the resident-memory plan, or disturbing the
already aligned DSpark attention and verifier schedule.
