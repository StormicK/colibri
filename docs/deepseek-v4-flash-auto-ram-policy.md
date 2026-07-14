# DeepSeek-V4-Flash automatic RAM tier policy

## Scope

This policy is based on memory available when the process starts. It does not
target a 128 GB machine class and does not define fixed 8/16/32/128 GB presets.
An 8 GB system is a supported low-memory target when its current available
memory can hold the measured minimum working set.

CUDA and VRAM placement are outside the current development phase. The planner
must produce a complete CPU plus SSD plan without probing or requiring a GPU.

## Inputs

The runtime obtains these values before allocating the expert cache:

- `available_bytes`: `GlobalMemoryStatusEx().ullAvailPhys` on Windows,
  `MemAvailable` on Linux, and the corresponding reclaimable-memory value on
  other platforms;
- exact tensor byte ranges from the checkpoint headers;
- maximum non-expert bytes of one transformer layer;
- routed-expert record bytes, sparse-layer count, and routing top-k;
- requested prompt and context capacity;
- hidden-state, attention-cache, and temporary-buffer dimensions;
- optional user RAM/cache limit.

The planner must not substitute total physical RAM for available memory. It
must never invent an 8 GB budget when the OS reports less memory available.

## Budget order

Memory is assigned in the following order:

1. adaptive system reserve;
2. exact runtime working reserve;
3. minimum routed-expert working set;
4. resident BF16 output head, only when it fits with safety margin;
5. additional per-layer expert-cache slots.

In automatic mode the system reserve is adaptive rather than tied to a machine
size:

```text
system_reserve = clamp(OS_available_bytes / 8, 512 MiB, 4 GiB)
```

An explicit CLI memory limit is a process cap. Memory outside that cap is
already reserved for the OS, so the planner does not subtract a second system
reserve from the cap:

```text
planner_available = min(OS_available_bytes, cli_memory_limit)
system_reserve = 0  # only when cli_memory_limit < OS_available_bytes
```

The runtime reserve is computed from model and request dimensions:

```text
runtime_reserve =
    2 * maximum_non_expert_layer_bytes       # current + async next layer
  + requested_KV_and_attention_cache_bytes
  + prompt_and_decode_hidden_buffers
  + maximum_attention_and_FFN_scratch_bytes
  + allocator_safety_margin
```

The minimum expert cache is derived from the model, not from a constant:

```text
minimum_expert_bytes =
    sparse_layer_count * routed_topk * expert_record_bytes
```

For the current checkpoint this is approximately 3.45 GB for six expert slots
per each of 43 layers. This explains why the measured process working set is
around 4 GB and why an 8 GB system can be viable, but the value remains derived
from checkpoint metadata.

If the minimum does not fit after reserves, startup fails with the exact byte
shortfall and suggestions to reduce context/prompt capacity. The planner does
not overcommit or silently depend on swap.

## Output-head policy

The 129,280 by 4,096 BF16 output head occupies about 1.06 GB. Residency is an
optional tier selected after the minimum inference and expert working sets are
safe:

- low-memory mode streams the BF16 head and retains the current exact path;
- when sufficient surplus exists, the raw BF16 head becomes resident;
- INT8 head conversion remains a separate opt-in accuracy experiment.

Therefore head residency cannot make an otherwise runnable 8 GB configuration
fail. It is a performance feature, not a minimum requirement.

## Expert-cache growth

After the mandatory allocations and optional resident head are accounted for,
all remaining planner budget may increase the per-layer slot count:

```text
slots_per_layer = floor(expert_cache_bytes /
                        (sparse_layer_count * expert_record_bytes))
slots_per_layer = clamp(slots_per_layer, routed_topk, experts_per_layer)
```

Slot payloads remain lazily committed. Capacity is still bounded so a long
conversation cannot grow the process beyond its startup plan.

## User override

The runtime is fully automatic by default. The supported resource override is
the optional `--memory-gb GiB` process cap:

```text
planner_available = min(OS_available, cli_memory_limit_if_set)
```

Expert-cache size, dense and DSpark residency, output-head residency, thread
split, verification window, and hot-expert pinning are derived automatically.
Ordinary inference never increases a budget beyond current OS availability.

## Explicit 24 GiB process tier

For the current DeepSeek-V4-Flash-DSpark checkpoint, `--memory-gb 24` selects
the same core residency strategy as the previous 32 GiB configuration while
reducing the target expert cache to fit the smaller cap:

```text
dense=resident(6.27GiB)
dspark=resident(10.45GiB)
target_slots=10 target_cache=5.35GiB
head=resident-bf16 projected=23.95GiB
```

The former environment-variable planner subtracted a 3 GiB system reserve
inside the 24 GiB process limit, reducing its effective budget to about 21 GiB
and incorrectly selecting the streamed tier. Automatic mode still reserves RAM
for the OS; only an explicit process cap uses the corrected accounting.

## Required startup log

Every run prints one resource line containing:

- OS available memory;
- system and runtime reserves;
- minimum and selected expert-cache bytes;
- expert slots per layer;
- output-head mode (`streamed-bf16` or `resident-bf16`);
- projected maximum committed memory.

This makes the same algorithm auditable on 8 GB and high-memory systems without
introducing hardware-specific profiles.

## Implementation order

1. Add a CPU-only V4 resource-plan structure and deterministic unit tests.
2. Replace the generator's fixed 4 GiB expert-cache argument with the plan.
3. Validate forced low-memory fixtures and the real checkpoint on Windows.
4. Add optional resident BF16 head selection.
5. Use the remaining planned memory for LRU growth and later pin/repin work.
6. Keep CUDA out of this phase and out of acceptance criteria.
