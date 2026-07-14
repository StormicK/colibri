# DeepSeek V4 CPU CLI

The DeepSeek V4 CPU runtime configures memory tiers, DSpark residency, expert
caches, verification windows, thread allocation, and hot-expert pinning
automatically. No environment variables are required.

## Build

From an MSYS2 UCRT64 shell on Windows:

```bash
export PATH=/ucrt64/bin:/usr/bin
cd /d/ai/colibri/c
make deepseek-v4
```

From the repository root:

```bash
make deepseek-v4
```

This produces `c/deepseek_v4.exe`. Rebuild after pulling engine changes.

## Run

`c/v4` is a Python launcher around `deepseek_v4.exe`. On Windows, invoke it with
an installed Python (MSYS2 may not ship `python3` on `PATH`):

```powershell
cd D:\ai\colibri\c
python v4 run --model D:/ai/DeepSeek-V4-Flash-DSpark --ram 32 `
  --stop-sentence "What is the capital of France?"
```

Verified smoke output:

```text
The capital of France is Paris.
[stats] RAM 23.95/24.00 GiB | TTFT 11.77s | prefill 1.106 tok/s | decode 0.921 tok/s | DSpark acceptance 100.0%
```

Assistant text is streamed to standard output. Compact stats and engine
diagnostics go to standard error.

## Chat

Interactive multi-turn chat:

```powershell
cd D:\ai\colibri\c
python v4 chat --model D:/ai/DeepSeek-V4-Flash-DSpark --ram 24
```

The model directory is an explicit argument; `COLI_MODEL` is not used by the
V4 launcher. Use `:reset` to clear conversation history and `:q` to exit.
The current chat implementation preserves multi-turn history and re-prefills
it on every turn; persistent cross-turn V4 KV cache reuse is a later step.

The default prompt uses the official DeepSeek V4 non-thinking chat encoding:

```text
<｜begin▁of▁sentence｜>[system]<｜User｜>[user]<｜Assistant｜></think>
```

## Options

- `--model PATH` (required) DeepSeek V4 checkpoint directory
- `--ram GiB` caps process memory. Without it, the planner uses current
  OS-available memory and retains an adaptive system reserve.
- `--ngen N` sets the generation limit; the default is 128.
- `--system TEXT` adds a system message before the first user turn (`chat` only).
- `--thinking` selects the official `<think>` generation prefix (`chat` only).
- `--stop-sentence` on `v4 run` stops after the first sentence terminator.

At 24 GiB, the current checkpoint keeps target dense tensors, DSpark stages and
experts, and the BF16 output head resident while reducing the target expert
cache to fit the cap. See `deepseek-v4-flash-auto-ram-policy.md` for details.
