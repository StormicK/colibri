# Windows + AMD ROCm/HIP setup notes (RX 9070, native, no WSL2)

Personal fork notes for building and running colibrì's Hy3/GLM engines with
native AMD GPU offload on Windows via the HIP SDK, until `HIP_DLL`/`hip-dll`
and the `backend_loader.c` fixes below are merged upstream.

## One-time environment prerequisites

- AMD HIP SDK for Windows installed at `C:\Program Files\AMD\ROCm\7.1`.
- A space-free junction to avoid hipcc's path-quoting bug:
  ```powershell
  cmd /c mklink /J C:\rocm71 "C:\Program Files\AMD\ROCm\7.1"
  ```
- Git for Windows installed (provides `sh.exe`, needed because GNU Make falls
  back to `cmd.exe` otherwise, which can't parse the Makefile's POSIX recipes).
- MSVC Build Tools (`vcvars64.bat`) — on this machine located at:

## Per-terminal-session setup (do this every new PowerShell window)

```powershell
# 1. Put ROCm's hipcc/clang on PATH
$env:PATH = "C:\rocm71\bin;" + $env:PATH

# 2. Import MSVC's cl.exe/link.exe into this session (hipcc needs link.exe)
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set' |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] } }

# 3. Re-add ROCm's bin AFTER vcvars64 (it overwrites PATH)
$env:PATH = "C:\rocm71\bin;" + $env:PATH
```

## Build commands

```powershell
cd c

# Build the GPU DLL (coli_cuda.dll) via hipcc
make HIP_DLL=1 hip-dll ROCM_HOME="C:/rocm71" SHELL="C:/Program Files/Git/usr/bin/sh.exe"

# Build hy3.exe / glm.exe linked against the DLL-loader path
make hy3 HIP_DLL=1 ROCM_HOME="C:/rocm71" SHELL="C:/Program Files/Git/usr/bin/sh.exe"
make glm HIP_DLL=1 ROCM_HOME="C:/rocm71" SHELL="C:/Program Files/Git/usr/bin/sh.exe"

# Or both at once
make HIP_DLL=1 ROCM_HOME="C:/rocm71" SHELL="C:/Program Files/Git/usr/bin/sh.exe" glm hy3

# Plain CPU-only build (no GPU), for comparison/fallback
make glm hy3
```

### After building: fix the DLL search order

Windows may load a stale/generic `amdhip64.dll` (e.g. bundled with the AMD
Adrenalin graphics driver in System32) instead of the ROCm SDK's own runtime,
which makes the engine misreport the GPU (wrong name, wrong VRAM). Fix: copy
the SDK's HIP runtime DLL next to the built executables, so it's found in
`hy3.exe`'s own directory before Windows ever searches System32:

```powershell
Copy-Item "C:\rocm71\bin\amdhip64_7.dll" -Destination .
```

Verify with `where.exe amdhip64*.dll` — if a copy shows up outside
`C:\rocm71\bin`, that's the one being loaded incorrectly without this fix.

## Fixes applied on top of upstream (not yet merged)

1. **`c/backend_loader.c`**: added the missing `coli_cuda_gqa_attention`
   typedef/struct-field/`RESOLVE`/wrapper. Without it, linking `hy3.exe`
   against the Windows DLL-loader path (`CUDA_DLL=1`/`HIP_DLL=1`) fails with
   `undefined reference to 'coli_cuda_gqa_attention'`, since GLM doesn't use
   that symbol but Hy3 does.

## Environment variables for running with GPU offload

```powershell
$env:COLI_CUDA = "1"           # enable GPU path
$env:COLI_GPU = "0"            # device index (0 = RX 9070, confirmed via hipInfo.exe)
$env:CUDA_EXPERT_GB = "12"     # VRAM budget for offloaded experts — tune to the card
$env:CUDA_RELEASE_HOST = "1"   # free host RAM copy of dense tensors once resident on GPU
```

These are read directly by `hy3.exe`/`glm.exe`. `coli`'s `--gpu`/`--vram`/
`--auto-tier` flags also set/forward them if you prefer the Python CLI route.

## Running the real model (once converted/downloaded)

```powershell
cd c

python coli info --model D:\path\to\your\hy3_model
python coli doctor --model D:\path\to\your\hy3_model

python coli run --model D:\path\to\your\hy3_model "your prompt"
python coli chat --model D:\path\to\your\hy3_model --ram 40 --topp 0.7

python coli serve --model D:\path\to\your\hy3_model

# Web UI: build once, then this serves the API + auto-opens the dashboard
cd web; npm install; npm run build; cd ..\c
python coli web --model D:\path\to\your\hy3_model
```

## Quick validation without the real model (tiny random oracle)

Generates a tiny random-weight `hy_v3` model and checks the C engine's
forward pass against a PyTorch reference — validates build + engine + GPU
detection without waiting on a multi-hundred-GB download.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.6.0" safetensors

cd c
python tools\make_hy3_oracle.py
python tools\convert_hy3.py --indir hy3_tiny --outdir hy3_tiny_i4 --ebits 4 --n-layers 5
```

Run the engine's built-in teacher-forcing self-test directly (bypasses
`coli`/tokenizer entirely — this tiny oracle has no real tokenizer):

```powershell
Remove-Item Env:PROMPT, Env:SERVE, Env:IDS -ErrorAction SilentlyContinue
$env:SNAP = ".\hy3_tiny"; $env:TF = "1"; .\hy3.exe 64 16 16
```

Expect `PREFILL (teacher-forcing) C vs oracle: 32/32 positions`. With the GPU
env vars above also set, expect a line like:
```
[CUDA] device 0: AMD Radeon RX 9070, 17.1 GB VRAM, sm_120
```

Clear `SNAP`/`TF` (or open a fresh terminal) before switching back to real
`coli` commands, since leftover values will make `hy3.exe` try to load
`hy3_tiny` again.

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| Garbled `cmd.exe` errors, German locale text | Make fell back to `cmd.exe` | pass `SHELL="C:/Program Files/Git/usr/bin/sh.exe"` |
| `No rule to make target 'hip-dll'` | Running from repo root | `cd c` first |
| `hipcc not found` | ROCm `bin` not on PATH | prepend `C:\rocm71\bin` to `$env:PATH` |
| `link.exe (MSVC) not in PATH` | vcvars64 not imported | run the `cmd /c '"...vcvars64.bat" && set' \| ForEach-Object {...}` snippet above |
| `clang: error: no such file or directory: 'Files/AMD/...'` | Space in `ROCM_HOME` path breaks hipcc's internal re-exec | use the `C:\rocm71` junction, not the real `Program Files` path |
| `undefined reference to 'coli_cuda_gqa_attention'` | Missing wrapper in `backend_loader.c` | already fixed in this fork (see above) |
| `SNAP=<dir>` when running `hy3.exe` | Env var set in a different terminal session | set `SNAP`/`TF` and run in the same command line |
| `tokenizer.json is missing` when running the tiny oracle | Leftover `PROMPT`/`SERVE` env var routes into `run_text`/`run_serve` instead of the `TF` self-test | `Remove-Item Env:PROMPT, Env:SERVE, Env:IDS` before setting `SNAP`/`TF` |
| GPU reports as "AMD Radeon(TM) Graphics" with wrong VRAM | Wrong `amdhip64.dll` loaded (stale copy in System32) | copy `amdhip64_7.dll` from `C:\rocm71\bin` into `c/` next to `hy3.exe` |
