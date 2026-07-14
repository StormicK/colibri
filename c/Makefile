# --- Target detection ---
# Ask the compiler what it actually targets (rofl0r's note on #129): a gcc/clang
# toolchain reports its target triple via `-dumpmachine`, e.g.
#   x86_64-w64-mingw32   x86_64-pc-cygwin   x86_64-unknown-linux-gnu
#   arm64-apple-darwin   powerpc64le-unknown-linux-gnu
# The triple follows the toolchain, not the host shell that `uname` reports, so
# it is correct under a native-Windows shell (no `uname` on PATH) AND when
# cross-compiling (e.g. make CC=x86_64-w64-mingw32-gcc on Linux). CC is chosen
# per-target below, so probe with the user's CC if they set one, else gcc
# (present on Linux and mingw, and a clang shim on macOS).
DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)

MINGW  := $(findstring mingw,$(TRIPLET))
CYGWIN := $(findstring cygwin,$(TRIPLET))
DARWIN := $(findstring darwin,$(TRIPLET))
PPC64  := $(findstring powerpc64,$(TRIPLET))
LINUX  := $(findstring linux,$(TRIPLET))
IS_WIN := $(MINGW)$(CYGWIN)

# Fallbacks for the rare toolchain that does not answer -dumpmachine: keep the
# #129 signal (OS=Windows_NT, set in every Windows shell) for Windows, then
# `uname` for everything else, so no host regresses.
ifeq ($(TRIPLET),)
IS_WIN := $(if $(filter Windows_NT,$(OS)),1,)
ifeq ($(IS_WIN),)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
DARWIN  := $(findstring Darwin,$(UNAME_S))
PPC64   := $(findstring ppc64,$(UNAME_M))
endif
endif

ifneq (,$(DARWIN))
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# Opt-in: ARCH=native appends -mcpu=native (arm64 clang uses -mcpu, not -march),
# which unlocks the i8mm SMMLA int8/int4 dot kernels in glm.c. ARCH unset ->
# no -mcpu, default build byte-identical. Apple clang knows apple-m4 / native.
ifneq ($(ARCH),)
CFLAGS += -mcpu=$(ARCH)
endif
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics - tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (portable binary with AVX2). For max speed on THIS
# machine use ARCH=native: on AVX-VNNI CPUs (Intel Alder Lake+, Meteor Lake+)
# it also unlocks the 128-bit VPDPBUSD int8/int4 dot kernel (dot_i8i8/dot_i4i8),
# which the x86-64-v3 baseline does not define. The #ifdef guards in glm.c mean
# a v3 build simply compiles out the VNNI path - safe on any x86-64.
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static
EXE     = .exe
else
ifneq (,$(PPC64))
# --- Linux PowerPC (POWER8/POWER9/POWER10) ---
# PowerPC GCC uses -mcpu, not -march. ARCH=native works on gcc >= 4.7.
# The AVX2/NEON kernels fall back to the portable scalar C path
# (validated token-exact vs the transformers oracle on a POWER8 S824).
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -mcpu=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
else
# --- Linux / *BSD x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
# -pthread: Linux lo tira dentro via -fopenmp, i *BSD no e le pthread_* non risolvono (#219).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
endif
endif

# --- install ---
PREFIX     ?= /usr/local
BINDIR     ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/colibri
INSTALL    ?= install

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
#
# Two paths:
#   - Linux/macOS: CUDA=1 links backend_cuda.o directly (cudart via -l).
#   - Windows:      CUDA_DLL=1 builds a standalone coli_cuda.dll (nvcc+MSVC),
#                   then the host glm.exe loads it at runtime via backend_loader.c
#                   (LoadLibrary/GetProcAddress). MinGW gcc cannot compile .cu
#                   (nvcc needs cl.exe), and cross-linking MSVC objects into a
#                   gcc binary is fragile — the DLL split keeps the toolchains
#                   clean. See backend_loader.c and README "cuda-dll" below.
CUDA      ?= 0
CUDA_DLL  ?= 0
ifneq ($(IS_WIN),)
# the CUDA installer sets CUDA_PATH system-wide (e.g. C:\Program Files\NVIDIA
# GPU Computing Toolkit\CUDA\v13.2); fall back to it before the POSIX default.
# NVCC defaults to plain `nvcc` from PATH: CUDA_PATH contains spaces, which the
# unquoted recipe checks cannot survive — and the cuda-dll recipe already
# requires an MSVC environment (vcvars64) on PATH, so requiring the CUDA bin
# directory too is symmetric. The installer adds it by default.
CUDA_HOME ?= $(subst \,/,$(CUDA_PATH))
NVCC      ?= nvcc
# Host compiler override for nvcc (e.g. Fedora ships g++16 but CUDA needs 15):
#   make glm CUDA=1 NVCC_CCBIN=g++-15        (opt-in: unset = nvcc's own default)
ifneq ($(NVCC_CCBIN),)
NVCCFLAGS += -ccbin $(NVCC_CCBIN)
endif
else
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
endif
CUDA_ARCH ?= native
ifneq ($(IS_WIN),)
# nvcc's host compiler on Windows is MSVC cl.exe: the GCC-style
# -Xcompiler=-Wall,-Wextra is rejected ("D8021 invalid numeric argument
# '/Wextra'"), which made `make cuda-dll` unbuildable as shipped. -W3 is
# the MSVC warning level (dash form, NOT /W3: MSYS make would mangle the
# slash-form into a filesystem path).
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-W3
else
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
endif
# PYTHON is a HOST tool (clean/test-c/test-python run it on the build machine),
# so key it off the host, NOT $(IS_WIN) — which is derived from the *target*
# triple ($(CC) -dumpmachine). Otherwise a cross build (make CC=x86_64-w64-
# mingw32-gcc ... on Linux) sets IS_WIN and picks `python`, breaking clean/test
# on hosts where only `python3` exists. $(OS)=Windows_NT in every Windows shell
# (the #129 signal) and is empty on Linux/macOS.
ifeq ($(OS),Windows_NT)
PYTHON    ?= python
else
PYTHON    ?= python3
endif
CUDA_OBJ  =
V4_TEST_BINS = tests/test_model_interfaces$(EXE) \
	tests/test_safetensors_index$(EXE) tests/test_native_quant$(EXE) \
	tests/test_tensor_io$(EXE) tests/test_deepseek_v4$(EXE)
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) \
	tests/test_grammar$(EXE) tests/test_schema_gbnf$(EXE) \
	tests/test_decode_batch$(EXE) tests/test_idot$(EXE) \
	tests/test_kv_alloc$(EXE) tests/test_i4_acc512$(EXE) \
	tests/test_compat_direct$(EXE) $(V4_TEST_BINS)
ifneq (,$(LINUX))
TEST_BINS += tests/test_uring$(EXE)
endif

# Windows CUDA DLL path: host links the loader, NOT cudart.
ifneq ($(IS_WIN),)
ifeq ($(CUDA_DLL),1)
CFLAGS  += -DCOLI_CUDA
CUDA_OBJ = backend_loader.o
endif
endif

# Linux CUDA direct-link path (unchanged).
ifeq ($(CUDA),1)
ifneq (,$(DARWIN))
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
# On Windows use CUDA_DLL=1 (runtime DLL), not CUDA=1 (direct link).
$(error On Windows use: make CUDA_DLL=1 cuda-dll  (see backend_loader.c))
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

# METAL=1 adds an opt-in Apple-GPU backend (macOS only). The shader is compiled at
# runtime, so no Xcode / offline metal compiler is required. Default build unchanged.
METAL     ?= 0
METAL_OBJ  =
METALXX    = clang++ -x objective-c++ -fobjc-arc -O3
ifeq ($(METAL),1)
ifeq (,$(DARWIN))
$(error METAL=1 is supported only on macOS)
endif
CFLAGS   += -DCOLI_METAL
LDFLAGS  += -framework Metal -framework Foundation -lc++
METAL_OBJ = backend_metal.o
endif

all: glm$(EXE)

.PHONY: deepseek-v4
deepseek-v4:
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4

# phony 'glm' → 'glm.exe' on Windows (so 'make glm' and 'coli build' work on every platform)
glm: glm$(EXE)

glm$(EXE): glm.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h $(CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) $(METAL_OBJ) -o glm$(EXE) $(LDFLAGS)

# Windows runtime loader object: resolves coli_cuda_* from coli_cuda.dll.
backend_loader.o: backend_loader.c backend_cuda.h compat.h
	$(CC) $(CFLAGS) -c backend_loader.c -o $@

# Windows CUDA DLL: compile backend_cuda.cu with nvcc (+MSVC cl.exe as host
# compiler, required by nvcc on Windows) into coli_cuda.dll. Run this from a
# shell that has the MSVC environment set (e.g. after vcvars64.bat, or from a
# "x64 Native Tools Command Prompt"). COLI_CUDA_BUILDING_DLL enables
# __declspec(dllexport) so the 15 API symbols are exported.
cuda-dll: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	@command -v cl >/dev/null 2>&1 || { echo "cl.exe (MSVC) not in PATH — run vcvars64.bat first" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -shared -DCOLI_CUDA_BUILDING_DLL \
		-L"$(CUDA_HOME)/lib/x64" -lcudart \
		backend_cuda.cu -o coli_cuda.dll

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

backend_metal.o: backend_metal.mm backend_metal.h
	$(METALXX) -c backend_metal.mm -o $@

metal-test: tests/test_backend_metal.mm backend_metal.mm backend_metal.h
	$(METALXX) tests/test_backend_metal.mm backend_metal.mm -framework Metal -framework Foundation -o backend_metal_test
	./backend_metal_test

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

cuda-bench: backend_cuda.cu backend_cuda.h tests/bench_tensor_core.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/bench_tensor_core.cu -o backend_cuda_bench$(EXE)
	./backend_cuda_bench$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_schema_gbnf$(EXE): tests/test_schema_gbnf.c schema_gbnf.h grammar.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_decode_batch$(EXE): tests/test_decode_batch.c decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_idot$(EXE): tests/test_idot.c glm.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_alloc$(EXE): tests/test_kv_alloc.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_acc512$(EXE): tests/test_i4_acc512.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_compat_direct$(EXE): tests/test_compat_direct.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_uring$(EXE): tests/test_uring.c glm.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_model_interfaces$(EXE): tests/test_model_interfaces.c model.h expert_store.h tensor.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_safetensors_index$(EXE): tests/test_safetensors_index.c safetensors_index.c safetensors_index.h json.h
	$(CC) $(CFLAGS) safetensors_index.c $< -o $@ $(LDFLAGS)

tests/test_native_quant$(EXE): tests/test_native_quant.c native_quant.c native_quant.h tensor.h
	$(CC) $(CFLAGS) native_quant.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_expert_store$(EXE): tests/test_deepseek_v4_expert_store.c deepseek_v4_expert_store.c deepseek_v4_expert_store.h safetensors_index.c safetensors_index.h expert_store.h tensor.h json.h
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_expert_store.c $< \
		-o $@ -pthread $(LDFLAGS)

tests/test_deepseek_v4_math$(EXE): tests/test_deepseek_v4_math.c deepseek_v4_math.c deepseek_v4_math.h
	$(CC) $(CFLAGS) deepseek_v4_math.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_expert$(EXE): tests/test_deepseek_v4_expert.c deepseek_v4_expert.c deepseek_v4_expert.h deepseek_v4_math.c deepseek_v4_math.h native_quant.c native_quant.h expert_store.h tensor.h
	$(CC) $(CFLAGS) native_quant.c deepseek_v4_math.c deepseek_v4_expert.c $< \
		-o $@ $(LDFLAGS)

tests/test_deepseek_v4_config$(EXE): tests/test_deepseek_v4_config.c deepseek_v4_config.c deepseek_v4_config.h json.h
	$(CC) $(CFLAGS) deepseek_v4_config.c $< -o $@ $(LDFLAGS)

tests/test_tensor_io$(EXE): tests/test_tensor_io.c tensor_io.c tensor_io.h native_quant.c native_quant.h safetensors_index.c safetensors_index.h tensor.h json.h
	$(CC) $(CFLAGS) safetensors_index.c native_quant.c tensor_io.c $< \
		-o $@ $(LDFLAGS)

tests/test_deepseek_v4_layer$(EXE): tests/test_deepseek_v4_layer.c deepseek_v4_layer.c deepseek_v4_layer.h deepseek_v4_config.h safetensors_index.c safetensors_index.h json.h
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_layer.c $< \
		-o $@ $(LDFLAGS)

tests/test_deepseek_v4_sparse_attention$(EXE): tests/test_deepseek_v4_sparse_attention.c deepseek_v4_sparse_attention.c deepseek_v4_sparse_attention.h native_quant.c native_quant.h
	$(CC) $(CFLAGS) deepseek_v4_sparse_attention.c native_quant.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_kv_cache$(EXE): tests/test_deepseek_v4_kv_cache.c deepseek_v4_kv_cache.c deepseek_v4_kv_cache.h
	$(CC) $(CFLAGS) deepseek_v4_kv_cache.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_attention_cache$(EXE): tests/test_deepseek_v4_attention_cache.c deepseek_v4_attention_cache.c deepseek_v4_attention_cache.h deepseek_v4_kv_cache.c deepseek_v4_kv_cache.h deepseek_v4_sparse_attention.c deepseek_v4_sparse_attention.h native_quant.c native_quant.h
	$(CC) $(CFLAGS) deepseek_v4_attention_cache.c deepseek_v4_kv_cache.c \
		deepseek_v4_sparse_attention.c native_quant.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_resource_plan$(EXE): tests/test_deepseek_v4_resource_plan.c deepseek_v4_resource_plan.c deepseek_v4_resource_plan.h
	$(CC) $(CFLAGS) deepseek_v4_resource_plan.c $< -o $@ $(LDFLAGS)

tests/test_deepseek_v4_prompt$(EXE): tests/test_deepseek_v4_prompt.c deepseek_v4_prompt.c deepseek_v4_prompt.h
	$(CC) $(CFLAGS) deepseek_v4_prompt.c $< -o $@ $(LDFLAGS)
probe-v4-layers: tools/probe_deepseek_v4_layers$(EXE)
	./tools/probe_deepseek_v4_layers$(EXE) $(MODEL) $(LAYER)

tools/probe_deepseek_v4_layers$(EXE): tools/probe_deepseek_v4_layers.c deepseek_v4_layer.c deepseek_v4_config.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c $< \
		-o $@ $(LDFLAGS)

probe-v4-attention: tools/probe_deepseek_v4_attention$(EXE)
	./tools/probe_deepseek_v4_attention$(EXE) $(MODEL) $(TOKEN) $(OUTPUT)

tools/probe_deepseek_v4_attention$(EXE): tools/probe_deepseek_v4_attention.c deepseek_v4_attention.c deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_sparse_attention.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_sparse_attention.c \
		deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_attention.c $< -o $@ $(LDFLAGS)

probe-v4-block: tools/probe_deepseek_v4_block$(EXE)
	./tools/probe_deepseek_v4_block$(EXE) $(MODEL) $(TOKEN) $(OUTPUT)

tools/probe_deepseek_v4_block$(EXE): tools/probe_deepseek_v4_block.c deepseek_v4_block.c deepseek_v4_attention.c deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_sparse_attention.c deepseek_v4_expert.c deepseek_v4_expert_store.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_sparse_attention.c \
		deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_attention.c \
		deepseek_v4_expert.c deepseek_v4_expert_store.c deepseek_v4_block.c $< \
		-o $@ -pthread $(LDFLAGS)

probe-v4-compressor: tools/probe_deepseek_v4_compressor$(EXE)
	./tools/probe_deepseek_v4_compressor$(EXE) $(MODEL) $(OUTPUT)

tools/probe_deepseek_v4_compressor$(EXE): tools/probe_deepseek_v4_compressor.c deepseek_v4_compressor.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_compressor.c $< -o $@ $(LDFLAGS)

probe-v4-ratio4-compressor: tools/probe_deepseek_v4_ratio4_compressor$(EXE)
	./tools/probe_deepseek_v4_ratio4_compressor$(EXE) $(MODEL) $(OUTPUT)

tools/probe_deepseek_v4_ratio4_compressor$(EXE): tools/probe_deepseek_v4_ratio4_compressor.c deepseek_v4_compressor.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_compressor.c $< -o $@ $(LDFLAGS)

probe-v4-indexer: tools/probe_deepseek_v4_indexer$(EXE)
	./tools/probe_deepseek_v4_indexer$(EXE) $(MODEL) $(OUTPUT)

tools/probe_deepseek_v4_indexer$(EXE): tools/probe_deepseek_v4_indexer.c deepseek_v4_indexer.c deepseek_v4_indexer.h deepseek_v4_compressor.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_compressor.c \
		deepseek_v4_indexer.c $< -o $@ $(LDFLAGS)

probe-v4-attention-sequence: tools/probe_deepseek_v4_attention_sequence$(EXE)
	./tools/probe_deepseek_v4_attention_sequence$(EXE) $(MODEL) $(OUTPUT)

tools/probe_deepseek_v4_attention_sequence$(EXE): tools/probe_deepseek_v4_attention_sequence.c deepseek_v4_attention.c deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_sparse_attention.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c deepseek_v4_sparse_attention.c \
		deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_attention.c $< -o $@ $(LDFLAGS)

deepseek-v4-first-token: tools/deepseek_v4_first_token$(EXE)
	./tools/deepseek_v4_first_token$(EXE) $(MODEL) $(TOKEN) $(COUNT)

deepseek-v4-tokenizer: tools/deepseek_v4_tokenizer$(EXE)
	./tools/deepseek_v4_tokenizer$(EXE) $(MODEL) "$(TEXT)"

tools/deepseek_v4_tokenizer$(EXE): tools/deepseek_v4_tokenizer.c tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tools/deepseek_v4_first_token$(EXE): tools/deepseek_v4_first_token.c tok.h tok_unicode.h json.h deepseek_v4_block.c deepseek_v4_attention.c deepseek_v4_sparse_attention.c deepseek_v4_compressor.c deepseek_v4_indexer.c deepseek_v4_expert.c deepseek_v4_expert_store.c deepseek_v4_layer.c deepseek_v4_config.c deepseek_v4_math.c native_quant.c tensor_io.c safetensors_index.c
	$(CC) $(CFLAGS) safetensors_index.c deepseek_v4_config.c deepseek_v4_layer.c \
		deepseek_v4_math.c native_quant.c tensor_io.c deepseek_v4_attention.c \
		deepseek_v4_compressor.c deepseek_v4_indexer.c \
		deepseek_v4_sparse_attention.c deepseek_v4_expert.c \
		deepseek_v4_expert_store.c deepseek_v4_block.c $< \
		-o $@ -pthread $(LDFLAGS)
# Merged DeepSeek V4 unit tests — objects from amalgamated units (test-only + deps).
include Makefile.deepseek-v4.units
V4_TEST_LINK_OBJS = \
	COLI_V4_UNIT_MATH.o \
	COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o \
	COLI_V4_UNIT_PROMPT.o \
	$(addsuffix .o,$(V4_TEST_UNITS)) \
	safetensors_index.o native_quant.o

tests/test_deepseek_v4$(EXE): tests/test_deepseek_v4.c deepseek_v4.c deepseek_v4.h \
		deepseek_v4_dspark.h Makefile.deepseek-v4.units
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4-test-objs \
		COLI_V4_UNIT_MATH.o COLI_V4_UNIT_SPARSE_ATTENTION.o \
		COLI_V4_UNIT_RESOURCE_PLAN.o COLI_V4_UNIT_PROMPT.o \
		safetensors_index.o
	$(CC) $(CFLAGS) native_quant.c -c -o native_quant.o
	$(CC) $(CFLAGS) $< $(V4_TEST_LINK_OBJS) -o $@ -pthread $(LDFLAGS)

test-c: $(TEST_BINS)
	$(PYTHON) tools/run_tests.py $(TEST_BINS)

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

install: glm$(EXE) olmoe$(EXE)
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 glm$(EXE) $(DESTDIR)$(LIBEXECDIR)/glm$(EXE)
	$(INSTALL) -m 755 olmoe$(EXE) $(DESTDIR)$(LIBEXECDIR)/olmoe$(EXE)
	$(INSTALL) -m 644 resource_plan.py doctor.py openai_server.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/coli
	rm -rf $(DESTDIR)$(LIBEXECDIR)

clean:
	$(PYTHON) tools/clean.py

bench: iobench$(EXE)
	@if [ -n "$(ARGS)" ]; then ./iobench$(EXE) $(ARGS); else echo "built iobench$(EXE) — run: ./iobench$(EXE) <file> <MB> <iters> <threads> <direct 0|1>"; fi
.PHONY: all glm deepseek-v4 cuda-test cuda-bench cuda-dll portable test-c test-python test check clean install uninstall bench \
	probe-v4-layers probe-v4-attention probe-v4-block probe-v4-compressor \
	probe-v4-ratio4-compressor probe-v4-indexer probe-v4-attention-sequence \
	deepseek-v4-first-token deepseek-v4-tokenizer
