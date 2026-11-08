# Vulkan Neural Network Compute Acceleration · From Zero to Mastery Course

> A hands-on course with paired "code + documentation" that progresses chapter by chapter. Each chapter = an independently compilable and runnable project + an explanatory document.
> Goal: take you from "not even knowing what Vulkan is" to "being able to build your own ncnn-like / ggml-vulkan-like inference engine".

---

## Bilingual / 多语言支持

- **Docs**: each chapter ships both Chinese (`<chapter>.md`) and English (`<chapter>.en.md`); this README also has a [Chinese version `README.md`](README.md).
- **Program output**: all chapter executables print **Chinese** by default; pass `--lang en` (or set the env var `COURSE_LANG=en`) to switch to **English**. Example:

  ```bash
  # Executables have a .exe suffix on Windows, no suffix on Linux/macOS
  ./build/ch01_vector_add/ch01_vector_add            # Chinese output
  ./build/ch01_vector_add/ch01_vector_add --lang en  # English output
  ```

## 0. Course Positioning and Prerequisites

- **Language**: C++17 + GLSL (compute shaders). C++ uses `vulkan.hpp` (the official Vulkan C++ wrapper).
- **Platform**: Works on both Windows / Linux; requires **Vulkan SDK ≥ 1.2** and a GPU that supports compute.
- **Toolchain**: CMake ≥ 3.16, `glslc` (bundled with the Vulkan SDK, compiles `.comp` → `.spv`), Validation Layers (enable for debugging).
- **Reference comparisons**: At every stage we compare against the equivalent implementations in **ncnn** (mobile CV), **llama.cpp / ggml-vulkan** (desktop LLM), **ncnn-MoE-Runtime** (sparse MoE engine) and **ncnn-compiler** (MLIR AOT compiler). All four references were re-verified against the 2026-11-08 snapshots; see Section 4.

### Project Structure (Unified Template)

```
course/
├── README.md                 # This file: course overview
├── CMakeLists.txt            # Top level: pulls in all chapters
├── cmake/helpers.cmake       # add_chapter() macro: auto-compiles shaders, links Vulkan
├── common/                   # Shared Vulkan wrappers (evolves with the course, becoming more like a small engine over time)
│   ├── vk_base.h / .cpp      # instance/device/buffer/shader/upload-download/one-click submit
│   └── CMakeLists.txt
├── ch01_vector_add/          # Chapter N
│   ├── CMakeLists.txt
│   ├── main.cpp              # Chapter main program (focus on concepts; boilerplate handed to common)
│   ├── shaders/*.comp        # This chapter's compute shaders
│   └── docs/chNN_xxx.md      # This chapter's docs (concepts + code walkthrough + exercises)
└── ...
```

Build method (any single chapter or the whole thing):

```bash
cd course
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Run a chapter (shaders are auto-compiled into the build directory)
./build/ch01_vector_add/ch01_vector_add
```

---

## 1. Course Outline (Beginner → Mastery)

### Part 1: Environment and Minimal Loop (Getting Started)
| Chapter | Topic | What You'll Learn | Industrial Equivalent |
|------|------|----------|------------|
| **ch00** | Environment Setup and Device Enumeration | Install SDK, enable Validation Layers, enumerate GPU/queues, print device capabilities | The starting point of everything |
| **ch01** | First Compute Shader: Vector Addition | buffer creation, shader compile/load, dispatch, read back results, push constant | Minimal runnable kernel |
| **ch02** | Synchronization Model: fence and barrier | Hand-written command buffer, submit + wait fence, inter-op pipeline barrier (resolving read/write hazards) | ncnn's `VkCompute::submit_and_wait` |
| **ch03** | Memory Management Primer | Memory types (host-visible / device-local / coherent), staging upload/download, FP16 storage | ncnn `VkStagingAllocator` |

### Part 2: Neural Network Operators (Common Operator Implementations)
| Chapter | Topic | What You'll Learn | Industrial Equivalent |
|------|------|----------|------------|
| **ch04** | Matrix Multiply GEMM | 2D block tiling, shared memory, coalesced access, timestamp timing, **Cooperative Matrix acceleration** | ncnn/llama matmul core |
| **ch05** | Convolution | Direct convolution → im2col+GEMM → **Winograd F(2×2,3×3)** three implementations, NC4HW4 packing | ncnn convolution family |
| **ch06** | Element-wise Operators | ReLU / GELU / SiLU, bandwidth-bound operator optimization | Fused activation |
| **ch07** | Normalization | LayerNorm / RMSNorm, reduction and workgroup cooperation | Transformer normalization |
| **ch08** | Softmax and Attention | Numerically stable softmax, 3-kernel simplified attention, **FlashAttention fused kernel** | FlashAttention |

### Part 3: Performance, Quantization, and Engineering (Mastery)
| Chapter | Topic | What You'll Learn | Industrial Equivalent |
|------|------|----------|------------|
| **ch09** | Performance Analysis Methodology | GPU timestamp query, Roofline model, Nsight/RGP measurements | Operator-level profiling |
| **ch10** | Advanced Memory Management | Memory pool / blob allocator, persistent weights, buffer reuse | ncnn `VkBlobAllocator` |
| **ch11** | Quantization and Fused Dequantization | int8 + **Q4_0/Q8_0** weights, in-place dequantization on GPU, fused matmul | llama.cpp `dequant_funcs` |
| **ch12** | Graph Execution and Scheduling | Record commands, deferred barrier, batched submission, multi-threaded queues | ggml compute graph |
| **ch13** | Capstone Project | Run a small model (MNIST or TinyLLM) with your own engine, compare against ncnn/llama.cpp | Complete inference engine |

### Part 4: Advanced GPU Programming and Frontier Technology (Mastery → Expert)
| Chapter | Topic | What You'll Learn | Industrial Equivalent |
|------|------|----------|------------|
| **ch14** | LLM Inference Engine | Full LLM inference pipeline (tokenizer→embedding→transformer→output), CPU reference implementation | ncnn / llama.cpp full engine |
| **ch15** | **Cooperative Matrix Programming** | Tensor Core programming, `coopmatMulAdd`, matrix fragments, specialization constant handshake, hardware constraints | ncnn `gemm_cm.comp` / ggml `mul_mm_cm2.comp` |
| **ch16** | **Subgroup Programming** | Warp-level ops: shuffle/broadcast/reduce/ballot, warp-level softmax, subgroup size control | ncnn/ggml subgroup optimizations |
| **ch17** | **FlashAttention Deep Dive** | Online softmax, tiling strategy, subgroup cooperative loading, fused kernel design, **KV cache incremental decode (split/reduce) advanced demo** | ncnn `sdpa_fa.comp` / ggml `flash_attn.comp` + `flash_attn_decode_phase_*` |
| **ch18** | **Deep Quantization** | Q4_0/Q8_0 block-wise quantization, fused dequantization GEMM, compression vs accuracy trade-offs | ggml `dequant_funcs` |
| **ch19** | **MoE Inference Engine** | Router (top-k), Expert dispatch, Combine, load balancing, **sparse Gather/Scatter advanced demo**, GPU MoE inference | ncnn-MoE-Runtime `forward_router/forward_moe` / ggml `ggml_vk_mul_mat_id` + `topk_moe` |
| **ch20** | **Frontier Quantization Formats** | K-quant family (Q2_K~Q6_K), MXFP4 microscaling (E2M1), **NVFP4 two-level E4M3 advanced demo**, super-block structure | ggml's 44 types: NVFP4 / TQ / Q1_0 / Q2_0 / F8_E4M3 |

---

## 2. Current Progress (All Complete ✅)

- ✅ **Course design overview** (this file)
- ✅ **common/ shared wrapper library** (evolves with the course: `vkInit/Buffer/upload-download` → `ComputePass` (mini-VkCompute) → `DeviceArena` (memory pool) → `Graph` (compute graph))
- ✅ **ch00 environment enumeration** (device/queue/memory/limits/feature probing report)
- ✅ **ch01 vector addition** (minimal compute loop: buffer → upload → pipeline → dispatch → read back)
- ✅ **ch02 synchronization model** (hand-written fence + pipeline barrier, resolving read/write hazards)
- ✅ **ch03 memory management** (memory types / staging transfer / FP16 storage buffer round-trip)
- ✅ **ch04 GEMM** (2D block tiling + shared memory, first use of ComputePass)
- ✅ **ch05 convolution** (direct convolution vs im2col+GEMM, result-consistency verification)
- ✅ **ch06 element-wise operators** (ReLU/GELU/SiLU parameterized kernel + fusion motivation)
- ✅ **ch07 normalization** (RMSNorm, intra-workgroup shared-memory reduction)
- ✅ **ch08 Softmax/attention** (numerically stable softmax + 3-kernel simplified attention)
- ✅ **ch09 performance analysis** (GPU timestamp timing + Roofline + vendor tools)
- ✅ **ch10 advanced memory management** (DeviceArena bump allocator, persistent weights / activation reuse)
- ✅ **ch11 quantization and fused dequantization** (int8 weights + in-kernel in-place dequantize GEMM)
- ✅ **ch12 graph execution and scheduling** (multiple operators recorded into one command buffer + barrier + single submit)
- ✅ **ch13 capstone project** (two-layer MLP forward: persistent weights + GEMM + bias + ReLU + graph)
- ✅ **ch14 LLM inference engine** (Tokenizer + Embedding + Transformer Block + output sampling, CPU reference implementation)
- ✅ **ch15 Cooperative Matrix programming** (Tensor Core GEMM, `coopmatMulAdd`, specialization constant handshake)
- ✅ **ch16 Subgroup programming** (warp-level ops: shuffle/broadcast/reduce/ballot, warp-level softmax)
- ✅ **ch17 FlashAttention deep dive** (online softmax, tiling strategy, fused kernel) + **KV cache incremental decode demo** (`ch17_kvcache`, two-phase split/reduce, see ch17 docs 17.8)
- ✅ **ch18 deep quantization** (Q4_0/Q8_0 block-wise quantization, fused dequantization GEMM)
- ✅ **ch19 MoE inference engine** (Router→Expert→Combine, top-k routing, GPU MoE inference) + **sparse Gather/Scatter advanced demo** (`ch19_moe_sparse`, a runnable reference for exercise #3, see ch19 docs 19.11)
- ✅ **ch20 frontier quantization formats** (K-quant family Q2_K~Q6_K, MXFP4 microscaling E2M1) + **NVFP4 advanced demo** (`ch20_nvfp4`, two-level E4M3 scale comparison, see ch20 docs)

> All **21 chapters (ch00–ch20)** pass **glslc shader compilation** and **full CMake C++ build** (`cmake -B build && cmake --build build` produces **24 executables**: 21 chapter programs + three advanced demos `ch19_moe_sparse` / `ch17_kvcache` / `ch20_nvfp4`). Running requires a Vulkan GPU with compute support and its driver; for debugging, enable Validation Layers (see ch00).

Each chapter's `docs/` contains a three-part structure of "concept explanation → line-by-line code breakdown → exercises/advanced". It is recommended to **read the docs first, then main.cpp, then change one parameter yourself and rerun**.

---

## 3. Study Recommendations

1. Do not skip ch01~ch03: the "command buffer / synchronization / memory" mental model they build is used in every later chapter.
2. Always **run each chapter with Validation Layers enabled** — 90% of "wrong results" are barrier or memory-visibility issues, which the Layers will report directly.
3. Read your own implementation alongside the chapter's labeled "industrial equivalent": you'll understand why ncnn / llama.cpp is written that way.
4. True mastery comes from the ch13 capstone project: when you can run a small model with your own engine and compare latency against ncnn/llama.cpp, the course goal is achieved.

---

## 4. Reference Resources

### Project Origin

The idea for this course started from [**AiChiTuDouPian/vulkan-nn**](https://github.com/AiChiTuDouPian/vulkan-nn) ("Vulkan Neural Network Compute Acceleration · From Zero to Mastery Course").

### Reference Industrial Implementations (re-verified 2026-11-08)

Each chapter labels its industrial counterpart. All three anchors from the first release (2026-08-19) are now stale; this pass re-verified them against the current snapshots and added the previously missing ncnn-compiler:

| Repository | Local path | Version / commit | Key features |
|------|----------|-------------|---------|
| **ncnn** | `C:\msys64\home\14616\fork\NCNNs\ncnn` | `20261020-52-g00357e2f` (2026-11-08) | **17** `*_cm.comp` shaders under `src/layer/vulkan/shader/` (GEMM/conv/Winograd/SDPA/int8/bf16); 6 SDPA shaders (FA / CM / cross-attention / KV cache); host-side MNK selection via `get_optimal_cooperative_matrix_mnk()`; **fp32 accumulation + subgroup** reductions; persistent PipelineCache; multi-queue `acquire_queue/reclaim_queue` |
| **ncnn-compiler** | `...\ncnn-compiler` | `867c6ab` (2026-10-31) | MLIR/LLVM AOT compiler; `--backend=vulkan` emits a **model-specific fixed-run** `.so` + C header: embedded SPIR-V/weights, generated `create/run/destroy` lifecycle, static-graph barriers, command buffer re-recorded each run; four verification profiles (release / sanitize / offline / software; `run_gate.py` also defines a `hardware` gate variant); no dynamic shapes, no transparent CPU fallback |
| **ncnn-MoE-Runtime** | `...\ncnn-MoE-Runtime` | `63745f1` (2026-11-07, v0.1.0) | C++20 CPU/Vulkan sparse MoE: **immutable execution graph**, expert weights outside the dense working set, ARC + Victim + Residency caches, HybridMode, BatchScheduler multi-session / speculative decoding; 8 model families incl. Qwen3 GGUF import; 60 Vulkan kernels under `src/kernels/vulkan/` (MXFP4 quintet, `router_select`, `expert_gather`, `moe_combine`…) |
| **llama.cpp / ggml-vulkan** | `...\llama.cpp` | `a691b23` (2026-11-08) | 155 `.comp` templates (plus 30 `.glsl` snippets) → SPIR-V generated at **build time** by `vulkan-shaders-gen` + glslc (no prebuilt `.spv`); **44** quantization types (new: Q1_0/Q2_0/TQ1_0/TQ2_0/MXFP4/NVFP4/F8_E4M3); FA split into `flash_attn_cm1/cm2/decode_phase_1/2/split_k_reduce/sparse_compact`; `ggml_vk_mul_mat_id` + `topk_moe/count_experts`; new WebGPU backend |

> Since 2026-08-19 the four repositories gained **147 / 106 / 192 / 1696** commits respectively. The "industrial comparison" sections in every chapter have been synced to the current paths, symbols and feature lists.

### Sync notes: stale → current (use the right column when reading upstream source)

| Old claim (first release) | Current implementation |
|------|------|
| `VulkanDevice::init_device()` | `ncnn::create_gpu_instance()` + `get_gpu_device()->info`; extension probing is `init_device_extension()` |
| `VkCompute::dispatch()` | `VkCompute::record_pipeline(...)` (recording) + `submit_and_wait()`; `delayed_records` / `pending_dispatch_total()` still exist |
| `VkBlobAllocator::stage_buffer()` / `VkStagingAllocator::current()` | Allocators use `fastMalloc()`; `VkBlobAllocator` (16MB default pool), `VkWeightAllocator` (8MB), `VkStagingAllocator` all retained |
| `vk-device.cpp` / `pipeline.cpp` / `dev->device_vk` | `ggml-vulkan.cpp` + `ggml-vulkan-types.h` (`vk_device_struct` / `vk_pipeline_struct`); `vk_pipeline_wrapper` no longer exists |
| `ggml::ggml_vk::graph_compute()` | `ggml_backend_vk_graph_compute()` in `ggml-vulkan.cpp` |
| `ggml/iq-quants.c` | `ggml/src/ggml-quants.c` + `ggml/src/ggml-vulkan/vulkan-shaders/dequant_*.comp` |
| `ncnn/src/layer/vulkan/shaders/` (plural) | `ncnn/src/layer/vulkan/shader/` (singular) |
| ncnn-MoE-Runtime `ExpertDispatcher` | **Removed**; now `graph/router.h::forward_router`, `engine/expert.h::forward_moe`, `backends/ncnn/{router_vulkan,moecombine_vulkan}.h` |
| "20+ quant formats / 15+ CM shaders" | 44 quant types / 17 `*_cm.comp` shaders |
| "FlashAttention 3 paths in one file" | 6 `sdpa_*.comp` in ncnn; 8 `flash_attn*.comp` variants + 3 `.glsl` snippets in ggml |
| "MoE = MUL_MAT_ID only" | `ggml_vk_mul_mat_id` + fused `topk_moe.comp` / `count_experts.comp` routing |

### Frontier Technologies in This Course from Research

| Technology | Source (current path) | Course Chapter |
|------|---------|---------|
| Cooperative Matrix | ncnn `shader/gemm_cm.comp` / ggml `vulkan-shaders/mul_mm_cm2.comp` | ch04 (extended) + ch15 |
| Subgroup Operations | ncnn `shader/reduction.comp` / ggml `flash_attn.comp` | ch16 |
| FlashAttention | ncnn `shader/sdpa_fa.comp` / ggml `vulkan-shaders/flash_attn.comp` | ch08 (extended) + ch17 |
| Q4_0/Q8_0 Quantization | ggml `vulkan-shaders/dequant_funcs.glsl` + `ggml-quants.c` | ch11 (extended) + ch18 |
| K-quant / MXFP4 / NVFP4 | Type enum in `ggml/include/ggml.h` + `dequant_*.comp` | ch20 |
| Winograd Convolution | ncnn `shader/convolution_*winograd*.comp` + `layer/vulkan/convolution_vulkan.cpp` | ch05 (extended) |
| MoE Inference | ncnn-MoE-Runtime `src/kernels/vulkan/` / ggml `ggml_vk_mul_mat_id` | ch19 |
| Fixed-graph AOT / command-buffer replay | ncnn-compiler `docs/ncnn-vulkan-backend-contract.md` + `tools/VulkanLifecycleCodegen.cpp` | ch12 (new subsection) |
