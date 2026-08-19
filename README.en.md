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
- **Reference comparisons**: At every stage we compare against the equivalent implementations in **ncnn** (mobile CV) and **llama.cpp / ggml-vulkan** (desktop LLM), showing you how industrial-grade code is written.

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
| **ch17** | **FlashAttention Deep Dive** | Online softmax, tiling strategy, subgroup cooperative loading, fused kernel design | ncnn `sdpa_fa.comp` / ggml `flash_attn.comp` |
| **ch18** | **Deep Quantization** | Q4_0/Q8_0 block-wise quantization, fused dequantization GEMM, compression vs accuracy trade-offs | ggml `dequant_funcs` |
| **ch19** | **MoE Inference Engine** | Router (top-k), Expert dispatch, Combine, load balancing, GPU MoE inference | ncnn-MoE-Runtime / ggml `MUL_MAT_ID` |
| **ch20** | **Frontier Quantization Formats** | K-quant family (Q2_K~Q6_K), MXFP4 microscaling (E2M1), super-block structure | ggml latest quantization formats |

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
- ✅ **ch17 FlashAttention deep dive** (online softmax, tiling strategy, fused kernel)
- ✅ **ch18 deep quantization** (Q4_0/Q8_0 block-wise quantization, fused dequantization GEMM)
- ✅ **ch19 MoE inference engine** (Router→Expert→Combine, top-k routing, GPU MoE inference)
- ✅ **ch20 frontier quantization formats** (K-quant family Q2_K~Q6_K, MXFP4 microscaling E2M1)

> All **20 chapters** pass **glslc shader compilation** and **full CMake C++ build** (`cmake -B build && cmake --build build` produces 20 executables). Running requires a Vulkan GPU with compute support and its driver; for debugging, enable Validation Layers (see ch00).

Each chapter's `docs/` contains a three-part structure of "concept explanation → line-by-line code breakdown → exercises/advanced". It is recommended to **read the docs first, then main.cpp, then change one parameter yourself and rerun**.

---

## 3. Study Recommendations

1. Do not skip ch01~ch03: the "command buffer / synchronization / memory" mental model they build is used in every later chapter.
2. Always **run each chapter with Validation Layers enabled** — 90% of "wrong results" are barrier or memory-visibility issues, which the Layers will report directly.
3. Read your own implementation alongside the chapter's labeled "industrial equivalent": you'll understand why ncnn / llama.cpp is written that way.
4. True mastery comes from the ch13 capstone project: when you can run a small model with your own engine and compare latency against ncnn/llama.cpp, the course goal is achieved.

---

## 4. Reference Resources

### Reference Industrial Implementations

Each chapter of this course labels the corresponding industrial implementation. The following local repositories were surveyed and used as reference:

| Repository | Version | Key Features |
|------|---------|---------|
| **ncnn** | `20260526-33-g946fe3fb` | 15+ Cooperative Matrix shaders (GEMM/conv/Winograd/SDPA), FlashAttention, Winograd F(2×2)/F(4×4), PipelineCache, multi-tier allocators, multi-queue |
| **ncnn MoE Runtime** | `3240ff4` | Standalone MoE inference engine: MoE IR, Expert multi-tier cache (ARC+Vulkan Victim), heterogeneous scheduling (Dense GPU + Expert CPU/GPU), cross-session microbatching, MXFP4 multi-platform kernels |
| **llama.cpp / ggml-vulkan** | Continuously evolving | 20+ quantization formats (Q4_0~Q8_0, K-quant, I-quant, MXFP4, NVFP4), Cooperative Matrix 2 (decode vector), FlashAttention 3-path (CM1/CM2), MoE, SSM, deep Subgroup usage |

### Frontier Technologies in This Course from Research

| Technology | Source | Course Chapter |
|------|---------|---------|
| Cooperative Matrix | ncnn `gemm_cm.comp` / ggml `mul_mm_cm2.comp` | ch04 (extended) + ch15 |
| Subgroup Operations | ncnn/ggml subgroup optimizations | ch16 |
| FlashAttention | ncnn `sdpa_fa.comp` / ggml `flash_attn.comp` | ch08 (extended) + ch17 |
| Q4_0/Q8_0 Quantization | ggml `dequant_funcs` | ch11 (extended) + ch18 |
| K-quant/MXFP4 | ggml latest quantization formats | ch20 |
| Winograd Convolution | ncnn convolution family | ch05 (extended) |
| MoE Inference | ncnn-MoE-Runtime / ggml `MUL_MAT_ID` | ch19 |
