# Chapter 15 · Cooperative Matrix Programming

> Goal: understand the **VK_KHR_cooperative_matrix** extension, and learn to leverage Tensor Core / Matrix Core / XMX hardware units on GPUs to accelerate matrix multiplication. This chapter implements a matrix multiply with both a CM path and a traditional tiled GEMM fallback, and compares their performance.

---

## 15.1 What is Cooperative Matrix / Tensor Core Hardware

### 15.1.1 The Bottleneck of Traditional ALUs

Traditional GPU compute cores (CUDA Cores / SPs) execute **scalar** operations: one thread can only do one multiply-add (FMA = 1 multiplication + 1 addition) at a time. For `C[M,N] = A[M,K] * B[K,N]`, you need `M * N * K` multiplications and `M * N * K` additions — `2 * M * N * K` FMA operations in total.

When matrices are large (e.g., `4096x4096` in LLMs), even thousands of CUDA Cores require many instruction cycles.

### 15.1.2 Dedicated Matrix Multiply Hardware

Starting with the NVIDIA Volta architecture (2017), GPUs introduced **Tensor Cores** — hardware units specialized for **small matrix multiply-accumulate** (`D = A * B + C`). A single Tensor Core can complete a **4×4 matrix multiply-accumulate** (64 FMA operations) in one clock cycle, equivalent to the throughput of 64 scalar FMAs.

Similarly:
- **AMD**: **Matrix Cores** on CDNA architectures (Matrix FMA instructions)
- **Intel**: **XMX** (Xe Matrix eXtension) on Xe architectures
- **Apple**: **Matrix Multiply** support in Metal

### 15.1.3 The Vulkan Abstraction: Cooperative Matrix

Vulkan exposes these hardware units through the **VK_KHR_cooperative_matrix** extension (and the earlier **VK_NV_cooperative_matrix**). The key idea:

> Transform a matrix multiply-accumulate from "N threads each doing N scalar FMAs" into "one subgroup cooperatively completing one small matrix multiply-accumulate".

This "small matrix" is called a **Matrix Fragment** in the shader, jointly held by all threads in the subgroup — each thread only holds a small portion of the fragment's data.

---

## 15.2 VK_KHR_cooperative_matrix Extension Overview

### 15.2.1 Extension Versions

| Extension | Status | Description |
|-----------|--------|-------------|
| `VK_NV_cooperative_matrix` | Proprietary (NVIDIA) | Earliest CM extension, NVIDIA only |
| `VK_KHR_cooperative_matrix` | Cross-vendor standard | Widely supported by NVIDIA / AMD / Intel |
| `VK_EXT_cooperative_matrix_maintenance1` | Supplementary | Adds flexible dimensions, etc. |

### 15.2.2 Core Concepts

- **Cooperative Matrix**: A special SPIR-V type representing a matrix fragment distributed across all threads in a subgroup.
- **Matrix Fragment**: A small `M_tile × N_tile` matrix cooperatively held by all threads in a subgroup.
- **coopmatMulAdd**: `C = C + A * B`, completed by hardware in one instruction, equivalent to `M_tile * K_tile * N_tile` scalar FMAs.
- **coopmatLoad / coopmatStore**: Load/store data between global memory and matrix fragments, also subgroup collective operations.

### 15.2.3 GLSL Syntax

```glsl
#extension GL_KHR_cooperative_matrix : require

// Define fragment types
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> fragA;
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> fragB;
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> fragC;

// Load fragments
cooperativeMatrixLoad(fragA, buffer, stride, gl_MatrixRowMajor, offset);
cooperativeMatrixLoad(fragB, buffer, stride, gl_MatrixRowMajor, offset);

// Matrix multiply-accumulate
fragC = cooperativeMatrixMulAdd(fragA, fragB, fragC);

// Store result
cooperativeMatrixStore(buffer, fragC, stride, gl_MatrixRowMajor, offset);
```

---

## 15.3 Matrix Fragment Concept (Subgroup-Scoped Distributed Storage)

### 15.3.1 Why Subgroup is Needed

In `C = A * B`, each output element `C[r][c]` depends on the r-th row of A and the c-th column of B — this is naturally **cross-thread** dependency. If a single thread were to compute an entire 16×16 output block, it would need to read 16×K elements from A and K×16 elements from B, far exceeding a single thread's register capacity.

Cooperative Matrix's solution: **all threads in the subgroup collaborate**, each holding only a small portion of the fragment, while collectively completing the load, compute, and store operations.

### 15.3.2 How Data is Distributed

Assume a `16×16` matrix fragment, and a subgroup of 32 threads:

- Each thread holds `16×16 / 32 = 8` elements
- The exact distribution is implementation-defined (transparent to the developer)
- When using `coopmatLoad` and `coopmatStore`, all threads collaborate to transfer data

### 15.3.3 Comparison with Shared Memory

| Aspect | Traditional Tiling (ch04) | Cooperative Matrix |
|--------|--------------------------|-------------------|
| Data storage | Shared memory (explicit load) | Registers (implicit distribution) |
| Synchronization | Explicit `barrier()` | Implicit via collective operations |
| Computation | Scalar FMAs in a loop | Single `coopmatMulAdd` instruction |
| Hardware | CUDA Cores / SPs | Tensor Cores / Matrix Cores |
| Programming complexity | High (manual tiling + bounds) | Low (collective operations encapsulate) |

---

## 15.4 Memory Layout (Row-Major vs Column-Major, Stride)

### 15.4.1 Row-Major

This is the default layout in C/C++: `A[i][j]` is stored contiguously, `&A[i][j+1] = &A[i][j] + 1`.

For CM load/store:

```glsl
cooperativeMatrixLoad(fragA, buffer, strideA, gl_MatrixRowMajor, offset);
```

- `strideA`: number of elements between adjacent rows (i.e., the column count K of A)
- `offset`: byte offset (or element index, depending on the exact syntax) of the starting position

### 15.4.2 Column-Major

Default layout in Fortran / MATLAB. Specified via `gl_MatrixColumnMajor` in CM:

```glsl
cooperativeMatrixStore(buffer, fragC, strideC, gl_MatrixColumnMajor, offset);
```

### 15.4.3 Stride Alignment Requirements

Hardware typically requires stride to be a multiple of a specific alignment value (e.g., 16, 32, or 64 bytes). If stride does not meet alignment requirements, the CM path may not be usable, requiring padding or a fallback path.

---

## 15.5 Specialization Constant Handshake for Tile Sizes

### 15.5.1 Why Specialization Constants

The fragment dimensions (`M_tile × N_tile × K_tile`) of a cooperative matrix must be determined at compile time, because different hardware supports different tile combinations:

- **NVIDIA Tensor Core (Ampere+)**：16×16×16 (FP16), 16×8×16 (BF16), 16×8×8 (INT8)
- **AMD Matrix Core (RDNA3+)**：16×16×16 (FP16)
- **Intel XMX (Xe)**：8×8×8 (FP16), 8×8×8 (INT8)

Using specialization constants, different tile sizes can be selected for different hardware without recompiling the shader:

```glsl
layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;
```

### 15.5.2 Setting from C++

```cpp
std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
pass.prepareWithSpec(ctx, spv, bindings, pcs, specInfo.info);
```

### 15.5.3 Hardware-Driven Tile Selection

In production engines (ncnn, ggml), the device's `VkCooperativeMatrixPropertiesKHR` is queried to get the list of supported tiles, then the optimal one is selected:

```cpp
// Query supported fragment types
uint32_t propCount = 0;
ctx.physical.getCooperativeMatrixPropertiesKHR(&propCount, nullptr);
std::vector<vk::CooperativeMatrixPropertiesKHR> props(propCount);
ctx.physical.getCooperativeMatrixPropertiesKHR(&propCount, props.data());
```

---

## 15.6 GPU Hardware Constraints

### 15.6.1 NVIDIA Tensor Core

| Architecture | Tensor Core Gen | Supported Data Types | Tile Sizes |
|-------------|-----------------|---------------------|------------|
| Volta (V100) | Gen 1 | FP16 | 16×16×16 |
| Turing (T4) | Gen 2 | FP16, INT8 | 16×16×16, 8×8×8 |
| Ampere (A100) | Gen 3 | FP16, BF16, TF32, INT8, INT4 | 16×16×16, 16×8×16, 8×8×8 |
| Hopper (H100) | Gen 4 | Above + FP8 | Same |
| Ada Lovelace (RTX 40xx) | Gen 4 | Above | Same |

### 15.6.2 AMD Matrix Core

| Architecture | Matrix Core Gen | Supported Data Types | Tile Sizes |
|-------------|----------------|---------------------|------------|
| RDNA3 (RX 7000) | Gen 1 | FP16, BF16 | 16×16×16 |
| CDNA2 (MI250) | Matrix Core | FP16, BF16, INT8 | 16×16×16 |
| CDNA3 (MI300) | Matrix Core | Above + FP8 | Same |

### 15.6.3 Intel XMX

| Architecture | XMX Gen | Supported Data Types | Tile Sizes |
|-------------|---------|---------------------|------------|
| Xe-LP (Tiger Lake) | Gen 1 | INT8 | 8×8×8 |
| Xe-HPG (Arc Alchemist) | XMX | FP16, INT8, INT4 | 8×8×8 |
| Xe-HPC (Ponte Vecchio) | XMX | FP16, BF16, INT8 | 8×8×8, 16×16×16 |

### 15.6.4 Key Constraints

1. **Subgroup size**: CM operations require the subgroup size to match the hardware-supported CM dimensions. Typically, subgroupSize must be ≥ one dimension of the tile.
2. **Stride alignment**: stride must be a multiple of 16 bytes (or larger).
3. **Data type alignment**: float16 typically requires 16-byte alignment, int8 requires 8-byte alignment.
4. **Workgroup size**: Some implementations require the workgroup size to match the subgroup size, or be an integer multiple of it.

---

## 15.7 When to Use CM vs Traditional Tiled GEMM

### 15.7.1 When to Use CM

- **Large matrix sizes**: M, N, K all ≥ 128 (fully utilize Tensor Core throughput)
- **FP16/BF16/INT8 data types**: Tensor Core FP16 throughput is 4~16× that of FP32
- **Batch inference**: larger batch sizes amplify CM's speedup
- **Hardware supports CM**: this is a prerequisite

### 15.7.2 When to Use Traditional Tiled GEMM

- **Small matrix sizes**: M, N, K are small (e.g., ≤ 64), CM startup overhead may exceed benefits
- **FP32 precision required**: some Tensor Cores don't support FP32 CM (need TF32 or emulation)
- **Hardware without CM**: older GPUs (Pascal, GCN, etc.) lack CM hardware
- **Irregular matrix shapes**: CM's fixed tile sizes incur padding overhead for non-standard shapes

### 15.7.3 Production Practice

**ncnn**: the shader directory is `src/layer/vulkan/shader/` (**singular**); the CM path is `gemm_cm.comp`, with scalar fallbacks `gemm.comp` / `gemm_sg.comp`. Host-side tile selection is wrapped in `VulkanDevice::get_optimal_cooperative_matrix_mnk(..., coopmat_acctype)` (`src/gpu.h`), driven by `Option::use_cooperative_matrix` (enabled by default) and the `GpuInfo` capability bits.

**llama.cpp / ggml-vulkan**: `vulkan-shaders/mul_mm_cm2.comp` is still the CM2 (incl. decode vector) main path; small matrices use `mul_mm.comp`. New additions include `mul_mmq_cm1.comp` and CM1/CM2 `conv2d/conv3d`. All 155 `.comp` templates are turned into SPIR-V at build time by `vulkan-shaders-gen` + glslc — there are no prebuilt `.spv` files.

---

## 15.8 Comparison with Production Implementations

### 15.8.1 ncnn's gemm_cm.comp (as of 2026-11)

- Path: `ncnn/src/layer/vulkan/shader/gemm_cm.comp`; the directory holds **17** `*_cm.comp` shaders covering GEMM, convolution/deconvolution, Winograd transforms and GEMM, 1D convolution, SDPA and cross-attention.
- Both KHR and NV coopmat paths are retained, with `UNROLL_SG/WG` macros controlling tile unrolling; recent fixes cover Adreno/Mali `coopMatLoad/Store` strides and four-row matrices.
- Data types extend from FP16 to **BF16 and INT8** (`gemm_int8_cm.comp`, `convolution_*_int8_cm.comp`); fp32 accumulation is explicit (`get_optimal_cooperative_matrix_mnk` also returns `coopmat_acctype`).
- Shared-memory padding was removed from the CM shader after measurements showed the unpadded variant is faster; `elempack=4` and int8 pack4 rules are documented in `docs/developer-guide/glsl-extension.md`.

### 15.8.2 ggml's mul_mm_cm2.comp (as of 2026-11)

- Path: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp`; `mul_mm_id_funcs.glsl` and `dequant_funcs_cm2.glsl` handle unpacking under CM2.
- Supports `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_EXT_cooperative_matrix_maintenance1` and subgroup size control.
- Quantization coverage grew to 44 types (MXFP4 / NVFP4 / Q1_0 / Q2_0 / TQ / F8_E4M3 included); the CM2 path cooperates with the `dequant_*` shaders.
- Device/pipeline structures live in `ggml-vulkan-types.h`; `vk-device.cpp`, `pipeline.cpp` and `vk_pipeline_wrapper` are gone.

### 15.8.3 New Capabilities Worth Trying (synced 2026-11)

| Capability | ncnn | ggml |
|------------|------|------|
| bf16 / int8 coopmat | `gemm_int8_cm.comp`, bf16 packed GEMM | `dequant_*` + CM1/CM2 quantized paths |
| Maintenance extensions | `VK_EXT_cooperative_matrix_maintenance1`, `VK_NV_cooperative_matrix2`, cooperative vector, float8, OCP microscaling | same + `mul_mmq_cm1.comp` |
| Accumulation precision | Reduction switched to fp32 accumulation (2026-11 #7247) | fp32 accumulation in FA decode |
| Host-side shape selection | `get_optimal_cooperative_matrix_mnk()` returns M/N/K/subgroup size/acctype | shape-specialized pipelines generated at build time |

### 15.8.4 This Chapter's Approach

This chapter's `gemm_cm.comp` includes both a CM path and a fallback path:
- **CM path**: uses a single `cooperativeMatrixMulAdd` instruction for matrix multiply, executed by Tensor Core hardware
- **Fallback path**: uses traditional shared memory + scalar FMAs (identical to ch04)

Using the timing comparison in `main.cpp`, you can directly observe the speedup from CM.

---

## 15.9 Code Walkthrough

### 15.9.1 Shader Structure

The two paths in `gemm_cm.comp` are selected via the compile-time macro `USE_COOPMAT`:

```glsl
#ifdef USE_COOPMAT
// CM path: uses cooperative matrix types and functions
#extension GL_KHR_cooperative_matrix : require
// ...
void path_cm() {
    cooperativeMatrixLoad(fragA, ...);
    cooperativeMatrixLoad(fragB, ...);
    fragC = cooperativeMatrixMulAdd(fragA, fragB, fragC);
    cooperativeMatrixStore(..., fragC, ...);
}
#else
// Fallback path: traditional tiled GEMM (uses shared memory)
void path_fallback() {
    // Same tiling logic as ch04
}
#endif
```

### 15.9.2 Main Program Flow

1. Calls `vkInitWithExtensions` for initialization, automatically enables CM extensions
2. Prints device CM capabilities via `printCoopMatCapabilities`
3. Generates random matrices, computes CPU reference
4. Runs the fallback path (tiled GEMM), times and verifies correctness
5. If device supports CM, runs the CM path, times and verifies
6. Prints speedup

### 15.9.3 Using Specialization Constants

```cpp
std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
pass.prepareWithSpec(ctx, spv, bindings, pcs, specInfo.info);
```

Corresponding shader side:

```glsl
layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;
```

---

## 15.10 Hands-on Exercises

### Basic Exercises

1. **Run the CM path**: Execute on a GPU with Tensor Cores and observe whether the CM path is faster than the fallback path.
2. **Change matrix sizes**: Change M, N, K from 256 to 512, 1024, and observe the speedup trend.
3. **Change tile sizes**: Try different tile sizes (e.g., 8×8×8, 32×32×32) and see which ones pass hardware validation.

### Advanced Exercises

4. **FP16 path**: Modify the shader to use `cooperative_matrix<float16_t, ...>` (requires `shaderFloat16` support), compare FP16 CM vs FP32 CM performance.
5. **Query hardware capabilities**: Modify the main program to print all supported `VkCooperativeMatrixPropertiesKHR`, understand the tile constraints of different hardware.
6. **Compare with ncnn**: Download the ncnn source, read `gemm_cm.comp`, and compare with our implementation.
7. **Maintenance1 extension**: If the device supports `VK_EXT_cooperative_matrix_maintenance1`, try using `cooperativeMatrixFlexibleDimensions` for more flexible tile sizes.

### Thought Questions

8. Why does the CM path's push constant use stride and K_total instead of M/N/K?
9. How should boundary handling be done when K is not a multiple of the tile size?
10. Why is Tensor Core FP16 throughput much higher than FP32? What does this mean for neural network inference?

---

## 15.11 Chapter Summary

| Concept | Description |
|---------|-------------|
| Cooperative Matrix | A matrix fragment distributed across subgroup threads |
| Tensor Core / Matrix Core / XMX | Dedicated matrix multiply-accumulate hardware units |
| Specialization Constants | Compile-time tile size selection, runtime choice |
| pass.prepareWithSpec | Pipeline creation with specialization constants |
| coopmatMulAdd | Single-instruction matrix multiply-accumulate |
| Hardware Constraints | Subgroup size, Stride alignment, Tile size limits |

In the next chapter (ch16), we will explore Subgroup Size Control and Shuffle instructions, deepening our understanding of subgroup roles in Vulkan compute.