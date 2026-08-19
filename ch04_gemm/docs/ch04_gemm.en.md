# Chapter 4 · Matrix Multiplication GEMM (the Compute Heart of Neural Networks)

> Goal: hand-write a **blocked + shared-memory** matrix multiply. It is the common compute kernel behind fully-connected layers, convolutions (after im2col), and attention (QKᵀ, PV) — once you understand this chapter, 80% of the operators that follow are just variants of it.

---

## 4.1 Why GEMM Is the Top Priority

Nearly every neural-network operation ultimately reduces to a matrix multiply:

- Fully-connected layer: `Y = X·W`
- Convolution: expanded via im2col into a GEMM (ch05)
- Self-attention: `QKᵀ`, `softmax(QKᵀ)·V`, `X·W_qkv`

So the performance of an inference engine is basically "how fast is the GEMM". ncnn and llama.cpp each ship a dozen-plus GEMM kernels (dispatched by matrix shape, quantization type, and hardware features), which shows its importance.

## 4.2 The Problem with Naive GEMM: Bandwidth Bottleneck

The most straightforward form: `c[r][col] = Σ_k a[r][k]*b[k][col]`. Problem: the column elements of B, `b[k][col]`, are **non-contiguous** in memory (column-major vs row-major), so every `k` requires strided reads — uncoalesced memory access, bandwidth saturated yet compute idle. This is the "bandwidth-bound" regime in the Roofline model.

## 4.3 Solution: 2D Block Tiling + Shared Memory

Core idea: split C into 16×16 tiles, each **workgroup** responsible for one tile. The 256 threads inside the workgroup cooperate, first moving the corresponding A/B tiles **entirely into shared memory** — shared memory is on-chip high-speed storage, an order of magnitude faster than global device memory, and the transfer is naturally **coalesced**; then do the multiply-adds inside shared memory.

Corresponding to `gemm_tiled.comp`:

1. `local_size_x=16, local_size_y=16` → 256 threads per workgroup.
2. `gl_GlobalInvocationID.y/x` is the global row/col of C; `gl_LocalInvocationID` is the in-workgroup coordinate.
3. Loop `numTiles` times: each iteration moves one A/B tile into `As`/`Bs` (shared memory), `barrier()` to wait for everyone, then do 16 multiply-adds, then `barrier()` again.
4. Accumulator `acc` uses **float32**, to avoid half/float16 error accumulation (mentioned in ch03).

> Why `barrier()`? Shared memory is shared within the workgroup. You must wait until all threads have written before reading, and after computing one tile you must also wait until everyone is done before loading the next — otherwise you read half-finished data. This is exactly the "synchronization" from ch02, the version inside a compute shader (`barrier()` ≈ an in-workgroup barrier, lighter than a pipeline barrier).

## 4.4 First Use of ComputePass (mini-VkCompute)

From this chapter on, the boilerplate is folded into `ComputePass` in `common/vk_base.cpp`:

```cpp
ComputePass pass;
pass.prepare(ctx, spv, {3 bindings}, {pushConstant range}); // build pipeline + descriptor set
pass.bindBuffers({{0,bufA},{1,bufB},{2,bufC}});           // bind buffers
pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);           // record dispatch
pass.destroy();                                           // release
```

This directly corresponds to **ncnn's `VkCompute`** (which wraps pipeline + descriptor + one dispatch into a single object) and **llama.cpp's `vk_pipeline_wrapper`**. The dozen-plus lines of pipeline-building code you wrote by hand in ch01 are now done in one line — but remember that those Vulkan calls are still underneath.

Number of workgroups: `gx = ceil(N/16)`, `gy = ceil(M/16)`, since the workgroup is 16×16.

## 4.5 Cooperative Matrix Acceleration

> This section demonstrates how to leverage GPU Tensor Core / Matrix Core hardware via the `VK_KHR_cooperative_matrix` extension to accelerate GEMM.

### 4.5.1 What is Cooperative Matrix

Cooperative Matrix (coopmat) is a Vulkan extension type that allows all threads within a Subgroup to **cooperatively hold** a matrix fragment. Each thread holds only a small portion of the fragment, and collective operations like `coopmatLoad` / `coopmatStore` / `coopmatMulAdd` handle loading, storing, and matrix multiply-add.

Key differences from traditional tiled GEMM:

| Feature | Tiled GEMM (gemm_tiled) | Cooperative Matrix (gemm_cm) |
|---------|------------------------|-------------------------------|
| Data movement | Manual copy to shared memory, `barrier()` sync | `coopmatLoad` auto-scheduled by hardware |
| Compute unit | Scalar ALU element-wise multiply-add | Tensor Core single-instruction `M×K×N` FMAs |
| Programming model | Each thread computes independently | Subgroup collective ops, threads share fragments |
| Synchronization | Explicit `barrier()` | Implicit in collective ops |

### 4.5.2 CM GEMM Shader Design

Corresponding to `gemm_cm.comp`, the core ideas:

1. **Fragment declaration**: Three `coopmat` fragments declared at file scope — A (`TILE_M×TILE_K`), B (`TILE_K×TILE_N`), C (accumulator, `TILE_M×TILE_N`). Dimensions are passed via **Specialization Constants** to select optimal tile sizes per hardware.

2. **Collective load**: `coopmatLoad` loads data from global memory into fragments, with all Subgroup threads collaborating — no explicit `barrier()` needed.

3. **Hardware-accelerated multiply-add**: `coopmatMulAdd` performs `TILE_M × TILE_K × TILE_N` floating-point multiply-adds in a single instruction, executed by Tensor Core hardware.

4. **Collective store**: `coopmatStore` writes results back to global memory.

The shader uses `push_constant` to pass strides and `K_total` rather than M/N/K directly, because `coopmatLoad/Store` require row strides (in elements).

### 4.5.3 Compilation and Runtime Detection

Compilation requires the `-DUSE_COOPMAT` flag to enable the extension:

```bash
glslc -DUSE_COOPMAT gemm_cm.comp -o gemm_cm.spv
```

The CMakeLists.txt handles this via a separate `add_custom_command`, leaving `gemm_tiled.comp`'s normal compilation undisturbed.

At runtime, `ctx.features.cooperativeMatrixKHR` detects whether the device supports `VK_KHR_cooperative_matrix`. The CM pipeline is only created and dispatched when supported.

### 4.5.4 C++ Integration of the CM Path

In `main.cpp`, the CM path uses `prepareWithSpec` to pass specialization constants (tile sizes), and the push constant uses the stride struct:

```cpp
struct CM_PC {
    uint32_t strideA;   // = K
    uint32_t strideB;   // = N
    uint32_t strideC;   // = N
    uint32_t K_total;   // = K
};

std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
passCM.prepareWithSpec(ctx, "gemm_cm.spv", bindings, pcs, specInfo.info);
```

### 4.5.5 CM vs Tiled GEMM Performance Comparison

GPU timestamp queries measure the time of each path, computing GFLOPS and speedup:

```
====== Performance Summary ======
  Matrix size: 256x256x256
  Tiled GEMM: 0.XX ms, XX.X GFLOPS
  CM GEMM:   0.XX ms, XX.X GFLOPS
  CM speedup: 1.XXx (correct)
```

On Tensor Core-capable hardware, the CM path typically achieves 2-5x speedup. Try changing `M/N/K` to `512` or `1024` to observe the speedup on larger matrices.

## 4.6 Hands-on Experiments

1. After it runs, change `M/N/K` to `512` or `1024` and observe that it is still correct.
2. Change `local_size_x/y` from 16 to 8 (also change `As[8][8]`/`Bs[8][8]` in the shader and `15→7` in `numTiles`) and compare the two results — to understand "workgroup size is a tunable knob".
3. (Advanced) Remove shared memory and switch to the naive column-wise read of B; use the timing method from ch09 to compare the bandwidth difference — you will see the value of tiling with your own eyes.
4. If your device supports Cooperative Matrix, compare the performance of tiled GEMM and CM GEMM. Note that the CM tile size (16×16×16) is a common Tensor Core spec; try changing it to 32×32×16 or 8×8×16 and observe the differences.
5. Think: when K is not a multiple of 16, why are the `ceil` of `numTiles` + the `if (aCol<K)` boundary guard necessary?

## 4.7 Industrial-Grade Comparison

- **ncnn**: the `gemm` family uses `optimal_local_size_xyz` to auto-select the workgroup, and writes different tile strategies for ARM Mali / Adreno / desktop GPUs; it also uses `elempack=4` to pack 4 elements into one vec4 for further coalesced access.
- **llama.cpp / ggml-vulkan**: `mul_mm.comp` dispatches **large / medium / small** kernels by matrix size; large matrices use cooperative matrix (coopmat, equivalent to Tensor Core), small/medium matrices use hand-written tiling; M/N/K are passed via `vkCmdPushConstants`.

In the next chapter (ch05) we break convolution into two paths, "direct convolution" and "im2col+GEMM", to see clearly why convolution also ultimately becomes a GEMM.
