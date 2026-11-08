# Chapter 11 · Quantization and Fused Dequantization

> Goal: understand the #1 technique for shrinking LLM inference memory footprint — **weight quantization** — and why "dequantization" should be **fused** into the matmul rather than performed separately. This chapter starts with int8, then dives into group-wise quantization with Q4_0 and Q8_0 formats for GPU fused dequant GEMM.

---

## 11.1 Why Quantize

Model weights often occupy over 90% of GPU memory. A FP32 weight costs 4 bytes each; switching to **int8** needs only 1 byte — **both memory and bandwidth are reduced by 4× at once**. Neural networks are not sensitive to weight precision, and the int8 error is usually negligible (thanks to the redundancy of later layers). This is also why a 7B LLaMA can fit into a 4GB GPU using Q4 (4 bits per weight).

## 11.2 Symmetric int8 Quantization (per-tensor)

The simplest form:

```
scale = max|W| / 127
w_q  = round( w / scale )          # clamp to [-127, 127]
dequantize: w ≈ w_q * scale
```

In `ch11_quant/main.cpp`, the `K*N` float weights are compressed into int8 and uploaded to GPU memory, occupying only 1/4 of the bytes. `q8_gemm.comp` reads the `int8_t` weights and **in place** restores them to float via `float(w) * scale`, then participates in the multiply-add — dequantization happens in the innermost loop of the GEMM, with **no intermediate float weight buffer at all**.

## 11.3 Key Point: Fused Dequantization vs. Separate Dequantization

- ❌ Separate: first decompress the int8 weights into a float temporary buffer, then run a normal GEMM. → Consumes an extra slice of memory plus one full extra copy, and **the bandwidth saved by quantization is handed right back**.
- ✅ Fused (this chapter's approach): dequantization is written directly into the matmul accumulation loop. → Weights stay in GPU memory as int8 at all times, bandwidth fully saved.

This is exactly the idea behind llama.cpp's `dequant_funcs.glsl`: each quantization format (q4_0, q8_0, iq4_nl…) has a corresponding set of "unpack + dequantize" functions on the GPU, called in place by `mul_mm` during accumulation.

## 11.4 Group-wise Quantization: Q4_0 and Q8_0 Formats

Per-tensor int8 uses a single scale for the entire matrix, which amplifies error in the presence of outliers. **Group-wise quantization** partitions the weight matrix by column, with every 32 elements sharing one scale, providing finer granularity.

### Q4_0 Format

Q4_0 is the most basic 4-bit quantization format in ggml/llama.cpp:

```
Q4_0 Block (32 elements, 20 bytes = 5 × uint32):
  ├── d: fp16 scale (2 bytes)           ← shared by 32 elements in this block
  ├── qs: 4-bit signed int packed (16 bytes) ← 2 elements/byte, low nibble = element i, high nibble = element i+1
  └── padding: 2 bytes to 20-byte alignment
```

- Storage per weight: 4 bits (theoretical compression 8×)
- Actual compression: 20 bytes / 32 elements = 5 bits/weight, ~6.4× (includes scale and padding overhead)
- Signed int4 range: [-8, 7]
- Dequantize: `w = float(q) * scale`

### Q8_0 Format

Q8_0 is the 8-bit group-wise quantization format in ggml, differing from per-tensor int8 in its group-wise nature:

```
Q8_0 Block (32 elements, 36 bytes = 9 × uint32):
  ├── d: fp16 scale (2 bytes)           ← shared by 32 elements in this block
  ├── qs: int8 data (32 bytes)
  └── padding: 2 bytes to 36-byte alignment
```

- Storage per weight: 8 bits (theoretical compression 4×)
- Actual compression: 36 bytes / 32 elements = 9 bits/weight, ~3.56× (includes scale overhead)
- Signed int8 range: [-127, 127] (avoiding -128)

### Compatibility with ggml Formats

The Q4_0/Q8_0 formats in this chapter are fully compatible with ggml's definitions:
- Same block size (32 elements)
- Same fp16 scale storage
- Same 4-bit/8-bit data layout
- The only difference is the 20/36 byte alignment (uint32 alignment) on the GPU side, making it convenient for shaders to read with `uint`

## 11.5 Fused Dequantization in Q4_0/Q8_0

For int8, dequantization is just `float(w) * scale`. For Q4_0, it's more involved:

```
1. Read fp16 scale from block header → convert to float
2. Locate the element's offset within the block
3. Extract nibble from packed byte (bit ops & 0xF / >> 4)
4. Sign extension (if q >= 8: q -= 16)
5. w = float(q) * scale
```

Q8_0 is similar but simpler (no nibble unpacking needed):

```
1. Read fp16 scale from block header → convert to float
2. Read int8 value
3. Sign extension (if q >= 128: q -= 256)
4. w = float(q) * scale
```

These operations are **fused** into the shared memory tiling loop of `q4_0_gemm.comp` and `q8_0_gemm.comp`: each thread, when loading the weight tile, unpacks and dequantizes in place, then stores the result in shared memory for the subsequent multiply-add. Compared to the int8 version, the Q4_0/Q8_0 versions use **16×16 shared memory tiling**, further reducing global memory accesses.

## 11.6 Where the Error Comes From, and How to Control It

The error comes only from the "quantization step": `w_q * scale` differs from the true value `w` by at most `scale/2`. After accumulating K terms, the error is averaged out and is usually smaller than at a single point.

Advantages of group-wise quantization (Q4_0/Q8_0) over per-tensor (int8):
- Each block has its own scale, more precisely representing local numerical ranges
- Less sensitive to outlier columns — outliers only affect their own block, not the entire matrix
- In practice, Q8_0 accuracy is close to FP32, while Q4_0 is acceptable within 5-10% relative error

Verification tolerances in this chapter:
- int8 / Q8_0: 5% relative tolerance
- Q4_0: 10% relative tolerance (int4 has lower precision, wider tolerance)

## 11.7 Hands-on Experiments

1. Change quantization granularity from per-tensor to **per-channel** (one scale per output column `col`), and re-test whether the maximum error drops significantly.
2. Replace int8 with "pseudo int4": pack two int4 values into one byte (unpack with bit operations), and experience the 8× compression of Q4 (note the use of `& 0xF` / `>> 4` for unpacking in GLSL).
3. Use the timing method from ch09 to compare the bandwidth usage and runtime of "int8 weights + fused dequantization" versus "pure float32 weights" — directly observe how quantization relieves the bandwidth bottleneck (Roofline shifts left).
4. Modify the Q4_0 block size (e.g., 64 elements/block) and observe the compression vs. accuracy trade-off.

## 11.8 Industrial-grade Comparison

- **ncnn**: int8 conv / fully-connected remains the main path; `src/layer/vulkan/shader/` now includes `convolution_*_int8.comp`, `gemm_int8.comp`, `gemm_quantize_{A,B}_int8.comp`, `gemm_quantize_B_{absmax,descale}_int8.comp`, plus int8 coopmat / Winograd variants (see ch15). Weights stay int8 in memory; "scale multiply + bias add" is fused into the kernel.
- **llama.cpp / ggml-vulkan** (as of 2026-11): dequantization is no longer "one `dequant_funcs` file" but three layers:
  - `ggml/src/ggml-quants.c` — CPU-side block layouts and quantization algorithms (the old `iq-quants.c` was merged into it);
  - `vulkan-shaders/dequant_funcs.glsl` / `dequant_funcs_cm2.glsl` / `dequant_head.glsl` — shared GLSL unpacking snippets;
  - 26 `vulkan-shaders/dequant_<format>.comp` — per-format kernels for `q4_0/q8_0/q*_k/iq*/mxfp4/nvfp4/q1_0/q2_0/tq1_0/tq2_0`.
  - SPIR-V is generated **at build time** from 155 `.comp` templates by `vulkan-shaders-gen` + glslc; the result is compiled into the binary as `.comp.cpp` arrays, and no prebuilt `.spv` ships in the repo. The "since v1.17 caches compiled SPIR-V to speed up startup" claim from the first release therefore no longer applies.
- **Quantization landscape**: `ggml/include/ggml.h` now has `GGML_TYPE_COUNT = 44`; besides Q4_0/Q8_0, K-quants and I-quants there are Q1_0/Q2_0, TQ1_0/TQ2_0, MXFP4/NVFP4 and F8_E4M3 (see ch20).
- **Q4_0 vs Q8_0 choice**: Q8_0 gives higher precision with cheaper unpacking (good for Q/K/V projections); Q4_0 gives higher compression for memory-bound setups. ggml supports per-layer mixed precision (`llama-quant.cpp`).

## 11.9 Code Structure

```
ch11_quant/
├── main.cpp               # Main program: int8/Q4_0/Q8_0 quantization, GPU fused GEMM, verification, benchmark
├── CMakeLists.txt          # Build config (auto-compiles .comp → .spv)
├── shaders/
│   ├── q8_gemm.comp        # int8 fused dequant GEMM (original, no shared memory tiling)
│   ├── q4_0_gemm.comp      # Q4_0 fused dequant GEMM (shared memory tiling 16×16)
│   ├── q8_0_gemm.comp      # Q8_0 fused dequant GEMM (shared memory tiling 16×16)
│   └── gemm_fp32.comp      # FP32 baseline GEMM (shared memory tiling 16×16)
└── docs/
    ├── ch11_quant.md        # Chinese documentation
    └── ch11_quant.en.md     # English documentation
```

In the next chapter (ch12) we chain multiple operators into **a single compute graph**: record them into the same command buffer, connect dependencies with barriers, and submit in one shot — turning scattered operators into an efficient pipeline.