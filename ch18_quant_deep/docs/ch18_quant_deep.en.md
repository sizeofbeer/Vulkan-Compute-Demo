# Chapter 18 · Deep Quantization (Q4_0 / Q8_0)

> Goal: master the **group-wise quantization** formats used in production LLM inference — Q4_0 and Q8_0. Understand block structure, fp16 scale, 4-bit packing/unpacking, and implement **fused dequantization GEMM** on the GPU. This chapter is the core foundation of the ggml/llama.cpp quantization system.

---

## 18.1 Quantization Fundamentals: Why Quantize

### 18.1.1 Memory Bottleneck

The primary bottleneck for large language models is **memory bandwidth**, not compute. Take LLaMA-7B as an example:

| Precision | Per weight | 7B model memory | Relative size |
|-----------|-----------|-----------------|---------------|
| FP32 | 4 bytes | ~28 GB | 1× |
| FP16 | 2 bytes | ~14 GB | 0.5× |
| Q8_0 | ~1 byte | ~7 GB | 0.25× |
| Q4_0 | ~0.5 byte | ~3.5 GB | 0.125× |

Q4_0 can fit a 7B model into a 4GB consumer GPU — the key to running LLM inference locally.

### 18.1.2 Compute Efficiency

Quantization not only saves memory, but also improves compute efficiency:

- **Less bandwidth consumption**: each weight read transfers fewer bytes, alleviating **memory-bound** bottlenecks
- **Higher arithmetic intensity**: arithmetic intensity = FLOPs / bytes accessed. In the Roofline model, higher intensity means closer to the compute ceiling
- **Fused dequantization**: unpack and dequantize in-place inside the GPU kernel, avoiding materializing intermediate FP32 buffers, further reducing memory traffic

### 18.1.3 Tolerance to Quantization Error

Neural networks have natural redundancy in weight precision:

- Trained weights are typically robust to noise
- Subsequent layers can "average out" quantization errors
- Errors measured by RMSE: Q8_0 typically < 0.1%, Q4_0 typically < 1%
- **Per-channel** or **group-wise** quantization granularity can further reduce error

---

## 18.2 Q4_0 Format Details

Q4_0 is the most basic 4-bit quantization format in ggml, and the core format originally supported by llama.cpp.

### 18.2.1 Block Structure

```
Q4_0 Block (32 elements, 20 bytes, GPU-aligned):
┌─────────────────────────────────────────────┐
│ Bytes 0-1 : d (fp16 scale)                  │  ← 2 bytes
│ Bytes 2-17: qs[16] (packed 4-bit, 2/byte)   │  ← 16 bytes
│ Bytes 18-19: padding                        │  ← 2 bytes (align to 20)
└─────────────────────────────────────────────┘
```

- **32 elements per block**: every 32 consecutive elements along the K dimension form a block
- **d (fp16 scale)**: 16-bit float representing the scaling factor for this block
- **qs (packed 4-bit)**: 16 bytes packing 32 signed 4-bit integers
- **padding**: 2 bytes to align the block to 20 bytes (5 × uint32), for efficient GPU access

### 18.2.2 Quantization Process

For the 32 weights w₀, w₁, ..., w₃₁ in a block:

1. Find max absolute value: `maxAbs = max(|w₀|, ..., |w₃₁|)`
2. Compute scale: `d = maxAbs / 7.0` (int4 max positive is 7)
3. Quantize each element: `qᵢ = round(wᵢ / d)`, clamp to [-8, 7]
4. Pack: every two int4 values pack into one byte — low nibble = element 2i, high nibble = element 2i+1

### 18.2.3 Dequantization Process

```
wᵢ ≈ float(qᵢ) × d
```

Where `qᵢ` is extracted from the packed byte:

```c
// Extract the i-th 4-bit value from a byte
byteIndex = i / 2;
nibbleShift = (i & 1) ? 4 : 0;
q = (byte[byteIndex] >> nibbleShift) & 0xF;
// Signed int4: values >= 8 are negative
if (q >= 8) q -= 16;
```

### 18.2.4 Compression Ratio

- Original FP32: 32 × 4 = 128 bytes
- Q4_0: 20 bytes (including 2 bytes padding)
- Effective compression: 128 / 18 ≈ 7.1× (without counting padding)
- Compared to FP16: 32 × 2 = 64 bytes → compression 64/20 = 3.2×

---

## 18.3 Q8_0 Format Details

Q8_0 is the 8-bit quantization format in ggml, with higher precision suitable for quantization-sensitive layers.

### 18.3.1 Block Structure

```
Q8_0 Block (32 elements, 36 bytes, GPU-aligned):
┌─────────────────────────────────────────────┐
│ Bytes 0-1 : d (fp16 scale)                  │  ← 2 bytes
│ Bytes 2-33: qs[32] (int8 data)              │  ← 32 bytes
│ Bytes 34-35: padding                        │  ← 2 bytes (align to 36)
└─────────────────────────────────────────────┘
```

### 18.3.2 Quantization Process

1. Find max absolute value: `maxAbs = max(|w₀|, ..., |w₃₁|)`
2. Compute scale: `d = maxAbs / 127.0` (symmetric int8 quantization, avoiding -128)
3. Quantize each element: `qᵢ = round(wᵢ / d)`, clamp to [-127, 127]

### 18.3.3 Dequantization and Compression Ratio

- Dequantization: `wᵢ ≈ float(qᵢ) × d`
- Original FP32: 128 bytes → Q8_0: 36 bytes → compression 128/36 ≈ 3.6×
- Compared to FP16: 64 bytes → compression 64/36 ≈ 1.8×

---

## 18.4 Group-Wise vs Per-Tensor Quantization

### Per-Tensor Quantization

The approach used in Chapter 11: the entire weight matrix shares one scale.

```
scale = max|W| / 127
w_q = round(w / scale)
```

**Drawback**: one abnormally large outlier inflates the scale, reducing quantization precision for all other values.

### Group-Wise (Block-Wise) Quantization

The approach used by Q4_0/Q8_0: every 32 elements share one scale.

```
block 0: scale₀ = max|w₀...w₃₁| / 7
block 1: scale₁ = max|w₃₂...w₆₃| / 7
...
```

**Advantages**:
- Each block's scale adapts to its local dynamic range
- Outliers only affect their own block, not the entire matrix
- Compared to per-tensor, group-wise quantization typically achieves 5-10× lower RMSE

**Cost**: each block needs 2 extra bytes for the scale, adding ~6.25% storage overhead (Q4_0).

---

## 18.5 Fused Dequantization

### Why Fused is More Efficient

**Standalone Path**:

```
Quantized weights (Q4_0) → Dequant kernel → FP32 weight buffer → FP32 GEMM → Output
                              ↑                                    ↑
                          one extra write                    one extra read
```

**Fused Path**:

```
Quantized weights (Q4_0) ──┐
                            ├→ Fused Dequant GEMM kernel → Output
Activations (FP32) ────────┘
```

Cost of the standalone path:
1. **Extra memory**: needs a full FP32 weight buffer (K×N×4 bytes)
2. **Extra bandwidth**: dequant writes once + GEMM reads once = 2× FP32 weight traffic
3. **Quantization benefits negated**: Q4_0 saves 7× on weight storage, but the standalone path doubles weight traffic

Fused path:
- Weights always stay in GPU memory as Q4_0 format
- Unpacking and dequantization happens on-demand inside the GEMM inner loop
- Dequantized values go directly into the multiply-accumulate accumulator, **never written back to memory**

### Fusing in Shared Memory

The `q4_0_gemm.comp` shader in this chapter uses 16×16 shared memory tiling:

1. Each thread loads one float element of A into `As[][]`
2. Each thread unpacks and dequantizes one element of W from the Q4_0 buffer into `Ws[][]`
3. After barrier synchronization, 16 multiply-adds are performed in shared memory
4. After another barrier, the next tile is loaded

This way, Q4_0 data is read at the global memory level, but only once before being cached in shared memory, where it's already in float format — balancing bandwidth savings and compute efficiency.

---

## 18.6 Dequantization Function Architecture in ggml (as of 2026-11)

Current ggml-vulkan splits dequantization into three layers, finer-grained than "one `dequant_funcs` file called from `mul_mm`":

| Layer | Location | Role |
|-------|----------|------|
| Data layout / quantization | `ggml/src/ggml-quants.c` (the old `iq-quants.c` was merged in), `ggml-cpu/quants.c` | Byte layout of each block, `quantize_row_*` / `dequantize_row_*` |
| Shared GLSL snippets | `vulkan-shaders/dequant_funcs.glsl`, `dequant_funcs_cm2.glsl`, `dequant_head.glsl` | `#include`-d unpack/dequant helpers; a separate copy for the CM2 path |
| Standalone dequant kernels | 26 `vulkan-shaders/dequant_<format>.comp` | `dequant_q4_0.comp`, `dequant_q8_0.comp`, `dequant_q*_k.comp`, `dequant_iq*.comp`, `dequant_{mxfp4,nvfp4,q1_0,q2_0,tq1_0,tq2_0}.comp`, etc. |

The `mul_mm` family (`mul_mm.comp` / `mul_mm_cm1.comp` / `mul_mm_cm2.comp` / `mul_mmq_cm1.comp`) picks a pipeline by weight type and unpacks weights in place while loading a tile into shared memory / fragment registers, accumulating in float. All 155 `.comp` templates are compiled by `vulkan-shaders-gen` + glslc **at build time**; no prebuilt `.spv` is shipped.

### Comparison with ggml

| Aspect | ggml implementation | This chapter |
|--------|-------------------|--------------|
| Q4_0 block size | 18 bytes (compact) | 20 bytes (aligned) |
| Shared memory tiling | Yes | Yes (16×16) |
| Tile size | Shape-specialized pipelines generated at build time | Fixed 16×16 |
| Multi-format support | 44 `GGML_TYPE_*` (Q4_0~Q8_0, K/I-quants, MXFP4/NVFP4, TQ, Q1_0/Q2_0, F8_E4M3) | Q4_0, Q8_0 |
| Bias fusion | Optional | Optional (push constant) |

The 20-byte aligned version in this chapter is more efficient on the GPU (avoiding unaligned access), at the cost of 2 bytes per block overhead (~11%). In ggml's compact version, data must be accessed via byte-address calculations, requiring more complex indexing logic. Among the new formats, NVFP4 uses an E4M3 scale per 4 blocks while this course's MXFP4 (ch20) uses E8M0 — a ready-made example of "same E2M1, different scale granularity".

---

## 18.7 K-Quant Overview (Stepping Stone to Chapter 20)

Building on Q4_0/Q8_0, ggml designed the more efficient **K-quant** series (Q2_K ~ Q6_K), as a preview for Chapter 20:

### K-quant Design Philosophy

K-quant's core idea is the **super-block**:

- **Super-block**: 256 elements form one super-block
- **Sub-blocks**: each super-block is divided into sub-blocks (e.g., Q4_K has 8 sub-blocks of 32 elements each)
- **Shared scale**: the super-block shares a larger scale (fp16), while sub-blocks use smaller deltas (6-bit or 4-bit)

### K-quant Format Comparison

| Format | Bits/weight | Super-block | Sub-block | Characteristics |
|--------|------------|-------------|-----------|-----------------|
| Q2_K | ~2.56 bit | 256 | 32 | Extreme compression, high accuracy loss |
| Q3_K | ~3.55 bit | 256 | 32 | Intermediate option |
| Q4_K | ~4.55 bit | 256 | 32 | Popular choice, balanced accuracy/compression |
| Q5_K | ~5.55 bit | 256 | 32 | High precision quantization |
| Q6_K | ~6.55 bit | 256 | 16 | Close to FP16 precision |

K-quant improvements over Q4_0/Q8_0:
1. More efficient scale encoding (6-bit floating-point quantization, not fp16)
2. Super-block shared scale reduces scale storage overhead
3. Sub-block deltas provide sufficient granularity

K-quant is ggml's primary quantization format since v2.0+, and will be implemented in detail in Chapter 20.

---

## 18.8 Chapter Code Structure

```
ch18_quant_deep/
├── CMakeLists.txt
├── main.cpp                    # Main program: quantization, GPU tests, verification, benchmark
├── shaders/
│   ├── q4_0_gemm.comp          # Q4_0 fused dequantization GEMM
│   ├── q8_0_gemm.comp          # Q8_0 fused dequantization GEMM
│   ├── q4_0_dequant.comp       # Q4_0 standalone dequantization (for verification)
│   └── gemm_tiled.comp         # FP32 tiled GEMM (reference)
└── docs/
    ├── ch18_quant_deep.md
    └── ch18_quant_deep.en.md
```

### Main Program Flow

1. Generate random matrix A (M×K) and weights W (K×N)
2. Quantize W with Q4_0 and Q8_0 formats
3. Upload data to GPU
4. Run four tests:
   - Q4_0 fused dequantization GEMM
   - Q8_0 fused dequantization GEMM
   - Standalone dequant + FP32 GEMM (comparison path)
   - FP32 GEMM reference
5. Verify results (RMSE, max error)
6. Benchmark with GPU timestamps
7. Report compression ratio and speedup

---

## 18.9 Hands-on Experiments

1. **Modify block size**: Change Q4_0 block size from 32 to 64 or 16, observe the change in quantization error and compression ratio. Larger blocks mean wider scale sharing, higher compression but also larger error.

2. **Implement per-channel quantization**: Change Q4_0 block granularity to the entire K dimension (one scale per column), compare the error difference with group-wise quantization.

3. **Test without bias**: Set `hasBias` to 0, test GEMM correctness without bias.

4. **Compare performance at different dimensions**: Change M, N, K to different values (e.g., M=128, N=512, K=1024), observe speed differences between quantization formats.

5. **Implement Q4_1 format**: Q4_1 adds an fp16 min value (offset) on top of Q4_0, dequantizing as `w = float(q) * d + min`. Think about why adding min can reduce quantization error.

6. **Analyze bandwidth-bound vs compute-bound**: Using this chapter's benchmark data and your GPU specifications (memory bandwidth, FP32 peak compute), determine the bottleneck type for each quantization format.

---

## 18.10 Industrial-grade Comparison

### ggml / llama.cpp

- **Quantization formats**: `GGML_TYPE_COUNT = 44` — Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q1_0/Q2_0, Q2_K~Q6_K/Q8_K, the IQ1~IQ4 family, TQ1_0/TQ2_0, MXFP4/NVFP4, F8_E4M3
- **Implementation**: the `mul_mm*` family picks a pipeline by weight type; shared unpacking snippets live in `dequant_funcs{,_cm2}.glsl`, with 26 standalone `dequant_<format>.comp` kernels; layouts are defined in `ggml-quants.c`
- **Optimization techniques**:
  - Using `read_first_invocation` or subgroup broadcasts to accelerate scale reads
  - Unpacking and dequantizing in shared memory to reduce global memory access
  - Using `uint` bulk reads for packed data to reduce instruction count
  - Compiling multiple kernel variants with specialization constants for hot paths
- **Performance**: Q4_0 is typically 2-4× faster than FP16 in bandwidth-bound scenarios

### ncnn

- **Quantization formats**: int8 symmetric/asymmetric quantization, supports per-channel
- **Implementation**: Dequantization fused into convolution and fully-connected layers
- **Feature**: Leverages hardware integer instructions (dp4a) for int8 computation

### TensorRT

- **Quantization formats**: INT8 quantization, supports per-tensor and per-channel
- **Calibration**: Requires a calibration dataset to determine scales (not simple max method)
- **Feature**: Uses saturating quantization instead of simple truncation

---

Next chapter (ch19) builds on this chapter's quantization understanding to implement Q4_0 and Q8_0 in actual LLM inference — using quantized weights in attention layers and feed-forward networks, comparing precision and speed differences against FP16 inference.