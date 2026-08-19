# Chapter 20: Frontier Quantization (K-quants / MXFP4)

## Overview

This chapter explores the frontier of quantization techniques: the **K-quant family** (Q2_K through Q6_K) and the **MXFP4 microscaling format**. These formats represent the latest advances in quantization for LLM inference, achieving a better balance between compression ratio and model accuracy.

## The K-quant Family: From Q4_0/Q8_0 to Q2_K ~ Q6_K

### Why K-quants?

In Chapters 11 and 18, we learned about Q4_0 and Q8_0 formats. These formats use **fixed-size blocks** (32 elements/block), each with its own fp16 scale. While simple and effective, this approach has limitations:

1. **Coarse quantization granularity**: 32 elements sharing a single scale leads to accuracy loss for layers with uneven weight distributions
2. **Different layer sensitivity**: Some layers need higher precision (e.g., attention output layers), while others can tolerate lower precision
3. **No outlier handling**: LLM weights frequently contain outliers that fixed block sizes handle poorly

The K-quant series addresses these issues through its **super-block structure** and **sub-block shared quantization scales**.

### Super-block Structure

K-quant's core innovation is a **two-level quantization structure**:

```
Super-block (256 elements)
├── Sub-block 0 (16 elements) — 6-bit scale + quantized values
├── Sub-block 1 (16 elements) — 6-bit scale + quantized values
├── ...
└── Sub-block 15 (16 elements) — 6-bit scale + quantized values
```

- **Super-block level**: Stores global fp16 scale (d) and minimum scale (dmin)
- **Sub-block level**: Each sub-block stores a 6-bit quantized scale (0-63)
- **Effective scale**: d × (6-bit scale)

Benefits of this two-level structure:
- **Finer granularity**: Every 16 elements has its own scale, reducing quantization error
- **Shared overhead**: 6-bit scale is more compact than fp16 (6-bit vs 16-bit)
- **Flexible adaptation**: Different sub-blocks can use different precision levels

### K-quant Family Members

| Format | Bit-width | Super-block | Sub-block | Characteristics |
|--------|-----------|-------------|-----------|-----------------|
| Q2_K | 2-bit | 256 | 16 | Extreme compression, good for large models |
| Q3_K | 3-bit | 256 | 16 | Balance between precision and compression |
| Q4_K | 4-bit | 256 | 32 | Better precision than Q4_0 |
| Q5_K | 5-bit | 256 | 32 | Near-lossless quantization |
| Q6_K | 6-bit | 256 | 16 | Almost lossless, moderate compression |

### Formats Implemented in This Chapter

**Q2_K**:
- 2-bit values (range 0-3, centered to -1, 0, 1, 2)
- 6-bit shared scale per sub-block
- Compression ratio: ~16x (vs FP32)
- Use case: Embedding layers, FFN intermediate layers

**Q6_K**:
- 6-bit values (range -32 to 31)
- 6-bit shared scale per sub-block
- 6-bit packing: 4 values in 3 bytes
- Compression ratio: ~5.3x (vs FP32)
- Use case: Layers requiring high precision

## MXFP4 Microscaling Format

### OCP Standard

MXFP4 (Microscaling FP4) is a microscaling format standard proposed by the **Open Compute Project (OCP)**. Unlike traditional block quantization, MXFP4 uses **microscaling** (shared per-tensor or per-block scales) combined with very low-precision floating-point formats.

### E2M1 Format

MXFP4 uses the 4-bit E2M1 format:

```
bit 3: sign
bit 2-1: exponent, bias = 1
bit 0: mantissa
```

Representable values (positive):

| Binary | Exponent | Mantissa | Value |
|--------|----------|----------|-------|
| 0000 | 0 | 0 | 0.0 |
| 0001 | 0 | 1 | 0.25 |
| 0010 | 1 | 0 | 1.0 |
| 0011 | 1 | 1 | 1.5 |
| 0100 | 2 | 0 | 2.0 |
| 0101 | 2 | 1 | 3.0 |
| 0110 | 3 | 0 | 4.0 |
| 0111 | 3 | 1 | 6.0 |

With the sign bit, the representable range is -6.0 to 6.0.

### Microscaling

The complete MXFP4 data format is:

```
[shared fp16 scale] + [N × 4-bit E2M1 values]
```

- Shared scale can be per-tensor, per-row, or per-block
- In this chapter, we use per-block (32 elements) shared scale
- Actual value = E2M1_decode(value) × scale

### NVFP4: NVIDIA's Variant

NVIDIA introduced NVFP4 in the Blackwell architecture (RTX 50 series). It is very similar to OCP MXFP4 but differs in:

- **NVFP4**: Uses E2M1 format with different scale sharing (per-group of 16)
- **Hardware support**: Blackwell's Tensor Core can directly process NVFP4 input
- **Training-friendly**: NVIDIA recommends FP8 for training, NVFP4 for inference

## Comparison with ggml Implementations

### K-quants in ggml

ggml (the library behind llama.cpp) originated the K-quant formats. Differences from our implementation:

| Aspect | ggml Implementation | Our Implementation |
|--------|--------------------|--------------------|
| Quantization algorithm | Calibration dataset optimization | Simple min-max quantization |
| 6-bit scale packing | Complex bit packing (high/low 6-bit) | Simplified storage (uint8) |
| Dequantization | Optimized assembly/NEON/CUDA | GLSL shader |
| Precision | Higher (calibration-optimized) | Sufficient for teaching |

### MXFP4 Implementation

Our MXFP4 implementation is a simplified version of the OCP standard. The full OCP MXFP4 specification includes:

- **E8M0 scale format**: 8-bit exponent-only scale format (no mantissa)
- **Grouping strategies**: Different granularities (per-tensor, per-row, per-block)
- **Fused operations**: MXFP4 matrix multiplication can be directly implemented in hardware

## The Future of Quantization

### 4-bit and Below

Quantization technology is advancing toward lower bit widths:

1. **NF4 (NormalFloat4)**: QLoRA's 4-bit format, based on quantile quantization
2. **FP4 (E2M1)**: The MXFP4 format implemented in this chapter
3. **2-bit quantization**: Q2_K and more aggressive 2-bit approaches
4. **Mixed-precision quantization**: Different layers at different precisions
5. **Dynamic quantization**: Adapting quantization parameters based on input data

### Hardware Trends

- **NVIDIA Blackwell**: Native support for NVFP4 and FP4 Tensor Core
- **AMD CDNA**: Planning MXFP4 support
- **Intel**: Following OCP MX standard
- **Mobile**: Qualcomm and MediaTek joining the MX standards alliance

## Exercises

1. **Implement Q3_K**: Extend Q2_K with 3-bit values, adjusting the packing scheme
2. **Compare different block sizes**: Change MXFP4 block size from 32 to 64 and 128, observing accuracy changes
3. **Implement per-row MXFP4**: Change MXFP4 scale granularity from per-block to per-row
4. **Mixed-precision experiment**: Use different quantization formats (e.g., Q4_0 and Q6_K mixed) for different parts of the weight matrix
5. **Implement symmetric quantization variant**: Change Q2_K's centering scheme to symmetric quantization (remove the -1 offset)
6. **Add calibration dataset**: Use a small calibration dataset to optimize K-quant scale selection