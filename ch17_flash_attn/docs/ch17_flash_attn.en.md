# Chapter 17 · FlashAttention: Fused Attention with Online Softmax

> Goal: Understand the O(N²) memory complexity bottleneck of standard attention, and implement **FlashAttention**—a fused kernel that reduces memory complexity to O(N) through tiling and online softmax. This is the core optimization for long-context LLM inference.

---

## 17.1 The Problem: Why Attention is Expensive

Standard scaled dot-product attention:

```
S = Q · Kᵀ / √d          # N×N score matrix
P = softmax(S, dim=-1)    # N×N probability matrix (row-wise)
O = P · V                 # N×d output
```

The intermediate matrices `S` and `P` are **N×N**. As sequence length N grows, memory usage grows O(N²):

| N | S/P size (FP32) | Memory/head |
|---|----------------|-------------|
| 512 | 512×512 = 262K | 1 MB |
| 1024 | 1024×1024 = 1M | 4 MB |
| 4096 | 4096×4096 = 16M | 64 MB |
| 32768 | 32768×32768 = 1G | 4 GB |

For 32 heads at 32K sequence length, attention scores alone would require 128 GB of VRAM—far exceeding any single GPU's capacity. This is the problem **FlashAttention** solves.

### Memory vs Compute

Attention's **compute** is also O(N²·d), but modern GPUs have sufficient compute throughput. The real bottleneck is **memory bandwidth**: writing the N×N score matrix to VRAM and reading it back is much slower than computing it.

FlashAttention's key insight: **do not materialize the N×N intermediate matrix**. Instead, compute scores, softmax, and accumulate output on-chip (registers and shared memory), writing back only the O(N×d) result.

---

## 17.2 FlashAttention Core Algorithm

### 17.2.1 Tiling Strategy

FlashAttention partitions Q, K, V into tiles:

- **Q tile**: `Br × d` (typically Br = 1, processing one query row at a time)
- **K/V tile**: `Bc × d` (typically Bc = 32 or 64, matching a subgroup size)

The outer loop iterates over K/V tiles, and the inner loop processes each key position within a tile.

### 17.2.2 Online Softmax

Standard softmax requires knowing all scores before normalization:
```
m = max(scores)          # find max first
p_i = exp(s_i - m)       # then exponentiate
sum = Σ p_i              # then sum
softmax_i = p_i / sum    # finally normalize
```

**Online softmax** can update incrementally without knowing all scores in advance:

```
Initialize: m = -inf, d = 0, O = 0

For each key position k:
  score = Q[q] · K[k] / √d
  m_new = max(m, score)
  rescale = exp(m - m_new)        # rescaling factor (≤1)
  p = exp(score - m_new)
  d = d * rescale + p
  O = O * rescale + p * V[k]
  m = m_new

After all keys: O = O / d
```

Key properties:
1. **Numerically stable**: always subtracts the current max, `exp` never overflows
2. **Mathematically equivalent**: final result is identical to standard softmax
3. **Incremental**: softmax can be computed without storing all scores

### 17.2.3 Numerical Derivation

Why is online softmax equivalent to standard softmax?

Standard softmax output:
```
O_std = Σ_k exp(s_k - m_global) · V[k] / Σ_k exp(s_k - m_global)
```

where `m_global = max(all scores)`.

Online softmax maintains a running maximum m. After processing the j-th key:
```
m_j = max(s_1, s_2, ..., s_j)
d_j = Σ_{i=1}^{j} exp(s_i - m_j)
O_j = Σ_{i=1}^{j} exp(s_i - m_j) · V[i]
```

When processing the (j+1)-th key, if `s_{j+1} > m_j`, all previous terms must be **rescaled**:
```
m_{j+1} = s_{j+1}
d_{j+1} = d_j · exp(m_j - m_{j+1}) + exp(s_{j+1} - m_{j+1})
O_{j+1} = O_j · exp(m_j - m_{j+1}) + exp(s_{j+1} - m_{j+1}) · V[j+1]
```

This is equivalent to dividing all previous terms by `exp(m_{j+1} - m_j)` and then adding the new term.

Finally:
```
O = O_N / d_N = Σ_k exp(s_k - m_N) · V[k] / Σ_k exp(s_k - m_N)
```

This matches standard softmax because `m_N = m_global`.

---

## 17.3 Implementation Details

### 17.3.1 Shader Design: `flash_attn.comp`

```
Workgroup structure:
  local_size_x = 32    ← one subgroup
  Br = 1               ← one query row per workgroup
  Bc = 32              ← K/V tile size

Each thread's responsibility:
  - Handles d/32 output dimensions (columns)
  - For each K/V tile, computes partial dot product and reduces
  - Maintains online softmax state
```

**State variables** (per-thread, in registers):
- `q_reg[elemsPerThread]`: local Q elements
- `m`: running max
- `d_sum`: running sum of exp
- `o_acc[elemsPerThread]`: output accumulator

**Main loop**:
```
for each K/V tile (k_start = 0, 32, 64, ..., N-1):
  for each key in tile (k = k_start ... k_start+31):
    score = dot(Q[q_idx], K[k]) / √d     # subgroupAdd reduction
    m_new = max(m, score)
    rescale = exp(m - m_new)
    p = exp(score - m_new)
    d_sum = d_sum * rescale + p
    o_acc = o_acc * rescale + p * V[k]
    m = m_new
```

**Final normalization**:
```
O[q_idx][col] = o_acc[col] / d_sum
```

### 17.3.2 Subgroup Reduction

The dot product uses `subgroupAdd` for reduction across 32 threads:

```glsl
float partial = 0.0;
for (uint i = 0; i < elemsPerThread; ++i)
    partial += q_reg[i] * K[k_idx * d + colBase + i];
float score = subgroupAdd(partial) * inv_sqrt_d;
```

Each thread contributes d/32 elements of the partial dot product, and `subgroupAdd` accumulates all 32 threads' contributions in one step. This requires the `GL_KHR_shader_subgroup_arithmetic` extension.

### 17.3.3 Naive Reference: `flash_attn_naive.comp`

The naive version first computes all N scores (stored in shared memory `scores[128]`), then performs softmax, and finally accumulates V. This requires O(N) shared memory and is infeasible for large N.

```
Step 1: Compute all scores[q, k] = Q[q]·K[k] / √d
Step 2: Row-wise softmax (max → exp → sum → normalize)
Step 3: Compute O[q, d] = Σ_k P[q, k] · V[k, d]
```

---

## 17.4 Performance Analysis

### 17.4.1 Memory Complexity Comparison

| Method | Intermediate Memory | Extra R/W |
|--------|-------------------|-----------|
| Standard attention (3 kernel) | N×N (scores + probs) | 2 × N² elements |
| Naive single kernel | N (shared memory) | 0 (on-chip) |
| FlashAttention | O(1) (registers) | 0 (on-chip) |

### 17.4.2 Compute Comparison

Both methods have O(N²·d) compute, but FlashAttention has better memory access patterns:
- **Standard attention**: writes 2×N² elements to VRAM (scores + probs), then reads them back
- **FlashAttention**: only reads/writes Q, K, V, O (total 4×N×d elements), loads K/V tiles in blocks

For N=1024, d=64:
- Standard attention: extra 2×1024² = 2M elements = 8 MB VRAM R/W
- FlashAttention: no extra VRAM R/W

### 17.4.3 Arithmetic Intensity

FlashAttention's arithmetic intensity (FLOP / byte) is much higher than standard attention because each K/V tile loaded from global memory is reused across multiple query rows.

---

## 17.5 Industrial Comparison

### 17.5.1 ncnn's `sdpa_fa.comp`

ncnn's FlashAttention implementation (`sdpa_fa.comp`) uses a 2×2 unroll strategy, where each workgroup processes 2 query rows and 2 key positions simultaneously, using local memory for tile caching. It supports a cooperative matrix accelerated version (`sdpa_fa_cm.comp`) that leverages Tensor Core hardware for the matrix multiplication portion.

### 17.5.2 ggml (llama.cpp)'s `flash_attn.comp`

ggml's FlashAttention implementation uses a complete tiling strategy combined with subgroup operations, similar to this chapter's implementation. It provides multiple variants:
- **Scalar path**: basic tiling, suitable for small sequences
- **CM path**: cooperative matrix acceleration (`flash_attn_cm1.comp`)
- **CM2 path**: NVIDIA-specific coopmat2 acceleration (`flash_attn_cm2.comp`)
- **Split-K path**: partitions K dimension across workgroups, then reduces (for large sequences)

### 17.5.3 Comparison Summary

| Feature | This chapter | ncnn sdpa_fa | ggml flash_attn |
|---------|-------------|-------------|-----------------|
| Subgroup reduction | ✓ (subgroupAdd) | ✓ | ✓ |
| Shared memory K/V tile | ✗ (direct global read) | ✓ (local mem) | ✓ |
| Online softmax | ✓ | ✓ | ✓ |
| CM acceleration | ✗ | ✓ (KHR+NV) | ✓ (KHR+NV+CM2) |
| Split-K | ✗ | ✗ | ✓ |
| KV cache support | ✗ | ✓ | ✓ |

---

## 17.6 Hands-on Experiments

1. **Change tile size**: Change `Bc` from 32 to 64 (adjust thread count), observe performance changes. Larger tiles mean fewer global memory loads but more shared memory usage.

2. **Compare across sequence lengths**: Run `ch17_flash_attn` and observe the speedup ratio of FlashAttention vs naive attention at N=128, 256, 512, 1024. Theoretically, larger N gives FlashAttention a bigger advantage.

3. **Implement KV cache version**: Modify the shader so K and V are read from a separate KV cache buffer (supporting incremental decoding).

4. **Implement multi-query rows**: Change `Br` from 1 to 4 (each workgroup processes 4 query rows), use shared memory to cache the Q tile and reduce K/V reloads.

5. **Add causal mask**: Introduce a causal mask in the score computation, making `score[q,k] = -inf` when `k > q`.

6. **Compare with ch08 3-kernel attention**: Use GPU timestamps to compare the performance difference between FlashAttention and ch08's 3-kernel attention.

---

## 17.7 Chapter Summary

- Standard attention has O(N²) memory complexity, making it infeasible for long sequences
- **FlashAttention** reduces memory complexity to O(N) through tiling and online softmax
- **Online softmax** uses a running maximum and rescaling factor to achieve numerically stable softmax without storing all scores
- Subgroup operations (`subgroupAdd`) enable efficient warp-level dot product reduction
- This chapter's implementation is a simplified Br=1 version; real industrial implementations support multi-query row tiling, cooperative matrix acceleration, causal masks, and other extensions
- Industrial implementations (ncnn, ggml) extend FlashAttention with CM acceleration, Split-K, KV cache, and other features

The next chapter (ch18) will explore **Advanced Quantization**—from int8 to Q4_0/Q8_0/K-quant to MXFP4 GPU-side dequantization and fused inference.