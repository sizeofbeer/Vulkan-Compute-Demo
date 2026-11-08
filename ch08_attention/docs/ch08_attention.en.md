# Chapter 8 · Softmax and Attention (Transformer Core)

> Goal: Assemble the building blocks from previous chapters (matrix multiply, reduction) into a **simplified self-attention**. You will see that attention is simply "two matrix multiplies with a softmax in between" — which is exactly why matmul optimization on the GPU can directly accelerate large models.

---

## 8.1 The Math of Single-Head Attention

```
S = Q · Kᵀ / √D        # scores, Q,K ∈ ℝ^{N×D}
P = softmax(S)         # normalize each row into a probability
O = P · V              # context, O ∈ ℝ^{N×D}
```

`N` is the sequence length and `D` is the head dimension. The whole pipeline is **two matrix multiplies plus a row-wise softmax**. The various optimizations of Flash Attention essentially avoid writing the huge `S`(N×N) and `P` back to memory in full, instead normalizing on-chip as they are computed — but the math is equivalent.

## 8.2 Wiring Up Three Kernels

In this chapter, `main.cpp` splits attention into 3 independent kernels, chained together with `vkSubmit` (which includes an implicit barrier/fence):

1. **`attn_scores.comp`**: each thread computes one score `(qi, ki)` as `Σ_d Q[qi,d]·K[ki,d]/√D` — this is "one element" of the ch04 matrix multiply.
2. **`softmax.comp`**: performs a **numerically stable** softmax row-wise (subtract the row maximum first to prevent `exp` overflow, then normalize). The `max` and `sum` reductions reuse the shared-memory tree reduction from ch07. One workgroup per row (`N≤64`).
3. **`attn_context.comp`**: each thread computes one `(qi, d)` as `Σ_k P[qi,k]·V[k,d]` — another matrix multiply.

> Why numerical stability matters: `exp(large number)` overflows to `inf`, causing softmax to become all `nan`. After subtracting the row maximum, the largest exponent becomes 0, so `exp` never overflows and the result is unchanged (softmax is invariant to a global shift).

## 8.3 Why Attention Is "Expensive"

The intermediate score matrices `S` and `P` are `N×N`. When the sequence length `N=4096`, `S` has 16 million elements (64MB, FP32) — and attention must compute **many heads**. This is the root cause of the memory explosion in long-context inference, and why Flash Attention (which does not materialize S/P) has become standard.

## 8.4 Hands-On Experiments

1. Change `N` to 128 (note the `softmax.comp` assumptions of `local_size_x=64` and `N≤64` — you would need to change it to 128 accordingly and adjust the reduction stride or split it into two passes).
2. Add a "scale" observation: remove `/√D` and see the absolute scores grow and softmax become "sharper"; check whether it affects the result (slightly, numerically).
3. (Advanced) Implement a prototype of Flash Attention: do **online softmax while computing scores** inside `attn_scores`, directly producing `O` and avoiding materializing the N×N `S`/`P`. This is the core idea of the `flash_attn` family in llama.cpp.

## 8.5 Compared with Industrial Implementations

- **llama.cpp / ggml-vulkan**: attention is the hottest path in LLM inference. It provides several implementations:
  - Scalar Flash Attention (for short sequences);
  - Flash Attention based on **cooperative matrix** (coopmat, approximating Tensor Cores);
  - NVIDIA-specific **coopmat2** path.
  The common goal is "do not write the N×N score matrix back to memory". The 3-kernel version in this chapter is exactly the "naive attention" that does not perform this optimization, for ease of understanding.
- **ncnn**: attention in CV models (e.g., ViT, MobileViT) is also split into matmul+softmax+matmul, and fused as much as possible.
- **Synced 2026-11**: production FlashAttention is no longer "one `flash_attn.comp`". ncnn's `shader/` has `sdpa_fa.comp` / `sdpa_fa_cm.comp` / `sdpa_cross*.comp` / `sdpa_kvcache_{append,copy}.comp`; ggml's `vulkan-shaders/` has `flash_attn{,_cm1,_cm2}.comp`, `flash_attn_decode_phase_{1,2}.comp`, `flash_attn_split_k_reduce.comp` and `flash_attn_sparse_compact.comp`. Splitting prefill and decode into distinct kernels is now the mainstream design; see ch17.

## 8.6 FlashAttention Fused Kernel

In addition to the original 3-kernel naive attention, this chapter now includes a **FlashAttention fused kernel** demonstration. The corresponding shader is `shaders/flash_attn.comp`.

### 8.6.1 Motivation

The 3-kernel naive attention writes the N×N score matrix `S` and probability matrix `P` to memory in full, then reads them back. When the sequence length N is large, this creates a severe memory bandwidth bottleneck. The core idea of FlashAttention is to **never materialize the N×N intermediate matrices**, instead computing scores, softmax, and output accumulation on-chip (in registers), writing back only the O(N×d) result.

### 8.6.2 Algorithm: Online Softmax

Standard softmax requires knowing all scores before normalization. Online softmax updates incrementally without knowing all scores in advance:

```
Initialize: m = -inf, d_sum = 0, O = 0

For each key position k:
  score = Q[q] · K[k] / √d
  m_new = max(m, score)
  rescale = exp(m - m_new)        // rescaling factor (≤1)
  p = exp(score - m_new)
  d_sum = d_sum * rescale + p
  O = O * rescale + p * V[k]
  m = m_new

After loop: O = O / d_sum
```

Key properties:
1. **Numerically stable**: always subtracts the current maximum, `exp` never overflows
2. **Mathematically equivalent**: the final result is identical to standard softmax
3. **Incremental**: softmax is completed without storing all scores

### 8.6.3 Shader Design

`flash_attn.comp` uses the following workgroup structure:
- `local_size_x = 32`: one subgroup per workgroup, 32 threads
- Each workgroup handles 1 query row (Br = 1)
- Each thread accumulates d/32 output dimensions
- K/V tile size Bc = 32

**State variables** (per-thread, in registers):
- `q_reg[elemsPerThread]`: local Q elements
- `m`: running max
- `d_sum`: running exponential sum (denominator)
- `o_acc[elemsPerThread]`: output accumulator

**Dot product reduction** uses `subgroupAdd` across 32 threads in a single step, requiring the `GL_KHR_shader_subgroup_arithmetic` extension.

### 8.6.4 Comparison with 3-Kernel Approach

The new code in `main.cpp`:
1. Preserves the original 3-kernel naive attention (for pedagogical understanding)
2. Adds a FlashAttention demonstration with GPU timestamp-based timing
3. Verifies that FlashAttention output matches the naive attention output
4. Prints timing comparison between both approaches

FlashAttention advantages:
- **Memory**: does not materialize N×N intermediate matrices, O(N) instead of O(N²)
- **Bandwidth**: reduces global memory reads/writes, higher arithmetic intensity
- **Latency**: single kernel replaces three kernels, reducing kernel launch overhead

In the next chapter (ch09) we formally enter **performance analysis**: use GPU hardware timestamps to measure time precisely, and apply the Roofline model to determine whether the bottleneck is bandwidth or compute.
