# Chapter 7 · Normalization (RMSNorm / LayerNorm)

> Goal: learn how to do **cross-element cooperation** on the GPU — use an "in-workgroup shared-memory reduction" to compute the statistics of one row of features, then normalize element-wise. This is the core of RMSNorm/LayerNorm in Transformers.

---

## 7.1 Why Normalization Needs "Cooperation"

Element-wise operators (ch06) each compute independently, with no dependencies. But normalization first needs the statistics (mean/variance/RMS) of the **entire row**, then uses them to normalize every element — this requires the threads of the same row to first "aggregate" and then "broadcast". On the GPU this intra-group cooperation is achieved via **shared memory + barrier**.

RMSNorm formula (used by models such as LLaMA):
```
rms = sqrt( mean_i(x_i^2) )
y_i = x_i / rms * gamma_i
```
(LayerNorm adds a mean-subtraction step: `y_i = (x_i - mean)/std * gamma + beta`; just compute one extra mean during the reduction.)

## 7.2 Shared-Memory Reduction

The idea of `rmsnorm.comp`: each workgroup is responsible for **one row**:

1. 256 threads, each computes the square of its own feature and writes it into the shared array `ssum[tid]`; `barrier()`.
2. **Tree reduction**: the step `s` halves from 128 down to 1; in each round, threads with `tid < s` do `ssum[tid] += ssum[tid+s]`, then `barrier()`. After the 32 → 16 → … → 1 rounds, `ssum[0]` holds the full sum of squares.
3. All threads read the same `rms = sqrt(ssum[0]/F)`, then each normalizes and multiplies by gamma.

> Why use "tree reduction + barrier" instead of letting thread 0 accumulate alone? Because 256 threads writing to `ssum[0]` simultaneously would cause a data race; reduction makes only half the threads write in each round, and the `barrier()` before writing guarantees the previous round's result is visible. This is the standard GPU reduction pattern (similar to the barrier in ch02, except it happens inside the workgroup).

## 7.3 Limitations and Extensions

This teaching version assumes `F ≤ 256` (the thread count of one workgroup). In real models `F` may be 4096, 8192:

- **Multi-pass reduction**: each thread first processes several elements to form a local sum, then performs the tree reduction;
- or use **warp/subgroup primitives** (`subgroupAdd`) to accelerate further — llama.cpp uses it for reduction on devices that support `subgroup`, which is faster than shared memory.

These are only touched upon in the docs; the core is to understand the "aggregate first, then broadcast" pattern.

## 7.4 Hands-on Experiments

1. Change `F` to 128 and rerun — the `tid < F` branch makes the extra threads idle, and the result is still correct.
2. (Advanced) Switch to LayerNorm: also compute `sum(x)` during reduction to obtain the mean, replace the normalization with `(x - mean)/std * gamma + beta`, and update the CPU reference accordingly.
3. (Advanced) Expand `F` to 1024: each thread loops over 4 elements to form a local sum (`for(uint j=tid;j<F;j+=256)`), then performs the tree reduction.

## 7.5 Comparison with Industrial Implementations

- **ncnn**: `Normalize`, `InstanceNorm`, etc. are done via two kernels — first a reduction to compute mean/variance (within the workgroup), then normalization; or use `pack` to pack channels for acceleration.
- **llama.cpp / ggml-vulkan**: the `rmsnorm` / `layernorm` kernels use shared-memory reduction to compute statistics; on devices supporting `VK_KHR_shader_subgroup_*`, they use `subgroupAdd` to converge the reduction into a single warp operation, which is significantly faster than pure shared memory. The weight `gamma` is dequantized from the quantized model before being passed in.

In the next chapter (ch08) we move on to the attention mechanism: a numerically stable softmax (another reduction), and chaining QKᵀ, softmax, and PV into a simplified attention.
