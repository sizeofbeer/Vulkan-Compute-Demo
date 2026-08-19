# Chapter 6 · Elementwise Operators (ReLU / GELU / SiLU)

> Goal: understand a class of operators that are "extremely cheap to compute, yet bottlenecked by memory bandwidth," and from there introduce one of the most important performance techniques in deep-learning inference — **operator fusion**.

---

## 6.1 What Elementwise Operators Do

Activation functions (ReLU/GELU/SiLU) apply the same operation to each element **independently**, with no dependence on neighbors. Besides activations, the elementwise parts of BN / bias-add / LayerNorm also belong to this class. They consume very few FLOPs, but share one characteristic: **every element must first be read from device memory, then written back to device memory**.

## 6.2 Why They Are "Bandwidth-Bound"

Recall the Roofline model (concept in ch00, quantization in ch09): an operator's **arithmetic intensity** = computation / memory traffic.

- Elementwise operator: read 4 bytes, write 4 bytes (8 bytes of traffic), doing only one multiply-add (~2 FLOPs). Arithmetic intensity ≈ 0.25 FLOP/byte.
- Modern GPU bandwidth ~1 TB/s, compute ~10 TFLOP/s → balance point ~10 FLOP/byte. 0.25 is far below it, so **compute sits idle the whole time, and the bottleneck is device memory bandwidth**.

`elemwise.comp` is readable in one line; the real optimization is not in that line of code, but in "moving the data fewer times."

## 6.3 Operator Fusion: Combine Multiple Elementwise Operators Into One

Classic pattern: `Conv/MatMul -> bias-add -> activation`. If split into three kernels, the data must be "written back → read out" of device memory twice; **fusing** into a single kernel lets the data flow through registers / shared memory only once:

```
// Before fusion (3 round trips to memory)
C = matmul(A,B);  C += bias;  Y = gelu(C);

// After fusion (1 write-out)
Y = gelu(matmul(A,B) + bias);   // done inline at matmul's accumulate/finalize stage
```

This is why ncnn directly appends `relu`/`sigmoid` at the end of `convolution`, and why llama.cpp folds `add+bias+act` into the finalize stage of `mul_mm`. Although this chapter does not write a fused kernel directly, keep this motivation in mind — the graph execution in ch12 will revisit it.

## 6.4 Parameterized Kernel and Passing an Enum via Push Constant

`elemwise.comp` uses the `mode` push constant to switch between three activations within a **single shader**, avoiding three near-identical copies of code. This is a common trick: small shape/enum-style parameters go through push constants, while large data blocks go through descriptors. Note that in GLSL `mode` is `uint`, so comparisons use `== 0u`.

## 6.5 Hands-on Experiments

1. Set `N` to `1<<22` (4 million) and compare the "bytes produced per byte" of ch04's GEMM versus this chapter's elementwise operator at the same scale — to get an intuitive feel for bandwidth-bound behavior.
2. Add a `mode==3` `LeakyReLU` (negatives multiplied by 0.01); you only need to change one branch in the shader, without touching the C++ structure.
3. (Advanced) Fuse `elemwise` with ch04's GEMM: at the finalize point `c[...] = acc` in `gemm_tiled.comp`, first add bias then apply gelu, making a "fused matmul with bias activation," and use ch09's timing to compare against the separated approach.

## 6.6 Industrial Comparisons

- **ncnn**: `relu`, `sigmoid`, `gelu`, etc. are in the `Elementwise` family; convolution / fully-connected kernels often inline the activation directly at the end, and fold the `bias` in during the weight-loading stage.
- **llama.cpp / ggml-vulkan**: the finalize stage of `mul_f32`/`mul_mm` fuses via `post_op` (such as bias-add, the elementwise part of RMSNorm); SiLU/GELU exist as standalone `vk_op`s, but within a transformer block they are merged with the preceding matmul submission wherever possible.

In the next chapter (ch07) we tackle operators that need **cross-element cooperation** — normalization (RMSNorm/LayerNorm), computing mean/variance via "reduction within a workgroup."
