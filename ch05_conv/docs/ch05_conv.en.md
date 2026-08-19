# Chapter 5 · Convolution: Direct Convolution vs im2col+GEMM vs Winograd

> Goal: understand why convolution ultimately reduces to matrix multiplication. This chapter runs three paths—direct convolution, im2col+GEMM, and Winograd F(2×2, 3×3)—on the **same input/weights** and verifies that the results match.

---

## 5.1 The Role of Convolution in Neural Networks

The convolution layer (Conv) of a CNN is the workhorse of CV models (classification, detection, segmentation). Looking purely at FLOPs, convolution often accounts for over 90% of the entire network. Therefore, how well an inference engine optimizes convolution essentially determines the speed of a CV model.

## 5.2 Direct Convolution (The Most Straightforward)

`direct_conv.comp`: for each output pixel `(oy,ox,oc)`, iterate over the kernel `(ky,kx)` and the input channel `cin` and do multiply-accumulate. The code is easy to read at a glance, but it has two fatal problems:

- **Non-contiguous memory access**: the input is read in `(cin, iy, ix)` order, yet the window slides, so neighboring output pixels read input addresses with large strides, preventing coalesced access.
- **Low arithmetic intensity**: each input element read is multiplied and accumulated only once, so memory bandwidth saturates quickly while compute sits idle (the classic "bandwidth-bound" case).

So production-grade code almost never writes it this way.

## 5.3 im2col: Laying Out Convolution into a Matrix Multiply

The key idea of `im2col.comp`: pull the **local input window** corresponding to each output pixel into a vector (a column), and stack the windows of all output positions into a large matrix `col` (rows = output positions, columns = flattened kernel). Convolution then becomes equivalent to:

```
C = col( M×KK )  ×  W_reshape( KK×Cout )     where KK = Cin*K*K
```

This is exactly what `gemm_naive.comp` does. In this chapter's main:

1. `im2col` computes the `256×9` column matrix;
2. reshape the weights `w[oc][k]` into `B[k][oc]` (transpose it);
3. a single GEMM produces the `256×2` output, then compare element-wise against the direct convolution—**the results match**.

> Why is im2col better? Because it reduces convolution to a GEMM, and GEMM has extremely mature tiling/fusion/quantization optimizations (ch04, ch11). The cost is that the `col` matrix is **KK times larger** than the original input, consuming extra VRAM/bandwidth—which is why better subsequent approaches exist, such as Winograd and indirect GEMM (which doesn't really expand, but fetches data on demand inside the GEMM kernel).

## 5.4 Winograd F(2×2, 3×3) Convolution

`winograd_filter_transform.comp` / `winograd_transform_input.comp` / `winograd_gemm.comp` / `winograd_transform_output.comp`: Winograd is a class of convolution acceleration algorithms that **trade multiplications for additions**, and it is particularly effective for 3×3 convolutions (the workhorse of VGG, ResNet, etc.).

### Core Idea

For F(2×2, 3×3), each 2×2 output tile is produced from a 4×4 input tile through the following transforms:

```
d (4×4 input)    ──BT @ d @ B──→  V (4×4 Winograd domain)
g (3×3 kernel)   ──G @ g @ G^T──→  U (4×4 Winograd domain)
M = V ⊙ U        (element-wise multiply, 4×4)
Y = AT @ M @ A   (4×4 Winograd domain → 2×2 output)
```

### Transform Matrices

```
B^T = [[1,  0, -1,  0],      G = [[1,    0,   0 ],
       [0,  1,  1,  0],           [0.5,  0.5, 0.5],
       [0, -1,  1,  0],           [0.5, -0.5, 0.5],
       [0,  1,  0, -1]]           [0,    0,   1 ]]

A^T = [[1,  1,  1,  0],
       [0,  1, -1, -1]]
```

### Why Is It Faster?

A standard 3×3 convolution requires 9 multiplications per output point, so 4 output points need 36 multiplications. Winograd needs only 16 multiplications (4×4 element-wise multiply), with the extra additions being essentially free on the GPU. This yields approximately a 2.25× reduction in compute.

### Implementation in This Chapter

1. `winograd_filter_transform.comp`: Computes U = G @ g @ G^T for each (oc, cin) on the GPU
2. `winograd_transform_input.comp`: Transforms each 4×4 input tile into the Winograd domain
3. `winograd_gemm.comp`: Element-wise multiply-accumulate in the Winograd domain, summing over cin
4. `winograd_transform_output.comp`: Transforms from the Winograd domain back to a 2×2 output

The main submits the pipeline in the order: filter_transform → input_transform → gemm → output_transform, and verifies the result against direct convolution element by element.

### Comparison with ncnn

ncnn's Winograd implementation (`convolution_winograd.cpp`) goes further with extreme optimizations:
- **NC4HW4 packing**: packs 4 channels into a `vec4`, computing 4 channels with one SIMD operation
- **Multi-tile merging**: merges multiple 4×4 tiles into a single large matrix multiply, increasing arithmetic density
- **F(6×6, 3×3)**: larger tiles further reduce multiplications (but require higher numerical precision)
- **Fused output transform**: fuses the output transform with bias addition / activation, reducing memory bandwidth

The implementation in this chapter is pedagogically equivalent to a "single-channel, single-tile" version, without packing / fusion / multi-tile merging, but the core mathematics is the same.

## 5.5 NC4HW4 and Data Packing (Important Optimization, Concept Only in This Chapter)

A real engine won't use something as naive as "one float per element." ncnn uses the **NC4HW4** layout: packs channels in groups of 4 into a `vec4`, so that:

- one memory access reads 4 channels, gaining both coalesced access and vector instructions;
- the convolution kernel is packed accordingly, and the GEMM inner loop computes 4 output channels at once.

llama.cpp also does `elempack` packing in similar embedding/convolution scenarios. This chapter still uses the plain NCHW layout for clarity; understanding "packing" is the key to reading ncnn's source code.

## 5.6 Hands-on Experiments

1. Change `Cout` to 4 or 8, rerun, and confirm the two paths still match (note that the N of `pcGemm` and the reshape loop must change accordingly—the code is already parameterized).
2. Change `K` to 5 (odd) and confirm `pad = K/2 = 2` takes effect automatically; what happens if you try K=4 (even)? (the same-padding definition needs adjustment—left as an exercise).
3. Use the timing method from ch09 to compare the time cost of "direct convolution" vs "im2col+GEMM" on larger feature maps (e.g. 64×64, Cout=32)—you will see the GEMM path overtake at moderate sizes.
4. Add the Winograd path to the comparison on the large feature maps above, and observe whether Winograd can further reduce latency.

## 5.7 Comparison with Industrial-Grade

- **ncnn**: convolution is its most deeply optimized part, providing a series of kernels prefixed `convolution_*`: `direct` / `winograd` / `im2col_gemm` / `blob_pack`, etc., dispatched by kernel size, whether 1×1, and whether depthwise. A 1×1 convolution degenerates directly into a GEMM; 3×3 commonly uses Winograd to reduce multiplications.
- **llama.cpp / ggml-vulkan**: convolution is rarer in LLMs, but the convolution idea (im2col+matmul) is reused in some vision/audio models; its `mul_mm` likewise handles this kind of "flattened matrix multiply."

In the next chapter (ch06) we turn our attention to **element-wise operators** (ReLU/GELU/SiLU)—individually simple, yet they are the typical "bandwidth-bound" case, leading to the performance-critical topic of "operator fusion."
