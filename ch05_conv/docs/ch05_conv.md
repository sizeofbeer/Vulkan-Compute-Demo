# 第 5 章 · 卷积：直接卷积 vs im2col+GEMM vs Winograd

> 目标：搞懂卷积为什么最终都变成矩阵乘。本章用**同一组输入/权重**跑三条路——直接卷积、im2col+GEMM、Winograd F(2×2, 3×3)——并验证结果一致。

---

## 5.1 卷积在神经网络里的地位

CNN 的卷积层（Conv）是 CV 模型（分类、检测、分割）的主力。单看 FLOPs，卷积往往占整个网络 90% 以上。所以推理引擎对卷积的优化程度，基本决定了 CV 模型的快慢。

## 5.2 直接卷积（最直白）

`direct_conv.comp`：对每个输出像素 `(oy,ox,oc)`，遍历卷积核 `(ky,kx)` 和输入通道 `cin`，做乘加。代码一眼能看懂，但有两个致命问题：

- **访存不连续**：输入按 `(cin, iy, ix)` 读取，而窗口在滑动，相邻输出像素读的输入地址跳跃很大，无法合并访存。
- **计算密度低**：每读一个输入元素只乘加一次，带宽很快打满、算力闲置（典型「带宽受限」）。

所以工业级几乎不直接这么写。

## 5.3 im2col：把卷积「铺平」成矩阵乘

`im2col.comp` 的关键思想：把每个输出像素对应的**输入局部窗口**拉成一个向量（列），所有输出位置的窗口排成一个大矩阵 `col`（行=输出位置，列=核展开）。随后卷积就等价于：

```
C = col( M×KK )  ×  W_reshape( KK×Cout )    其中 KK = Cin*K*K
```

这正是 `gemm_naive.comp` 做的。本章 main 里：

1. `im2col` 把 `256×9` 的列矩阵算出来；
2. 把权重 `w[oc][k]` reshape 成 `B[k][oc]`（转置一下）；
3. 一次 GEMM 得到 `256×2` 输出，与直接卷积逐元素比对——**结果一致**。

> 为什么 im2col 更优？因为它把卷积化归为 GEMM，而 GEMM 有极其成熟的 tiling/融合/量化优化（ch04、ch11）。代价是 `col` 矩阵比原输入**大 KK 倍**，要吃显存/带宽——所以才有后续的 Winograd、间接 GEMM（不真的展开，而是在 GEMM 内核里按需取数）等更优方案。

## 5.4 Winograd F(2×2, 3×3) 卷积

`winograd_filter_transform.comp` / `winograd_transform_input.comp` / `winograd_gemm.comp` / `winograd_transform_output.comp`：Winograd 是一类**用加法换乘法**的卷积加速算法，在 3×3 卷积（VGG、ResNet 等的主力）上效果显著。

### 核心思想

对于 F(2×2, 3×3)，每个 2×2 输出块由 4×4 输入块经过以下变换得到：

```
d (4×4 输入)  ──BT @ d @ B──→  V (4×4 Winograd 域)
g (3×3 核)    ──G @ g @ G^T──→  U (4×4 Winograd 域)
M = V ⊙ U     (逐元素乘加，4×4)
Y = AT @ M @ A (4×4 Winograd 域 → 2×2 输出)
```

### 变换矩阵

```
B^T = [[1,  0, -1,  0],      G = [[1,    0,   0 ],
       [0,  1,  1,  0],           [0.5,  0.5, 0.5],
       [0, -1,  1,  0],           [0.5, -0.5, 0.5],
       [0,  1,  0, -1]]           [0,    0,   1 ]]

A^T = [[1,  1,  1,  0],
       [0,  1, -1, -1]]
```

### 为什么更快？

标准卷积 3×3 每输出点需要 9 次乘法，4 个输出点共 36 次乘法。Winograd 只需 16 次乘法（4×4 逐元素乘），加法的增加在 GPU 上基本免费。计算量减少约 2.25 倍。

### 本章实现

1. `winograd_filter_transform.comp`：GPU 上对每个 (oc, cin) 做 U = G @ g @ G^T
2. `winograd_transform_input.comp`：将每个 4×4 输入 tile 变换到 Winograd 域
3. `winograd_gemm.comp`：Winograd 域逐元素乘加，跨 cin 累加
4. `winograd_transform_output.comp`：从 Winograd 域变换回 2×2 输出

main 中按 filter_transform → input_transform → gemm → output_transform 顺序提交，结果与直接卷积逐元素比对验证。

### 对照 ncnn

ncnn 的 Winograd 实现（层代码 `src/layer/vulkan/convolution_vulkan.cpp`，着色器 `src/layer/vulkan/shader/convolution_*winograd*.comp`；首版写的 `convolution_winograd.cpp` 并不存在）做了更极致的优化：
- **NC4HW4 打包**：4 通道一组打包成 `vec4`，一次 SIMD 算 4 通道
- **多 tile 合并**：多个 4×4 tile 合并成一次大矩阵乘，提高计算密度
- **F(6×6, 3×3)**：更大的 tile 能进一步减少乘法（但数值稳定性要求更高）
- **融合输出变换**：输出变换与偏置加 / 激活函数融合，减少显存带宽
- **int8 / coopmat 变体（2026-11 新增）**：`convolution_3x3s1d1_winograd{23,43}_transform_input_int8_cm.comp`、`convolution_winograd_gemm_cm.comp`、`convolution_gemm_int8_cm.comp`、`convolution1d_*_cm.comp`（详见 ch15）

本章的实现在教学上等价于「单通道单 tile 版」，不涉及打包 / 融合 / 多 tile 合并，但核心数学一致。

## 5.6 NC4HW4 与数据打包（重要优化，本章仅讲概念）

真实引擎不会用「每元素一个 float」这么朴素。ncnn 用 **NC4HW4** 布局：把通道 4 个一组打包进一个 `vec4`，这样：

- 一次访存读 4 个通道，合并访存 + 向量指令双收益；
- 卷积核也相应打包，GEMM 内层一次算 4 个输出通道。

llama.cpp 在 embedding/卷积类似场景也做 `elempack` 打包。本章为清晰仍用 NCHW 平面布局；理解「打包」是看懂 ncnn 源码的钥匙。

## 5.6 动手实验

1. 把 `Cout` 改成 4、8，重跑，确认两路仍一致（注意 `pcGemm` 的 N 和 reshape 循环要跟着变——代码已参数化）。
2. 把 `K` 改成 5（奇数），确认 `pad = K/2 = 2` 自动生效；试 K=4（偶数）会怎样？（same padding 定义需调整，留作思考）
3. 用 ch09 的计时法对比「直接卷积」与「im2col+GEMM」在更大特征图（如 64×64、Cout=32）上的耗时——你会看到 GEMM 路径在中等尺寸下反超。
4. 在上述大特征图上再加入 Winograd 路径对比，观察 Winograd 是否能进一步降低延迟。

## 5.7 对照工业级

- **ncnn**：卷积是它优化最深的部分，提供 `convolution_*` 一系列 kernel：`direct` / `winograd` / `im2col_gemm` / `blob_pack` 等，并按卷积核尺寸、是否 1×1、是否 depthwise 分派。1×1 卷积直接退化成 GEMM；3×3 常用 Winograd 减少乘法。
- **llama.cpp / ggml-vulkan**：LLM 里卷积较少，但卷积思路（im2col+matmul）被复用在部分视觉/音频模型中；其 `mul_mm` 同样承接这类「铺平后的矩阵乘」。

下一章（ch06）我们把注意力转向**逐元素算子**（ReLU/GELU/SiLU）——它们单个很简单，却是「带宽受限」的典型，引出「算子融合」这个性能关键话题。
