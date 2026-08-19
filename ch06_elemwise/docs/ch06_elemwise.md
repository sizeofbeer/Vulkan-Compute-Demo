# 第 6 章 · 逐元素算子（ReLU / GELU / SiLU）

> 目标：认识一类「计算极轻、却被带宽卡死」的算子，并由此引出深度学习推理最重要的性能手段之一——**算子融合（fusion）**。

---

## 6.1 逐元素算子是干什么的

激活函数（ReLU/GELU/SiLU）对每个元素**独立**做同一运算，不依赖邻居。除了激活，BN/加偏置/LayerNorm 的逐元素部分也属此类。它们占的 FLOPs 极少，但有一个共同特征：**每个元素都要先从显存读、再写回显存**。

## 6.2 为什么它们是「带宽受限」

回顾 Roofline 模型（ch00 概念、ch09 量化）：算子的**算术强度** = 计算量 / 访存量。

- 逐元素算子：每读 4 字节、写 4 字节（8 字节访存），只做 1 次乘加（~2 FLOP）。算术强度 ≈ 0.25 FLOP/字节。
- 现代 GPU 带宽 ~1 TB/s、算力 ~10 TFLOP/s → 平衡点 ~10 FLOP/字节。0.25 远低于它，于是**算力全程闲置，瓶颈在显存带宽**。

`elemwise.comp` 一行就能看懂，真正的优化不在这行代码，而在「少搬几次数据」。

## 6.3 算子融合：把多个逐元素算子并成一个

经典套路：`Conv/MatMul -> 加偏置 -> 激活`。若拆成三个 kernel，数据要在显存里「写回→读出」两次；**融合**成一个 kernel，数据只在寄存器/共享内存里流转一次：

```
// 融合前（3 次访存往返）
C = matmul(A,B);  C += bias;  Y = gelu(C);

// 融合后（1 次写出）
Y = gelu(matmul(A,B) + bias);   // 在 matmul 的累加/收尾阶段顺手做
```

这就是 ncnn `convolution` 末尾直接接 `relu`/`sigmoid`，以及 llama.cpp 把 `add+bias+act` 融进 `mul_mm` 收尾的原因。本章虽未直接写融合 kernel，但请记住这个动机——ch12 的图执行会再次用到。

## 6.4 参数化 kernel 与 push constant 传枚举

`elemwise.comp` 用 `mode` 这个 push constant 在**一个 shader** 里切换三种激活，避免复制三份几乎相同的代码。这是常见技巧：形状/枚举类小参数走 push constant，大块数据走描述符。注意 GLSL 里 `mode` 是 `uint`，比较用 `== 0u`。

## 6.5 动手实验

1. 把 `N` 调到 `1<<22`（400 万），对比 ch04 的 GEMM 与本章逐元素在同一规模下的「每字节产出」——直观感受带宽受限。
2. 加一个 `mode==3` 的 `LeakyReLU`（负值乘 0.01），只需改 shader 的一个分支，无需动 C++ 结构。
3. （进阶）把 `elemwise` 与 ch04 的 GEMM 融合：在 `gemm_tiled.comp` 收尾 `c[...] = acc` 处，先加 bias 再套 gelu，做成一个「带偏置激活的融合矩阵乘」，用 ch09 计时对比分离写法的延迟。

## 6.6 对照工业级

- **ncnn**：`relu`、`sigmoid`、`gelu` 等是 `Elementwise` 系列；卷积/全连接 kernel 末尾常直接内联激活，权重加载阶段就把 `bias` 融进去。
- **llama.cpp / ggml-vulkan**：`mul_f32`/`mul_mm` 的收尾阶段通过 `post_op`（如加偏置、RMSNorm 的逐元素部分）融合；SiLU/GELU 作为独立 `vk_op` 存在，但在 transformer block 里尽量与前一个 matmul 合并提交。

下一章（ch07）我们处理需要**跨元素协作**的算子——归一化（RMSNorm/LayerNorm），用「工作组内归约」算出均值/方差。
