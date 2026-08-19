# 第 20 章：前沿量化（K-quants / MXFP4）

## 概述

本章深入探索量化技术的前沿：**K-quant 系列**（Q2_K 到 Q6_K）和 **MXFP4 微观缩放格式**。这些格式代表了当前 LLM 推理中量化技术的最新进展，在压缩比和模型精度之间取得了更好的平衡。

## K-quant 系列：从 Q4_0/Q8_0 到 Q2_K ~ Q6_K

### 为什么需要 K-quants？

在第 11 章和第 18 章中，我们学习了 Q4_0 和 Q8_0 格式。这些格式使用**固定大小的 block**（32 元素/block），每个 block 有独立的 fp16 scale。这种方法简单有效，但存在以下局限：

1. **量化粒度太粗**：32 个元素共享一个 scale，对于权重分布不均匀的层，精度损失较大
2. **不同层的敏感性不同**：有些层需要更高的精度（如 attention 的输出层），有些层可以容忍更低的精度
3. **没有考虑 outlier**：LLM 权重中经常出现 outlier，固定 block 大小难以有效处理

K-quant 系列通过引入**super-block 结构**和**子 block 共享量化尺度**来解决这些问题。

### Super-block 结构

K-quant 的核心创新是**两级量化结构**：

```
Super-block (256 元素)
├── Sub-block 0 (16 元素) — 6-bit scale + 量化值
├── Sub-block 1 (16 元素) — 6-bit scale + 量化值
├── ...
└── Sub-block 15 (16 元素) — 6-bit scale + 量化值
```

- **Super-block level**：存储全局 fp16 scale（d）和最小 scale（dmin）
- **Sub-block level**：每个 sub-block 存储一个 6-bit 量化 scale（0-63）
- 实际使用的 scale = d × (6-bit scale)

这种两级结构的好处：
- **更精细的粒度**：每 16 个元素就有独立的 scale，减少了量化误差
- **共享开销**：6-bit scale 比 fp16 scale 节省空间（6-bit vs 16-bit）
- **灵活适应**：不同 sub-block 可以用不同的精度级别

### K-quant 家族成员

| 格式 | 位宽 | Super-block 大小 | 子 block 大小 | 特点 |
|------|------|-----------------|--------------|------|
| Q2_K | 2-bit | 256 | 16 | 极高压缩比，适合大模型 |
| Q3_K | 3-bit | 256 | 16 | 精度与压缩的平衡点 |
| Q4_K | 4-bit | 256 | 32 | 接近 Q4_0 但精度更高 |
| Q5_K | 5-bit | 256 | 32 | 接近无损的量化 |
| Q6_K | 6-bit | 256 | 16 | 几乎无损，压缩比适中 |

### 本课程实现的格式

在本章中，我们实现了两种 K-quant 格式：

**Q2_K**：
- 2-bit 值（范围 0-3，中心化到 -1, 0, 1, 2）
- 每个 sub-block 共享 6-bit scale
- 压缩比：~16x（vs FP32）
- 适用场景：大模型的嵌入层、FFN 中间层

**Q6_K**：
- 6-bit 值（范围 -32 到 31）
- 每个 sub-block 共享 6-bit scale
- 6-bit 值打包：4 个值占 3 字节
- 压缩比：~5.3x（vs FP32）
- 适用场景：需要高精度的层

## MXFP4 微观缩放格式

### OCP 标准

MXFP4（Microscaling FP4）是 **Open Compute Project (OCP)** 提出的微观缩放格式标准。与传统 block 量化不同，MXFP4 使用**微观缩放**（per-tensor 或 per-block 的共享 scale）配合极低精度的浮点格式。

### E2M1 格式

MXFP4 使用 4-bit E2M1 格式：

```
bit 3: 符号位 (sign)
bit 2-1: 指数位 (exponent), bias = 1
bit 0: 尾数位 (mantissa)
```

可表示的值（正数）：

| 二进制 | 指数 | 尾数 | 值 |
|--------|------|------|-----|
| 0000 | 0 | 0 | 0.0 |
| 0001 | 0 | 1 | 0.25 |
| 0010 | 1 | 0 | 1.0 |
| 0011 | 1 | 1 | 1.5 |
| 0100 | 2 | 0 | 2.0 |
| 0101 | 2 | 1 | 3.0 |
| 0110 | 3 | 0 | 4.0 |
| 0111 | 3 | 1 | 6.0 |

加上符号位后，可表示 -6.0 到 6.0 的范围。

### 微观缩放

MXFP4 的完整数据格式为：

```
[shared fp16 scale] + [N × 4-bit E2M1 值]
```

- 共享 scale 可以是 per-tensor、per-row 或 per-block
- 在本章中，我们使用 per-block（32 元素）共享 scale
- 实际值 = E2M1_decode(value) × scale

### NVFP4：NVIDIA 的变体

NVIDIA 在 Blackwell 架构（RTX 50 系列）中引入了 NVFP4 格式，与 OCP MXFP4 非常相似但有以下区别：

- **NVFP4**：使用 E2M1 格式，但 scale 共享方式不同（per-group of 16）
- **硬件支持**：Blackwell 的 Tensor Core 可以直接处理 NVFP4 输入
- **训练友好**：NVIDIA 的推荐是在训练中使用 FP8，推理中使用 NVFP4

## 与 ggml 实现的比较

### K-quants in ggml

ggml（llama.cpp 使用的底层库）是 K-quant 格式的起源。我们的实现与 ggml 的差异：

| 方面 | ggml 实现 | 本课程实现 |
|------|-----------|-----------|
| 量化算法 | 大规模校准数据集优化 | 简单 min-max 量化 |
| 6-bit scale 打包 | 复杂位打包（高 6 位/低 6 位） | 简化存储（uint8） |
| 反量化 | 优化汇编/NEON/CUDA | GLSL 着色器 |
| 精度 | 更高（校准优化） | 足够教学演示 |

### MXFP4 实现

我们的 MXFP4 实现是 OCP 标准的简化版。完整的 OCP MXFP4 规范还包含：

- **E8M0 scale 格式**：8 位指数格式的 scale（无尾数）
- **分组策略**：不同粒度（per-tensor, per-row, per-block）
- **融合运算**：MXFP4 矩阵乘可以直接在硬件上实现

## 量化技术的未来

### 4-bit 及以下

量化技术正朝着更低位宽发展：

1. **NF4 (NormalFloat4)**：QLoRA 使用的 4-bit 格式，基于分位数量化
2. **FP4 (E2M1)**：本章实现的 MXFP4 格式
3. **2-bit 量化**：Q2_K 和更激进的 2-bit 方法
4. **混合精度量化**：不同层使用不同精度
5. **动态量化**：根据输入数据动态调整量化参数

### 硬件趋势

- **NVIDIA Blackwell**：原生支持 NVFP4 和 FP4 Tensor Core
- **AMD CDNA**：计划支持 MXFP4
- **Intel**：关注 OCP MX 标准
- **移动端**：高通和联发科加入 MX 标准联盟

## 练习

1. **实现 Q3_K**：在 Q2_K 的基础上，将 2-bit 值改为 3-bit 值，调整打包方式
2. **比较不同 block 大小**：将 MXFP4 的 block 大小从 32 改为 64 和 128，观察精度变化
3. **实现 per-row MXFP4**：将 MXFP4 的 scale 粒度从 per-block 改为 per-row
4. **混合精度实验**：对权重矩阵的不同部分使用不同的量化格式（如 Q4_0 和 Q6_K 混合）
5. **实现对称量化变体**：将 Q2_K 的中心化方案改为对称量化（去掉 -1 偏移）
6. **添加校准数据集**：使用少量校准数据优化 K-quant 的 scale 选择